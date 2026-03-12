"""Integration tests for EnergySimulationEngine.

These tests exercise the full engine pipeline end-to-end using two minimal
but complete concrete devices:

``ConstantCostDevice``
    A non-genome device (genome_requirements returns None). Always returns
    a fixed scalar cost per individual per objective. Used to verify cost
    accumulation without genome complexity.

``ScheduleDevice``
    A genome-controlled device (one gene per step). Schedules power on a
    single port. Cost = sum of |power| across the horizon (total energy
    magnitude). Optionally clamps genome values to test repair detection.
    Uses a fixed device_index for arbitrator round-tripping.

Both devices participate in bus arbitration via a single AC bus.

Covers:
    Engine lifecycle
        - Construction validates topology
        - setup_run transitions CREATED → RUN_CONFIGURED
        - genome_requirements transitions RUN_CONFIGURED → STRUCTURE_FROZEN
        - evaluate_population requires STRUCTURE_FROZEN
        - setup_run raises in RUN_CONFIGURED state
        - genome_requirements raises outside RUN_CONFIGURED
        - evaluate_population raises outside STRUCTURE_FROZEN
        - setup_run from STRUCTURE_FROZEN resets to RUN_CONFIGURED (re-run)

    objective_names
        - Unavailable before setup_run (RuntimeError)
        - Available after setup_run
        - Correct names in correct order
        - Shared objective name from two devices appears once

    evaluate_population — fitness
        - Fitness shape is (population_size, num_objectives)
        - Fitness column order matches objective_names
        - Cost from single device accumulates correctly
        - Costs from two devices sharing an objective name are summed
        - Costs from two devices with different objectives go to separate columns
        - All-zero genome produces correct cost (zero energy magnitude)

    evaluate_population — repair detection
        - Feasible genome: repaired_genomes is empty
        - Infeasible genome: repaired_genomes contains device_id
        - Repair detection uses value comparison, not only identity
        - Two devices: only the repaired one appears in repaired_genomes

    evaluate_population — population independence
        - Each individual's fitness is computed independently
        - Population size 1 and population size 8 both work correctly

    EvaluationResult
        - objective_names matches engine.objective_names
        - repaired_genomes values have correct shape

NOTE — device_index contract
-----------------------------
DeviceRequest.device_index is set by the device's own build_device_request().
The engine does not inject indices. ScheduleDevice uses a constructor-assigned
device_index. This is a conscious design decision documented in engine.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from akkudoktoreos.devices.devicesabc import (
    EnergyBus,
    EnergyCarrier,
    EnergyPort,
    PortDirection,
    SingleStateBatchState,
    SingleStateEnergyDevice,
)
from akkudoktoreos.optimization.genetic2.genome import GenomeSlice
from akkudoktoreos.simulation.genetic2.arbitrator import (
    BusTopology,
    DeviceGrant,
    DeviceRequest,
    PortRequest,
    VectorizedBusArbitrator,
)
from akkudoktoreos.simulation.genetic2.engine import (
    EnergySimulationEngine,
    EngineState,
    EvaluationResult,
)
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.simulation.genetic2.topology import TopologyValidationError
from akkudoktoreos.utils.datetimeutil import to_datetime

# ============================================================
# Concrete test devices
# ============================================================

class ConstantCostDevice(SingleStateEnergyDevice):
    """Non-genome device that returns a fixed cost every generation.

    genome_requirements() returns None — this device is not genome-controlled.
    Cost = fixed_cost_per_step * horizon for every individual.
    Participates in the bus as a sink (consumer) to satisfy topology.
    """

    def __init__(
        self,
        device_id: str,
        device_index: int,
        objective: str,
        fixed_cost_per_step: float,
        bus_id: str = "bus_ac",
    ) -> None:
        super().__init__()
        self.device_id = device_id
        self._device_index = device_index
        self._objective = objective
        self._fixed_cost = fixed_cost_per_step
        self._bus_id = bus_id

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return (EnergyPort(
            port_id="p_sink",
            bus_id=self._bus_id,
            direction=PortDirection.SINK,
        ),)

    @property
    def objective_names(self) -> list[str]:
        return [self._objective]

    def genome_requirements(self) -> GenomeSlice | None:
        return None

    def initial_state(self) -> float:
        return 0.0

    def state_transition_batch(self, state, power, step_interval):
        return state + power * step_interval / 3600.0

    def power_bounds(self) -> tuple[float, float]:
        return (0.0, 100.0)

    def build_device_request(self, state: SingleStateBatchState) -> DeviceRequest | None:
        # Request fixed consumption across all individuals and steps
        pop_size = state.population_size
        horizon = state.horizon
        energy = np.full((pop_size, horizon), 10.0)   # 10 Wh per step
        min_e = np.zeros((pop_size, horizon))
        pr = PortRequest(port_index=0, energy_wh=energy, min_energy_wh=min_e)
        return DeviceRequest(device_index=self._device_index, port_requests=(pr,))

    def apply_device_grant(self, state, grant: DeviceGrant) -> None:
        pass

    def compute_cost(self, state: SingleStateBatchState) -> np.ndarray:
        # Fixed cost: same for all individuals
        cost_per_individual = self._fixed_cost * state.horizon
        return np.full((state.population_size, 1), cost_per_individual)


class ScheduleDevice(SingleStateEnergyDevice):
    """Genome-controlled device: one gene per step, power in [-limit, limit] W.

    Cost = sum of abs(power) across the horizon (total energy magnitude).
    Optionally clamps the genome to [0, limit] to force repair.

    Participates in the bus as a source (producer) — injects power.
    """

    def __init__(
        self,
        device_id: str,
        device_index: int,
        objective: str,
        limit: float = 100.0,
        force_clamp_positive: bool = False,
        bus_id: str = "bus_ac",
    ) -> None:
        super().__init__()
        self.device_id = device_id
        self._device_index = device_index
        self._objective = objective
        self._limit = limit
        self._force_clamp = force_clamp_positive
        self._bus_id = bus_id

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return (EnergyPort(
            port_id="p_source",
            bus_id=self._bus_id,
            direction=PortDirection.SOURCE,
        ),)

    @property
    def objective_names(self) -> list[str]:
        return [self._objective]

    def initial_state(self) -> float:
        return 0.0

    def state_transition_batch(self, state, power, step_interval):
        return state + power * step_interval / 3600.0

    def power_bounds(self) -> tuple[float, float]:
        return (-self._limit, self._limit)

    def repair_batch(self, step, requested_power, current_state):
        power = np.clip(requested_power, -self._limit, self._limit)
        if self._force_clamp:
            # Clamp negative values to 0 — forces repair when genome has negatives
            power = np.maximum(power, 0.0)
        return power

    def build_device_request(self, state: SingleStateBatchState) -> DeviceRequest | None:
        pop_size = state.population_size
        horizon = state.horizon
        # Inject schedule as negative energy (source = injection into bus)
        energy = -np.abs(state.schedule)
        min_e = np.zeros((pop_size, horizon))
        pr = PortRequest(port_index=0, energy_wh=energy, min_energy_wh=min_e)
        return DeviceRequest(device_index=self._device_index, port_requests=(pr,))

    def apply_device_grant(self, state, grant: DeviceGrant) -> None:
        pass

    def compute_cost(self, state: SingleStateBatchState) -> np.ndarray:
        # Cost = total energy magnitude per individual
        cost = np.sum(np.abs(state.schedule), axis=1, keepdims=True)
        return cost


# ============================================================
# Fixtures and shared setup helpers
# ============================================================

STEP_INTERVAL = 3600.0
HORIZON = 4
STEP_TIMES = tuple(to_datetime(i * 3600) for i in range(HORIZON))

AC_BUS = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)


def make_topology(num_ports: int) -> BusTopology:
    """All ports on bus 0."""
    return BusTopology(
        port_to_bus=np.zeros(num_ports, dtype=int),
        num_buses=1,
    )


def make_engine(
    devices: list,
    num_bus_ports: int | None = None,
) -> EnergySimulationEngine:
    """Build a fully wired engine from a device list."""
    registry = DeviceRegistry()
    for dev in devices:
        registry.register(dev)

    # Count total ports across all devices for topology mapping
    total_ports = num_bus_ports or sum(len(d.ports) for d in devices)
    topo = make_topology(total_ports)
    arb = VectorizedBusArbitrator(topo, horizon=HORIZON)

    return EnergySimulationEngine(registry, [AC_BUS], arb)


def make_genome(device: ScheduleDevice, pop_size: int, value: float = 50.0) -> dict:
    return {device.device_id: np.full((pop_size, HORIZON), value)}


def setup_engine(engine: EnergySimulationEngine) -> None:
    """Run engine through to STRUCTURE_FROZEN."""
    context = SimulationContext(step_times=STEP_TIMES, step_interval=STEP_INTERVAL)
    engine.setup_run(context)
    engine.genome_requirements()


# ============================================================
# TestEngineConstruction
# ============================================================

class TestEngineConstruction:
    def test_initial_state_is_created(self):
        sched = ScheduleDevice("s0", 0, "cost", bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 1.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        assert engine._state == EngineState.CREATED

    def test_invalid_topology_raises_at_construction(self):
        """A source-only topology (no sink) must raise TopologyValidationError."""
        sched = ScheduleDevice("s0", 0, "cost")
        registry = DeviceRegistry()
        registry.register(sched)
        topo = make_topology(1)
        arb = VectorizedBusArbitrator(topo, horizon=HORIZON)
        with pytest.raises(TopologyValidationError):
            EnergySimulationEngine(registry, [AC_BUS], arb)


# ============================================================
# TestEngineLifecycle
# ============================================================

class TestEngineLifecycle:
    @pytest.fixture()
    def engine(self):
        sched = ScheduleDevice("s0", 0, "cost", bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 1.0, bus_id="bus_ac")
        return make_engine([sched, sink])

    def test_setup_run_transitions_to_run_configured(self, engine):
        context = SimulationContext(STEP_TIMES, STEP_INTERVAL)
        engine.setup_run(context)
        assert engine._state == EngineState.RUN_CONFIGURED

    def test_genome_requirements_transitions_to_structure_frozen(self, engine):
        context = SimulationContext(STEP_TIMES, STEP_INTERVAL)
        engine.setup_run(context)
        engine.genome_requirements()
        assert engine._state == EngineState.STRUCTURE_FROZEN

    def test_setup_run_raises_in_run_configured_state(self, engine):
        context = SimulationContext(STEP_TIMES, STEP_INTERVAL)
        engine.setup_run(context)
        with pytest.raises(RuntimeError, match="RUN_CONFIGURED"):
            engine.setup_run(context)

    def test_genome_requirements_raises_before_setup_run(self, engine):
        with pytest.raises(RuntimeError):
            engine.genome_requirements()

    def test_genome_requirements_raises_in_structure_frozen(self, engine):
        setup_engine(engine)
        with pytest.raises(RuntimeError):
            engine.genome_requirements()

    def test_evaluate_population_raises_before_genome_requirements(self, engine):
        context = SimulationContext(STEP_TIMES, STEP_INTERVAL)
        engine.setup_run(context)
        genome = {"s0": np.zeros((2, HORIZON))}
        with pytest.raises(RuntimeError):
            engine.evaluate_population(genome)

    def test_evaluate_population_raises_in_created_state(self, engine):
        genome = {"s0": np.zeros((2, HORIZON))}
        with pytest.raises(RuntimeError):
            engine.evaluate_population(genome)

    def test_setup_run_from_structure_frozen_allows_re_run(self, engine):
        """Engine can be reconfigured from STRUCTURE_FROZEN for a new run."""
        setup_engine(engine)
        new_context = SimulationContext(
            step_times=tuple(to_datetime(i * 3600) for i in range(8)),
            step_interval=1800.0,
        )
        engine.setup_run(new_context)
        assert engine._state == EngineState.RUN_CONFIGURED


# ============================================================
# TestObjectiveNames
# ============================================================

class TestObjectiveNames:
    def test_objective_names_raises_before_setup_run(self):
        sched = ScheduleDevice("s0", 0, "cost")
        sink = ConstantCostDevice("c0", 1, "cost", 1.0)
        engine = make_engine([sched, sink])
        with pytest.raises(RuntimeError):
            _ = engine.objective_names

    def test_objective_names_available_after_setup_run(self):
        sched = ScheduleDevice("s0", 0, "cost", bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 1.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        engine.setup_run(SimulationContext(STEP_TIMES, STEP_INTERVAL))
        assert engine.objective_names == ["cost"]

    def test_two_different_objectives_both_appear(self):
        sched = ScheduleDevice("s0", 0, "energy_cost", bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "peak_power", 1.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        engine.setup_run(SimulationContext(STEP_TIMES, STEP_INTERVAL))
        names = engine.objective_names
        assert "energy_cost" in names
        assert "peak_power" in names
        assert len(names) == 2

    def test_shared_objective_name_appears_once(self):
        sched = ScheduleDevice("s0", 0, "cost", bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 1.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        engine.setup_run(SimulationContext(STEP_TIMES, STEP_INTERVAL))
        assert engine.objective_names.count("cost") == 1

    def test_objective_order_follows_device_registration_order(self):
        """First device's objective must occupy column 0."""
        sched = ScheduleDevice("s0", 0, "energy_cost", bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "peak_power", 1.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        engine.setup_run(SimulationContext(STEP_TIMES, STEP_INTERVAL))
        assert engine.objective_names[0] == "energy_cost"
        assert engine.objective_names[1] == "peak_power"


# ============================================================
# TestEvaluatePopulationFitness
# ============================================================

class TestEvaluatePopulationFitness:
    @pytest.fixture()
    def single_device_engine(self):
        sched = ScheduleDevice("s0", 0, "cost", limit=100.0, bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 0.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)
        return engine, sched

    def test_fitness_shape_is_pop_size_by_num_objectives(
        self, single_device_engine
    ):
        engine, sched = single_device_engine
        pop_size = 5
        genome = make_genome(sched, pop_size, value=50.0)
        result = engine.evaluate_population(genome)
        assert result.fitness.shape == (pop_size, 1)

    def test_fitness_objective_names_match_engine(self, single_device_engine):
        engine, sched = single_device_engine
        genome = make_genome(sched, 3)
        result = engine.evaluate_population(genome)
        assert result.objective_names == engine.objective_names

    def test_cost_from_schedule_device_correct(self, single_device_engine):
        """cost = sum(|power|) = 50 W * 4 steps = 200 for each individual."""
        engine, sched = single_device_engine
        pop_size = 3
        genome = make_genome(sched, pop_size, value=50.0)
        result = engine.evaluate_population(genome)
        # ConstantCostDevice has cost=0, ScheduleDevice cost=50*4=200
        np.testing.assert_array_almost_equal(
            result.fitness[:, 0], [200.0] * pop_size
        )

    def test_shared_objective_costs_are_summed(self):
        """Two devices with the same objective name must sum into one column."""
        sched = ScheduleDevice("s0", 0, "cost", limit=100.0, bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 10.0, bus_id="bus_ac")
        # ConstantCostDevice cost = 10 * HORIZON = 40
        # ScheduleDevice cost = 50 * HORIZON = 200
        # Total = 240
        engine = make_engine([sched, sink])
        setup_engine(engine)
        genome = make_genome(sched, 2, value=50.0)
        result = engine.evaluate_population(genome)
        assert result.fitness.shape == (2, 1)
        np.testing.assert_array_almost_equal(
            result.fitness[:, 0], [240.0, 240.0]
        )

    def test_separate_objectives_go_to_separate_columns(self):
        """Costs from different objective names must not mix."""
        sched = ScheduleDevice("s0", 0, "energy_cost", limit=100.0, bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "peak_power", 5.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)
        genome = make_genome(sched, 2, value=50.0)
        result = engine.evaluate_population(genome)
        assert result.fitness.shape == (2, 2)

        energy_col = result.objective_names.index("energy_cost")
        peak_col = result.objective_names.index("peak_power")
        # energy_cost = 50 * 4 = 200
        np.testing.assert_array_almost_equal(result.fitness[:, energy_col], [200.0, 200.0])
        # peak_power = 5 * 4 = 20
        np.testing.assert_array_almost_equal(result.fitness[:, peak_col], [20.0, 20.0])

    def test_zero_genome_produces_zero_schedule_cost(self, single_device_engine):
        engine, sched = single_device_engine
        genome = make_genome(sched, 2, value=0.0)
        result = engine.evaluate_population(genome)
        np.testing.assert_array_almost_equal(result.fitness[:, 0], [0.0, 0.0])


# ============================================================
# TestEvaluatePopulationRepair
# ============================================================

class TestEvaluatePopulationRepair:
    def test_feasible_genome_produces_empty_repaired_genomes(self):
        sched = ScheduleDevice("s0", 0, "cost", limit=100.0, bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 0.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)
        genome = make_genome(sched, 3, value=50.0)   # within (-100, 100)
        result = engine.evaluate_population(genome)
        assert result.repaired_genomes == {}

    def test_infeasible_genome_appears_in_repaired_genomes(self):
        sched = ScheduleDevice(
            "s0", 0, "cost", limit=100.0,
            force_clamp_positive=True,   # clamps negatives → repair
            bus_id="bus_ac",
        )
        sink = ConstantCostDevice("c0", 1, "cost", 0.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)
        # Negative values will be clamped to 0 by force_clamp
        genome = {"s0": np.full((3, HORIZON), -50.0)}
        result = engine.evaluate_population(genome)
        assert "s0" in result.repaired_genomes

    def test_repaired_genome_has_correct_shape(self):
        pop_size = 4
        sched = ScheduleDevice(
            "s0", 0, "cost", limit=100.0,
            force_clamp_positive=True,
            bus_id="bus_ac",
        )
        sink = ConstantCostDevice("c0", 1, "cost", 0.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)
        genome = {"s0": np.full((pop_size, HORIZON), -50.0)}
        result = engine.evaluate_population(genome)
        assert result.repaired_genomes["s0"].shape == (pop_size, HORIZON)

    def test_only_repaired_device_appears_in_repaired_genomes(self):
        """When only one of two devices repairs, only that one appears."""
        sched_repaired = ScheduleDevice(
            "s0", 0, "energy_cost", limit=100.0,
            force_clamp_positive=True,
            bus_id="bus_ac",
        )
        sink = ConstantCostDevice("c0", 1, "peak_power", 0.0, bus_id="bus_ac")
        engine = make_engine([sched_repaired, sink])
        setup_engine(engine)
        genome = {"s0": np.full((2, HORIZON), -50.0)}
        result = engine.evaluate_population(genome)
        assert "s0" in result.repaired_genomes
        assert "c0" not in result.repaired_genomes

    def test_repaired_genome_values_reflect_clamping(self):
        """The repaired array must contain the clamped (non-negative) values."""
        sched = ScheduleDevice(
            "s0", 0, "cost", limit=100.0,
            force_clamp_positive=True,
            bus_id="bus_ac",
        )
        sink = ConstantCostDevice("c0", 1, "cost", 0.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)
        genome = {"s0": np.full((2, HORIZON), -50.0)}
        result = engine.evaluate_population(genome)
        np.testing.assert_array_equal(result.repaired_genomes["s0"], 0.0)


# ============================================================
# TestEvaluatePopulationIndependence
# ============================================================

class TestEvaluatePopulationIndependence:
    def test_different_individuals_get_different_fitness(self):
        """Each row of the population must produce its own fitness value."""
        sched = ScheduleDevice("s0", 0, "cost", limit=200.0, bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 0.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)

        # Three individuals with different constant power schedules
        genome = np.array([
            [50.0] * HORIZON,    # cost = 200
            [100.0] * HORIZON,   # cost = 400
            [150.0] * HORIZON,   # cost = 600
        ])
        result = engine.evaluate_population({"s0": genome})
        np.testing.assert_array_almost_equal(
            result.fitness[:, 0], [200.0, 400.0, 600.0]
        )

    def test_population_size_one_works(self):
        sched = ScheduleDevice("s0", 0, "cost", limit=100.0, bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 0.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)
        genome = make_genome(sched, pop_size=1, value=50.0)
        result = engine.evaluate_population(genome)
        assert result.fitness.shape == (1, 1)

    def test_population_size_eight_works(self):
        sched = ScheduleDevice("s0", 0, "cost", limit=100.0, bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 0.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)
        genome = make_genome(sched, pop_size=8, value=50.0)
        result = engine.evaluate_population(genome)
        assert result.fitness.shape == (8, 1)

    def test_multiple_evaluate_population_calls_produce_consistent_results(self):
        """Calling evaluate_population twice with the same input must give
        identical results — no state leaks between generations."""
        sched = ScheduleDevice("s0", 0, "cost", limit=100.0, bus_id="bus_ac")
        sink = ConstantCostDevice("c0", 1, "cost", 0.0, bus_id="bus_ac")
        engine = make_engine([sched, sink])
        setup_engine(engine)
        genome = make_genome(sched, pop_size=4, value=75.0)

        result_1 = engine.evaluate_population(genome)
        result_2 = engine.evaluate_population(genome)

        np.testing.assert_array_equal(result_1.fitness, result_2.fitness)
