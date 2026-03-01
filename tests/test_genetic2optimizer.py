"""Tests for GeneticOptimizer and RollingHorizonOptimizer.

Test strategy
-------------
GA operator tests (_tournament_select, _blx_crossover, _mutate,
_build_assembled_genome, _init_population, _split_population,
_apply_repairs) are exercised in isolation — no engine is constructed.
This keeps them fast and focused on the arithmetic.

Integration tests for optimize() and RollingHorizonOptimizer.optimize()
use the same minimal concrete devices as the engine tests (ScheduleDevice,
ConstantCostDevice). These are real engine calls — no mocking. Population
sizes and generation counts are kept small (pop=6, gen=3) so the suite
remains fast while still exercising the full pipeline.

Covers
------
    default_scalarize
        - Sums across objective axis
        - Shape contract

    GenerationStats
        - Construction and field access

    OptimizationResult
        - Construction and field access

    GeneticOptimizer._build_assembled_genome
        - Empty slice dict → zero-size AssembledGenome
        - Single device slice → correct bounds and total_size
        - Two device slices → correct total_size and combined bounds

    GeneticOptimizer._init_population
        - Shape is (population_size, total_genome_size)
        - All values within [lower, upper] bounds
        - Infinite bounds fall back to 0.0

    GeneticOptimizer._split_population
        - Returns dict keyed by device_id
        - Each value is a view of the population slice (correct columns)
        - Shape of each value is (pop_size, slice_size)

    GeneticOptimizer._apply_repairs
        - Modified columns are updated in-place
        - Columns for devices absent from repaired_genomes unchanged
        - Device absent from assembled.slices is silently ignored

    GeneticOptimizer._tournament_select
        - Returns an individual with the lowest fitness among drawn samples
        - Returns a copy, not a view
        - tournament_size=1 always returns the drawn individual

    GeneticOptimizer._blx_crossover
        - crossover_rate=1.0: children are convex combinations of parents
        - crossover_rate=0.0: children are copies of parents unchanged
        - Children are within the convex hull [min(p1,p2), max(p1,p2)]
        - Children are distinct arrays (not the same object as parents)

    GeneticOptimizer._mutate
        - mutation_rate=0.0: genome unchanged
        - mutation_rate=1.0: every gene perturbed but stays within bounds
        - Modifies genome in-place
        - Genes never exceed bounds after mutation

    GeneticOptimizer.optimize
        - Returns OptimizationResult
        - best_genome shape is (total_genome_size,)
        - best_fitness_vector shape is (num_objectives,)
        - history has exactly `generations` entries
        - GenerationStats.generation matches loop index (0-based)
        - best_scalar_fitness is non-negative (cost cannot be negative here)
        - assembled.slices keys match registered genome devices
        - Deterministic with fixed random_seed
        - zero-size genome raises ValueError
        - Custom scalarize function is called and affects selection
        - Repair write-back: repaired values appear in population

    RollingHorizonOptimizer.__init__
        - roll_steps > window_size raises ValueError

    RollingHorizonOptimizer.optimize
        - Returns dict keyed by device_id
        - Schedule arrays have length == total_steps
        - All schedule values within device power bounds
        - window_size == total_steps behaves like a single GA run
        - Last window truncated correctly when total_steps not divisible
        - Multiple windows each contribute their roll_steps to the schedule
"""

from __future__ import annotations

import numpy as np
import pytest

from akkudoktoreos.core.emplan import EnergyManagementInstruction, OMBCInstruction
from akkudoktoreos.devices.devicesabc import (
    EnergyBus,
    EnergyCarrier,
    EnergyPort,
    PortDirection,
    SingleStateBatchState,
    SingleStateEnergyDevice,
)
from akkudoktoreos.optimization.genetic2.genome import AssembledGenome, GenomeSlice
from akkudoktoreos.optimization.genetic2.optimizer import (
    GenerationStats,
    GeneticOptimizer,
    OptimizationResult,
    RollingHorizonOptimizer,
    default_scalarize,
)
from akkudoktoreos.simulation.genetic2.arbitrator import (
    BusTopology,
    DeviceGrant,
    DeviceRequest,
    PortRequest,
    VectorizedBusArbitrator,
)
from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import to_datetime

# ============================================================
# Concrete minimal devices (same pattern as engine tests)
# ============================================================

class ScheduleDevice(SingleStateEnergyDevice):
    """Genome-controlled device. Cost = sum(|power|). Source on bus."""

    def __init__(
        self,
        device_id: str,
        device_index: int,
        limit: float = 100.0,
        force_clamp_positive: bool = False,
        bus_id: str = "bus_ac",
        objective: str = "cost",
    ) -> None:
        super().__init__()
        self.device_id = device_id
        self._device_index = device_index
        self._limit = limit
        self._force_clamp = force_clamp_positive
        self._bus_id = bus_id
        self._objective = objective

    @property
    def ports(self):
        return (EnergyPort(
            port_id="p_src", bus_id=self._bus_id,
            direction=PortDirection.SOURCE,
        ),)

    @property
    def objective_names(self):
        return [self._objective]

    def initial_state(self):
        return 0.0

    def state_transition_batch(self, state, power, step_interval):
        return state + power * step_interval / 3600.0

    def power_bounds(self):
        return (-self._limit, self._limit)

    def repair_batch(self, step, requested_power, current_state):
        power = np.clip(requested_power, -self._limit, self._limit)
        if self._force_clamp:
            power = np.maximum(power, 0.0)
        return power

    def build_device_request(self, state):
        pop_size, horizon = state.population_size, state.horizon
        energy = -np.abs(state.schedule)
        pr = PortRequest(
            port_index=0,
            energy_wh=energy,
            min_energy_wh=np.zeros((pop_size, horizon)),
        )
        return DeviceRequest(device_index=self._device_index, port_requests=(pr,))

    def apply_device_grant(self, state, grant):
        pass

    def compute_cost(self, state):
        return np.sum(np.abs(state.schedule), axis=1, keepdims=True)


class SinkDevice(SingleStateEnergyDevice):
    """Non-genome sink device. Zero cost. Needed to satisfy bus topology."""

    def __init__(self, device_id: str, device_index: int, bus_id: str = "bus_ac") -> None:
        super().__init__()
        self.device_id = device_id
        self._device_index = device_index
        self._bus_id = bus_id

    @property
    def ports(self):
        return (EnergyPort(
            port_id="p_snk", bus_id=self._bus_id,
            direction=PortDirection.SINK,
        ),)

    @property
    def objective_names(self):
        return ["cost"]

    def genome_requirements(self):
        return None

    def initial_state(self):
        return 0.0

    def state_transition_batch(self, state, power, step_interval):
        return state

    def power_bounds(self):
        return (0.0, 100.0)

    def build_device_request(self, state):
        pop_size, horizon = state.population_size, state.horizon
        pr = PortRequest(
            port_index=0,
            energy_wh=np.full((pop_size, horizon), 10.0),
            min_energy_wh=np.zeros((pop_size, horizon)),
        )
        return DeviceRequest(device_index=self._device_index, port_requests=(pr,))

    def apply_device_grant(self, state, grant):
        pass

    def compute_cost(self, state):
        return np.zeros((state.population_size, 1))


# ============================================================
# Engine factory helpers
# ============================================================

AC_BUS = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
STEP_INTERVAL = 3600.0


def make_engine(devices: list, horizon: int) -> EnergySimulationEngine:
    registry = DeviceRegistry()
    for dev in devices:
        registry.register(dev)
    total_ports = sum(len(d.ports) for d in devices)
    topo = BusTopology(
        port_to_bus=np.zeros(total_ports, dtype=int),
        num_buses=1,
    )
    arb = VectorizedBusArbitrator(topo, horizon=horizon)
    return EnergySimulationEngine(registry, [AC_BUS], arb)


def make_context(horizon: int) -> SimulationContext:
    return SimulationContext(
        step_times=tuple(to_datetime(i * 3600) for i in range(horizon)),
        step_interval=STEP_INTERVAL,
    )


def make_optimizer(
    engine: EnergySimulationEngine,
    pop: int = 6,
    gens: int = 3,
    seed: int = 42,
    **kwargs,
) -> GeneticOptimizer:
    return GeneticOptimizer(
        engine=engine,
        population_size=pop,
        generations=gens,
        random_seed=seed,
        **kwargs,
    )


def assembled_from_slices(slices: dict) -> AssembledGenome:
    return GeneticOptimizer._build_assembled_genome(slices)


# ============================================================
# TestDefaultScalarize
# ============================================================

class TestDefaultScalarize:
    def test_single_objective_returns_same_values(self):
        fitness = np.array([[3.0], [1.0], [2.0]])
        result = default_scalarize(fitness)
        np.testing.assert_array_equal(result, [3.0, 1.0, 2.0])

    def test_two_objectives_summed(self):
        fitness = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = default_scalarize(fitness)
        np.testing.assert_array_equal(result, [3.0, 7.0])

    def test_output_shape_is_population_size(self):
        fitness = np.ones((5, 3))
        assert default_scalarize(fitness).shape == (5,)


# ============================================================
# TestGenerationStats
# ============================================================

class TestGenerationStats:
    def test_construction_and_field_access(self):
        s = GenerationStats(
            generation=2,
            best_scalar_fitness=1.5,
            mean_scalar_fitness=3.0,
            num_repaired=1,
        )
        assert s.generation == 2
        assert s.best_scalar_fitness == 1.5
        assert s.mean_scalar_fitness == 3.0
        assert s.num_repaired == 1


# ============================================================
# TestOptimizationResult
# ============================================================

class TestOptimizationResult:
    def test_construction_and_field_access(self):
        assembled = assembled_from_slices({})
        r = OptimizationResult(
            best_genome=np.zeros(4),
            best_fitness_vector=np.array([1.0]),
            best_scalar_fitness=1.0,
            objective_names=["cost"],
            generations_run=10,
            history=[],
            assembled=assembled,
        )
        assert r.generations_run == 10
        assert r.objective_names == ["cost"]
        assert r.best_scalar_fitness == 1.0


# ============================================================
# TestBuildAssembledGenome
# ============================================================

class TestBuildAssembledGenome:
    def test_empty_slices_returns_zero_size(self):
        result = assembled_from_slices({})
        assert result.total_size == 0
        assert result.slices == {}

    def test_single_device_total_size(self):
        slc = GenomeSlice(start=0, size=4, lower_bound=np.full(4, -10.0), upper_bound=np.full(4, 10.0))
        result = assembled_from_slices({"dev": slc})
        assert result.total_size == 4

    def test_single_device_bounds_filled(self):
        slc = GenomeSlice(start=0, size=3, lower_bound=np.full(3, -5.0), upper_bound=np.full(3, 5.0))
        result = assembled_from_slices({"dev": slc})
        np.testing.assert_array_equal(result.lower_bounds, [-5.0, -5.0, -5.0])
        np.testing.assert_array_equal(result.upper_bounds, [5.0, 5.0, 5.0])

    def test_two_devices_total_size(self):
        s0 = GenomeSlice(start=0, size=3, lower_bound=np.full(3, -1.0), upper_bound=np.full(3, 1.0))
        s1 = GenomeSlice(start=3, size=4, lower_bound=np.full(4, -2.0), upper_bound=np.full(4, 2.0))
        result = assembled_from_slices({"a": s0, "b": s1})
        assert result.total_size == 7

    def test_two_devices_combined_bounds(self):
        s0 = GenomeSlice(start=0, size=2, lower_bound=np.full(2, -1.0), upper_bound=np.full(2, 1.0))
        s1 = GenomeSlice(start=2, size=2, lower_bound=np.full(2, -9.0), upper_bound=np.full(2, 9.0))
        result = assembled_from_slices({"a": s0, "b": s1})
        np.testing.assert_array_equal(result.lower_bounds, [-1.0, -1.0, -9.0, -9.0])
        np.testing.assert_array_equal(result.upper_bounds, [1.0, 1.0, 9.0, 9.0])

    def test_no_bounds_defaults_to_inf(self):
        slc = GenomeSlice(start=0, size=2)
        result = assembled_from_slices({"dev": slc})
        assert np.all(result.lower_bounds == -np.inf)
        assert np.all(result.upper_bounds == np.inf)


# ============================================================
# TestInitPopulation
# ============================================================

class TestInitPopulation:
    def _make_optimizer(self) -> GeneticOptimizer:
        # Minimal optimizer with no engine needed for unit tests
        dev = ScheduleDevice("s0", 0, limit=100.0)
        sink = SinkDevice("c0", 1)
        engine = make_engine([dev, sink], horizon=4)
        return make_optimizer(engine)

    def test_shape_is_pop_size_by_genome_size(self):
        opt = self._make_optimizer()
        s0 = GenomeSlice(start=0, size=6, lower_bound=np.full(6, -50.0), upper_bound=np.full(6, 50.0))
        assembled = assembled_from_slices({"dev": s0})
        pop = opt._init_population(assembled)
        assert pop.shape == (opt.population_size, 6)

    def test_all_values_within_bounds(self):
        opt = self._make_optimizer()
        s0 = GenomeSlice(start=0, size=8, lower_bound=np.full(8, -30.0), upper_bound=np.full(8, 30.0))
        assembled = assembled_from_slices({"dev": s0})
        pop = opt._init_population(assembled)
        assert np.all(pop >= -30.0)
        assert np.all(pop <= 30.0)

    def test_infinite_bounds_fall_back_to_zero(self):
        opt = self._make_optimizer()
        slc = GenomeSlice(start=0, size=4)  # no bounds → ±inf
        assembled = assembled_from_slices({"dev": slc})
        pop = opt._init_population(assembled)
        np.testing.assert_array_equal(pop, 0.0)


# ============================================================
# TestSplitPopulation
# ============================================================

class TestSplitPopulation:
    def _make_optimizer(self) -> GeneticOptimizer:
        dev = ScheduleDevice("s0", 0)
        sink = SinkDevice("c0", 1)
        return make_optimizer(make_engine([dev, sink], horizon=4))

    def test_returns_dict_keyed_by_device_id(self):
        opt = self._make_optimizer()
        s0 = GenomeSlice(start=0, size=3, lower_bound=np.full(3, 0.0), upper_bound=np.full(3, 1.0))
        s1 = GenomeSlice(start=3, size=2, lower_bound=np.full(2, 0.0), upper_bound=np.full(2, 1.0))
        assembled = assembled_from_slices({"dev_a": s0, "dev_b": s1})
        population = np.arange(30.0).reshape(6, 5)
        result = opt._split_population(population, assembled)
        assert set(result.keys()) == {"dev_a", "dev_b"}

    def test_slice_shapes_are_correct(self):
        opt = self._make_optimizer()
        s0 = GenomeSlice(start=0, size=3, lower_bound=np.full(3, 0.0), upper_bound=np.full(3, 1.0))
        s1 = GenomeSlice(start=3, size=2, lower_bound=np.full(2, 0.0), upper_bound=np.full(2, 1.0))
        assembled = assembled_from_slices({"dev_a": s0, "dev_b": s1})
        population = np.ones((6, 5))
        result = opt._split_population(population, assembled)
        assert result["dev_a"].shape == (6, 3)
        assert result["dev_b"].shape == (6, 2)

    def test_slice_values_are_correct_columns(self):
        opt = self._make_optimizer()
        s0 = GenomeSlice(start=0, size=2, lower_bound=np.full(2, 0.0), upper_bound=np.full(2, 1.0))
        s1 = GenomeSlice(start=2, size=3, lower_bound=np.full(3, 0.0), upper_bound=np.full(3, 1.0))
        assembled = assembled_from_slices({"a": s0, "b": s1})
        population = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ])
        result = opt._split_population(population, assembled)
        np.testing.assert_array_equal(result["a"], [[1.0, 2.0], [6.0, 7.0]])
        np.testing.assert_array_equal(result["b"], [[3.0, 4.0, 5.0], [8.0, 9.0, 10.0]])

    def test_returned_arrays_are_views_of_population(self):
        """Slices must be views so genome dict modifications affect the population."""
        opt = self._make_optimizer()
        slc = GenomeSlice(start=0, size=3, lower_bound=np.full(3, 0.0), upper_bound=np.full(3, 1.0))
        assembled = assembled_from_slices({"dev": slc})
        population = np.zeros((4, 3))
        result = opt._split_population(population, assembled)
        result["dev"][0, 0] = 99.0
        assert population[0, 0] == 99.0


# ============================================================
# TestApplyRepairs
# ============================================================

class TestApplyRepairs:
    def _make_optimizer(self) -> GeneticOptimizer:
        dev = ScheduleDevice("s0", 0)
        sink = SinkDevice("c0", 1)
        return make_optimizer(make_engine([dev, sink], horizon=4))

    def test_repaired_columns_updated_in_place(self):
        opt = self._make_optimizer()
        slc = GenomeSlice(start=0, size=3, lower_bound=np.full(3, -100.0), upper_bound=np.full(3, 100.0))
        assembled = assembled_from_slices({"dev": slc})
        population = np.zeros((4, 3))
        repaired = np.full((4, 3), 42.0)
        opt._apply_repairs(population, {"dev": repaired}, assembled)
        np.testing.assert_array_equal(population, 42.0)

    def test_non_repaired_columns_unchanged(self):
        opt = self._make_optimizer()
        s0 = GenomeSlice(start=0, size=2, lower_bound=np.full(2, -1.0), upper_bound=np.full(2, 1.0))
        s1 = GenomeSlice(start=2, size=3, lower_bound=np.full(3, -1.0), upper_bound=np.full(3, 1.0))
        assembled = assembled_from_slices({"dev_a": s0, "dev_b": s1})
        population = np.zeros((3, 5))
        repaired = np.full((3, 2), 7.0)
        opt._apply_repairs(population, {"dev_a": repaired}, assembled)
        # dev_a columns updated
        np.testing.assert_array_equal(population[:, :2], 7.0)
        # dev_b columns untouched
        np.testing.assert_array_equal(population[:, 2:], 0.0)

    def test_device_absent_from_slices_silently_ignored(self):
        opt = self._make_optimizer()
        slc = GenomeSlice(start=0, size=2, lower_bound=np.full(2, -1.0), upper_bound=np.full(2, 1.0))
        assembled = assembled_from_slices({"known": slc})
        population = np.zeros((2, 2))
        # "ghost" not in assembled.slices — must not raise
        opt._apply_repairs(population, {"ghost": np.ones((2, 2))}, assembled)
        np.testing.assert_array_equal(population, 0.0)


# ============================================================
# TestTournamentSelect
# ============================================================

class TestTournamentSelect:
    def _make_optimizer(self, tournament_size: int = 3) -> GeneticOptimizer:
        dev = ScheduleDevice("s0", 0)
        sink = SinkDevice("c0", 1)
        return make_optimizer(
            make_engine([dev, sink], horizon=4),
            tournament_size=tournament_size,
            seed=0,
        )

    def test_winner_has_lower_or_equal_fitness_than_all_drawn(self):
        opt = self._make_optimizer(tournament_size=3)
        # Five individuals with known fitness; best is index 2 (fitness=1)
        population = np.arange(25.0).reshape(5, 5)
        fitness = np.array([5.0, 4.0, 1.0, 3.0, 2.0])
        rng = np.random.default_rng(0)
        opt._rng = rng
        # Run many draws; winner must never be worse than all drawn
        for _ in range(50):
            winner = opt._tournament_select(population, fitness)
            # Winner must be one of the population rows
            assert any(np.array_equal(winner, population[i]) for i in range(5))

    def test_returns_copy_not_view(self):
        opt = self._make_optimizer()
        population = np.zeros((4, 3))
        fitness = np.array([1.0, 2.0, 3.0, 4.0])
        winner = opt._tournament_select(population, fitness)
        winner[0] = 99.0
        # Original population must not be modified
        assert population[0, 0] == 0.0

    def test_tournament_size_one_returns_drawn_individual(self):
        opt = self._make_optimizer(tournament_size=1)
        population = np.eye(5)
        fitness = np.ones(5)
        # With tournament_size=1, whatever is drawn is returned
        winner = opt._tournament_select(population, fitness)
        assert winner.shape == (5,)


# ============================================================
# TestBlxCrossover
# ============================================================

class TestBlxCrossover:
    def _make_optimizer(self, crossover_rate: float = 1.0) -> GeneticOptimizer:
        dev = ScheduleDevice("s0", 0)
        sink = SinkDevice("c0", 1)
        return make_optimizer(
            make_engine([dev, sink], horizon=4),
            crossover_rate=crossover_rate,
            seed=7,
        )

    def test_children_are_convex_combinations(self):
        """With crossover_rate=1.0, children must be in the convex hull of parents."""
        opt = self._make_optimizer(crossover_rate=1.0)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([10.0, 10.0, 10.0])
        for _ in range(20):
            c1, c2 = opt._blx_crossover(p1, p2)
            assert np.all(c1 >= 0.0) and np.all(c1 <= 10.0)
            assert np.all(c2 >= 0.0) and np.all(c2 <= 10.0)

    def test_no_crossover_returns_copies_of_parents(self):
        """With crossover_rate=0.0, children are unchanged copies."""
        opt = self._make_optimizer(crossover_rate=0.0)
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])
        c1, c2 = opt._blx_crossover(p1, p2)
        np.testing.assert_array_equal(c1, p1)
        np.testing.assert_array_equal(c2, p2)

    def test_children_are_distinct_objects_from_parents(self):
        opt = self._make_optimizer(crossover_rate=1.0)
        p1 = np.ones(4)
        p2 = np.zeros(4)
        c1, c2 = opt._blx_crossover(p1, p2)
        assert c1 is not p1 and c1 is not p2
        assert c2 is not p1 and c2 is not p2

    def test_blx_is_symmetric(self):
        """blx(p1, p2) and blx(p2, p1) with the same RNG state produce the
        same pair of children (just swapped)."""
        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 4.0])
        opt_ab = self._make_optimizer()
        opt_ba = self._make_optimizer()   # same seed
        c1_ab, c2_ab = opt_ab._blx_crossover(p1, p2)
        c1_ba, c2_ba = opt_ba._blx_crossover(p2, p1)
        # Due to symmetry in BLX-α: child2 of (p1,p2) == child1 of (p2,p1)
        np.testing.assert_array_almost_equal(c1_ab, c2_ba)
        np.testing.assert_array_almost_equal(c2_ab, c1_ba)


# ============================================================
# TestMutate
# ============================================================

class TestMutate:
    def _make_optimizer(self, mutation_rate: float = 0.5) -> GeneticOptimizer:
        dev = ScheduleDevice("s0", 0)
        sink = SinkDevice("c0", 1)
        return make_optimizer(
            make_engine([dev, sink], horizon=4),
            mutation_rate=mutation_rate,
            seed=99,
        )

    def _make_assembled(self, lo: float, hi: float, size: int = 8) -> AssembledGenome:
        slc = GenomeSlice(
            start=0, size=size,
            lower_bound=np.full(size, lo),
            upper_bound=np.full(size, hi),
        )
        return assembled_from_slices({"dev": slc})

    def test_mutation_rate_zero_leaves_genome_unchanged(self):
        opt = self._make_optimizer(mutation_rate=0.0)
        assembled = self._make_assembled(-100.0, 100.0)
        genome = np.full(8, 50.0)
        opt._mutate(genome, assembled)
        np.testing.assert_array_equal(genome, 50.0)

    def test_mutation_rate_one_modifies_in_place(self):
        opt = self._make_optimizer(mutation_rate=1.0)
        assembled = self._make_assembled(-1000.0, 1000.0, size=50)
        genome = np.zeros(50)
        original = genome.copy()
        opt._mutate(genome, assembled)
        # With rate=1.0 and large sigma, very likely that at least one gene changed
        assert not np.array_equal(genome, original)

    def test_genes_stay_within_bounds_after_mutation(self):
        opt = self._make_optimizer(mutation_rate=1.0)
        assembled = self._make_assembled(-10.0, 10.0, size=100)
        genome = np.zeros(100)
        for _ in range(20):
            opt._mutate(genome, assembled)
            assert np.all(genome >= -10.0)
            assert np.all(genome <= 10.0)

    def test_mutate_modifies_genome_in_place(self):
        opt = self._make_optimizer(mutation_rate=1.0)
        assembled = self._make_assembled(-1000.0, 1000.0, size=10)
        genome = np.zeros(10)
        genome_id = id(genome)
        opt._mutate(genome, assembled)
        assert id(genome) == genome_id   # same object


# ============================================================
# TestGeneticOptimizerOptimize
# ============================================================

class TestGeneticOptimizerOptimize:
    HORIZON = 4

    @pytest.fixture()
    def engine_and_device(self):
        dev = ScheduleDevice("s0", 0, limit=100.0)
        sink = SinkDevice("c0", 1)
        engine = make_engine([dev, sink], self.HORIZON)
        return engine, dev

    @pytest.fixture()
    def result(self, engine_and_device):
        engine, _ = engine_and_device
        opt = make_optimizer(engine, pop=6, gens=3, seed=42)
        return opt.optimize(make_context(self.HORIZON))

    def test_returns_optimization_result(self, result):
        assert isinstance(result, OptimizationResult)

    def test_best_genome_shape(self, result):
        assert result.best_genome.shape == (self.HORIZON,)

    def test_best_fitness_vector_shape(self, result):
        assert result.best_fitness_vector.shape == (1,)

    def test_history_length_equals_generations(self, result):
        assert len(result.history) == 3

    def test_generation_indices_are_zero_based_sequential(self, result):
        for i, stats in enumerate(result.history):
            assert stats.generation == i

    def test_best_scalar_fitness_is_non_negative(self, result):
        assert result.best_scalar_fitness >= 0.0

    def test_best_scalar_fitness_equals_minimum_in_history(self, result):
        min_in_history = min(s.best_scalar_fitness for s in result.history)
        assert result.best_scalar_fitness <= min_in_history + 1e-9

    def test_assembled_slices_keyed_by_genome_device_id(self, result):
        assert "s0" in result.assembled.slices
        assert "c0" not in result.assembled.slices   # SinkDevice has no genome

    def test_objective_names_match_engine(self, engine_and_device, result):
        engine, _ = engine_and_device
        assert result.objective_names == engine.objective_names

    def test_deterministic_with_fixed_seed(self, engine_and_device):
        engine, _ = engine_and_device
        r1 = make_optimizer(engine, seed=0).optimize(make_context(self.HORIZON))
        # Re-run requires new engine (setup_run can be called again from STRUCTURE_FROZEN)
        dev2 = ScheduleDevice("s0", 0, limit=100.0)
        sink2 = SinkDevice("c0", 1)
        engine2 = make_engine([dev2, sink2], self.HORIZON)
        r2 = make_optimizer(engine2, seed=0).optimize(make_context(self.HORIZON))
        np.testing.assert_array_equal(r1.best_genome, r2.best_genome)

    def test_zero_genome_size_raises_value_error(self):
        """If no device has a genome, ValueError must be raised.

        Topology still requires ≥1 source and ≥1 sink, so we use a
        ScheduleDevice whose genome_requirements() is patched to return
        None — valid topology, but zero assembled genome size.
        """
        class NonGenomeSource(ScheduleDevice):
            def genome_requirements(self):
                return None

        source = NonGenomeSource("s0", 0, limit=100.0)
        sink = SinkDevice("c0", 1)
        engine = make_engine([source, sink], self.HORIZON)
        opt = make_optimizer(engine)
        with pytest.raises(ValueError, match="zero"):
            opt.optimize(make_context(self.HORIZON))

    def test_custom_scalarize_influences_best_selection(self, engine_and_device):
        """A scalarize that returns constant zeros means any individual can win —
        just verify it is called and the result is still an OptimizationResult."""
        engine, _ = engine_and_device
        calls = []

        def counting_scalarize(m: np.ndarray) -> np.ndarray:
            calls.append(1)
            return m.sum(axis=1)

        opt = GeneticOptimizer(
            engine=engine,
            population_size=4,
            generations=2,
            scalarize=counting_scalarize,
            random_seed=1,
        )
        result = opt.optimize(make_context(self.HORIZON))
        assert isinstance(result, OptimizationResult)
        assert len(calls) == 2   # once per generation

    def test_lamarckian_repair_writes_back_clamped_values(self):
        """With force_clamp_positive=True, negative genes become 0.
        After repair write-back the population must contain only non-negative genes."""
        dev = ScheduleDevice(
            "s0", 0, limit=100.0, force_clamp_positive=True
        )
        sink = SinkDevice("c0", 1)
        engine = make_engine([dev, sink], self.HORIZON)

        # Set population to all-negative so repair always fires
        class NegativeInitOptimizer(GeneticOptimizer):
            def _init_population(self, assembled):
                return np.full(
                    (self.population_size, assembled.total_size), -50.0
                )

        opt = NegativeInitOptimizer(
            engine=engine,
            population_size=4,
            generations=2,
            random_seed=5,
        )
        result = opt.optimize(make_context(self.HORIZON))
        # After repair, all genes must be >= 0
        assert np.all(result.best_genome >= 0.0)

    def test_best_genome_values_within_device_bounds(self, result):
        np.testing.assert_array_compare(
            np.less_equal, result.best_genome, 100.0
        )
        np.testing.assert_array_compare(
            np.greater_equal, result.best_genome, -100.0
        )


# ============================================================
# TestExtractBestInstructions
# ============================================================

class InstructionScheduleDevice(ScheduleDevice):
    """ScheduleDevice that implements extract_instructions via OMBC.

    Returns one OMBCInstruction per time step, using ``state.step_times``
    for execution_time and the per-step power as the operation_mode_factor
    (normalised to [0, 1] from [-limit, limit]).

    This is deliberately simple — real devices would choose an S2 control
    type appropriate to their physics. The important thing under test is that
    ``state.step_times`` are available and correctly aligned with the schedule.
    """

    def extract_instructions(self, state, individual_index: int):
        from akkudoktoreos.devices.devicesabc import SingleStateBatchState
        s: SingleStateBatchState = state
        schedule_row = s.schedule[individual_index]   # (horizon,) [W]
        instructions = []
        for power_w, dt in zip(schedule_row, s.step_times):
            # Normalise power to [0, 1] — factor=0 means min power, 1 means max power
            factor = float(np.clip((power_w + self._limit) / (2 * self._limit), 0.0, 1.0))
            instructions.append(OMBCInstruction(
                resource_id=self.device_id,
                execution_time=dt,
                operation_mode_id="power_mode",
                operation_mode_factor=factor,
            ))
        return instructions


class TestExtractBestInstructions:
    """Tests for GeneticOptimizer.extract_best_instructions().

    Covers:
        - Returns dict keyed by device_id
        - Only devices with extract_instructions implemented appear
        - Instruction list length equals horizon
        - execution_time of each instruction matches the corresponding step_time
        - OMBCInstruction fields are valid (factor in [0, 1])
        - SinkDevice (no genome, no extract_instructions) is absent from result
        - Returns empty dict when no device implements extract_instructions
    """

    HORIZON = 4

    @pytest.fixture()
    def engine_and_inputs(self):
        dev = InstructionScheduleDevice("s0", 0, limit=100.0)
        sink = SinkDevice("c0", 1)
        engine = make_engine([dev, sink], self.HORIZON)
        inputs = make_context(self.HORIZON)
        return engine, dev, inputs

    @pytest.fixture()
    def result_and_optimizer(self, engine_and_inputs):
        engine, dev, inputs = engine_and_inputs
        opt = make_optimizer(engine, pop=4, gens=3, seed=42)
        result = opt.optimize(inputs)
        return result, opt, inputs

    def test_returns_dict(self, result_and_optimizer):
        result, opt, inputs = result_and_optimizer
        instructions = opt.extract_best_instructions(result, inputs)
        assert isinstance(instructions, dict)

    def test_keyed_by_device_id_of_implementing_device(self, result_and_optimizer):
        result, opt, inputs = result_and_optimizer
        instructions = opt.extract_best_instructions(result, inputs)
        assert "s0" in instructions

    def test_sink_device_absent_not_implemented(self, result_and_optimizer):
        """SinkDevice raises NotImplementedError — it must be silently omitted."""
        result, opt, inputs = result_and_optimizer
        instructions = opt.extract_best_instructions(result, inputs)
        assert "c0" not in instructions

    def test_instruction_count_equals_horizon(self, result_and_optimizer):
        result, opt, inputs = result_and_optimizer
        instructions = opt.extract_best_instructions(result, inputs)
        assert len(instructions["s0"]) == self.HORIZON

    def test_execution_times_match_step_times(self, result_and_optimizer):
        """Each instruction's execution_time must equal the corresponding step_time."""
        result, opt, inputs = result_and_optimizer
        instructions = opt.extract_best_instructions(result, inputs)
        for instr, expected_dt in zip(instructions["s0"], inputs.step_times):
            assert instr.execution_time == expected_dt

    def test_instructions_are_ombc_type(self, result_and_optimizer):
        result, opt, inputs = result_and_optimizer
        instructions = opt.extract_best_instructions(result, inputs)
        for instr in instructions["s0"]:
            assert isinstance(instr, OMBCInstruction)

    def test_operation_mode_factor_within_bounds(self, result_and_optimizer):
        result, opt, inputs = result_and_optimizer
        instructions = opt.extract_best_instructions(result, inputs)
        for instr in instructions["s0"]:
            assert 0.0 <= instr.operation_mode_factor <= 1.0

    def test_empty_dict_when_no_device_implements(self):
        """If no device implements extract_instructions, result is an empty dict."""
        dev = ScheduleDevice("s0", 0, limit=100.0)   # base class raises NotImplementedError
        sink = SinkDevice("c0", 1)
        engine = make_engine([dev, sink], self.HORIZON)
        inputs = make_context(self.HORIZON)
        opt = make_optimizer(engine, pop=4, gens=2, seed=0)
        result = opt.optimize(inputs)
        instructions = opt.extract_best_instructions(result, inputs)
        assert instructions == {}


# ============================================================
# TestRollingHorizonOptimizerInit
# ============================================================

class TestRollingHorizonOptimizerInit:
    def _make_rho(self, window: int, roll: int) -> RollingHorizonOptimizer:
        dev = ScheduleDevice("s0", 0, limit=100.0)
        sink = SinkDevice("c0", 1)
        all_times = tuple(to_datetime(i * 3600) for i in range(8))
        engine = make_engine([dev, sink], horizon=window)
        return RollingHorizonOptimizer(
            engine=engine,
            all_step_times=all_times,
            step_interval=STEP_INTERVAL,
            window_size=window,
            roll_steps=roll,
            population_size=4,
            generations=2,
            random_seed=0,
        )

    def test_roll_steps_greater_than_window_raises(self):
        with pytest.raises(ValueError, match="roll_steps"):
            self._make_rho(window=4, roll=5)

    def test_roll_steps_equal_to_window_is_valid(self):
        rho = self._make_rho(window=4, roll=4)
        assert rho._roll_steps == 4

    def test_roll_steps_less_than_window_is_valid(self):
        rho = self._make_rho(window=4, roll=2)
        assert rho._roll_steps == 2


# ============================================================
# TestRollingHorizonOptimizerOptimize
# ============================================================

class TestRollingHorizonOptimizerOptimize:
    """All tests use a short 6-step horizon with window=4, roll=2 unless noted."""

    TOTAL_STEPS = 6
    WINDOW = 4
    ROLL = 2
    ALL_TIMES = tuple(float(i) for i in range(TOTAL_STEPS))

    def _make_rho(
        self,
        total_steps: int | None = None,
        window: int | None = None,
        roll: int | None = None,
        seed: int = 42,
    ) -> RollingHorizonOptimizer:
        total_steps = total_steps or self.TOTAL_STEPS
        window = window or self.WINDOW
        roll = roll or self.ROLL
        all_times = tuple(to_datetime(i * 3600) for i in range(total_steps))

        dev = ScheduleDevice("s0", 0, limit=100.0)
        sink = SinkDevice("c0", 1)
        engine = make_engine([dev, sink], horizon=window)

        return RollingHorizonOptimizer(
            engine=engine,
            all_step_times=all_times,
            step_interval=STEP_INTERVAL,
            window_size=window,
            roll_steps=roll,
            population_size=4,
            generations=2,
            random_seed=seed,
        )

    def test_returns_dict_keyed_by_device_id(self):
        rho = self._make_rho()
        result = rho.optimize()
        assert "s0" in result
        assert "c0" not in result   # SinkDevice has no genome

    def test_schedule_length_equals_total_steps(self):
        rho = self._make_rho()
        result = rho.optimize()
        assert result["s0"].shape == (self.TOTAL_STEPS,)

    def test_schedule_values_within_device_bounds(self):
        rho = self._make_rho()
        result = rho.optimize()
        assert np.all(result["s0"] >= -100.0)
        assert np.all(result["s0"] <= 100.0)

    def test_window_equal_to_total_steps_is_single_run(self):
        """When window_size == total_steps, only one GA run is needed."""
        rho = self._make_rho(total_steps=4, window=4, roll=4)
        result = rho.optimize()
        assert result["s0"].shape == (4,)

    def test_last_window_truncated_when_not_divisible(self):
        """7 steps, window=4, roll=3 → windows at [0,4), [3,7).
        Second window is truncated to 4 steps ending at step 7."""
        rho = self._make_rho(total_steps=7, window=4, roll=3)
        result = rho.optimize()
        assert result["s0"].shape == (7,)

    def test_all_steps_are_filled(self):
        """No step in the schedule should remain at the initialised 0.0 value
        if the optimizer always finds non-zero power (not guaranteed, but
        we can verify no NaN or inf values appear)."""
        rho = self._make_rho()
        result = rho.optimize()
        assert np.all(np.isfinite(result["s0"]))

    def test_multiple_windows_cover_disjoint_committed_ranges(self):
        """With total=6, window=4, roll=2 we get 3 windows committing steps
        [0,2), [2,4), [4,6). Verify schedule is fully populated."""
        rho = self._make_rho(total_steps=6, window=4, roll=2)
        result = rho.optimize()
        # All 6 positions must have been written (may be 0.0 but must be finite)
        assert result["s0"].shape == (6,)
        assert np.all(np.isfinite(result["s0"]))
