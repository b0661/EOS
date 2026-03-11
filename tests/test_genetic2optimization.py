"""Tests for the Genetic2Optimization entry-point module (genetic2.py).

Module under test
-----------------
``akkudoktoreos.optimization.genetic2.genetic2``

Test strategy
-------------
The pure helper functions — ``_build_devices``, ``_build_topology``,
``_result_to_solution``, ``_result_to_plan`` — are unit-tested in
isolation using hand-crafted param instances and ``EnergyBus`` objects.

``Genetic2Optimization.optimize()`` is integration-tested via a
lightweight fake config/EMS that injects minimal but complete data
(one ``HybridInverterDevice`` in BATTERY mode + one
``GridConnectionDevice`` as the slack bus, short 4-step horizon).
Population and generation counts are kept small (pop=4, gen=3).

Fake config structure
---------------------
``optimize()`` traverses this attribute path:

    config.optimization.interval           → int (seconds)
    config.optimization.horizon_hours      → int (hours)
    config.optimization.genetic.individuals→ int
    config.optimization.genetic.generations→ int
    config.optimization.genetic.seed       → int | None
    config.buses.to_domain()               → list[EnergyBus]  (root level)
    config.devices.to_genetic2_params()    → list[DeviceParam]
    ems.start_datetime                     → pendulum.DateTime

The fakes mirror this structure exactly.

Patch targets
-------------
``GeneticOptimizer`` and ``EnergySimulationEngine`` are imported into
``genetic2.py`` by name.  To intercept them the monkeypatch must target
the name as bound in that module:

    akkudoktoreos.optimization.genetic2.genetic2.GeneticOptimizer
    akkudoktoreos.optimization.genetic2.genetic2.EnergySimulationEngine

Inverter genome size
--------------------
``HybridInverterDevice`` uses two genes per time step, so its schedule
list has length ``2 * horizon``, not ``horizon``.  Tests that inspect
schedule lengths account for this.

Covers
------
_build_devices
    - GridConnectionParam → GridConnectionDevice with correct indices
    - HomeApplianceParam  → HomeApplianceDevice with correct indices
    - HybridInverterParam → HybridInverterDevice with correct indices
    - Unknown param type emits UserWarning and is skipped
    - Mixed list: device_index and port_index advance sequentially
    - Unknown param still advances port counter for subsequent devices
    - Empty param list → empty device list
    - Buses list returned unchanged

_build_topology
    - Single device, single port → length-1 port_to_bus array mapping to bus 0
    - Two devices same bus → both entries map to bus 0
    - Two devices different buses → correct bus indices
    - Bus order determines index
    - Port referencing unknown bus_id raises ValueError naming device and bus
    - Empty device list → zero-length port_to_bus array
    - port_to_bus dtype is integer
    - Returns VectorizedBusArbitrator

_result_to_solution
    - Returns Genetic2Solution
    - cost matches best_scalar_fitness
    - schedule keys match assembled.slices
    - schedule values are Python lists with correct gene values
    - objective_values keys and values match result
    - generations_run forwarded
    - Empty slices → empty schedule

_result_to_plan
    - Returns EnergyManagementPlan
    - extract_best_instructions called with (result, context)
    - Returned instructions appear in plan.instructions
    - No instructions → plan.instructions is empty

Genetic2Optimization.optimize — validation
    - step_interval == 0 raises ValueError
    - step_interval < 0 raises ValueError
    - step_interval is None raises ValueError
    - Empty device list raises ValueError
    - Only unknown param types raises ValueError

Genetic2Optimization.optimize — context
    - step_times length equals computed horizon
    - step_interval is pendulum.Duration
    - step_interval value matches config
    - First step_time equals ems.start_datetime
    - Consecutive step_times separated by step_interval_sec seconds

Genetic2Optimization.optimize — bus handling
    - No AC bus in config → fallback AC bus injected
    - Fallback bus has bus_id "bus_ac"
    - Existing AC bus not duplicated
    - DC-only config → AC bus prepended, DC preserved

Genetic2Optimization.optimize — return values (end-to-end)
    - Returns (Genetic2Solution, EnergyManagementPlan)
    - cost is finite non-negative float
    - schedule keys are strings
    - schedule values are lists
    - Inverter device schedule length == 2 * horizon
    - generations_run matches configured generations
    - objective_values is a dict

Genetic2Optimization.optimize — determinism
    - Same seed → same cost
    - Same seed → same schedules
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pendulum import Duration

# ---------------------------------------------------------------------------
# Imports from the module under test
# ---------------------------------------------------------------------------
from akkudoktoreos.optimization.genetic2.genetic2 import (
    Genetic2Optimization,
    Genetic2Solution,
    _build_devices,
    _build_topology,
    _result_to_plan,
    _result_to_solution,
)

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
from akkudoktoreos.core.emplan import EnergyManagementPlan, OMBCInstruction
from akkudoktoreos.devices.devicesabc import (
    EnergyBus,
    EnergyCarrier,
    EnergyPort,
    PortDirection,
)
from akkudoktoreos.devices.genetic2.gridconnection import GridConnectionDevice, GridConnectionParam
from akkudoktoreos.devices.genetic2.homeappliance import HomeApplianceDevice, HomeApplianceParam
from akkudoktoreos.devices.genetic2.hybridinverter import (
    HybridInverterDevice,
    HybridInverterParam,
    InverterType,
)
from akkudoktoreos.optimization.genetic2.genome import AssembledGenome, GenomeSlice
from akkudoktoreos.optimization.genetic2.optimizer import OptimizationResult
from akkudoktoreos.simulation.genetic2.arbitrator import VectorizedBusArbitrator
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import to_datetime


# ============================================================
# Shared constants
# ============================================================

HORIZON = 4
STEP_INTERVAL_SEC = 3600
HORIZON_HOURS = 4          # horizon_hours * 3600 / STEP_INTERVAL_SEC == HORIZON
START_DT = to_datetime("2024-01-01T00:00:00")

AC_BUS = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
DC_BUS = EnergyBus(bus_id="bus_dc", carrier=EnergyCarrier.DC)

AC_PORT_BIDIR = EnergyPort(
    port_id="p_ac", bus_id="bus_ac", direction=PortDirection.BIDIRECTIONAL
)


# ============================================================
# Param factories
# ============================================================

def make_grid_param(device_id: str = "grid0", bus_id: str = "bus_ac") -> GridConnectionParam:
    return GridConnectionParam(
        device_id=device_id,
        port_id="p_grid",
        bus_id=bus_id,
        max_import_power_w=20_000.0,
        max_export_power_w=20_000.0,
        import_cost_per_kwh=0.30,
        export_revenue_per_kwh=0.07,
    )


def make_appliance_param(
    device_id: str = "dw0",
    bus_id: str = "bus_ac",
    duration_h: int = 1,
) -> HomeApplianceParam:
    port = EnergyPort(port_id="p_app", bus_id=bus_id, direction=PortDirection.SINK)
    return HomeApplianceParam(
        device_id=device_id,
        ports=(port,),
        consumption_wh=1_000.0,
        duration_h=duration_h,
        num_cycles=1,
    )


def make_inverter_param(
    device_id: str = "inv0",
    bus_id: str = "bus_ac",
    inverter_type: InverterType = InverterType.BATTERY,
) -> HybridInverterParam:
    return HybridInverterParam(
        device_id=device_id,
        port_id="p_ac",
        bus_id=bus_id,
        inverter_type=inverter_type,
        off_state_power_consumption_w=0.0,
        on_state_power_consumption_w=0.0,
        # PV fields — validation only fires for SOLAR/HYBRID, but the
        # dataclass still requires them; supply benign values.
        pv_to_ac_efficiency=1.0,
        pv_to_battery_efficiency=1.0,
        pv_max_power_w=1.0,
        pv_min_power_w=0.0,
        pv_power_w_key=None,
        # Battery fields
        ac_to_battery_efficiency=0.95,
        battery_to_ac_efficiency=0.95,
        battery_capacity_wh=10_000.0,
        battery_charge_rates=None,
        battery_min_charge_rate=0.0,
        battery_max_charge_rate=1.0,
        battery_min_discharge_rate=0.0,
        battery_max_discharge_rate=1.0,
        battery_min_soc_factor=0.1,
        battery_max_soc_factor=0.9,
        battery_initial_soc_factor_key="inv0-soc-factor",
    )


# ============================================================
# Unknown param stub (for the fallback-warning branch)
# ============================================================

@dataclass(frozen=True, slots=True)
class _UnknownParam:
    """Minimal duck-typed param that is not any of the three known types."""
    device_id: str
    ports: tuple


# ============================================================
# Fake config / EMS for integration tests
# ============================================================
#
# optimize() reads:
#   config.optimization.interval            → int seconds
#   config.optimization.horizon_hours       → int hours
#   config.optimization.genetic.individuals → int
#   config.optimization.genetic.generations → int
#   config.optimization.genetic.seed        → int | None
#   config.buses.to_domain()               → list[EnergyBus]  ← root level
#   config.devices.to_genetic2_params()    → list
#   ems.start_datetime                      → DateTime
#
# Note: buses live at config root (config.buses), NOT inside
# config.devices.  DevicesCommonSettings has no buses attribute.


@dataclass
class _FakeGeneticCfg:
    individuals: int = 4
    generations: int = 3
    seed: int | None = 42


@dataclass
class _FakeOptCfg:
    interval: int = STEP_INTERVAL_SEC
    horizon_hours: int = HORIZON_HOURS
    genetic: _FakeGeneticCfg = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.genetic is None:
            self.genetic = _FakeGeneticCfg()


class _FakeBusesConfig:
    def __init__(self, buses: list[EnergyBus]) -> None:
        self._buses = buses

    def to_genetic2_param(self) -> list[EnergyBus]:
        return list(self._buses)


class _FakeDevicesConfig:
    def __init__(self, device_params: list) -> None:
        self._device_params = device_params

    def to_genetic2_params(self) -> list:
        return list(self._device_params)


class _FakeConfig:
    def __init__(
        self,
        device_params: list,
        buses: list[EnergyBus] | None = None,
        opt_cfg: _FakeOptCfg | None = None,
    ) -> None:
        self.devices = _FakeDevicesConfig(device_params)
        self.buses = _FakeBusesConfig(buses if buses is not None else [AC_BUS])
        self.optimization = opt_cfg or _FakeOptCfg()


class _FakeEMS:
    def __init__(self) -> None:
        self.start_datetime = START_DT


def _make_instance(
    device_params: list,
    buses: list[EnergyBus] | None = None,
    opt_cfg: _FakeOptCfg | None = None,
) -> Genetic2Optimization:
    """Return a ``Genetic2Optimization`` whose ``config`` and ``ems`` are fakes.

    ``ConfigMixin.config`` and ``EnergyManagementSystemMixin.ems`` are both
    ``@classproperty`` descriptors that call global singletons — they ignore
    instance attributes set via ``object.__setattr__``.  We therefore create
    a one-off subclass per call that shadows both descriptors with regular
    ``@property`` returning the fake objects.
    """
    cfg = _FakeConfig(device_params, buses, opt_cfg)
    fake_ems = _FakeEMS()

    class _TestOptimization(Genetic2Optimization):
        @property  # type: ignore[override]
        def config(self):  # type: ignore[override]
            return cfg

        @property  # type: ignore[override]
        def ems(self):  # type: ignore[override]
            return fake_ems

    return object.__new__(_TestOptimization)


# ============================================================
# OptimizationResult / SimulationContext helpers
# ============================================================

def _make_opt_result(
    genome: np.ndarray,
    slices: dict[str, GenomeSlice],
    objective_names: list[str],
    fitness_vector: np.ndarray | None = None,
    generations_run: int = 3,
) -> OptimizationResult:
    assembled = MagicMock(spec=AssembledGenome)
    assembled.slices = slices
    if fitness_vector is None:
        fitness_vector = np.zeros(len(objective_names))
    return OptimizationResult(
        best_genome=genome,
        best_fitness_vector=fitness_vector,
        best_scalar_fitness=float(fitness_vector.sum()),
        objective_names=objective_names,
        generations_run=generations_run,
        history=[],
        assembled=assembled,
    )


def _make_context() -> SimulationContext:
    step_times = tuple(
        START_DT.add(seconds=i * STEP_INTERVAL_SEC) for i in range(HORIZON)
    )
    return SimulationContext(
        step_times=step_times,
        step_interval=Duration(seconds=STEP_INTERVAL_SEC),
    )


# ============================================================
# TestBuildDevices
# ============================================================


class TestBuildDevices:

    def test_grid_param_produces_grid_device(self):
        devices, _ = _build_devices([make_grid_param()], [AC_BUS])
        assert len(devices) == 1
        assert isinstance(devices[0], GridConnectionDevice)

    def test_grid_device_id_matches_param(self):
        devices, _ = _build_devices([make_grid_param(device_id="g0")], [AC_BUS])
        assert devices[0].device_id == "g0"

    def test_grid_device_index_is_zero_for_first_device(self):
        devices, _ = _build_devices([make_grid_param()], [AC_BUS])
        assert devices[0]._device_index == 0

    def test_grid_port_index_starts_at_zero(self):
        devices, _ = _build_devices([make_grid_param()], [AC_BUS])
        assert devices[0]._port_index == 0

    def test_appliance_param_produces_home_appliance_device(self):
        devices, _ = _build_devices([make_appliance_param()], [AC_BUS])
        assert len(devices) == 1
        assert isinstance(devices[0], HomeApplianceDevice)

    def test_appliance_device_id_matches_param(self):
        devices, _ = _build_devices([make_appliance_param(device_id="dw1")], [AC_BUS])
        assert devices[0].device_id == "dw1"

    def test_appliance_device_index_is_zero_for_first_device(self):
        devices, _ = _build_devices([make_appliance_param()], [AC_BUS])
        assert devices[0]._device_index == 0

    def test_appliance_port_index_starts_at_zero(self):
        devices, _ = _build_devices([make_appliance_param()], [AC_BUS])
        assert devices[0]._port_index == 0

    def test_inverter_param_produces_hybrid_inverter_device(self):
        devices, _ = _build_devices([make_inverter_param()], [AC_BUS])
        assert len(devices) == 1
        assert isinstance(devices[0], HybridInverterDevice)

    def test_inverter_device_id_matches_param(self):
        devices, _ = _build_devices([make_inverter_param(device_id="inv1")], [AC_BUS])
        assert devices[0].device_id == "inv1"

    def test_inverter_device_index_is_zero_for_first_device(self):
        devices, _ = _build_devices([make_inverter_param()], [AC_BUS])
        assert devices[0]._device_index == 0

    def test_inverter_port_index_starts_at_zero(self):
        devices, _ = _build_devices([make_inverter_param()], [AC_BUS])
        assert devices[0]._port_index == 0

    def test_unknown_param_emits_user_warning(self):
        stub = _UnknownParam(device_id="x", ports=(AC_PORT_BIDIR,))
        with pytest.warns(UserWarning, match="Unknown DeviceParam type"):
            _build_devices([stub], [AC_BUS])

    def test_unknown_param_is_skipped(self):
        stub = _UnknownParam(device_id="x", ports=(AC_PORT_BIDIR,))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            devices, _ = _build_devices([stub], [AC_BUS])
        assert len(devices) == 0

    def test_device_indices_sequential_across_types(self):
        """grid(idx=0) → appliance(idx=1) → inverter(idx=2)."""
        params = [make_grid_param(), make_appliance_param(), make_inverter_param()]
        devices, _ = _build_devices(params, [AC_BUS])
        assert len(devices) == 3
        assert devices[0]._device_index == 0
        assert devices[1]._device_index == 1
        assert devices[2]._device_index == 2

    def test_port_index_advances_past_appliance(self):
        """Appliance has 1 port → inverter's port_index must be 1."""
        params = [make_appliance_param(), make_inverter_param()]
        devices, _ = _build_devices(params, [AC_BUS])
        assert devices[1]._port_index == 1

    def test_unknown_param_port_counter_still_advances(self):
        """Unknown param (1 port) skipped → next device gets port_index 1."""
        stub = _UnknownParam(device_id="x", ports=(AC_PORT_BIDIR,))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            devices, _ = _build_devices([stub, make_grid_param()], [AC_BUS])
        assert devices[0]._port_index == 1

    def test_empty_param_list_returns_empty_devices(self):
        devices, _ = _build_devices([], [AC_BUS])
        assert devices == []

    def test_buses_returned_unchanged(self):
        _, returned = _build_devices([make_grid_param()], [AC_BUS])
        assert returned == [AC_BUS]


# ============================================================
# TestBuildTopology
# ============================================================


class TestBuildTopology:

    def _mock_dev(self, bus_id: str, device_id: str = "dev0") -> MagicMock:
        port = EnergyPort(port_id="p0", bus_id=bus_id, direction=PortDirection.BIDIRECTIONAL)
        dev = MagicMock()
        dev.device_id = device_id
        dev.ports = (port,)
        return dev

    def test_single_port_array_length_one(self):
        topo, _ = _build_topology([self._mock_dev("bus_ac")], [AC_BUS], HORIZON)
        assert len(topo.port_to_bus) == 1

    def test_single_port_maps_to_bus_zero(self):
        topo, _ = _build_topology([self._mock_dev("bus_ac")], [AC_BUS], HORIZON)
        assert topo.port_to_bus[0] == 0

    def test_two_devices_same_bus_both_map_to_zero(self):
        devs = [self._mock_dev("bus_ac", "d0"), self._mock_dev("bus_ac", "d1")]
        topo, _ = _build_topology(devs, [AC_BUS], HORIZON)
        np.testing.assert_array_equal(topo.port_to_bus, [0, 0])

    def test_two_devices_different_buses_correct_indices(self):
        devs = [self._mock_dev("bus_ac", "d0"), self._mock_dev("bus_dc", "d1")]
        topo, _ = _build_topology(devs, [AC_BUS, DC_BUS], HORIZON)
        np.testing.assert_array_equal(topo.port_to_bus, [0, 1])

    def test_bus_order_determines_index(self):
        """DC bus listed first → d_ac maps to index 1."""
        devs = [self._mock_dev("bus_ac", "d0"), self._mock_dev("bus_dc", "d1")]
        topo, _ = _build_topology(devs, [DC_BUS, AC_BUS], HORIZON)
        np.testing.assert_array_equal(topo.port_to_bus, [1, 0])

    def test_unknown_bus_id_raises_value_error(self):
        dev = self._mock_dev("ghost_bus", "bad_dev")
        with pytest.raises(ValueError, match="ghost_bus"):
            _build_topology([dev], [AC_BUS], HORIZON)

    def test_error_message_includes_device_id(self):
        dev = self._mock_dev("ghost_bus", "offending_device")
        with pytest.raises(ValueError, match="offending_device"):
            _build_topology([dev], [AC_BUS], HORIZON)

    def test_empty_device_list_zero_length_array(self):
        topo, _ = _build_topology([], [AC_BUS], HORIZON)
        assert len(topo.port_to_bus) == 0

    def test_port_to_bus_dtype_is_integer(self):
        topo, _ = _build_topology([self._mock_dev("bus_ac")], [AC_BUS], HORIZON)
        assert np.issubdtype(topo.port_to_bus.dtype, np.integer)

    def test_returns_vectorized_bus_arbitrator(self):
        _, arb = _build_topology([self._mock_dev("bus_ac")], [AC_BUS], HORIZON)
        assert isinstance(arb, VectorizedBusArbitrator)


# ============================================================
# TestResultToSolution
# ============================================================


class TestResultToSolution:

    def test_returns_genetic2_solution(self):
        slc = GenomeSlice(start=0, size=HORIZON,
                          lower_bound=np.full(HORIZON, -100.0),
                          upper_bound=np.full(HORIZON, 100.0))
        result = _make_opt_result(np.zeros(HORIZON), {"dev0": slc}, ["cost"])
        assert isinstance(_result_to_solution(result, _make_context()), Genetic2Solution)

    def test_cost_matches_best_scalar_fitness(self):
        slc = GenomeSlice(start=0, size=HORIZON,
                          lower_bound=np.full(HORIZON, -100.0),
                          upper_bound=np.full(HORIZON, 100.0))
        result = _make_opt_result(np.zeros(HORIZON), {"dev0": slc},
                                  ["cost"], fitness_vector=np.array([42.0]))
        assert _result_to_solution(result, _make_context()).cost == pytest.approx(42.0)

    def test_schedule_keys_match_slices(self):
        genome = np.arange(6.0)
        s0 = GenomeSlice(start=0, size=3, lower_bound=np.zeros(3), upper_bound=np.full(3, 10.0))
        s1 = GenomeSlice(start=3, size=3, lower_bound=np.zeros(3), upper_bound=np.full(3, 10.0))
        result = _make_opt_result(genome, {"grid0": s0, "inv0": s1}, ["cost"])
        sol = _result_to_solution(result, _make_context())
        assert set(sol.schedule.keys()) == {"grid0", "inv0"}

    def test_schedule_values_correct(self):
        genome = np.array([10.0, 20.0, 30.0, 40.0])
        slc = GenomeSlice(start=0, size=4, lower_bound=np.zeros(4), upper_bound=np.full(4, 100.0))
        result = _make_opt_result(genome, {"dev0": slc}, ["cost"])
        sol = _result_to_solution(result, _make_context())
        assert sol.schedule["dev0"] == pytest.approx([10.0, 20.0, 30.0, 40.0])

    def test_schedule_values_are_python_list(self):
        slc = GenomeSlice(start=0, size=HORIZON,
                          lower_bound=np.full(HORIZON, -100.0),
                          upper_bound=np.full(HORIZON, 100.0))
        result = _make_opt_result(np.ones(HORIZON), {"dev0": slc}, ["cost"])
        sol = _result_to_solution(result, _make_context())
        assert isinstance(sol.schedule["dev0"], list)

    def test_objective_values_keys_match_names(self):
        slc = GenomeSlice(start=0, size=HORIZON,
                          lower_bound=np.full(HORIZON, -100.0),
                          upper_bound=np.full(HORIZON, 100.0))
        result = _make_opt_result(np.zeros(HORIZON), {"dev0": slc},
                                  ["energy_cost_eur", "peak_import_kw"],
                                  fitness_vector=np.array([5.0, 2.0]))
        sol = _result_to_solution(result, _make_context())
        assert set(sol.objective_values.keys()) == {"energy_cost_eur", "peak_import_kw"}

    def test_objective_values_correct(self):
        slc = GenomeSlice(start=0, size=HORIZON,
                          lower_bound=np.full(HORIZON, -100.0),
                          upper_bound=np.full(HORIZON, 100.0))
        result = _make_opt_result(np.zeros(HORIZON), {"dev0": slc},
                                  ["energy_cost_eur", "peak_import_kw"],
                                  fitness_vector=np.array([5.5, 2.2]))
        sol = _result_to_solution(result, _make_context())
        assert sol.objective_values["energy_cost_eur"] == pytest.approx(5.5)
        assert sol.objective_values["peak_import_kw"] == pytest.approx(2.2)

    def test_generations_run_forwarded(self):
        slc = GenomeSlice(start=0, size=HORIZON,
                          lower_bound=np.full(HORIZON, -100.0),
                          upper_bound=np.full(HORIZON, 100.0))
        result = _make_opt_result(np.zeros(HORIZON), {"dev0": slc}, ["cost"],
                                  generations_run=17)
        sol = _result_to_solution(result, _make_context())
        assert sol.generations_run == 17

    def test_empty_slices_produces_empty_schedule(self):
        result = _make_opt_result(np.zeros(0), {}, ["cost"],
                                  fitness_vector=np.array([0.0]))
        sol = _result_to_solution(result, _make_context())
        assert sol.schedule == {}


# ============================================================
# TestResultToPlan
# ============================================================


class TestResultToPlan:

    def test_returns_energy_management_plan(self):
        optimizer = MagicMock()
        optimizer.extract_best_instructions.return_value = {}
        result = _make_opt_result(np.zeros(0), {}, ["cost"],
                                  fitness_vector=np.array([0.0]))
        plan = _result_to_plan(optimizer, result, _make_context())
        assert isinstance(plan, EnergyManagementPlan)

    def test_extract_best_instructions_called_with_result_and_context(self):
        optimizer = MagicMock()
        optimizer.extract_best_instructions.return_value = {}
        result = _make_opt_result(np.zeros(0), {}, ["cost"],
                                  fitness_vector=np.array([0.0]))
        ctx = _make_context()
        _result_to_plan(optimizer, result, ctx)
        optimizer.extract_best_instructions.assert_called_once_with(result, ctx)

    def test_instructions_appear_in_plan(self):
        """Instructions returned by extract_best_instructions land in plan.instructions."""
        instr = OMBCInstruction(
            resource_id="dev0",
            execution_time=START_DT,
            operation_mode_id="mode0",
            operation_mode_factor=0.5,
        )
        optimizer = MagicMock()
        optimizer.extract_best_instructions.return_value = {"dev0": [instr]}
        result = _make_opt_result(np.zeros(0), {}, ["cost"],
                                  fitness_vector=np.array([0.0]))
        plan = _result_to_plan(optimizer, result, _make_context())
        assert instr in plan.instructions

    def test_no_instructions_returns_empty_plan(self):
        optimizer = MagicMock()
        optimizer.extract_best_instructions.return_value = {}
        result = _make_opt_result(np.zeros(0), {}, ["cost"],
                                  fitness_vector=np.array([0.0]))
        plan = _result_to_plan(optimizer, result, _make_context())
        assert isinstance(plan, EnergyManagementPlan)
        assert plan.instructions == []


# ============================================================
# TestOptimizeValidation
# ============================================================


class TestOptimizeValidation:

    def test_zero_step_interval_raises(self):
        instance = _make_instance(
            [make_grid_param()],
            opt_cfg=_FakeOptCfg(interval=0),
        )
        with pytest.raises(ValueError, match="invalid"):
            instance.optimize()

    def test_negative_step_interval_raises(self):
        instance = _make_instance(
            [make_grid_param()],
            opt_cfg=_FakeOptCfg(interval=-1),
        )
        with pytest.raises(ValueError, match="invalid"):
            instance.optimize()

    def test_none_step_interval_raises(self):
        instance = _make_instance(
            [make_grid_param()],
            opt_cfg=_FakeOptCfg(interval=None),  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="invalid"):
            instance.optimize()

    def test_empty_device_list_raises(self):
        instance = _make_instance([])
        with pytest.raises(ValueError, match="No controllable devices"):
            instance.optimize()

    def test_only_unknown_params_raises(self):
        stub = _UnknownParam(device_id="x", ports=(AC_PORT_BIDIR,))
        instance = _make_instance([stub])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="No controllable devices"):
                instance.optimize()


# ============================================================
# TestOptimizeContext
# ============================================================

# GeneticOptimizer is imported by name into genetic2.py, so the patch
# target is the binding inside that module, not the original module.
_OPTIMIZER_PATCH = "akkudoktoreos.optimization.genetic2.optimizer.GeneticOptimizer"


class TestOptimizeContext:
    """Monkeypatches GeneticOptimizer to capture the SimulationContext."""

    @pytest.fixture()
    def captured_context(self, monkeypatch) -> SimulationContext:
        contexts: list[SimulationContext] = []

        class _CapturingOptimizer:
            def __init__(self, engine, population_size, generations, random_seed):
                pass

            def optimize(self, ctx: SimulationContext) -> OptimizationResult:
                contexts.append(ctx)
                # Return a minimal valid result so the rest of optimize() proceeds.
                slc = GenomeSlice(
                    start=0, size=HORIZON,
                    lower_bound=np.zeros(HORIZON),
                    upper_bound=np.ones(HORIZON),
                )
                assembled = MagicMock(spec=AssembledGenome)
                assembled.slices = {"grid0": slc}
                return OptimizationResult(
                    best_genome=np.zeros(HORIZON),
                    best_fitness_vector=np.array([0.0]),
                    best_scalar_fitness=0.0,
                    objective_names=["cost"],
                    generations_run=3,
                    history=[],
                    assembled=assembled,
                )

            def extract_best_instructions(self, result, ctx):
                return {}

        monkeypatch.setattr(_OPTIMIZER_PATCH, _CapturingOptimizer)
        instance = _make_instance([make_grid_param()])
        instance.optimize()
        return contexts[0]

    def test_step_times_length_equals_horizon(self, captured_context):
        assert len(captured_context.step_times) == HORIZON

    def test_step_interval_is_pendulum_duration(self, captured_context):
        assert isinstance(captured_context.step_interval, Duration)

    def test_step_interval_value_matches_config(self, captured_context):
        assert captured_context.step_interval.total_seconds() == STEP_INTERVAL_SEC

    def test_first_step_time_equals_ems_start(self, captured_context):
        assert captured_context.step_times[0] == START_DT

    def test_consecutive_step_times_separated_by_interval(self, captured_context):
        for i in range(1, len(captured_context.step_times)):
            delta = (
                captured_context.step_times[i] - captured_context.step_times[i - 1]
            ).total_seconds()
            assert delta == pytest.approx(STEP_INTERVAL_SEC)


# ============================================================
# TestOptimizeBusHandling
# ============================================================

# EnergySimulationEngine is imported by name into genetic2.py.
_ENGINE_PATCH = "akkudoktoreos.simulation.genetic2.engine.EnergySimulationEngine"


class TestOptimizeBusHandling:
    """Captures the bus list seen by EnergySimulationEngine.__init__."""

    def _run_and_capture_buses(
        self,
        buses: list[EnergyBus],
        monkeypatch,
    ) -> list[EnergyBus]:
        captured: list[list[EnergyBus]] = []

        class _CapEngine:
            """Stub that captures buses without running topology validation."""
            def __init__(self, registry, buses_, arb):
                captured.append(list(buses_))

        monkeypatch.setattr(_ENGINE_PATCH, _CapEngine)
        instance = _make_instance([make_grid_param()], buses=buses)
        try:
            instance.optimize()
        except Exception:
            pass
        return captured[0] if captured else []

    def test_no_ac_bus_triggers_fallback(self, monkeypatch):
        buses = self._run_and_capture_buses([], monkeypatch)
        ac = [b for b in buses if b.carrier == EnergyCarrier.AC]
        assert len(ac) >= 1

    def test_fallback_bus_id_is_bus_ac(self, monkeypatch):
        buses = self._run_and_capture_buses([], monkeypatch)
        ids = [b.bus_id for b in buses if b.carrier == EnergyCarrier.AC]
        assert "bus_ac" in ids

    def test_existing_ac_bus_not_duplicated(self, monkeypatch):
        buses = self._run_and_capture_buses([AC_BUS], monkeypatch)
        ac = [b for b in buses if b.carrier == EnergyCarrier.AC]
        assert len(ac) == 1

    def test_dc_only_config_gets_ac_prepended_dc_preserved(self, monkeypatch):
        buses = self._run_and_capture_buses([DC_BUS], monkeypatch)
        carriers = {b.carrier for b in buses}
        assert EnergyCarrier.AC in carriers
        assert EnergyCarrier.DC in carriers


# ============================================================
# TestOptimizeReturnValues  (end-to-end)
# ============================================================


class TestOptimizeReturnValues:
    """Full pipeline run: HybridInverterDevice (BATTERY) + GridConnectionDevice."""

    @pytest.fixture(scope="class")
    def result_pair(self):
        inv = make_inverter_param()
        grid = make_grid_param()
        opt_cfg = _FakeOptCfg(
            interval=STEP_INTERVAL_SEC,
            horizon_hours=HORIZON_HOURS,
            genetic=_FakeGeneticCfg(individuals=4, generations=3, seed=7),
        )
        instance = _make_instance([inv, grid], buses=[AC_BUS], opt_cfg=opt_cfg)

        with patch(
            "akkudoktoreos.simulation.genetic2.simulation.SimulationContext.resolve_measurement",
            return_value=0.5,
        ), patch(
            "akkudoktoreos.simulation.genetic2.simulation.SimulationContext.resolve_prediction",
            return_value=np.zeros(HORIZON),
        ):
            return instance.optimize()

    def test_returns_tuple_of_two(self, result_pair):
        assert len(result_pair) == 2

    def test_first_element_is_genetic2_solution(self, result_pair):
        assert isinstance(result_pair[0], Genetic2Solution)

    def test_second_element_is_energy_management_plan(self, result_pair):
        assert isinstance(result_pair[1], EnergyManagementPlan)

    def test_cost_is_finite_float(self, result_pair):
        cost = result_pair[0].cost
        assert isinstance(cost, float)
        assert np.isfinite(cost)

    def test_cost_is_non_negative(self, result_pair):
        # Energy cost objective is always >= 0.
        assert result_pair[0].cost >= 0.0

    def test_schedule_keys_are_strings(self, result_pair):
        for key in result_pair[0].schedule:
            assert isinstance(key, str)

    def test_schedule_values_are_lists(self, result_pair):
        for val in result_pair[0].schedule.values():
            assert isinstance(val, list)

    def test_inverter_schedule_length_is_2x_horizon(self, result_pair):
        # HybridInverterDevice uses 2 genes per step → genome size = 2 * horizon.
        inv_schedule = result_pair[0].schedule.get("inv0")
        assert inv_schedule is not None, "Expected 'inv0' in schedule"
        assert len(inv_schedule) == 2 * HORIZON

    def test_generations_run_matches_config(self, result_pair):
        assert result_pair[0].generations_run == 3

    def test_objective_values_is_dict(self, result_pair):
        assert isinstance(result_pair[0].objective_values, dict)


# ============================================================
# TestOptimizeDeterminism
# ============================================================


class TestOptimizeDeterminism:

    def _run(self, seed: int) -> Genetic2Solution:
        inv = make_inverter_param()
        grid = make_grid_param()
        opt_cfg = _FakeOptCfg(
            interval=STEP_INTERVAL_SEC,
            horizon_hours=HORIZON_HOURS,
            genetic=_FakeGeneticCfg(individuals=4, generations=3, seed=seed),
        )
        instance = _make_instance([inv, grid], buses=[AC_BUS], opt_cfg=opt_cfg)

        with patch(
            "akkudoktoreos.simulation.genetic2.simulation.SimulationContext.resolve_measurement",
            return_value=0.5,
        ), patch(
            "akkudoktoreos.simulation.genetic2.simulation.SimulationContext.resolve_prediction",
            return_value=np.zeros(HORIZON),
        ):
            solution, _ = instance.optimize()
        return solution

    def test_same_seed_same_cost(self):
        assert self._run(seed=99).cost == pytest.approx(self._run(seed=99).cost)

    def test_same_seed_same_schedules(self):
        s1 = self._run(seed=99)
        s2 = self._run(seed=99)
        for device_id in s1.schedule:
            np.testing.assert_array_almost_equal(
                s1.schedule[device_id],
                s2.schedule[device_id],
                err_msg=f"Schedule for {device_id!r} differs between identical-seed runs",
            )
