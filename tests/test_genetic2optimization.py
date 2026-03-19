"""Tests for the Genetic2Optimization entry-point module (genetic2.py).

Module under test
-----------------
``akkudoktoreos.optimization.genetic2.genetic2``

Test strategy
-------------
The pure helper functions — ``_build_devices``, ``_build_topology``,
``_best_to_solution``, ``_best_to_plan`` — are unit-tested in isolation
using hand-crafted param instances and ``EnergyBus`` objects.

``Genetic2Optimization.optimize()`` is integration-tested via a
lightweight fake config/EMS that injects minimal but complete data
(one ``HybridInverterDevice`` in BATTERY mode + one
``GridConnectionDevice`` as the slack bus, short 4-step horizon).
Population and generation counts are kept small (pop=4, gen=3).

Fake config structure
---------------------
``optimize()`` traverses this attribute path:

    config.optimization.interval           → int (seconds)
    config.optimization.horizon            → int (steps, computed)
    config.optimization.genetic.individuals→ int
    config.optimization.genetic.generations→ int
    config.optimization.genetic.seed       → int | None
    config.buses.to_genetic2_param()       → list[EnergyBus]
    config.devices.to_genetic2_params()    → list[DeviceParam]
    ems.start_datetime                     → pendulum.DateTime

The fakes mirror this structure exactly.

Param construction
------------------
``GridConnectionParam`` and ``HybridInverterParam`` both inherit from
``DeviceParam``, which declares ``device_id: str`` and
``ports: tuple[EnergyPort, ...]`` as the base fields.  There are no
separate ``port_id`` or ``bus_id`` constructor arguments on the param
classes themselves — the port identity lives inside the ``EnergyPort``
objects passed as ``ports``.

Patch targets
-------------
``GeneticOptimizer`` lives in
``akkudoktoreos.optimization.genetic2.optimizer``, which is where
``genetic2.py`` imports it from.  To intercept it the monkeypatch must
target the name as bound in that module:

    akkudoktoreos.optimization.genetic2.optimizer.GeneticOptimizer

Likewise ``EnergySimulationEngine``:

    akkudoktoreos.simulation.genetic2.engine.EnergySimulationEngine

Inverter genome size
--------------------
``HybridInverterDevice`` uses two genes per time step, so its schedule
contribution in the genome has length ``2 * horizon``, not ``horizon``.
Tests that inspect schedule lengths account for this.

Return type
-----------
``optimize()`` returns ``(OptimizationSolution, EnergyManagementPlan)``.
``OptimizationSolution`` carries ``fitness_score: set[float]``,
``total_costs_amt``, ``total_revenues_amt``, and
``solution: PydanticDateTimeDataFrame`` — *not* a ``schedule`` dict.
``_best_to_solution`` and ``_best_to_plan`` work from
``BestIndividualResult``, not from ``OptimizationResult`` slices.

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

_best_to_solution
    - Returns OptimizationSolution
    - fitness_score contains best_scalar_fitness
    - total_costs_amt comes from BestIndividualResult
    - total_revenues_amt comes from BestIndividualResult
    - valid_from equals first step_time
    - valid_until equals last step_time
    - solution is PydanticDateTimeDataFrame

_best_to_plan
    - Returns EnergyManagementPlan
    - Instructions from BestIndividualResult appear in plan.instructions
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
    - Returns (OptimizationSolution, EnergyManagementPlan)
    - fitness_score is a set of floats
    - total_costs_amt is a finite float
    - total_revenues_amt is a finite float
    - solution is a PydanticDateTimeDataFrame

Genetic2Optimization.optimize — determinism
    - Same seed → same fitness_score
    - Same seed → same total_costs_amt
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Union, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pendulum import Duration

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
from akkudoktoreos.core.emplan import EnergyManagementPlan, OMBCInstruction
from akkudoktoreos.core.pydantic import PydanticDateTimeDataFrame
from akkudoktoreos.devices.devicesabc import (
    EnergyBus,
    EnergyCarrier,
    EnergyPort,
    PortDirection,
)
from akkudoktoreos.devices.genetic2.gridconnection import (
    GridConnectionDevice,
    GridConnectionParam,
)
from akkudoktoreos.devices.genetic2.homeappliance import (
    HomeApplianceDevice,
    HomeApplianceParam,
)
from akkudoktoreos.devices.genetic2.hybridinverter import (
    HybridInverterDevice,
    HybridInverterParam,
    InverterType,
)

# ---------------------------------------------------------------------------
# Imports from the module under test
# ---------------------------------------------------------------------------
from akkudoktoreos.optimization.genetic2.genetic2 import (
    Genetic2Optimization,
    _best_to_plan,
    _best_to_solution,
    _build_devices,
    _build_topology,
)
from akkudoktoreos.optimization.genetic2.genome import AssembledGenome, GenomeSlice
from akkudoktoreos.optimization.genetic2.optimizer import (
    BestIndividualResult,
    OptimizationResult,
)
from akkudoktoreos.optimization.optimization import OptimizationSolution
from akkudoktoreos.simulation.genetic2.arbitrator import VectorizedBusArbitrator
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import to_datetime

# ============================================================
# Shared type alias
# ============================================================

# The union type that _build_devices accepts as its first argument.
_ParamUnion = Union[GridConnectionParam, HomeApplianceParam, HybridInverterParam]

# ============================================================
# Shared constants
# ============================================================

HORIZON = 4
STEP_INTERVAL_SEC = 3600
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
    port = EnergyPort(port_id="p_grid", bus_id=bus_id, direction=PortDirection.BIDIRECTIONAL)
    return GridConnectionParam(
        device_id=device_id,
        ports=(port,),
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
    port = EnergyPort(port_id="p_ac", bus_id=bus_id, direction=PortDirection.BIDIRECTIONAL)
    return HybridInverterParam(
        device_id=device_id,
        ports=(port,),
        inverter_type=inverter_type,
        off_state_power_consumption_w=0.0,
        on_state_power_consumption_w=0.0,
        pv_to_ac_efficiency=1.0,
        pv_to_battery_efficiency=1.0,
        pv_max_power_w=1.0,
        pv_min_power_w=0.0,
        pv_power_w_key=None,
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

@dataclass
class _FakeGeneticCfg:
    individuals: int = 4
    generations: int = 3
    seed: int | None = 42


@dataclass
class _FakeOptCfg:
    interval: int = STEP_INTERVAL_SEC
    horizon: int = HORIZON
    genetic: _FakeGeneticCfg = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.genetic is None:
            self.genetic = _FakeGeneticCfg()


class _FakeBusesCommonSettings:
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
        self.buses = _FakeBusesCommonSettings(buses if buses is not None else [AC_BUS])
        self.optimization = opt_cfg or _FakeOptCfg()


class _FakeEMS:
    def __init__(self) -> None:
        self.start_datetime = START_DT


def _make_instance(
    device_params: list,
    buses: list[EnergyBus] | None = None,
    opt_cfg: _FakeOptCfg | None = None,
) -> Genetic2Optimization:
    """Return a ``Genetic2Optimization`` whose ``config`` and ``ems`` are fakes."""
    cfg = _FakeConfig(device_params, buses, opt_cfg)
    fake_ems = _FakeEMS()

    class _TestOptimization(Genetic2Optimization):
        @property
        def config(self):
            return cfg

        @property
        def ems(self):
            return fake_ems

    return object.__new__(_TestOptimization)


# ============================================================
# BestIndividualResult helper
# ============================================================

def _make_best_result(
    total_costs_amt: float = 5.0,
    total_revenues_amt: float = 1.0,
    instructions: dict | None = None,
) -> BestIndividualResult:
    step_index = pd.DatetimeIndex(
        [START_DT.add(seconds=i * STEP_INTERVAL_SEC) for i in range(HORIZON)]
    )
    solution_df = pd.DataFrame(
        {
            "grid_energy_wh": np.zeros(HORIZON),
            "costs_amt": np.zeros(HORIZON),
            "revenue_amt": np.zeros(HORIZON),
            "load_energy_wh": np.zeros(HORIZON),
            "losses_energy_wh": np.zeros(HORIZON),
        },
        index=step_index,
    )
    return BestIndividualResult(
        instructions=instructions if instructions is not None else {},
        solution_df=solution_df,
        total_costs_amt=total_costs_amt,
        total_revenues_amt=total_revenues_amt,
    )


def _make_opt_result(
    generations_run: int = 3,
    objective_names: list[str] | None = None,
    best_scalar_fitness: float = 0.0,
) -> OptimizationResult:
    if objective_names is None:
        objective_names = ["energy_cost_eur"]
    assembled = MagicMock(spec=AssembledGenome)
    assembled.slices = {}
    return OptimizationResult(
        best_genome=np.zeros(0),
        best_fitness_vector=np.array([best_scalar_fitness]),
        best_scalar_fitness=best_scalar_fitness,
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
        stub = cast(GridConnectionParam, _UnknownParam(device_id="x", ports=(AC_PORT_BIDIR,)))
        with pytest.warns(UserWarning, match="Unknown DeviceParam type"):
            _build_devices([stub], [AC_BUS])

    def test_unknown_param_is_skipped(self):
        stub = cast(GridConnectionParam, _UnknownParam(device_id="x", ports=(AC_PORT_BIDIR,)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            devices, _ = _build_devices([stub], [AC_BUS])
        assert len(devices) == 0

    def test_device_indices_sequential_across_types(self):
        """grid(idx=0) → appliance(idx=1) → inverter(idx=2)."""
        params: list[_ParamUnion] = [
            make_grid_param(), make_appliance_param(), make_inverter_param()
        ]
        devices, _ = _build_devices(params, [AC_BUS])
        assert len(devices) == 3
        assert devices[0]._device_index == 0
        assert devices[1]._device_index == 1
        assert devices[2]._device_index == 2

    def test_port_index_advances_past_appliance(self):
        """Appliance has 1 port → inverter's port_index must be 1."""
        params: list[_ParamUnion] = [make_appliance_param(), make_inverter_param()]
        devices, _ = _build_devices(params, [AC_BUS])
        assert devices[1]._port_index == 1

    def test_unknown_param_port_counter_still_advances(self):
        """Unknown param (1 port) skipped → next device gets port_index 1."""
        stub = cast(GridConnectionParam, _UnknownParam(device_id="x", ports=(AC_PORT_BIDIR,)))
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
# TestBestToSolution
# ============================================================

_SOLUTION_CLS_PATCH = (
    "akkudoktoreos.optimization.genetic2.genetic2.OptimizationSolution"
)


class TestBestToSolution:
    def _call(self, best, result, ctx):
        with patch(_SOLUTION_CLS_PATCH) as mock_cls:
            mock_cls.return_value = MagicMock(spec=OptimizationSolution)
            retval = _best_to_solution(best, result, ctx)
        return mock_cls, retval

    def test_calls_optimization_solution_constructor(self):
        mock_cls, _ = self._call(_make_best_result(), _make_opt_result(), _make_context())
        mock_cls.assert_called_once()

    def test_fitness_score_contains_best_scalar_fitness(self):
        result = _make_opt_result(best_scalar_fitness=99.5)
        mock_cls, _ = self._call(_make_best_result(), result, _make_context())
        kwargs = mock_cls.call_args.kwargs
        assert 99.5 in kwargs["fitness_score"]

    def test_total_costs_amt_comes_from_best_result(self):
        best = _make_best_result(total_costs_amt=12.34)
        mock_cls, _ = self._call(best, _make_opt_result(), _make_context())
        assert mock_cls.call_args.kwargs["total_costs_amt"] == pytest.approx(12.34)

    def test_total_revenues_amt_comes_from_best_result(self):
        best = _make_best_result(total_revenues_amt=3.21)
        mock_cls, _ = self._call(best, _make_opt_result(), _make_context())
        assert mock_cls.call_args.kwargs["total_revenues_amt"] == pytest.approx(3.21)

    def test_valid_from_equals_first_step_time(self):
        ctx = _make_context()
        mock_cls, _ = self._call(_make_best_result(), _make_opt_result(), ctx)
        assert mock_cls.call_args.kwargs["valid_from"] == ctx.step_times[0]

    def test_valid_until_equals_last_step_time(self):
        ctx = _make_context()
        mock_cls, _ = self._call(_make_best_result(), _make_opt_result(), ctx)
        assert mock_cls.call_args.kwargs["valid_until"] == ctx.step_times[-1]

    def test_total_losses_energy_wh_is_zero(self):
        mock_cls, _ = self._call(_make_best_result(), _make_opt_result(), _make_context())
        assert mock_cls.call_args.kwargs["total_losses_energy_wh"] == pytest.approx(0.0)


# ============================================================
# TestBestToPlan
# ============================================================


class TestBestToPlan:

    def test_returns_energy_management_plan(self):
        best = _make_best_result()
        assert isinstance(_best_to_plan(best), EnergyManagementPlan)

    def test_instructions_appear_in_plan(self):
        instr = OMBCInstruction(
            resource_id="dev0",
            execution_time=START_DT,
            operation_mode_id="mode0",
            operation_mode_factor=0.5,
        )
        best = _make_best_result(instructions={"dev0": [instr]})
        plan = _best_to_plan(best)
        assert instr in plan.instructions

    def test_no_instructions_returns_empty_plan(self):
        plan = _best_to_plan(_make_best_result(instructions={}))
        assert isinstance(plan, EnergyManagementPlan)
        assert plan.instructions == []

    def test_multiple_devices_all_instructions_collected(self):
        instr1 = OMBCInstruction(
            resource_id="dev0",
            execution_time=START_DT,
            operation_mode_id="mode0",
            operation_mode_factor=1.0,
        )
        instr2 = OMBCInstruction(
            resource_id="dev1",
            execution_time=START_DT,
            operation_mode_id="mode1",
            operation_mode_factor=0.0,
        )
        best = _make_best_result(instructions={"dev0": [instr1], "dev1": [instr2]})
        plan = _best_to_plan(best)
        assert instr1 in plan.instructions
        assert instr2 in plan.instructions


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
        stub = cast(GridConnectionParam, _UnknownParam(device_id="x", ports=(AC_PORT_BIDIR,)))
        instance = _make_instance([stub])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="No controllable devices"):
                instance.optimize()


# ============================================================
# TestOptimizeContext
# ============================================================

_OPTIMIZER_PATCH = "akkudoktoreos.optimization.genetic2.optimizer.GeneticOptimizer"
_GET_MEASUREMENT_PATCH = "akkudoktoreos.simulation.genetic2.simulation.get_measurement"
_GET_PREDICTION_PATCH = "akkudoktoreos.simulation.genetic2.simulation.get_prediction"


def _make_fake_measurement_store(value: float = 0.5):
    store = MagicMock()
    store.key_to_value.return_value = value
    return store


def _make_fake_prediction_store(horizon: int, fill: float = 0.0):
    store = MagicMock()
    store.key_to_array.return_value = np.full(horizon, fill)
    return store


def _patch_context_resolution(monkeypatch) -> None:
    monkeypatch.setattr(
        _GET_MEASUREMENT_PATCH,
        lambda: _make_fake_measurement_store(),
    )
    monkeypatch.setattr(
        _GET_PREDICTION_PATCH,
        lambda: _make_fake_prediction_store(HORIZON),
    )


class TestOptimizeContext:

    @pytest.fixture()
    def captured_context(self, monkeypatch) -> SimulationContext:
        _patch_context_resolution(monkeypatch)
        contexts: list[SimulationContext] = []

        class _CapturingOptimizer:
            def __init__(self, engine, population_size, generations, random_seed):
                pass

            def optimize(self, ctx: SimulationContext) -> OptimizationResult:
                contexts.append(ctx)
                assembled = MagicMock(spec=AssembledGenome)
                assembled.slices = {}
                return OptimizationResult(
                    best_genome=np.zeros(0),
                    best_fitness_vector=np.array([0.0]),
                    best_scalar_fitness=0.0,
                    objective_names=["energy_cost_eur"],
                    generations_run=3,
                    history=[],
                    assembled=assembled,
                )

            def extract_best_result(self, result, ctx):
                return _make_best_result()

        monkeypatch.setattr(_OPTIMIZER_PATCH, _CapturingOptimizer)

        with patch(_SOLUTION_CLS_PATCH) as mock_sol_cls:
            mock_sol_cls.return_value = MagicMock(spec=OptimizationSolution)
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

_ENGINE_PATCH = "akkudoktoreos.simulation.genetic2.engine.EnergySimulationEngine"


class TestOptimizeBusHandling:

    def _run_and_capture_buses(
        self,
        buses: list[EnergyBus],
        monkeypatch,
    ) -> list[EnergyBus]:
        captured: list[list[EnergyBus]] = []

        class _CapEngine:
            def __init__(self, registry, buses_, arb):
                captured.append(list(buses_))

        _patch_context_resolution(monkeypatch)
        monkeypatch.setattr(_ENGINE_PATCH, _CapEngine)
        instance = _make_instance([make_grid_param()], buses=buses)
        try:
            with patch(_SOLUTION_CLS_PATCH) as mock_sol_cls:
                mock_sol_cls.return_value = MagicMock(spec=OptimizationSolution)
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

    @pytest.fixture()
    def result_pair(self, monkeypatch):
        _patch_context_resolution(monkeypatch)
        inv = make_inverter_param()
        grid = make_grid_param()
        opt_cfg = _FakeOptCfg(
            interval=STEP_INTERVAL_SEC,
            horizon=HORIZON,
            genetic=_FakeGeneticCfg(individuals=4, generations=3, seed=7),
        )
        instance = _make_instance([inv, grid], buses=[AC_BUS], opt_cfg=opt_cfg)
        return instance.optimize()

    def test_returns_tuple_of_two(self, result_pair):
        assert len(result_pair) == 2

    def test_first_element_is_optimization_solution(self, result_pair):
        assert isinstance(result_pair[0], OptimizationSolution)

    def test_second_element_is_energy_management_plan(self, result_pair):
        assert isinstance(result_pair[1], EnergyManagementPlan)

    def test_fitness_score_is_set(self, result_pair):
        assert isinstance(result_pair[0].fitness_score, set)

    def test_total_costs_amt_is_finite_float(self, result_pair):
        cost = result_pair[0].total_costs_amt
        assert isinstance(cost, float)
        assert np.isfinite(cost)

    def test_total_revenues_amt_is_finite_float(self, result_pair):
        rev = result_pair[0].total_revenues_amt
        assert isinstance(rev, float)
        assert np.isfinite(rev)

    def test_solution_has_horizon_rows(self, result_pair):
        df = result_pair[0].solution.to_dataframe()
        assert len(df) == HORIZON

    def test_solution_is_pydantic_datetime_dataframe(self, result_pair):
        assert isinstance(result_pair[0].solution, PydanticDateTimeDataFrame)


# ============================================================
# TestOptimizeDeterminism
# ============================================================


class TestOptimizeDeterminism:

    def _run(self, seed: int, monkeypatch) -> tuple[float, dict]:
        from akkudoktoreos.optimization.genetic2.optimizer import (
            GeneticOptimizer as _RealOptimizer,
        )

        _patch_context_resolution(monkeypatch)
        inv = make_inverter_param()
        grid = make_grid_param()
        opt_cfg = _FakeOptCfg(
            interval=STEP_INTERVAL_SEC,
            horizon=HORIZON,
            genetic=_FakeGeneticCfg(individuals=4, generations=3, seed=seed),
        )
        instance = _make_instance([inv, grid], buses=[AC_BUS], opt_cfg=opt_cfg)

        captured_results: list[OptimizationResult] = []

        class _CapturingOptimizer:
            def __init__(self, engine, population_size, generations, random_seed):
                self._real = _RealOptimizer(
                    engine=engine,
                    population_size=population_size,
                    generations=generations,
                    random_seed=random_seed,
                )

            def optimize(self, ctx):
                result = self._real.optimize(ctx)
                captured_results.append(result)
                return result

            def extract_best_result(self, result, ctx):
                return self._real.extract_best_result(result, ctx)

        with patch(_OPTIMIZER_PATCH, _CapturingOptimizer):
            instance.optimize()

        opt_result = captured_results[0]
        genome_by_device: dict[str, np.ndarray] = {
            dev_id: opt_result.best_genome[slc.start : slc.end]
            for dev_id, slc in opt_result.assembled.slices.items()
        }
        return opt_result.best_scalar_fitness, genome_by_device

    def test_same_seed_same_fitness(self, monkeypatch):
        f1, _ = self._run(seed=99, monkeypatch=monkeypatch)
        f2, _ = self._run(seed=99, monkeypatch=monkeypatch)
        assert f1 == pytest.approx(f2)

    def test_same_seed_same_genome(self, monkeypatch):
        _, g1 = self._run(seed=99, monkeypatch=monkeypatch)
        _, g2 = self._run(seed=99, monkeypatch=monkeypatch)
        for device_id in g1:
            np.testing.assert_array_almost_equal(
                g1[device_id],
                g2[device_id],
                err_msg=f"Genome for {device_id!r} differs between identical-seed runs",
            )


# ============================================================
# TestSimulationContextPredictionTracking
# ============================================================


from akkudoktoreos.core.cache import CacheEnergyManagementStore
from akkudoktoreos.simulation.genetic2.simulation import _prediction_key_tracker


class TestSimulationContextPredictionTracking:

    def setup_method(self):
        """Clear cache and tracker before each test."""
        CacheEnergyManagementStore().clear()
        _prediction_key_tracker.clear()

    @pytest.fixture()
    def context(self) -> SimulationContext:
        return _make_context()

    def test_resolved_prediction_keys_empty_on_init(self, context):
        assert _prediction_key_tracker.get(id(context), set()) == set()

    def test_resolve_prediction_records_key(self, monkeypatch):
        context = _make_context()
        monkeypatch.setattr(
            _GET_PREDICTION_PATCH,
            lambda: _make_fake_prediction_store(HORIZON),
        )
        context.resolve_prediction("pv_forecast_w")
        assert "pv_forecast_w" in _prediction_key_tracker.get(id(context), set())

    def test_resolve_prediction_records_multiple_keys(self, monkeypatch):
        context = _make_context()
        monkeypatch.setattr(
            _GET_PREDICTION_PATCH,
            lambda: _make_fake_prediction_store(HORIZON),
        )
        context.resolve_prediction("pv_forecast_w")
        context.resolve_prediction("elec_price_amt_kwh")
        keys = _prediction_key_tracker.get(id(context), set())
        assert "pv_forecast_w" in keys
        assert "elec_price_amt_kwh" in keys

    def test_resolve_prediction_same_key_recorded_once(self, monkeypatch):
        context = _make_context()
        monkeypatch.setattr(
            _GET_PREDICTION_PATCH,
            lambda: _make_fake_prediction_store(HORIZON),
        )
        context.resolve_prediction("pv_forecast_w")
        context.resolve_prediction("pv_forecast_w")
        assert len(_prediction_key_tracker.get(id(context), set())) == 1

    def test_resolved_predictions_returns_dict(self, monkeypatch):
        context = _make_context()
        monkeypatch.setattr(
            _GET_PREDICTION_PATCH,
            lambda: _make_fake_prediction_store(HORIZON),
        )
        context.resolve_prediction("pv_forecast_w")
        result = context.resolved_predictions()
        assert isinstance(result, dict)

    def test_resolved_predictions_contains_resolved_keys(self, monkeypatch):
        context = _make_context()
        monkeypatch.setattr(
            _GET_PREDICTION_PATCH,
            lambda: _make_fake_prediction_store(HORIZON),
        )
        context.resolve_prediction("pv_forecast_w")
        context.resolve_prediction("elec_price_amt_kwh")
        result = context.resolved_predictions()
        assert "pv_forecast_w" in result
        assert "elec_price_amt_kwh" in result

    def test_resolved_predictions_values_are_numpy_arrays(self, monkeypatch):
        context = _make_context()
        monkeypatch.setattr(
            _GET_PREDICTION_PATCH,
            lambda: _make_fake_prediction_store(HORIZON),
        )
        context.resolve_prediction("pv_forecast_w")
        result = context.resolved_predictions()
        assert isinstance(result["pv_forecast_w"], np.ndarray)

    def test_resolved_predictions_array_length_equals_horizon(self, monkeypatch):
        context = _make_context()
        monkeypatch.setattr(
            _GET_PREDICTION_PATCH,
            lambda: _make_fake_prediction_store(HORIZON),
        )
        context.resolve_prediction("pv_forecast_w")
        result = context.resolved_predictions()
        assert len(result["pv_forecast_w"]) == HORIZON

    def test_resolved_predictions_empty_when_no_keys_resolved(self):
        context = _make_context()
        assert context.resolved_predictions() == {}

    def test_new_context_instance_has_independent_tracking(self, monkeypatch):
        monkeypatch.setattr(
            _GET_PREDICTION_PATCH,
            lambda: _make_fake_prediction_store(HORIZON),
        )
        ctx1 = _make_context()
        ctx2 = _make_context()
        ctx1.resolve_prediction("pv_forecast_w")
        assert "pv_forecast_w" not in _prediction_key_tracker.get(id(ctx2), set())

    def test_measurement_resolution_does_not_pollute_prediction_keys(self, monkeypatch):
        context = _make_context()
        monkeypatch.setattr(
            _GET_MEASUREMENT_PATCH,
            lambda: _make_fake_measurement_store(),
        )
        context.resolve_measurement("battery_soc")
        assert "battery_soc" not in _prediction_key_tracker.get(id(context), set())
