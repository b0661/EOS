"""Tests for the continuous two-gene hybrid inverter device.

Module under test
-----------------
``akkudoktoreos.devices.genetic2.hybridinverter``

Design under test
-----------------
Two genes per time step, interleaved as
``[bat_factor₀, pv_util₀, bat_factor₁, pv_util₁, …]``:

* ``bat_factor ∈ [−1, +1]``:
  - ``> 0`` charge at ``bat_factor × max_charge_rate × capacity`` W (bat side)
  - ``< 0`` discharge at ``|bat_factor| × max_discharge_rate × capacity`` W (bat side)
  - ``= 0`` battery idle
* ``pv_util ∈ [0, 1]``:
  - Fraction of available clipped PV DC power to utilise.
  - Battery charging takes priority; surplus goes to the AC bus.

Param construction
------------------
``HybridInverterParam`` inherits ``device_id`` and ``ports`` from
``DeviceParam``.  There are no separate ``port_id`` / ``bus_id`` kwargs
on the param itself — the port identity lives inside ``EnergyPort``
objects passed as ``ports=(EnergyPort(...),)``.

Test strategy
-------------
Internal helpers (_repair_genes, _compute_ac_power, _advance_soc) are
tested in isolation.  Integration tests on apply_genome_batch confirm
composition and Lamarckian write-back.

Tolerances: rtol=1e-9 throughout; zero checks use approx(0.0, abs=1e-9).
"""

from __future__ import annotations

import numpy as np
import pytest
from pendulum import Duration

from akkudoktoreos.core.emplan import OMBCInstruction
from akkudoktoreos.devices.devicesabc import (
    EnergyPort,
    InstructionContext,
    PortDirection,
)
from akkudoktoreos.devices.genetic2.hybridinverter import (
    HybridInverterBatchState,
    HybridInverterDevice,
    HybridInverterParam,
    InverterType,
)
from akkudoktoreos.simulation.genetic2.arbitrator import DeviceGrant, PortGrant
from akkudoktoreos.utils.datetimeutil import to_datetime

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STEP_INTERVAL = 3600.0   # 1-hour steps [s]
HORIZON = 4
POP = 3
PV_KEY = "pv_forecast"
SOC_KEY = "bat_soc"

# Shared port descriptors used in param factories.
_AC_PORT = EnergyPort(port_id="p_ac", bus_id="bus_ac", direction=PortDirection.BIDIRECTIONAL)


def make_step_times(n: int = HORIZON) -> tuple:
    return tuple(to_datetime(i * STEP_INTERVAL) for i in range(n))


# ---------------------------------------------------------------------------
# FakeContext
# ---------------------------------------------------------------------------

class FakeContext:
    """Minimal SimulationContext stand-in for unit tests."""

    def __init__(
        self,
        step_times: tuple,
        step_interval_sec: float = STEP_INTERVAL,
        pv_w: np.ndarray | None = None,
        initial_soc_factor: float = 0.5,
    ) -> None:
        self.step_times = step_times
        self.step_interval = Duration(seconds=step_interval_sec)
        self.horizon = len(step_times)
        self._pv_w = pv_w if pv_w is not None else np.zeros(len(step_times))
        self._initial_soc_factor = initial_soc_factor

    def resolve_prediction(self, key: str) -> np.ndarray:
        return self._pv_w.copy()

    def resolve_measurement(self, key: str) -> float:
        return self._initial_soc_factor


def make_context(
    horizon: int = HORIZON,
    step_interval_sec: float = STEP_INTERVAL,
    pv_w: float | np.ndarray = 4_000.0,
    initial_soc_factor: float = 0.5,
) -> FakeContext:
    times = make_step_times(horizon)
    if np.ndim(pv_w) == 0:
        pv_array = np.full(horizon, float(pv_w))
    else:
        pv_array = np.asarray(pv_w, dtype=np.float64)
    return FakeContext(times, step_interval_sec, pv_array, initial_soc_factor)


# ---------------------------------------------------------------------------
# Param factories
# ---------------------------------------------------------------------------

def make_battery_param(
    device_id: str = "bat",
    capacity_wh: float = 10_000.0,
    min_soc: float = 0.1,
    max_soc: float = 0.9,
    min_charge: float = 0.05,
    max_charge: float = 1.0,
    min_discharge: float = 0.05,
    max_discharge: float = 1.0,
    ac_to_bat_eff: float = 0.95,
    bat_to_ac_eff: float = 0.95,
    charge_rates: tuple | None = None,
    off_w: float = 5.0,
    on_w: float = 10.0,
) -> HybridInverterParam:
    return HybridInverterParam(
        device_id=device_id,
        ports=(_AC_PORT,),
        inverter_type=InverterType.BATTERY,
        off_state_power_consumption_w=off_w,
        on_state_power_consumption_w=on_w,
        pv_to_ac_efficiency=0.97,
        pv_to_battery_efficiency=0.95,
        pv_max_power_w=10_000.0,
        pv_min_power_w=0.0,
        pv_power_w_key=None,
        ac_to_battery_efficiency=ac_to_bat_eff,
        battery_to_ac_efficiency=bat_to_ac_eff,
        battery_capacity_wh=capacity_wh,
        battery_charge_rates=charge_rates,
        battery_min_charge_rate=min_charge,
        battery_max_charge_rate=max_charge,
        battery_min_discharge_rate=min_discharge,
        battery_max_discharge_rate=max_discharge,
        battery_min_soc_factor=min_soc,
        battery_max_soc_factor=max_soc,
        battery_initial_soc_factor_key=SOC_KEY,
    )


def make_solar_param(
    device_id: str = "solar",
    pv_max_w: float = 8_000.0,
    off_w: float = 5.0,
    on_w: float = 10.0,
) -> HybridInverterParam:
    return HybridInverterParam(
        device_id=device_id,
        ports=(_AC_PORT,),
        inverter_type=InverterType.SOLAR,
        off_state_power_consumption_w=off_w,
        on_state_power_consumption_w=on_w,
        pv_to_ac_efficiency=0.97,
        pv_to_battery_efficiency=0.95,
        pv_max_power_w=pv_max_w,
        pv_min_power_w=0.0,
        pv_power_w_key=PV_KEY,
        ac_to_battery_efficiency=0.95,
        battery_to_ac_efficiency=0.95,
        # SOLAR type: battery fields are not validated, supply minimal sentinel values.
        battery_capacity_wh=1.0,
        battery_charge_rates=None,
        battery_min_charge_rate=0.0,
        battery_max_charge_rate=0.0,
        battery_min_discharge_rate=0.0,
        battery_max_discharge_rate=0.0,
        battery_min_soc_factor=0.0,
        battery_max_soc_factor=0.1,
        battery_initial_soc_factor_key=SOC_KEY,
    )


def make_hybrid_param(
    device_id: str = "hybrid",
    capacity_wh: float = 10_000.0,
    min_soc: float = 0.1,
    max_soc: float = 0.9,
    min_charge: float = 0.05,
    max_charge: float = 1.0,
    min_discharge: float = 0.05,
    max_discharge: float = 1.0,
    pv_max_w: float = 8_000.0,
    ac_to_bat_eff: float = 0.95,
    bat_to_ac_eff: float = 0.95,
    pv_to_ac_eff: float = 0.97,
    pv_to_bat_eff: float = 0.95,
    off_w: float = 5.0,
    on_w: float = 10.0,
) -> HybridInverterParam:
    return HybridInverterParam(
        device_id=device_id,
        ports=(_AC_PORT,),
        inverter_type=InverterType.HYBRID,
        off_state_power_consumption_w=off_w,
        on_state_power_consumption_w=on_w,
        pv_to_ac_efficiency=pv_to_ac_eff,
        pv_to_battery_efficiency=pv_to_bat_eff,
        pv_max_power_w=pv_max_w,
        pv_min_power_w=0.0,
        pv_power_w_key=PV_KEY,
        ac_to_battery_efficiency=ac_to_bat_eff,
        battery_to_ac_efficiency=bat_to_ac_eff,
        battery_capacity_wh=capacity_wh,
        battery_charge_rates=None,
        battery_min_charge_rate=min_charge,
        battery_max_charge_rate=max_charge,
        battery_min_discharge_rate=min_discharge,
        battery_max_discharge_rate=max_discharge,
        battery_min_soc_factor=min_soc,
        battery_max_soc_factor=max_soc,
        battery_initial_soc_factor_key=SOC_KEY,
    )


# ---------------------------------------------------------------------------
# Device factory
# ---------------------------------------------------------------------------

def make_device(
    param: HybridInverterParam,
    horizon: int = HORIZON,
    device_index: int = 0,
    port_index: int = 0,
    pv_w: float | np.ndarray = 4_000.0,
    initial_soc_factor: float = 0.5,
) -> HybridInverterDevice:
    context = make_context(horizon, STEP_INTERVAL, pv_w, initial_soc_factor)
    device = HybridInverterDevice(param, device_index, port_index)
    device.setup_run(context)
    return device


def make_genome(
    pop: int,
    horizon: int,
    bat_factors: list[float],
    pv_utils: list[float],
) -> np.ndarray:
    """Build interleaved genome array from per-step gene lists."""
    genes = []
    for b, p in zip(bat_factors, pv_utils):
        genes += [float(b), float(p)]
    return np.tile(genes, (pop, 1))


# ============================================================
# TestHybridInverterParamValidation
# ============================================================

class TestHybridInverterParamValidation:
    def test_negative_off_state_power_raises(self):
        with pytest.raises(ValueError, match="off_state_power_consumption_w"):
            make_battery_param(off_w=-1.0)

    def test_negative_on_state_power_raises(self):
        with pytest.raises(ValueError, match="on_state_power_consumption_w"):
            make_battery_param(on_w=-0.1)

    def test_zero_off_state_power_valid(self):
        make_battery_param(off_w=0.0)

    def test_solar_pv_efficiency_zero_raises(self):
        with pytest.raises(ValueError, match="pv_to_ac_efficiency"):
            HybridInverterParam(
                device_id="s",
                ports=(_AC_PORT,),
                inverter_type=InverterType.SOLAR,
                off_state_power_consumption_w=0.0,
                on_state_power_consumption_w=0.0,
                pv_to_ac_efficiency=0.0,
                pv_to_battery_efficiency=0.95,
                pv_max_power_w=8000.0,
                pv_min_power_w=0.0,
                pv_power_w_key=PV_KEY,
                ac_to_battery_efficiency=0.95,
                battery_to_ac_efficiency=0.95,
                battery_capacity_wh=1.0,
                battery_charge_rates=None,
                battery_min_charge_rate=0.0,
                battery_max_charge_rate=0.0,
                battery_min_discharge_rate=0.0,
                battery_max_discharge_rate=0.0,
                battery_min_soc_factor=0.0,
                battery_max_soc_factor=0.1,
                battery_initial_soc_factor_key=SOC_KEY,
            )

    def test_battery_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="battery_capacity_wh"):
            make_battery_param(capacity_wh=0.0)

    def test_battery_ac_to_bat_efficiency_zero_raises(self):
        with pytest.raises(ValueError, match="ac_to_battery_efficiency"):
            make_battery_param(ac_to_bat_eff=0.0)

    def test_battery_min_charge_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="battery charge rates"):
            make_battery_param(min_charge=0.6, max_charge=0.5)

    def test_battery_min_discharge_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="battery discharge rates"):
            make_battery_param(min_discharge=0.8, max_discharge=0.3)

    def test_battery_soc_min_equals_max_raises(self):
        with pytest.raises(ValueError, match="SoC factors"):
            make_battery_param(min_soc=0.5, max_soc=0.5)

    def test_battery_charge_rates_empty_raises(self):
        with pytest.raises(ValueError, match="battery_charge_rates must not be empty"):
            make_battery_param(charge_rates=())

    def test_battery_charge_rates_zero_raises(self):
        with pytest.raises(ValueError, match="battery_charge_rates values"):
            make_battery_param(charge_rates=(0.0, 0.5))

    def test_valid_battery_constructs(self):
        p = make_battery_param()
        assert p.inverter_type == InverterType.BATTERY

    def test_valid_solar_constructs(self):
        p = make_solar_param()
        assert p.inverter_type == InverterType.SOLAR

    def test_valid_hybrid_constructs(self):
        p = make_hybrid_param()
        assert p.inverter_type == InverterType.HYBRID


# ============================================================
# TestHybridInverterParamDerivedProperties
# ============================================================

class TestHybridInverterParamDerivedProperties:
    def test_battery_soc_wh_min(self):
        p = make_battery_param(capacity_wh=10_000.0, min_soc=0.1)
        assert p.battery_min_soc_wh == pytest.approx(1_000.0)

    def test_battery_soc_wh_max(self):
        p = make_battery_param(capacity_wh=10_000.0, max_soc=0.9)
        assert p.battery_max_soc_wh == pytest.approx(9_000.0)


# ============================================================
# TestHybridInverterDeviceTopology
# ============================================================

class TestHybridInverterDeviceTopology:
    def test_single_bidirectional_port(self):
        device = make_device(make_battery_param())
        assert len(device.ports) == 1
        assert device.ports[0].direction == PortDirection.BIDIRECTIONAL

    def test_port_ids_match_param(self):
        param = make_battery_param()
        device = make_device(param)
        assert device.ports[0].port_id == param.ports[0].port_id
        assert device.ports[0].bus_id == param.ports[0].bus_id

    def test_objective_names_empty(self):
        assert make_device(make_battery_param()).objective_names == []

    def test_device_id_matches(self):
        param = make_battery_param(device_id="inv_007")
        assert make_device(param).device_id == "inv_007"


# ============================================================
# TestSetupRun
# ============================================================

class TestSetupRun:
    def test_stores_num_steps(self):
        assert make_device(make_battery_param(), horizon=6)._num_steps == 6

    def test_stores_step_interval(self):
        assert make_device(make_battery_param())._step_interval_sec == STEP_INTERVAL

    def test_stores_step_times(self):
        ctx = make_context(horizon=4)
        dev = HybridInverterDevice(make_battery_param(), 0, 0)
        dev.setup_run(ctx)
        assert dev._step_times == ctx.step_times

    def test_solar_pv_key_none_raises(self):
        param = HybridInverterParam(
            device_id="s",
            ports=(_AC_PORT,),
            inverter_type=InverterType.SOLAR,
            off_state_power_consumption_w=0.0,
            on_state_power_consumption_w=0.0,
            pv_to_ac_efficiency=0.97,
            pv_to_battery_efficiency=0.95,
            pv_max_power_w=8000.0,
            pv_min_power_w=0.0,
            pv_power_w_key=None,
            ac_to_battery_efficiency=0.95,
            battery_to_ac_efficiency=0.95,
            battery_capacity_wh=1.0,
            battery_charge_rates=None,
            battery_min_charge_rate=0.0,
            battery_max_charge_rate=0.0,
            battery_min_discharge_rate=0.0,
            battery_max_discharge_rate=0.0,
            battery_min_soc_factor=0.0,
            battery_max_soc_factor=0.1,
            battery_initial_soc_factor_key=SOC_KEY,
        )
        with pytest.raises(ValueError, match="pv_power_w_key"):
            HybridInverterDevice(param, 0, 0).setup_run(make_context())

    def test_pv_wrong_shape_raises(self):
        dev = HybridInverterDevice(make_solar_param(), 0, 0)
        ctx = FakeContext(make_step_times(4), pv_w=np.full(2, 3000.0))
        with pytest.raises(ValueError):
            dev.setup_run(ctx)

    def test_initial_soc_out_of_bounds_raises(self):
        dev = HybridInverterDevice(make_battery_param(min_soc=0.2, max_soc=0.8), 0, 0)
        with pytest.raises(ValueError, match="initial SoC factor"):
            dev.setup_run(make_context(initial_soc_factor=-0.1))
        with pytest.raises(ValueError, match="initial SoC factor"):
            dev.setup_run(make_context(initial_soc_factor=1.1))

    def test_battery_resolves_initial_soc_wh(self):
        dev = HybridInverterDevice(
            make_battery_param(capacity_wh=10_000.0, min_soc=0.1, max_soc=0.9), 0, 0
        )
        dev.setup_run(make_context(initial_soc_factor=0.6))
        assert dev._battery_initial_soc_wh == pytest.approx(6_000.0)

    def test_solar_resolves_pv_array(self):
        dev = HybridInverterDevice(make_solar_param(), 0, 0)
        dev.setup_run(make_context(pv_w=np.full(HORIZON, 5_000.0)))
        np.testing.assert_allclose(dev._pv_power_w, 5_000.0)


# ============================================================
# TestGenomeRequirements
# ============================================================

class TestGenomeRequirements:
    """Genome has 2 genes per step for all inverter types."""

    def test_battery_genome_size(self):
        assert make_device(make_battery_param(), horizon=4).genome_requirements().size == 8

    def test_solar_genome_size(self):
        assert make_device(make_solar_param(), horizon=4).genome_requirements().size == 8

    def test_hybrid_genome_size(self):
        assert make_device(make_hybrid_param(), horizon=4).genome_requirements().size == 8

    def test_bat_factor_lower_bound_is_minus_one(self):
        slc = make_device(make_battery_param(), horizon=4).genome_requirements()
        np.testing.assert_array_equal(slc.lower_bound[0::2], np.full(4, -1.0))

    def test_bat_factor_upper_bound_is_plus_one(self):
        slc = make_device(make_battery_param(), horizon=4).genome_requirements()
        np.testing.assert_array_equal(slc.upper_bound[0::2], np.full(4, +1.0))

    def test_pv_util_lower_bound_is_zero(self):
        slc = make_device(make_hybrid_param(), horizon=4).genome_requirements()
        np.testing.assert_array_equal(slc.lower_bound[1::2], np.zeros(4))

    def test_pv_util_upper_bound_is_one(self):
        slc = make_device(make_hybrid_param(), horizon=4).genome_requirements()
        np.testing.assert_array_equal(slc.upper_bound[1::2], np.ones(4))

    def test_before_setup_run_raises(self):
        with pytest.raises(RuntimeError):
            HybridInverterDevice(make_battery_param(), 0, 0).genome_requirements()


# ============================================================
# TestCreateBatchState
# ============================================================

class TestCreateBatchState:
    def test_bat_factors_shape_and_zeros(self):
        state = make_device(make_battery_param()).create_batch_state(POP, HORIZON)
        assert state.bat_factors.shape == (POP, HORIZON)
        np.testing.assert_array_equal(state.bat_factors, 0.0)

    def test_pv_util_factors_shape_and_zeros(self):
        state = make_device(make_battery_param()).create_batch_state(POP, HORIZON)
        assert state.pv_util_factors.shape == (POP, HORIZON)
        np.testing.assert_array_equal(state.pv_util_factors, 0.0)

    def test_soc_wh_prefilled(self):
        param = make_battery_param(capacity_wh=10_000.0, min_soc=0.0, max_soc=1.0)
        state = make_device(param, initial_soc_factor=0.6).create_batch_state(POP, HORIZON)
        np.testing.assert_allclose(state.soc_wh, 6_000.0)

    def test_ac_power_w_zeros(self):
        state = make_device(make_battery_param()).create_batch_state(POP, HORIZON)
        np.testing.assert_array_equal(state.ac_power_w, 0.0)

    def test_before_setup_run_raises(self):
        with pytest.raises(RuntimeError):
            HybridInverterDevice(make_battery_param(), 0, 0).create_batch_state(POP, HORIZON)


# ============================================================
# TestRepairGenes
# ============================================================

class TestRepairGenes:
    """Tests for _repair_genes called directly on one-step vectors."""

    def _call(self, device, bat_list, pv_list, soc_list, pv_dc_avail=0.0):
        bat = np.array(bat_list, dtype=np.float64)
        pv  = np.array(pv_list,  dtype=np.float64)
        soc = np.array(soc_list, dtype=np.float64)
        return device._repair_genes(bat, pv, soc, pv_dc_avail)

    # ---- BATTERY: pv_util always zero ----

    def test_battery_pv_util_zeroed(self):
        device = make_device(make_battery_param())
        _, pv = self._call(device, [0.5], [1.0], [5000.0], pv_dc_avail=3000.0)
        assert pv[0] == pytest.approx(0.0)

    # ---- SOLAR: bat_factor always zero ----

    def test_solar_bat_factor_zeroed(self):
        device = make_device(make_solar_param())
        bat, _ = self._call(device, [0.8], [0.7], [0.0], pv_dc_avail=3000.0)
        assert bat[0] == pytest.approx(0.0)

    def test_solar_pv_util_clipped_above_one(self):
        device = make_device(make_solar_param())
        _, pv = self._call(device, [0.0], [1.5], [0.0], pv_dc_avail=3000.0)
        assert pv[0] == pytest.approx(1.0)

    def test_solar_pv_util_preserved_in_range(self):
        device = make_device(make_solar_param())
        _, pv = self._call(device, [0.0], [0.7], [0.0], pv_dc_avail=3000.0)
        assert pv[0] == pytest.approx(0.7)

    # ---- BATTERY: charging (bat > 0) ----

    def test_charge_gene_maps_through_max_charge_rate(self):
        param = make_battery_param(max_charge=0.8, min_charge=0.0,
                                   min_soc=0.0, max_soc=1.0)
        device = make_device(param)
        bat, _ = self._call(device, [1.0], [0.0], [500.0])
        assert bat[0] == pytest.approx(1.0)

    def test_charge_half_gene_gives_half_rate(self):
        param = make_battery_param(max_charge=1.0, min_charge=0.0)
        device = make_device(param)
        bat, _ = self._call(device, [0.5], [0.0], [1000.0])
        assert bat[0] == pytest.approx(0.5)

    def test_charge_clipped_to_max_charge_rate(self):
        param = make_battery_param(max_charge=0.5, min_charge=0.0)
        device = make_device(param)
        bat, _ = self._call(device, [1.0], [0.0], [1000.0])
        assert bat[0] == pytest.approx(1.0)

    def test_charge_below_min_deadband_clips_to_min(self):
        param = make_battery_param(min_charge=0.05, max_charge=1.0,
                                   min_soc=0.0, max_soc=1.0)
        device = make_device(param)
        bat, _ = self._call(device, [0.02], [0.0], [500.0])
        assert bat[0] == pytest.approx(0.05)

    def test_charge_discrete_snap_to_nearest(self):
        param = make_battery_param(charge_rates=(0.25, 0.5, 0.75))
        device = make_device(param)
        bat, _ = self._call(device, [0.6], [0.0], [1000.0])
        assert bat[0] == pytest.approx(0.5)

    def test_charge_soc_headroom_reduces_gene(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=0.9, ac_to_bat_eff=0.95,
            min_charge=0.05, max_charge=1.0,
        )
        device = make_device(param)
        bat, _ = self._call(device, [1.0], [0.0], [8800.0])
        assert bat[0] == pytest.approx(0.0)

    def test_charge_soc_at_max_zeroed(self):
        param = make_battery_param(capacity_wh=10_000.0, max_soc=0.9, min_charge=0.05)
        device = make_device(param)
        bat, _ = self._call(device, [1.0], [0.0], [9000.0])
        assert bat[0] == pytest.approx(0.0)

    # ---- BATTERY: discharging (bat < 0) ----

    def test_discharge_half_gene_gives_half_rate(self):
        param = make_battery_param(max_discharge=1.0, min_discharge=0.0,
                                   min_soc=0.0, max_soc=1.0)
        device = make_device(param)
        bat, _ = self._call(device, [-0.5], [0.0], [9000.0])
        assert bat[0] == pytest.approx(-0.5)

    def test_discharge_below_min_deadband_clips_to_min(self):
        param = make_battery_param(min_discharge=0.05, max_discharge=1.0,
                                   min_soc=0.0, max_soc=1.0)
        device = make_device(param)
        bat, _ = self._call(device, [-0.02], [0.0], [9000.0])
        assert bat[0] == pytest.approx(-0.05)

    def test_discharge_soc_at_min_zeroed(self):
        param = make_battery_param(capacity_wh=10_000.0, min_soc=0.1, min_discharge=0.05)
        device = make_device(param)
        bat, _ = self._call(device, [-1.0], [0.0], [1000.0])
        assert bat[0] == pytest.approx(0.0)

    def test_discharge_soc_floor_reduces_gene(self):
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.1, min_discharge=0.05, max_discharge=1.0,
        )
        device = make_device(param)
        bat, _ = self._call(device, [-1.0], [0.0], [2000.0])
        assert bat[0] == pytest.approx(-0.1, rel=1e-5)

    # ---- Idle battery (bat = 0) ----

    def test_idle_bat_stays_zero(self):
        device = make_device(make_battery_param())
        bat, _ = self._call(device, [0.0], [0.0], [5000.0])
        assert bat[0] == pytest.approx(0.0)

    # ---- HYBRID: both genes active ----

    def test_hybrid_pv_util_not_zeroed_when_charging(self):
        device = make_device(make_hybrid_param())
        bat, pv = self._call(device, [0.5], [0.8], [5000.0], pv_dc_avail=4000.0)
        assert bat[0] > 0.0
        assert pv[0] == pytest.approx(0.8)

    def test_hybrid_pv_util_zeroed_when_no_pv_available(self):
        """With pv_dc_avail=0.0, pv_util must be zeroed regardless of gene."""
        device = make_device(make_hybrid_param())
        _, pv = self._call(device, [0.5], [1.0], [5000.0], pv_dc_avail=0.0)
        assert pv[0] == pytest.approx(0.0)


# ============================================================
# TestComputeAcPower
# ============================================================

class TestComputeAcPower:
    """Tests for _compute_ac_power called directly on shape-(pop,) arrays."""

    def _call(self, device, bat_list, pv_list, pv_dc_avail=0.0, soc_list=None):
        bat = np.array(bat_list, dtype=np.float64)
        pv  = np.array(pv_list,  dtype=np.float64)
        soc = np.zeros(len(bat_list)) if soc_list is None else np.array(soc_list)
        return device._compute_ac_power(bat, pv, pv_dc_avail, soc)

    # ---- BATTERY: idle ----

    def test_battery_idle_no_pv_returns_off_state(self):
        param = make_battery_param(off_w=8.0)
        device = make_device(param)
        result = self._call(device, [0.0], [0.0])
        assert result[0] == pytest.approx(8.0)

    # ---- BATTERY: charging ----

    def test_battery_charge_consumes_correct_ac(self):
        param = make_battery_param(capacity_wh=10_000.0, max_charge=1.0,
                                   ac_to_bat_eff=0.95, on_w=10.0)
        device = make_device(param)
        result = self._call(device, [0.5], [0.0])
        expected = (0.5 * 1.0 * 10_000.0 / 0.95) + 10.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_charge_positive_power(self):
        result = self._call(make_device(make_battery_param()), [0.3], [0.0])
        assert result[0] > 0.0

    # ---- BATTERY: discharging ----

    def test_battery_discharge_injects_correct_ac(self):
        param = make_battery_param(capacity_wh=10_000.0, max_discharge=1.0,
                                   bat_to_ac_eff=0.95, on_w=10.0)
        device = make_device(param)
        result = self._call(device, [-0.5], [0.0])
        expected = -(0.5 * 1.0 * 10_000.0 * 0.95) + 10.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_discharge_negative_power(self):
        result = self._call(make_device(make_battery_param(on_w=1.0)), [-1.0], [0.0])
        assert result[0] < 0.0

    # ---- SOLAR: PV only ----

    def test_solar_pv_injects_correctly(self):
        param = make_solar_param(on_w=10.0)
        device = make_device(param)
        result = self._call(device, [0.0], [1.0], pv_dc_avail=4000.0)
        expected = -(4000.0 * 0.97) + 10.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_solar_pv_util_zero_returns_off_state(self):
        param = make_solar_param(off_w=5.0)
        device = make_device(param)
        result = self._call(device, [0.0], [0.0], pv_dc_avail=4000.0)
        assert result[0] == pytest.approx(5.0)

    def test_solar_partial_pv_util(self):
        param = make_solar_param(on_w=0.0)
        device = make_device(param)
        result = self._call(device, [0.0], [0.5], pv_dc_avail=4000.0)
        expected = -(4000.0 * 0.5 * 0.97)
        assert result[0] == pytest.approx(expected, rel=1e-9)

    # ---- HYBRID: charging with PV assist ----

    def test_hybrid_charge_pv_reduces_ac_draw(self):
        param = make_hybrid_param(
            capacity_wh=10_000.0, max_charge=1.0,
            ac_to_bat_eff=0.95, pv_to_bat_eff=0.95, pv_to_ac_eff=0.97, on_w=10.0,
        )
        device = make_device(param, pv_w=6000.0)
        pv_dc = 6000.0
        pv_used = 0.5 * pv_dc
        charge_bat = 0.5 * 1.0 * 10_000.0
        pv_for_bat = min(pv_used * 0.95, charge_bat)
        pv_surplus_dc = pv_used - pv_for_bat / 0.95
        pv_surplus_ac = pv_surplus_dc * 0.97
        ac_for_bat = max(0.0, charge_bat - pv_for_bat) / 0.95
        expected = ac_for_bat - pv_surplus_ac + 10.0
        result = self._call(device, [0.5], [0.5], pv_dc_avail=pv_dc)
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_hybrid_charge_pv_surplus_makes_net_negative(self):
        param = make_hybrid_param(capacity_wh=500.0, max_charge=1.0, on_w=0.0)
        device = make_device(param, pv_w=10_000.0)
        result = self._call(device, [0.1], [1.0], pv_dc_avail=10_000.0)
        assert result[0] < 0.0

    def test_hybrid_pv_util_zero_same_as_battery_only(self):
        bat_param  = make_battery_param(capacity_wh=10_000.0, max_charge=1.0,
                                        ac_to_bat_eff=0.95, on_w=10.0)
        hybr_param = make_hybrid_param(capacity_wh=10_000.0, max_charge=1.0,
                                       ac_to_bat_eff=0.95, on_w=10.0)
        bat_device  = make_device(bat_param)
        hybr_device = make_device(hybr_param, pv_w=4000.0)
        bat_result  = self._call(bat_device,  [0.4], [0.0])
        hybr_result = self._call(hybr_device, [0.4], [0.0], pv_dc_avail=4000.0)
        assert bat_result[0] == pytest.approx(hybr_result[0], rel=1e-9)

    # ---- HYBRID: discharging with PV ----

    def test_hybrid_discharge_battery_and_pv_combine(self):
        param = make_hybrid_param(
            capacity_wh=10_000.0, max_discharge=1.0,
            bat_to_ac_eff=0.95, pv_to_ac_eff=0.97, on_w=5.0,
        )
        device = make_device(param, pv_w=2000.0)
        result = self._call(device, [-0.5], [1.0], pv_dc_avail=2000.0)
        dis_bat_ac = 0.5 * 1.0 * 10_000.0 * 0.95
        pv_ac      = 2000.0 * 1.0 * 0.97
        expected   = -(dis_bat_ac + pv_ac) + 5.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    # ---- HYBRID: idle battery, PV only ----

    def test_hybrid_idle_battery_pv_injects(self):
        param = make_hybrid_param(on_w=5.0)
        device = make_device(param, pv_w=3000.0)
        result = self._call(device, [0.0], [1.0], pv_dc_avail=3000.0)
        expected = -(3000.0 * 0.97) + 5.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_hybrid_idle_battery_zero_pv_returns_off_state(self):
        param = make_hybrid_param(off_w=7.0)
        device = make_device(param, pv_w=0.0)
        result = self._call(device, [0.0], [0.0], pv_dc_avail=0.0)
        assert result[0] == pytest.approx(7.0)

    # ---- Population independence ----

    def test_mixed_population_independent(self):
        param = make_battery_param(capacity_wh=10_000.0, off_w=5.0, on_w=10.0)
        device = make_device(param)
        bat = np.array([0.0, 0.5, -0.5])
        pv  = np.zeros(3)
        result = device._compute_ac_power(bat, pv, 0.0, np.zeros(3))
        assert result[0] == pytest.approx(5.0)
        assert result[1] > 0.0
        assert result[2] < 0.0


# ============================================================
# TestAdvanceSoc
# ============================================================

class TestAdvanceSoc:
    """Tests for _advance_soc called directly."""

    def _call(self, device, soc_list, bat_list, pv_list, pv_dc_avail=0.0):
        return device._advance_soc(
            np.array(soc_list),
            np.array(bat_list),
            np.array(pv_list),
            pv_dc_avail,
        )

    def test_idle_soc_unchanged(self):
        device = make_device(make_battery_param())
        result = self._call(device, [5000.0], [0.0], [0.0])
        assert result[0] == pytest.approx(5000.0)

    def test_battery_charge_increases_soc(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_charge=1.0, ac_to_bat_eff=0.95,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, initial_soc_factor=0.2)
        result = self._call(device, [2000.0], [0.5], [0.0])
        expected = 2000.0 + 0.5 * 1.0 * 10_000.0 * 0.95 * 1.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_discharge_decreases_soc(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_discharge=1.0, min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        result = self._call(device, [8000.0], [-0.3], [0.0])
        expected = 8000.0 - 0.3 * 1.0 * 10_000.0 * 1.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_soc_clamped_at_max(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=0.9, min_soc=0.0,
            ac_to_bat_eff=1.0, max_charge=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        result = self._call(device, [8900.0], [1.0], [0.0])
        assert result[0] == pytest.approx(9000.0)

    def test_soc_clamped_at_min(self):
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.1, max_soc=1.0, max_discharge=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        result = self._call(device, [1100.0], [-1.0], [0.0])
        assert result[0] == pytest.approx(1000.0)

    def test_hybrid_charge_pv_contributes_first(self):
        param = make_hybrid_param(
            capacity_wh=10_000.0, max_charge=1.0,
            ac_to_bat_eff=0.95, pv_to_bat_eff=0.95, min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, pv_w=4000.0, initial_soc_factor=0.5)
        result = self._call(device, [3000.0], [0.5], [0.5], pv_dc_avail=4000.0)
        pv_dc_used = 0.5 * 4000.0
        charge_bat = 0.5 * 1.0 * 10_000.0
        pv_for_bat = min(pv_dc_used * 0.95, charge_bat)
        ac_used    = max(0.0, charge_bat - pv_for_bat)
        stored_wh  = (pv_for_bat + ac_used * 0.95) * 1.0
        expected   = 3000.0 + stored_wh
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_hybrid_pv_exceeds_charge_demand(self):
        param = make_hybrid_param(
            capacity_wh=1_000.0, max_charge=1.0,
            ac_to_bat_eff=0.95, pv_to_bat_eff=0.95, min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, pv_w=5000.0, initial_soc_factor=0.5)
        result = self._call(device, [200.0], [0.1], [1.0], pv_dc_avail=5000.0)
        stored_wh = 0.1 * 1.0 * 1_000.0
        expected  = 200.0 + stored_wh
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_solar_soc_stays_at_zero(self):
        device = make_device(make_solar_param())
        result = self._call(device, [0.0], [0.0], [1.0], pv_dc_avail=4000.0)
        assert result[0] == pytest.approx(0.0)


# ============================================================
# TestApplyGenomeBatch
# ============================================================

class TestApplyGenomeBatch:
    """Integration tests for the full apply_genome_batch pipeline."""

    def test_genome_shape_is_pop_by_2_horizon(self):
        device = make_device(make_battery_param(), horizon=4)
        state  = device.create_batch_state(POP, 4)
        genome = np.zeros((POP, 8))
        device.apply_genome_batch(state, genome)
        assert genome.shape == (POP, 8)

    def test_bat_factors_stored_in_state(self):
        param = make_battery_param(min_soc=0.0, max_soc=1.0)
        device = make_device(param, initial_soc_factor=0.1)
        state  = device.create_batch_state(1, 2)
        genome = np.array([[0.5, 0.0, -0.4, 0.0]])
        device.apply_genome_batch(state, genome)
        assert state.bat_factors[0, 0] == pytest.approx(0.5, rel=1e-6)

    def test_pv_util_stored_in_state(self):
        device = make_device(make_hybrid_param(), horizon=2)
        state  = device.create_batch_state(1, 2)
        genome = np.array([[0.0, 0.7, 0.0, 0.3]])
        device.apply_genome_batch(state, genome)
        assert state.pv_util_factors[0, 0] == pytest.approx(0.7, rel=1e-6)
        assert state.pv_util_factors[0, 1] == pytest.approx(0.3, rel=1e-6)

    def test_solar_bat_factors_zeroed(self):
        device = make_device(make_solar_param(), horizon=2)
        state  = device.create_batch_state(1, 2)
        genome = np.array([[0.9, 0.5, -0.7, 0.8]])
        device.apply_genome_batch(state, genome)
        np.testing.assert_array_equal(state.bat_factors, 0.0)

    def test_battery_pv_util_zeroed(self):
        device = make_device(make_battery_param(), horizon=2)
        state  = device.create_batch_state(1, 2)
        genome = np.array([[0.5, 0.9, -0.3, 0.7]])
        device.apply_genome_batch(state, genome)
        np.testing.assert_array_equal(state.pv_util_factors, 0.0)

    def test_lamarckian_writeback_bat_factor(self):
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.0, max_soc=1.0,
            min_charge=0.0, max_charge=1.0, ac_to_bat_eff=0.95,
        )
        device = make_device(param, horizon=1, initial_soc_factor=0.5)
        state  = device.create_batch_state(1, 1)
        genome = np.array([[0.3, 0.0]])
        device.apply_genome_batch(state, genome)
        assert genome[0, 0] == pytest.approx(0.3, rel=1e-6)

    def test_lamarckian_writeback_pv_util(self):
        device = make_device(make_hybrid_param(), horizon=1)
        state  = device.create_batch_state(1, 1)
        genome = np.array([[0.0, 1.5]])
        device.apply_genome_batch(state, genome)
        assert genome[0, 1] <= 1.0

    def test_soc_evolves_over_steps(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_charge=1.0, ac_to_bat_eff=1.0,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, horizon=3, initial_soc_factor=0.5)
        state  = device.create_batch_state(1, 3)
        genome = np.tile([0.1, 0.0], (1, 3))
        device.apply_genome_batch(state, genome)
        assert state.soc_wh[0, 0] == pytest.approx(5000.0 + 1000.0, rel=1e-6)
        assert state.soc_wh[0, 1] == pytest.approx(5000.0 + 2000.0, rel=1e-6)
        assert state.soc_wh[0, 2] == pytest.approx(5000.0 + 3000.0, rel=1e-6)

    def test_population_axis_independent(self):
        device = make_device(make_battery_param(capacity_wh=10_000.0), horizon=1)
        state  = device.create_batch_state(2, 1)
        genome = np.array([
            [0.5, 0.0],
            [-0.5, 0.0],
        ])
        device.apply_genome_batch(state, genome)
        assert state.ac_power_w[0, 0] > 0.0
        assert state.ac_power_w[1, 0] < 0.0


# ============================================================
# TestBuildDeviceRequest
# ============================================================

class TestBuildDeviceRequest:
    def _setup(self, param=None, pop=POP, horizon=HORIZON, device_index=7, port_index=3):
        if param is None:
            param = make_battery_param()
        device = HybridInverterDevice(param, device_index, port_index)
        device.setup_run(make_context(horizon=horizon))
        state  = device.create_batch_state(pop, horizon)
        genome = np.tile([0.3, 0.0], (pop, horizon))
        device.apply_genome_batch(state, genome)
        return device, state

    def test_device_index_matches(self):
        from akkudoktoreos.simulation.genetic2.arbitrator import DeviceRequest
        device, state = self._setup(device_index=7)
        req = device.build_device_request(state)
        assert isinstance(req, DeviceRequest)
        assert req.device_index == 7

    def test_port_index_matches(self):
        device, state = self._setup(port_index=3)
        req = device.build_device_request(state)
        assert req.port_requests[0].port_index == 3

    def test_energy_wh_equals_ac_power_times_step_h(self):
        device, state = self._setup()
        req    = device.build_device_request(state)
        step_h = STEP_INTERVAL / 3600.0
        np.testing.assert_allclose(
            req.port_requests[0].energy_wh, state.ac_power_w * step_h, rtol=1e-9,
        )

    def test_energy_wh_shape(self):
        device, state = self._setup(pop=POP, horizon=HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].energy_wh.shape == (POP, HORIZON)

    def test_min_energy_wh_shape(self):
        device, state = self._setup(pop=POP, horizon=HORIZON)
        assert device.build_device_request(state).port_requests[0].min_energy_wh.shape == (POP, HORIZON)

    def test_idle_steps_zero_min_energy(self):
        device = HybridInverterDevice(make_battery_param(), 0, 0)
        device.setup_run(make_context(horizon=HORIZON))
        state  = device.create_batch_state(1, HORIZON)
        genome = np.zeros((1, 2 * HORIZON))
        device.apply_genome_batch(state, genome)
        np.testing.assert_array_equal(
            device.build_device_request(state).port_requests[0].min_energy_wh, 0.0,
        )

    def test_charge_steps_positive_min_energy(self):
        device = HybridInverterDevice(make_battery_param(min_charge=0.1), 0, 0)
        device.setup_run(make_context(horizon=1))
        state  = device.create_batch_state(1, 1)
        genome = np.array([[0.5, 0.0]])
        device.apply_genome_batch(state, genome)
        assert device.build_device_request(state).port_requests[0].min_energy_wh[0, 0] > 0.0

    def test_discharge_steps_negative_min_energy(self):
        device = HybridInverterDevice(make_battery_param(min_discharge=0.1), 0, 0)
        device.setup_run(make_context(horizon=1))
        state  = device.create_batch_state(1, 1)
        genome = np.array([[-0.5, 0.0]])
        device.apply_genome_batch(state, genome)
        assert device.build_device_request(state).port_requests[0].min_energy_wh[0, 0] < 0.0


# ============================================================
# TestApplyDeviceGrant
# ============================================================

class TestApplyDeviceGrant:
    def _make_grant(self, granted_wh: np.ndarray) -> DeviceGrant:
        return DeviceGrant(
            device_index=0,
            port_grants=(PortGrant(port_index=0, granted_wh=granted_wh),),
        )

    def test_ac_power_w_updated(self):
        device = make_device(make_battery_param())
        state  = device.create_batch_state(POP, HORIZON)
        device.apply_device_grant(state, self._make_grant(np.full((POP, HORIZON), 1000.0)))
        np.testing.assert_allclose(
            state.ac_power_w, 1000.0 / (STEP_INTERVAL / 3600.0), rtol=1e-9,
        )

    def test_other_arrays_unchanged(self):
        device = make_device(make_battery_param())
        state  = device.create_batch_state(POP, HORIZON)
        state.bat_factors[:] = 0.3
        before = state.bat_factors.copy()
        device.apply_device_grant(state, self._make_grant(np.zeros((POP, HORIZON))))
        np.testing.assert_array_equal(state.bat_factors, before)


# ============================================================
# TestComputeCost
# ============================================================

class TestComputeCost:
    def test_shape_pop_zero(self):
        device = make_device(make_battery_param())
        state  = device.create_batch_state(POP, HORIZON)
        assert device.compute_cost(state).shape == (POP, 0)

    def test_all_zeros(self):
        device = make_device(make_battery_param())
        state  = device.create_batch_state(POP, HORIZON)
        np.testing.assert_array_equal(device.compute_cost(state), 0.0)


# ============================================================
# TestExtractInstructions
# ============================================================

class TestExtractInstructions:
    """extract_instructions emits 2 OMBCInstructions per time step."""

    def _run(self, param, bat_per_step: list, pv_per_step: list):
        horizon = len(bat_per_step)
        device  = make_device(param, horizon=horizon)
        state   = device.create_batch_state(1, horizon)
        genes   = []
        for b, p in zip(bat_per_step, pv_per_step):
            genes += [float(b), float(p)]
        genome = np.array([genes])
        device.apply_genome_batch(state, genome)
        return device, state

    def test_two_instructions_per_step(self):
        param = make_battery_param()
        device, state = self._run(param, [0.3] * HORIZON, [0.0] * HORIZON)
        assert len(device.extract_instructions(state, 0)) == 2 * HORIZON

    def test_all_are_ombc_instructions(self):
        param = make_battery_param()
        device, state = self._run(param, [0.3] * HORIZON, [0.0] * HORIZON)
        assert all(
            isinstance(i, OMBCInstruction)
            for i in device.extract_instructions(state, 0)
        )

    def test_battery_instruction_mode_id_charge(self):
        param = make_battery_param()
        device, state = self._run(param, [0.5], [0.0])
        assert device.extract_instructions(state, 0)[0].operation_mode_id == "CHARGE"

    def test_battery_instruction_mode_id_discharge(self):
        param = make_battery_param()
        device, state = self._run(param, [-0.5], [0.0])
        assert device.extract_instructions(state, 0)[0].operation_mode_id == "DISCHARGE"

    def test_battery_instruction_mode_id_idle(self):
        param = make_battery_param()
        device, state = self._run(param, [0.0], [0.0])
        assert device.extract_instructions(state, 0)[0].operation_mode_id == "IDLE"

    def test_battery_instruction_factor_is_bat_factor(self):
        param = make_battery_param(min_charge=0.0, max_charge=1.0)
        device, state = self._run(param, [0.3], [0.0])
        instrs = device.extract_instructions(state, 0)
        assert instrs[0].operation_mode_factor == pytest.approx(
            state.bat_factors[0, 0], rel=1e-9,
        )

    def test_pv_instruction_mode_id(self):
        param = make_hybrid_param()
        device, state = self._run(param, [0.0], [0.7])
        assert device.extract_instructions(state, 0)[1].operation_mode_id == "PV_UTILISE"

    def test_pv_instruction_factor_is_pv_util(self):
        param = make_hybrid_param()
        device, state = self._run(param, [0.0], [0.6])
        instrs = device.extract_instructions(state, 0)
        assert instrs[1].operation_mode_factor == pytest.approx(
            state.pv_util_factors[0, 0], rel=1e-9,
        )

    def test_resource_id_matches(self):
        param = make_battery_param(device_id="inv_X")
        device, state = self._run(param, [0.3] * HORIZON, [0.0] * HORIZON)
        assert all(
            i.resource_id == "inv_X"
            for i in device.extract_instructions(state, 0)
        )

    def test_execution_times_match_step_times(self):
        param = make_battery_param()
        device, state = self._run(param, [0.3] * HORIZON, [0.0] * HORIZON)
        instrs = device.extract_instructions(state, 0)
        expected_times = make_step_times(HORIZON)
        for step_idx, dt in enumerate(expected_times):
            assert instrs[2 * step_idx].execution_time     == dt
            assert instrs[2 * step_idx + 1].execution_time == dt

    def test_individual_index_selects_correct_row(self):
        param  = make_battery_param()
        device = make_device(param, horizon=1)
        state  = device.create_batch_state(2, 1)
        genome = np.array([
            [0.4, 0.0],
            [-0.4, 0.0],
        ])
        device.apply_genome_batch(state, genome)
        assert device.extract_instructions(state, 0)[0].operation_mode_id == "CHARGE"
        assert device.extract_instructions(state, 1)[0].operation_mode_id == "DISCHARGE"

    # ------------------------------------------------------------------
    # SELF_CONSUMPTION via InstructionContext
    # ------------------------------------------------------------------

    def _make_context_with_grid(
        self,
        grid_wh_per_step: list[float],
        step_interval_sec: float = STEP_INTERVAL,
    ) -> InstructionContext:
        return InstructionContext(
            grid_granted_wh=np.array(grid_wh_per_step, dtype=np.float64),
            step_interval_sec=step_interval_sec,
        )

    def test_no_context_gives_explicit_modes(self):
        param = make_battery_param()
        device, state = self._run(param, [-0.5], [0.0])
        instr = device.extract_instructions(state, 0, instruction_context=None)
        assert instr[0].operation_mode_id == "DISCHARGE"
        assert instr[0].operation_mode_factor > 0.0

    def test_self_consumption_emitted_when_grid_near_zero_and_battery_discharging(self):
        param = make_battery_param()
        device, state = self._run(param, [-0.5], [0.0])
        ctx = self._make_context_with_grid([0.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "SELF_CONSUMPTION"
        assert instr[0].operation_mode_factor == 0.0

    def test_self_consumption_emitted_when_grid_near_zero_and_battery_charging(self):
        param = make_battery_param()
        device, state = self._run(param, [0.5], [0.0])
        ctx = self._make_context_with_grid([0.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "SELF_CONSUMPTION"
        assert instr[0].operation_mode_factor == 0.0

    def test_self_consumption_within_threshold_wh(self):
        """40 Wh < 50 Wh threshold → SELF_CONSUMPTION."""
        param = make_battery_param()
        device, state = self._run(param, [-0.5], [0.0])
        ctx = self._make_context_with_grid([40.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "SELF_CONSUMPTION"

    def test_explicit_discharge_above_threshold(self):
        """200 Wh > 50 Wh threshold → explicit DISCHARGE."""
        param = make_battery_param()
        device, state = self._run(param, [-0.5], [0.0])
        ctx = self._make_context_with_grid([200.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "DISCHARGE"
        assert instr[0].operation_mode_factor > 0.0

    def test_idle_not_promoted_to_self_consumption(self):
        """bat_factor == 0 stays IDLE even when grid exchange is zero."""
        param = make_battery_param()
        device, state = self._run(param, [0.0], [0.0])
        ctx = self._make_context_with_grid([0.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "IDLE"

    def test_context_with_none_grid_falls_back_to_explicit(self):
        """InstructionContext present but grid_granted_wh=None → explicit modes."""
        param = make_battery_param()
        device, state = self._run(param, [-0.5], [0.0])
        ctx = InstructionContext(grid_granted_wh=None, step_interval_sec=STEP_INTERVAL)
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "DISCHARGE"

    def test_mixed_steps_per_horizon(self):
        """Step 0: grid≈0 → SELF_CONSUMPTION; step 1: large export → DISCHARGE."""
        param = make_battery_param()
        device, state = self._run(param, [-0.1, -0.1], [0.0, 0.0])
        ctx = self._make_context_with_grid([0.0, 500.0])
        instrs = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instrs[0].operation_mode_id == "SELF_CONSUMPTION"
        assert instrs[2].operation_mode_id == "DISCHARGE"

    def test_pv_instruction_unaffected_by_self_consumption(self):
        """PV_UTILISE is always emitted unchanged regardless of battery mode."""
        param = make_hybrid_param()
        device, state = self._run(param, [-0.3], [0.7])
        ctx = self._make_context_with_grid([0.0])
        instrs = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instrs[0].operation_mode_id == "SELF_CONSUMPTION"
        assert instrs[1].operation_mode_id == "PV_UTILISE"
        assert instrs[1].operation_mode_factor == pytest.approx(
            state.pv_util_factors[0, 0], rel=1e-9
        )
