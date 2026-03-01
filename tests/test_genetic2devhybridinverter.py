"""Tests for the hybrid inverter energy device.

Module under test
-----------------
``akkudoktoreos.devices.genetic2.hybridinverter``

Test strategy
-------------
Each class exercises one cohesive responsibility. Internal helpers
(_repair_mode, _repair_factor, _compute_ac_power, _advance_soc) are
tested in isolation by calling them directly on a device instance that
has been through setup_run. This lets us verify the physics without
running the full apply_genome_batch pipeline.

apply_genome_batch integration tests then confirm that the helpers
compose correctly and that Lamarckian repair writes back to the genome.

All numerical assertions use relative tolerances (rtol=1e-9) except
where exact equality is expected (dtype checks, zero checks, enum
values). Horizon of 4 and population of 3 is the default unless a test
needs a specific configuration.

Covers
------
    InverterType
        - Enum values and names

    InverterMode
        - Enum values and names

    HybridInverterParam — validation
        - Off/on-state negative power rejected
        - SOLAR: bad efficiency, bad pv_max, bad pv_min_power rejected
        - BATTERY/HYBRID: bad efficiencies, capacity ≤ 0 rejected
        - BATTERY/HYBRID: charge rate order violation rejected
        - BATTERY/HYBRID: discharge rate order violation rejected
        - BATTERY/HYBRID: SoC factor order violation rejected
        - battery_charge_rates=[] rejected
        - battery_charge_rates containing 0.0 rejected
        - Valid BATTERY, SOLAR, HYBRID each construct without error

    HybridInverterParam — derived properties
        - valid_modes returns correct tuple per type
        - has_factor_gene True for BATTERY/HYBRID, False for SOLAR
        - n_modes matches valid_modes length
        - battery_min/max_soc_wh computed correctly from factors

    HybridInverterDevice — topology and identity
        - ports returns single BIDIRECTIONAL EnergyPort
        - port_id and bus_id match param
        - objective_names is empty
        - device_id matches param.device_id

    TestSetupRun
        - Stores _num_steps, _step_interval, _step_times from SimulationContext
        - SOLAR/HYBRID: pv_power_w_key=None raises ValueError
        - SOLAR/HYBRID: resolved PV array wrong shape raises ValueError
        - BATTERY/HYBRID: initial SoC factor out of bounds raises ValueError
        - BATTERY: resolve_measurement provides initial SoC Wh

    TestGenomeRequirements
        - BATTERY: size == 2 × horizon; mode bounds [0, 2]; factor bounds [0, 1]
        - SOLAR: size == horizon; mode bounds [0, 1]
        - HYBRID: size == 2 × horizon; mode bounds [0, 3]; factor bounds [0, 1]
        - Before setup_run raises RuntimeError

    TestCreateBatchState
        - Array shapes and dtypes
        - soc_wh pre-filled with resolved battery_initial_soc_wh from context
        - step_times reference matches setup_run argument
        - Before setup_run raises RuntimeError

    TestRepairMode
        - BATTERY positions 0/1/2 → OFF/CHARGE/DISCHARGE values
        - SOLAR positions 0/1 → OFF/PV values
        - HYBRID positions 0/1/2/3 → OFF/PV/CHARGE/DISCHARGE values
        - Fractional values round to nearest position
        - Out-of-range high clipped to last mode
        - Out-of-range low clipped to OFF
        - Mixed population vector processed correctly

    TestRepairFactor
        - OFF/PV modes: factor forced to 0.0
        - CHARGE: factor clipped to [min, max] charge rate
        - CHARGE: factor below min_charge_rate clipped up to min_charge_rate
        - CHARGE: discrete rates snapped to nearest
        - CHARGE: SoC headroom cap reduces factor
        - CHARGE: SoC at max → factor zeroed
        - DISCHARGE: factor clipped to [min, max] discharge rate
        - DISCHARGE: SoC at min → factor zeroed
        - DISCHARGE: partial SoC headroom reduces factor

    TestComputeAcPower
        - OFF returns off_state_power_consumption_w
        - SOLAR PV: -(pv × pv_to_ac_eff) + on_state
        - BATTERY CHARGE active: (factor × cap / ac_eff) + on_state
        - BATTERY CHARGE factor=0: returns off_state (standby)
        - BATTERY DISCHARGE active: -(factor × cap × batt_to_ac_eff) + on_state
        - BATTERY DISCHARGE factor=0: returns on_state (no injection)
        - HYBRID CHARGE: PV offsets AC demand; surplus PV injects
        - HYBRID DISCHARGE: battery + PV both inject

    TestAdvanceSoc
        - OFF/PV: SoC unchanged
        - BATTERY CHARGE: SoC increased by correct stored_wh
        - BATTERY DISCHARGE: SoC decreased by drawn_wh (battery side)
        - SoC clamped at battery_max_soc_wh on over-charge
        - SoC clamped at battery_min_soc_wh on over-discharge
        - HYBRID CHARGE: PV contributes first, AC covers remainder
        - HYBRID CHARGE: PV exceeds charge demand → only PV used

    TestApplyGenomeBatch
        - SOLAR: genome size == horizon; factors all zero; modes decoded
        - BATTERY: modes and factors decoded from interleaved genome
        - Lamarckian write-back: genome contains repaired values after call
        - SoC evolves correctly across a 3-step horizon
        - Population axis independent: two individuals with different modes
        - Mixed modes in one population processed per-individual

    TestBuildDeviceRequest
        - Returns DeviceRequest with correct device_index
        - Single PortRequest with correct port_index
        - energy_wh == ac_power_w × step_h
        - energy_wh shape (pop, horizon)
        - min_energy_wh shape (pop, horizon)
        - OFF/PV steps: min_energy_wh == 0
        - CHARGE steps: min_energy_wh > 0 (BATTERY, no PV)
        - DISCHARGE steps: min_energy_wh < 0 (BATTERY, no PV)

    TestApplyDeviceGrant
        - ac_power_w updated to granted_wh / step_h
        - Other state arrays unchanged

    TestComputeCost
        - Returns shape (population_size, 0)
        - All values zero

    TestExtractInstructions
        - Length == horizon
        - All elements are OMBCInstruction
        - resource_id matches device_id
        - execution_time matches step_times
        - operation_mode_id is the mode's enum name
        - operation_mode_factor matches state.factors
        - individual_index selects correct population row
"""

from __future__ import annotations

import numpy as np
import pytest
from pendulum import Duration

from akkudoktoreos.core.emplan import OMBCInstruction
from akkudoktoreos.devices.devicesabc import EnergyPort, PortDirection
from akkudoktoreos.devices.genetic2.hybridinverter import (
    HybridInverterBatchState,
    HybridInverterDevice,
    HybridInverterParam,
    InverterMode,
    InverterType,
)
from akkudoktoreos.simulation.genetic2.arbitrator import DeviceGrant, PortGrant
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import to_datetime

# ---------------------------------------------------------------------------
# Shared step-time helpers
# ---------------------------------------------------------------------------

STEP_INTERVAL = 3600.0  # 1-hour steps [s]
HORIZON = 4
POP = 3

# Keys used by FakeContext — any stable string will do.
PV_KEY = "pv_forecast"
SOC_KEY = "bat_soc"


def make_step_times(n: int = HORIZON) -> tuple:
    return tuple(to_datetime(i * STEP_INTERVAL) for i in range(n))


class FakeContext:
    """Minimal SimulationContext stand-in for unit tests.

    Avoids any dependency on the real prediction/measurement stores.
    ``resolve(key)`` returns a pre-configured PV array.
    ``resolve_measurement(key)`` returns a pre-configured SoC factor.

    Pass ``pv_w`` as a flat array of length ``horizon`` (values in W).
    Pass ``initial_soc_factor`` as a float in [0, 1].
    """

    def __init__(
        self,
        step_times: tuple,
        step_interval_sec: float = STEP_INTERVAL,
        pv_w: np.ndarray | None = None,
        initial_soc_factor: float = 0.5,
    ) -> None:
        self.step_times = step_times
        self.step_interval = Duration(seconds = step_interval_sec)
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
    """Build a FakeContext with uniform PV and given SoC factor."""
    times = make_step_times(horizon)
    if np.ndim(pv_w) == 0:
        pv_array = np.full(horizon, float(pv_w))
    else:
        pv_array = np.asarray(pv_w, dtype=np.float64)
    return FakeContext(times, step_interval_sec, pv_array, initial_soc_factor)


# ---------------------------------------------------------------------------
# Param factory helpers — one per inverter type
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
        port_id="p_ac",
        bus_id="bus_ac",
        inverter_type=InverterType.BATTERY,
        off_state_power_consumption_w=off_w,
        on_state_power_consumption_w=on_w,
        pv_to_ac_efficiency=0.97,         # unused for BATTERY but field is required
        pv_to_battery_efficiency=0.95,    # unused for BATTERY
        pv_max_power_w=10_000.0,          # unused
        pv_min_power_w=0.0,               # unused
        pv_power_w_key=None,              # unused for BATTERY
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
        port_id="p_ac",
        bus_id="bus_ac",
        inverter_type=InverterType.SOLAR,
        off_state_power_consumption_w=off_w,
        on_state_power_consumption_w=on_w,
        pv_to_ac_efficiency=0.97,
        pv_to_battery_efficiency=0.95,    # unused for SOLAR
        pv_max_power_w=pv_max_w,
        pv_min_power_w=0.0,
        pv_power_w_key=PV_KEY,
        ac_to_battery_efficiency=0.95,    # unused for SOLAR
        battery_to_ac_efficiency=0.95,    # unused for SOLAR
        battery_capacity_wh=0.0,          # unused for SOLAR
        battery_charge_rates=None,
        battery_min_charge_rate=0.0,
        battery_max_charge_rate=0.0,
        battery_min_discharge_rate=0.0,
        battery_max_discharge_rate=0.0,
        battery_min_soc_factor=0.0,
        battery_max_soc_factor=0.1,       # non-zero so min < max
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
        port_id="p_ac",
        bus_id="bus_ac",
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
# Device factory helper
# ---------------------------------------------------------------------------

def make_device(
    param: HybridInverterParam,
    horizon: int = HORIZON,
    device_index: int = 0,
    port_index: int = 0,
    pv_w: float | np.ndarray = 4_000.0,
    initial_soc_factor: float = 0.5,
) -> HybridInverterDevice:
    """Construct a device and call setup_run with a FakeContext."""
    context = make_context(horizon, STEP_INTERVAL, pv_w, initial_soc_factor)
    device = HybridInverterDevice(param, device_index, port_index)
    device.setup_run(context)
    return device


# ============================================================
# TestInverterType
# ============================================================

class TestInverterType:
    def test_battery_value(self):
        assert InverterType.BATTERY == 0

    def test_solar_value(self):
        assert InverterType.SOLAR == 1

    def test_hybrid_value(self):
        assert InverterType.HYBRID == 2

    def test_names(self):
        assert InverterType.BATTERY.name == "BATTERY"
        assert InverterType.SOLAR.name == "SOLAR"
        assert InverterType.HYBRID.name == "HYBRID"


# ============================================================
# TestInverterMode
# ============================================================

class TestInverterMode:
    def test_off_value(self):
        assert InverterMode.OFF == 0

    def test_pv_value(self):
        assert InverterMode.PV == 1

    def test_charge_value(self):
        assert InverterMode.CHARGE == 2

    def test_discharge_value(self):
        assert InverterMode.DISCHARGE == 3

    def test_names(self):
        assert InverterMode.OFF.name == "OFF"
        assert InverterMode.PV.name == "PV"
        assert InverterMode.CHARGE.name == "CHARGE"
        assert InverterMode.DISCHARGE.name == "DISCHARGE"


# ============================================================
# TestHybridInverterParamValidation
# ============================================================

class TestHybridInverterParamValidation:
    # ---- auxiliary power ----

    def test_negative_off_state_power_raises(self):
        with pytest.raises(ValueError, match="off_state_power_consumption_w"):
            make_battery_param(off_w=-1.0)

    def test_negative_on_state_power_raises(self):
        with pytest.raises(ValueError, match="on_state_power_consumption_w"):
            make_battery_param(on_w=-0.1)

    def test_zero_off_state_power_is_valid(self):
        make_battery_param(off_w=0.0)  # should not raise

    # ---- SOLAR-specific PV validation ----

    def test_solar_pv_to_ac_efficiency_zero_raises(self):
        with pytest.raises(ValueError, match="pv_to_ac_efficiency"):
            HybridInverterParam(
                device_id="s",
                port_id="p",
                bus_id="b",
                inverter_type=InverterType.SOLAR,
                off_state_power_consumption_w=0.0,
                on_state_power_consumption_w=0.0,
                pv_to_ac_efficiency=0.0,   # invalid
                pv_to_battery_efficiency=0.95,
                pv_max_power_w=8000.0,
                pv_min_power_w=0.0,
                pv_power_w_key=PV_KEY,
                ac_to_battery_efficiency=0.95,
                battery_to_ac_efficiency=0.95,
                battery_capacity_wh=0.0,
                battery_charge_rates=None,
                battery_min_charge_rate=0.0,
                battery_max_charge_rate=0.0,
                battery_min_discharge_rate=0.0,
                battery_max_discharge_rate=0.0,
                battery_min_soc_factor=0.0,
                battery_max_soc_factor=0.1,
                battery_initial_soc_factor_key=SOC_KEY,
            )

    def test_solar_pv_max_power_zero_raises(self):
        with pytest.raises(ValueError, match="pv_max_power_w"):
            HybridInverterParam(
                device_id="s",
                port_id="p",
                bus_id="b",
                inverter_type=InverterType.SOLAR,
                off_state_power_consumption_w=0.0,
                on_state_power_consumption_w=0.0,
                pv_to_ac_efficiency=0.97,
                pv_to_battery_efficiency=0.95,
                pv_max_power_w=0.0,   # invalid
                pv_min_power_w=0.0,
                pv_power_w_key=PV_KEY,
                ac_to_battery_efficiency=0.95,
                battery_to_ac_efficiency=0.95,
                battery_capacity_wh=0.0,
                battery_charge_rates=None,
                battery_min_charge_rate=0.0,
                battery_max_charge_rate=0.0,
                battery_min_discharge_rate=0.0,
                battery_max_discharge_rate=0.0,
                battery_min_soc_factor=0.0,
                battery_max_soc_factor=0.1,
                battery_initial_soc_factor_key=SOC_KEY,
            )

    def test_solar_pv_min_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="pv_min_power_w"):
            HybridInverterParam(
                device_id="s",
                port_id="p",
                bus_id="b",
                inverter_type=InverterType.SOLAR,
                off_state_power_consumption_w=0.0,
                on_state_power_consumption_w=0.0,
                pv_to_ac_efficiency=0.97,
                pv_to_battery_efficiency=0.95,
                pv_max_power_w=8000.0,
                pv_min_power_w=9000.0,   # > max
                pv_power_w_key=PV_KEY,
                ac_to_battery_efficiency=0.95,
                battery_to_ac_efficiency=0.95,
                battery_capacity_wh=0.0,
                battery_charge_rates=None,
                battery_min_charge_rate=0.0,
                battery_max_charge_rate=0.0,
                battery_min_discharge_rate=0.0,
                battery_max_discharge_rate=0.0,
                battery_min_soc_factor=0.0,
                battery_max_soc_factor=0.1,
                battery_initial_soc_factor_key=SOC_KEY,
            )

    # ---- BATTERY-specific validation ----

    def test_battery_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="battery_capacity_wh"):
            make_battery_param(capacity_wh=0.0)

    def test_battery_ac_to_bat_efficiency_zero_raises(self):
        with pytest.raises(ValueError, match="ac_to_battery_efficiency"):
            make_battery_param(ac_to_bat_eff=0.0)

    def test_battery_bat_to_ac_efficiency_exceeds_one_raises(self):
        with pytest.raises(ValueError, match="battery_to_ac_efficiency"):
            make_battery_param(bat_to_ac_eff=1.01)

    def test_battery_min_charge_exceeds_max_charge_raises(self):
        with pytest.raises(ValueError, match="battery charge rates"):
            make_battery_param(min_charge=0.6, max_charge=0.5)

    def test_battery_min_discharge_exceeds_max_discharge_raises(self):
        with pytest.raises(ValueError, match="battery discharge rates"):
            make_battery_param(min_discharge=0.8, max_discharge=0.3)

    def test_battery_soc_min_equals_max_raises(self):
        with pytest.raises(ValueError, match="SoC factors"):
            make_battery_param(min_soc=0.5, max_soc=0.5)

    def test_battery_charge_rates_empty_raises(self):
        with pytest.raises(ValueError, match="battery_charge_rates must not be empty"):
            make_battery_param(charge_rates=())

    def test_battery_charge_rates_contains_zero_raises(self):
        with pytest.raises(ValueError, match="battery_charge_rates values"):
            make_battery_param(charge_rates=(0.0, 0.5, 1.0))

    def test_battery_charge_rates_exceeds_one_raises(self):
        with pytest.raises(ValueError, match="battery_charge_rates values"):
            make_battery_param(charge_rates=(0.5, 1.1))

    # ---- Valid constructions ----

    def test_valid_battery_constructs(self):
        p = make_battery_param()
        assert p.inverter_type == InverterType.BATTERY

    def test_valid_solar_constructs(self):
        p = make_solar_param()
        assert p.inverter_type == InverterType.SOLAR

    def test_valid_hybrid_constructs(self):
        p = make_hybrid_param()
        assert p.inverter_type == InverterType.HYBRID

    def test_valid_battery_with_discrete_rates(self):
        p = make_battery_param(charge_rates=(0.25, 0.5, 0.75, 1.0))
        assert p.battery_charge_rates == (0.25, 0.5, 0.75, 1.0)


# ============================================================
# TestHybridInverterParamDerivedProperties
# ============================================================

class TestHybridInverterParamDerivedProperties:
    def test_battery_valid_modes(self):
        p = make_battery_param()
        assert p.valid_modes == (InverterMode.OFF, InverterMode.CHARGE, InverterMode.DISCHARGE)

    def test_solar_valid_modes(self):
        p = make_solar_param()
        assert p.valid_modes == (InverterMode.OFF, InverterMode.PV)

    def test_hybrid_valid_modes(self):
        p = make_hybrid_param()
        assert p.valid_modes == (
            InverterMode.OFF,
            InverterMode.PV,
            InverterMode.CHARGE,
            InverterMode.DISCHARGE,
        )

    def test_battery_has_factor_gene(self):
        assert make_battery_param().has_factor_gene is True

    def test_solar_has_no_factor_gene(self):
        assert make_solar_param().has_factor_gene is False

    def test_hybrid_has_factor_gene(self):
        assert make_hybrid_param().has_factor_gene is True

    def test_battery_n_modes(self):
        assert make_battery_param().n_modes == 3

    def test_solar_n_modes(self):
        assert make_solar_param().n_modes == 2

    def test_hybrid_n_modes(self):
        assert make_hybrid_param().n_modes == 4

    def test_battery_soc_wh_properties(self):
        p = make_battery_param(capacity_wh=10_000.0, min_soc=0.1, max_soc=0.9)
        assert p.battery_min_soc_wh == pytest.approx(1_000.0)
        assert p.battery_max_soc_wh == pytest.approx(9_000.0)


# ============================================================
# TestHybridInverterDeviceTopology
# ============================================================

class TestHybridInverterDeviceTopology:
    def test_ports_returns_single_port(self):
        device = make_device(make_battery_param())
        assert len(device.ports) == 1

    def test_port_is_bidirectional(self):
        device = make_device(make_battery_param())
        assert device.ports[0].direction == PortDirection.BIDIRECTIONAL

    def test_port_id_matches_param(self):
        param = make_battery_param()
        device = make_device(param)
        assert device.ports[0].port_id == param.port_id

    def test_bus_id_matches_param(self):
        param = make_battery_param()
        device = make_device(param)
        assert device.ports[0].bus_id == param.bus_id

    def test_objective_names_empty(self):
        device = make_device(make_battery_param())
        assert device.objective_names == []

    def test_device_id_matches_param(self):
        param = make_battery_param(device_id="inverter_42")
        device = make_device(param)
        assert device.device_id == "inverter_42"


# ============================================================
# TestSetupRun
# ============================================================

class TestSetupRun:
    def test_stores_num_steps(self):
        device = make_device(make_battery_param(), horizon=6)
        assert device._num_steps == 6

    def test_stores_step_interval(self):
        device = make_device(make_battery_param())
        assert device._step_interval_sec == STEP_INTERVAL

    def test_stores_step_times(self):
        context = make_context(horizon=4)
        device = HybridInverterDevice(make_battery_param(), 0, 0)
        device.setup_run(context)
        assert device._step_times == context.step_times

    def test_solar_pv_power_w_key_none_raises(self):
        """SOLAR with pv_power_w_key=None must raise at setup_run."""
        param = HybridInverterParam(
            device_id="s",
            port_id="p",
            bus_id="b",
            inverter_type=InverterType.SOLAR,
            off_state_power_consumption_w=0.0,
            on_state_power_consumption_w=0.0,
            pv_to_ac_efficiency=0.97,
            pv_to_battery_efficiency=0.95,
            pv_max_power_w=8000.0,
            pv_min_power_w=0.0,
            pv_power_w_key=None,   # invalid for SOLAR
            ac_to_battery_efficiency=0.95,
            battery_to_ac_efficiency=0.95,
            battery_capacity_wh=0.0,
            battery_charge_rates=None,
            battery_min_charge_rate=0.0,
            battery_max_charge_rate=0.0,
            battery_min_discharge_rate=0.0,
            battery_max_discharge_rate=0.0,
            battery_min_soc_factor=0.0,
            battery_max_soc_factor=0.1,
            battery_initial_soc_factor_key=SOC_KEY,
        )
        device = HybridInverterDevice(param, 0, 0)
        with pytest.raises(ValueError, match="pv_power_w_key"):
            device.setup_run(make_context())

    def test_solar_pv_wrong_shape_raises(self):
        """SOLAR with resolved PV array of wrong length raises at setup_run."""
        param = make_solar_param()
        device = HybridInverterDevice(param, 0, 0)
        # Context with PV array of length 2 but horizon 4
        ctx = FakeContext(
            step_times=make_step_times(4),
            step_interval_sec=STEP_INTERVAL,
            pv_w=np.full(2, 3000.0),   # wrong length
            initial_soc_factor=0.5,
        )
        with pytest.raises(ValueError):
            device.setup_run(ctx)

    def test_hybrid_pv_wrong_shape_raises(self):
        """HYBRID with resolved PV array of wrong length raises at setup_run."""
        param = make_hybrid_param()
        device = HybridInverterDevice(param, 0, 0)
        ctx = FakeContext(
            step_times=make_step_times(4),
            step_interval_sec=STEP_INTERVAL,
            pv_w=np.full(1, 3000.0),   # wrong length
            initial_soc_factor=0.5,
        )
        with pytest.raises(ValueError):
            device.setup_run(ctx)

    def test_initial_soc_out_of_bounds_raises(self):
        """Initial SoC factor outside [min_soc, max_soc] raises at setup_run."""
        param = make_battery_param(min_soc=0.2, max_soc=0.8)
        device = HybridInverterDevice(param, 0, 0)
        # SoC factor 0.1 < min_soc_factor 0.2
        ctx = make_context(initial_soc_factor=0.1)
        with pytest.raises(ValueError, match="initial SoC factor"):
            device.setup_run(ctx)

    def test_battery_resolves_initial_soc_wh(self):
        """setup_run stores the resolved initial SoC in Wh."""
        param = make_battery_param(capacity_wh=10_000.0, min_soc=0.1, max_soc=0.9)
        device = HybridInverterDevice(param, 0, 0)
        ctx = make_context(initial_soc_factor=0.6)
        device.setup_run(ctx)
        assert device._battery_initial_soc_wh == pytest.approx(6_000.0)

    def test_solar_resolves_pv_array(self):
        """setup_run stores the resolved PV array for SOLAR type."""
        param = make_solar_param()
        device = HybridInverterDevice(param, 0, 0)
        ctx = make_context(pv_w=np.full(HORIZON, 5_000.0))
        device.setup_run(ctx)
        np.testing.assert_allclose(device._pv_power_w, 5_000.0)


# ============================================================
# TestGenomeRequirements
# ============================================================

class TestGenomeRequirements:
    def test_battery_genome_size(self):
        device = make_device(make_battery_param(), horizon=4)
        slc = device.genome_requirements()
        assert slc.size == 2 * 4

    def test_battery_mode_upper_bound(self):
        # 3 modes → position indices 0, 1, 2 → upper bound 2.0
        device = make_device(make_battery_param(), horizon=4)
        slc = device.genome_requirements()
        mode_ub = slc.upper_bound[0::2]  # even positions
        np.testing.assert_array_equal(mode_ub, np.full(4, 2.0))

    def test_battery_mode_lower_bound(self):
        device = make_device(make_battery_param(), horizon=4)
        slc = device.genome_requirements()
        mode_lb = slc.lower_bound[0::2]
        np.testing.assert_array_equal(mode_lb, np.zeros(4))

    def test_battery_factor_bounds(self):
        device = make_device(make_battery_param(), horizon=4)
        slc = device.genome_requirements()
        np.testing.assert_array_equal(slc.lower_bound[1::2], np.zeros(4))
        np.testing.assert_array_equal(slc.upper_bound[1::2], np.ones(4))

    def test_solar_genome_size(self):
        device = make_device(make_solar_param(), horizon=4)
        slc = device.genome_requirements()
        assert slc.size == 4

    def test_solar_mode_upper_bound(self):
        # 2 modes → upper bound 1.0
        device = make_device(make_solar_param(), horizon=4)
        slc = device.genome_requirements()
        np.testing.assert_array_equal(slc.upper_bound, np.ones(4))

    def test_solar_mode_lower_bound(self):
        device = make_device(make_solar_param(), horizon=4)
        slc = device.genome_requirements()
        np.testing.assert_array_equal(slc.lower_bound, np.zeros(4))

    def test_hybrid_genome_size(self):
        device = make_device(make_hybrid_param(), horizon=4)
        slc = device.genome_requirements()
        assert slc.size == 2 * 4

    def test_hybrid_mode_upper_bound(self):
        # 4 modes → upper bound 3.0
        device = make_device(make_hybrid_param(), horizon=4)
        slc = device.genome_requirements()
        mode_ub = slc.upper_bound[0::2]
        np.testing.assert_array_equal(mode_ub, np.full(4, 3.0))

    def test_before_setup_run_raises(self):
        param = make_battery_param()
        device = HybridInverterDevice(param, 0, 0)
        with pytest.raises(RuntimeError):
            device.genome_requirements()


# ============================================================
# TestCreateBatchState
# ============================================================

class TestCreateBatchState:
    def test_modes_shape_and_dtype(self):
        device = make_device(make_battery_param(), horizon=HORIZON)
        state = device.create_batch_state(POP, HORIZON)
        assert state.modes.shape == (POP, HORIZON)
        assert state.modes.dtype == np.int8

    def test_modes_initialised_to_zero(self):
        device = make_device(make_battery_param(), horizon=HORIZON)
        state = device.create_batch_state(POP, HORIZON)
        np.testing.assert_array_equal(state.modes, 0)

    def test_factors_shape(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
        assert state.factors.shape == (POP, HORIZON)

    def test_factors_initialised_to_zero(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
        np.testing.assert_array_equal(state.factors, 0.0)

    def test_soc_wh_shape(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
        assert state.soc_wh.shape == (POP, HORIZON)

    def test_soc_wh_prefilled_with_initial_soc(self):
        param = make_battery_param(capacity_wh=10_000.0, min_soc=0.0, max_soc=1.0)
        device = make_device(param, initial_soc_factor=0.6)
        state = device.create_batch_state(POP, HORIZON)
        np.testing.assert_allclose(state.soc_wh, 6_000.0, rtol=1e-9)

    def test_ac_power_w_shape(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
        assert state.ac_power_w.shape == (POP, HORIZON)

    def test_ac_power_w_initialised_to_zero(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
        np.testing.assert_array_equal(state.ac_power_w, 0.0)

    def test_step_times_stored(self):
        context = make_context(horizon=HORIZON)
        device = HybridInverterDevice(make_battery_param(), 0, 0)
        device.setup_run(context)
        state = device.create_batch_state(POP, HORIZON)
        assert state.step_times == context.step_times

    def test_before_setup_run_raises(self):
        param = make_battery_param()
        device = HybridInverterDevice(param, 0, 0)
        with pytest.raises(RuntimeError):
            device.create_batch_state(POP, HORIZON)


# ============================================================
# TestRepairMode
# ============================================================

class TestRepairMode:
    # BATTERY: valid_modes = (OFF, CHARGE, DISCHARGE) → values (0, 2, 3)
    # Genome position 0 → OFF(0), position 1 → CHARGE(2), position 2 → DISCHARGE(3)

    def test_battery_position_0_maps_to_off(self):
        device = make_device(make_battery_param())
        raw = np.array([0.0])
        result = device._repair_mode(raw)
        assert result[0] == InverterMode.OFF

    def test_battery_position_1_maps_to_charge(self):
        device = make_device(make_battery_param())
        raw = np.array([1.0])
        result = device._repair_mode(raw)
        assert result[0] == InverterMode.CHARGE

    def test_battery_position_2_maps_to_discharge(self):
        device = make_device(make_battery_param())
        raw = np.array([2.0])
        result = device._repair_mode(raw)
        assert result[0] == InverterMode.DISCHARGE

    def test_battery_never_produces_pv_mode(self):
        # All positions 0..2 should never yield InverterMode.PV (value 1)
        device = make_device(make_battery_param())
        raw = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = device._repair_mode(raw)
        assert InverterMode.PV not in result

    def test_solar_position_0_maps_to_off(self):
        device = make_device(make_solar_param())
        raw = np.array([0.0])
        result = device._repair_mode(raw)
        assert result[0] == InverterMode.OFF

    def test_solar_position_1_maps_to_pv(self):
        device = make_device(make_solar_param())
        raw = np.array([1.0])
        result = device._repair_mode(raw)
        assert result[0] == InverterMode.PV

    def test_hybrid_position_0_maps_to_off(self):
        device = make_device(make_hybrid_param())
        raw = np.array([0.0])
        assert device._repair_mode(raw)[0] == InverterMode.OFF

    def test_hybrid_position_1_maps_to_pv(self):
        device = make_device(make_hybrid_param())
        raw = np.array([1.0])
        assert device._repair_mode(raw)[0] == InverterMode.PV

    def test_hybrid_position_2_maps_to_charge(self):
        device = make_device(make_hybrid_param())
        raw = np.array([2.0])
        assert device._repair_mode(raw)[0] == InverterMode.CHARGE

    def test_hybrid_position_3_maps_to_discharge(self):
        device = make_device(make_hybrid_param())
        raw = np.array([3.0])
        assert device._repair_mode(raw)[0] == InverterMode.DISCHARGE

    def test_fractional_rounds_to_nearest(self):
        # 0.4 rounds to 0 (OFF), 0.6 rounds to 1 (CHARGE for BATTERY)
        device = make_device(make_battery_param())
        raw = np.array([0.4, 0.6])
        result = device._repair_mode(raw)
        assert result[0] == InverterMode.OFF
        assert result[1] == InverterMode.CHARGE

    def test_out_of_range_high_clipped_to_last_mode(self):
        device = make_device(make_battery_param())  # n_modes=3, last position=2
        raw = np.array([99.0])
        result = device._repair_mode(raw)
        assert result[0] == InverterMode.DISCHARGE  # last valid mode

    def test_out_of_range_low_clipped_to_off(self):
        device = make_device(make_battery_param())
        raw = np.array([-5.0])
        result = device._repair_mode(raw)
        assert result[0] == InverterMode.OFF

    def test_result_dtype_is_int8(self):
        device = make_device(make_battery_param())
        raw = np.array([0.0, 1.0, 2.0])
        result = device._repair_mode(raw)
        assert result.dtype == np.int8

    def test_mixed_population_processed_correctly(self):
        device = make_device(make_hybrid_param())
        # Population of 4: positions 0, 1, 2, 3
        raw = np.array([0.0, 1.0, 2.0, 3.0])
        result = device._repair_mode(raw)
        assert result[0] == InverterMode.OFF
        assert result[1] == InverterMode.PV
        assert result[2] == InverterMode.CHARGE
        assert result[3] == InverterMode.DISCHARGE


# ============================================================
# TestRepairFactor
# ============================================================

class TestRepairFactor:
    """Tests for _repair_factor. Called with shape-(pop,) arrays."""

    def _call(self, device, modes_list, factors_list, soc_list):
        modes = np.array(modes_list, dtype=np.int8)
        factors = np.array(factors_list, dtype=np.float64)
        soc = np.array(soc_list, dtype=np.float64)
        return device._repair_factor(modes, factors, soc)

    # ---- OFF and PV modes always zero ----

    def test_off_mode_zeroes_factor(self):
        device = make_device(make_battery_param())
        result = self._call(device, [InverterMode.OFF], [0.8], [5000.0])
        assert result[0] == pytest.approx(0.0)

    def test_pv_mode_zeroes_factor_for_hybrid(self):
        device = make_device(make_hybrid_param())
        result = self._call(device, [InverterMode.PV], [0.9], [5000.0])
        assert result[0] == pytest.approx(0.0)

    # ---- CHARGE: rate bounds clipping ----

    def test_charge_clips_to_max_charge_rate(self):
        param = make_battery_param(max_charge=0.7, capacity_wh=10_000.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [1.0], [1000.0])
        # Factor should be capped at max_charge_rate = 0.7
        assert result[0] == pytest.approx(0.7, rel=1e-6)

    def test_charge_below_min_charge_rate_clipped_to_min(self):
        param = make_battery_param(min_charge=0.1, max_charge=1.0, capacity_wh=10_000.0)
        device = make_device(param)
        # Request factor 0.05 < min_charge_rate 0.1 → clipped up to 0.1
        # (zeroing only happens when SoC cap pushes the factor below min)
        result = self._call(device, [InverterMode.CHARGE], [0.05], [5000.0])
        assert result[0] == pytest.approx(0.1)

    # ---- CHARGE: discrete rate snapping ----

    def test_charge_discrete_rates_snap_to_nearest(self):
        param = make_battery_param(charge_rates=(0.25, 0.5, 0.75, 1.0))
        device = make_device(param)
        # Request 0.6 → nearest discrete rate is 0.5
        result = self._call(device, [InverterMode.CHARGE], [0.6], [1000.0])
        assert result[0] == pytest.approx(0.5)

    def test_charge_discrete_rates_snap_rounds_up(self):
        param = make_battery_param(charge_rates=(0.25, 0.75, 1.0))
        device = make_device(param)
        # Request 0.55 → nearest is 0.75 (distance 0.20) vs 0.25 (distance 0.30)
        result = self._call(device, [InverterMode.CHARGE], [0.55], [1000.0])
        assert result[0] == pytest.approx(0.75)

    # ---- CHARGE: SoC cap ----

    def test_charge_soc_cap_reduces_factor(self):
        # capacity=10000, max_soc=0.9 → max_soc_wh=9000
        # current soc = 8500, headroom = 500 Wh
        # max_factor = headroom / (capacity × step_h × ac_eff)
        #            = 500 / (10000 × 1.0 × 0.95)
        #            = 500 / 9500 ≈ 0.05263
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=0.9, ac_to_bat_eff=0.95,
            min_charge=0.01, max_charge=1.0,
        )
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [1.0], [8500.0])
        expected_max = 500.0 / (10_000.0 * 1.0 * 0.95)
        assert result[0] == pytest.approx(expected_max, rel=1e-5)

    def test_charge_soc_at_max_zeroes_factor(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=0.9, min_charge=0.05
        )
        device = make_device(param)
        # soc exactly at max → headroom = 0 → max_factor = 0 < min_charge → zeroed
        result = self._call(device, [InverterMode.CHARGE], [1.0], [9000.0])
        assert result[0] == pytest.approx(0.0)

    # ---- DISCHARGE: rate bounds ----

    def test_discharge_clips_to_max_discharge_rate(self):
        # soc=9000, min_soc=0.1 → available=8000 Wh, max_factor=0.8 > max_discharge=0.5
        # so max_discharge is the binding constraint, not the SoC cap
        param = make_battery_param(max_discharge=0.5, capacity_wh=10_000.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [1.0], [9000.0])
        assert result[0] == pytest.approx(0.5)

    # ---- DISCHARGE: SoC floor ----

    def test_discharge_soc_at_min_zeroes_factor(self):
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.1, min_discharge=0.05
        )
        device = make_device(param)
        # soc at min → available = 0 → factor 0 < min_discharge → zeroed
        result = self._call(device, [InverterMode.DISCHARGE], [1.0], [1000.0])
        assert result[0] == pytest.approx(0.0)

    def test_discharge_partial_soc_reduces_factor(self):
        # capacity=10000, min_soc=0.1 → min_soc_wh=1000
        # current soc=2000, available=1000 Wh
        # max_factor = 1000 / (10000 × 1.0) = 0.1
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.1, min_discharge=0.05, max_discharge=1.0
        )
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [1.0], [2000.0])
        expected = 1000.0 / (10_000.0 * 1.0)  # step_h = 1.0
        assert result[0] == pytest.approx(expected, rel=1e-5)


# ============================================================
# TestComputeAcPower
# ============================================================

class TestComputeAcPower:
    """Tests for _compute_ac_power. Called directly with shape-(pop,) arrays."""

    def _call(self, device, modes_list, factors_list, pv_dc_w=0.0):
        modes = np.array(modes_list, dtype=np.int8)
        factors = np.array(factors_list, dtype=np.float64)
        return device._compute_ac_power(modes, factors, pv_dc_w)

    # ---- OFF mode ----

    def test_off_returns_off_state_power(self):
        param = make_battery_param(off_w=8.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.OFF], [0.0])
        assert result[0] == pytest.approx(8.0)

    # ---- SOLAR PV mode ----

    def test_solar_pv_injects_correct_power(self):
        # pv_dc=4000, pv_to_ac=0.97, on_state=10
        # expected = -(4000 × 0.97) + 10 = -3870
        param = make_solar_param(on_w=10.0)
        param_pv_to_ac = param.pv_to_ac_efficiency  # 0.97
        device = make_device(param)
        result = self._call(device, [InverterMode.PV], [0.0], pv_dc_w=4000.0)
        expected = -(4000.0 * param_pv_to_ac) + 10.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_solar_pv_negative_power(self):
        """AC power must be negative (injection) for PV with no auxiliary."""
        param = make_solar_param(on_w=0.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.PV], [0.0], pv_dc_w=1000.0)
        assert result[0] < 0.0

    # ---- BATTERY CHARGE ----

    def test_battery_charge_active_consumes_from_bus(self):
        # factor=0.5, capacity=10000, ac_eff=0.95, on_w=10
        # charge_power_w = 0.5 × 10000 = 5000
        # gross_ac_w = 5000 / 0.95 ≈ 5263.16
        # expected = 5263.16 + 10
        param = make_battery_param(capacity_wh=10_000.0, ac_to_bat_eff=0.95, on_w=10.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [0.5])
        expected = (0.5 * 10_000.0 / 0.95) + 10.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_charge_active_positive_power(self):
        """AC power must be positive (consuming) during CHARGE."""
        device = make_device(make_battery_param())
        result = self._call(device, [InverterMode.CHARGE], [0.3])
        assert result[0] > 0.0

    def test_battery_charge_factor_zero_falls_back_to_off_state(self):
        param = make_battery_param(off_w=5.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [0.0])
        assert result[0] == pytest.approx(5.0)

    # ---- BATTERY DISCHARGE ----

    def test_battery_discharge_active_injects_into_bus(self):
        # factor=0.5, capacity=10000, bat_to_ac=0.95, on_w=10
        # battery_ac_w = 0.5 × 10000 × 0.95 = 4750
        # expected = -4750 + 10 = -4740
        param = make_battery_param(capacity_wh=10_000.0, bat_to_ac_eff=0.95, on_w=10.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [0.5])
        expected = -(0.5 * 10_000.0 * 0.95) + 10.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_discharge_negative_power(self):
        """AC power must be negative (injection) during DISCHARGE with large factor."""
        param = make_battery_param(on_w=1.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [1.0])
        assert result[0] < 0.0

    def test_battery_discharge_factor_zero_returns_on_state(self):
        # factor=0 → total_inject=0 → ac = -0 + on_state = on_state
        param = make_battery_param(on_w=15.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [0.0])
        assert result[0] == pytest.approx(15.0)

    # ---- HYBRID CHARGE: PV offsets AC demand ----

    def test_hybrid_charge_pv_reduces_ac_demand(self):
        # capacity=10000, factor=0.5, charge_power=5000W, ac_eff=0.95
        # gross_ac = 5000/0.95 = 5263.16
        # pv_dc=3000, pv_to_bat=0.95 → pv_for_bat=2850W
        # pv_used = min(2850, 5000) = 2850
        # ac_offset = 2850 / 0.95 = 3000
        # net_ac = 5263.16 - 3000 = 2263.16 (+ surplus injection)
        # pv_surplus_dc = 3000 - 2850/0.95 = 3000 - 3000 = 0
        # ac_w = 2263.16 - 0 + on_state
        param = make_hybrid_param(
            capacity_wh=10_000.0,
            ac_to_bat_eff=0.95,
            pv_to_bat_eff=0.95,
            pv_to_ac_eff=0.97,
            on_w=10.0,
        )
        device = make_device(param, pv_w=3000.0)
        pv_dc = 3000.0
        charge_power = 0.5 * 10_000.0
        pv_for_bat = pv_dc * 0.95
        pv_used = min(pv_for_bat, charge_power)
        gross_ac = charge_power / 0.95
        ac_offset = pv_used / 0.95
        pv_surplus_dc = pv_dc - pv_used / 0.95
        pv_surplus_ac = pv_surplus_dc * 0.97
        expected = gross_ac - ac_offset - pv_surplus_ac + 10.0
        result = self._call(device, [InverterMode.CHARGE], [0.5], pv_dc_w=pv_dc)
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_hybrid_charge_pv_surplus_injects(self):
        # PV power greatly exceeds charge demand → net AC should be negative (injection)
        param = make_hybrid_param(
            capacity_wh=1_000.0,   # small battery, factor 0.1 → charge 100W
            on_w=0.0,
        )
        device = make_device(param, pv_w=10_000.0)  # large PV
        result = self._call(device, [InverterMode.CHARGE], [0.1], pv_dc_w=10_000.0)
        assert result[0] < 0.0  # net injection

    # ---- HYBRID DISCHARGE: battery + PV both inject ----

    def test_hybrid_discharge_combines_battery_and_pv(self):
        # factor=0.5, capacity=10000, bat_to_ac=0.95 → battery_ac=4750
        # pv_dc=2000, pv_to_ac=0.97 → pv_ac=1940
        # total_inject = 4750 + 1940 = 6690
        # expected = -6690 + on_state
        param = make_hybrid_param(
            capacity_wh=10_000.0,
            bat_to_ac_eff=0.95,
            pv_to_ac_eff=0.97,
            on_w=5.0,
        )
        device = make_device(param, pv_w=2000.0)
        result = self._call(device, [InverterMode.DISCHARGE], [0.5], pv_dc_w=2000.0)
        battery_ac = 0.5 * 10_000.0 * 0.95
        pv_ac = 2000.0 * 0.97
        expected = -(battery_ac + pv_ac) + 5.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    # ---- population vectorisation ----

    def test_mixed_modes_independent(self):
        param = make_battery_param(off_w=5.0, on_w=10.0, capacity_wh=10_000.0)
        device = make_device(param)
        modes = np.array([InverterMode.OFF, InverterMode.CHARGE, InverterMode.DISCHARGE], dtype=np.int8)
        factors = np.array([0.0, 0.5, 0.5])
        result = device._compute_ac_power(modes, factors, 0.0)
        assert result[0] == pytest.approx(5.0)   # OFF
        assert result[1] > 0.0                    # CHARGE consumes
        assert result[2] < 0.0                    # DISCHARGE injects


# ============================================================
# TestAdvanceSoc
# ============================================================

class TestAdvanceSoc:
    """Tests for _advance_soc. Called directly."""

    def _call(self, device, soc_list, modes_list, factors_list, pv_dc_w=0.0):
        soc = np.array(soc_list, dtype=np.float64)
        modes = np.array(modes_list, dtype=np.int8)
        factors = np.array(factors_list, dtype=np.float64)
        return device._advance_soc(soc, modes, factors, pv_dc_w)

    def test_off_mode_soc_unchanged(self):
        device = make_device(make_battery_param())
        result = self._call(device, [5000.0], [InverterMode.OFF], [0.0])
        assert result[0] == pytest.approx(5000.0)

    def test_pv_mode_soc_unchanged_for_solar(self):
        device = make_device(make_solar_param())
        result = self._call(device, [0.0], [InverterMode.PV], [0.0], pv_dc_w=3000.0)
        # SoC should not change for SOLAR (no battery)
        assert result[0] == pytest.approx(0.0)

    def test_battery_charge_increases_soc(self):
        # factor=0.5, capacity=10000, step_h=1.0, ac_eff=0.95
        # stored_wh = 0.5 × 10000 × 1.0 × 0.95 = 4750
        # new_soc = 2000 + 4750 = 6750
        param = make_battery_param(
            capacity_wh=10_000.0, ac_to_bat_eff=0.95,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        result = self._call(device, [2000.0], [InverterMode.CHARGE], [0.5])
        expected = 2000.0 + 0.5 * 10_000.0 * 1.0 * 0.95
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_discharge_decreases_soc(self):
        # factor=0.3, capacity=10000, step_h=1.0
        # drawn_wh = 0.3 × 10000 × 1.0 = 3000 (battery side, pre-efficiency)
        # new_soc = 8000 - 3000 = 5000
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        result = self._call(device, [8000.0], [InverterMode.DISCHARGE], [0.3])
        expected = 8000.0 - 0.3 * 10_000.0 * 1.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_soc_clamped_at_max_on_overcharge(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=0.9, min_soc=0.0,
            ac_to_bat_eff=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        # Start near max, charge full — result must be clamped to max_soc_wh=9000
        result = self._call(device, [8900.0], [InverterMode.CHARGE], [1.0])
        assert result[0] == pytest.approx(9000.0)

    def test_soc_clamped_at_min_on_over_discharge(self):
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.1, max_soc=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        # Start near min, discharge max — result must be clamped to min_soc_wh=1000
        result = self._call(device, [1100.0], [InverterMode.DISCHARGE], [1.0])
        assert result[0] == pytest.approx(1000.0)

    def test_hybrid_charge_pv_contributes_first(self):
        # capacity=10000, factor=0.5, charge_power=5000W
        # pv_dc=2000, pv_to_bat=0.95 → pv_for_bat=1900W
        # pv_used=min(1900, 5000)=1900
        # ac_used=5000-1900=3100
        # stored_wh = (1900 + 3100 × 0.95) × 1.0
        #           = (1900 + 2945) = 4845
        param = make_hybrid_param(
            capacity_wh=10_000.0,
            ac_to_bat_eff=0.95,
            pv_to_bat_eff=0.95,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, pv_w=2000.0, initial_soc_factor=0.5)
        result = self._call(device, [3000.0], [InverterMode.CHARGE], [0.5], pv_dc_w=2000.0)
        pv_for_bat = 2000.0 * 0.95
        charge_power = 0.5 * 10_000.0
        pv_used = min(pv_for_bat, charge_power)
        ac_used = charge_power - pv_used
        stored_wh = (pv_used + ac_used * 0.95) * 1.0
        expected = 3000.0 + stored_wh
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_hybrid_charge_pv_exceeds_demand_uses_only_pv(self):
        # capacity=1000, factor=0.1, charge_power=100W
        # pv_dc=5000, pv_to_bat=0.95 → pv_for_bat=4750W > 100W
        # pv_used=min(4750,100)=100, ac_used=0
        # stored_wh = (100 + 0) × 1.0 = 100
        param = make_hybrid_param(
            capacity_wh=1_000.0,
            ac_to_bat_eff=0.95,
            pv_to_bat_eff=0.95,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, pv_w=5000.0, initial_soc_factor=0.5)
        result = self._call(device, [200.0], [InverterMode.CHARGE], [0.1], pv_dc_w=5000.0)
        stored_wh = 0.1 * 1_000.0  # pv fully covers → no AC efficiency loss
        expected = 200.0 + stored_wh
        assert result[0] == pytest.approx(expected, rel=1e-9)


# ============================================================
# TestApplyGenomeBatch
# ============================================================

class TestApplyGenomeBatch:
    """Integration tests for the full apply_genome_batch pipeline."""

    # ---- SOLAR ----

    def test_solar_genome_size_is_horizon(self):
        device = make_device(make_solar_param(), horizon=4)
        state = device.create_batch_state(POP, 4)
        genome = np.zeros((POP, 4))  # size == horizon for SOLAR
        device.apply_genome_batch(state, genome)
        assert state.modes.shape == (POP, 4)

    def test_solar_all_factors_zero(self):
        device = make_device(make_solar_param(), horizon=4)
        state = device.create_batch_state(POP, 4)
        # All mode genes at 0 (OFF)
        genome = np.zeros((POP, 4))
        device.apply_genome_batch(state, genome)
        np.testing.assert_array_equal(state.factors, 0.0)

    def test_solar_pv_mode_decoded(self):
        device = make_device(make_solar_param(), horizon=4)
        state = device.create_batch_state(1, 4)
        # Gene value 1.0 → position 1 → InverterMode.PV
        genome = np.ones((1, 4))
        device.apply_genome_batch(state, genome)
        assert all(state.modes[0, :] == InverterMode.PV)

    # ---- BATTERY ----

    def test_battery_mode_and_factor_decoded(self):
        device = make_device(make_battery_param(), horizon=2)
        state = device.create_batch_state(1, 2)
        # Genome: [mode=1(→CHARGE), factor=0.5, mode=2(→DISCHARGE), factor=0.3]
        genome = np.array([[1.0, 0.5, 2.0, 0.3]])
        device.apply_genome_batch(state, genome)
        # Step 0: position 1 → CHARGE (value 2)
        assert state.modes[0, 0] == InverterMode.CHARGE
        # Step 1: position 2 → DISCHARGE (value 3)
        assert state.modes[0, 1] == InverterMode.DISCHARGE

    def test_battery_genome_writeback_lamarckian(self):
        # After apply_genome_batch, genome contains repaired mode integers as floats
        param = make_battery_param(max_charge=0.6)  # caps factor at 0.6
        device = make_device(param, horizon=1)
        state = device.create_batch_state(1, 1)
        # Genome: position 1 (CHARGE), factor 1.0 (will be capped to 0.6)
        genome = np.array([[1.0, 1.0]])
        device.apply_genome_batch(state, genome)
        # Repaired factor in genome[:, 1] should be ≤ 0.6
        assert genome[0, 1] <= 0.6

    def test_battery_mode_writeback(self):
        device = make_device(make_battery_param(), horizon=1)
        state = device.create_batch_state(1, 1)
        # Position 0.2 → rounds to 0 → OFF (value 0)
        genome = np.array([[0.2, 0.0]])
        device.apply_genome_batch(state, genome)
        assert genome[0, 0] == pytest.approx(float(InverterMode.OFF))

    # ---- SoC evolution ----

    def test_soc_evolves_across_steps(self):
        param = make_battery_param(
            capacity_wh=10_000.0,
            min_soc=0.0, max_soc=1.0, ac_to_bat_eff=1.0,
        )
        device = make_device(param, horizon=3, initial_soc_factor=0.5)
        state = device.create_batch_state(1, 3)
        # 3 steps all CHARGE (position 1) with factor 0.1 each
        genome = np.array([[1.0, 0.1, 1.0, 0.1, 1.0, 0.1]])
        device.apply_genome_batch(state, genome)
        # each step stores 0.1 × 10000 × 1.0 × 1.0 = 1000 Wh
        expected_soc_t0 = 5000.0 + 1000.0
        expected_soc_t1 = expected_soc_t0 + 1000.0
        expected_soc_t2 = expected_soc_t1 + 1000.0
        assert state.soc_wh[0, 0] == pytest.approx(expected_soc_t0, rel=1e-6)
        assert state.soc_wh[0, 1] == pytest.approx(expected_soc_t1, rel=1e-6)
        assert state.soc_wh[0, 2] == pytest.approx(expected_soc_t2, rel=1e-6)

    # ---- Population axis independence ----

    def test_population_axis_independent(self):
        device = make_device(make_battery_param(capacity_wh=10_000.0), horizon=1)
        state = device.create_batch_state(2, 1)
        # Individual 0: CHARGE (pos=1) with factor 0.5
        # Individual 1: DISCHARGE (pos=2) with factor 0.3
        genome = np.array([
            [1.0, 0.5],  # individual 0
            [2.0, 0.3],  # individual 1
        ])
        device.apply_genome_batch(state, genome)
        assert state.modes[0, 0] == InverterMode.CHARGE
        assert state.modes[1, 0] == InverterMode.DISCHARGE
        assert state.ac_power_w[0, 0] > 0.0   # charging consumes
        assert state.ac_power_w[1, 0] < 0.0   # discharging injects


# ============================================================
# TestBuildDeviceRequest
# ============================================================

class TestBuildDeviceRequest:
    def _setup(self, param=None, pop=POP, horizon=HORIZON, device_index=7, port_index=3):
        if param is None:
            param = make_battery_param()
        device = HybridInverterDevice(param, device_index, port_index)
        device.setup_run(make_context(horizon=horizon))
        state = device.create_batch_state(pop, horizon)
        # All CHARGE mode with factor 0.2 — gives a predictable ac_power_w
        genome = np.tile([1.0, 0.2], (pop, horizon))
        device.apply_genome_batch(state, genome)
        return device, state

    def test_returns_device_request(self):
        from akkudoktoreos.simulation.genetic2.arbitrator import DeviceRequest
        device, state = self._setup()
        req = device.build_device_request(state)
        assert isinstance(req, DeviceRequest)

    def test_device_index_matches(self):
        device, state = self._setup(device_index=7)
        req = device.build_device_request(state)
        assert req.device_index == 7

    def test_single_port_request(self):
        device, state = self._setup()
        req = device.build_device_request(state)
        assert len(req.port_requests) == 1

    def test_port_index_matches(self):
        device, state = self._setup(port_index=3)
        req = device.build_device_request(state)
        assert req.port_requests[0].port_index == 3

    def test_energy_wh_equals_ac_power_times_step_h(self):
        device, state = self._setup()
        req = device.build_device_request(state)
        step_h = STEP_INTERVAL / 3600.0
        np.testing.assert_allclose(
            req.port_requests[0].energy_wh,
            state.ac_power_w * step_h,
            rtol=1e-9,
        )

    def test_energy_wh_shape(self):
        device, state = self._setup(pop=POP, horizon=HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].energy_wh.shape == (POP, HORIZON)

    def test_min_energy_wh_shape(self):
        device, state = self._setup(pop=POP, horizon=HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].min_energy_wh.shape == (POP, HORIZON)

    def test_off_steps_have_zero_min_energy(self):
        param = make_battery_param()
        device = HybridInverterDevice(param, 0, 0)
        device.setup_run(make_context(horizon=HORIZON))
        state = device.create_batch_state(1, HORIZON)
        # All OFF (genome pos 0 = OFF for BATTERY)
        genome = np.zeros((1, 2 * HORIZON))
        device.apply_genome_batch(state, genome)
        req = device.build_device_request(state)
        np.testing.assert_array_equal(req.port_requests[0].min_energy_wh, 0.0)

    def test_charge_steps_have_positive_min_energy(self):
        param = make_battery_param(min_charge=0.1)
        device = HybridInverterDevice(param, 0, 0)
        device.setup_run(make_context(horizon=1))
        state = device.create_batch_state(1, 1)
        # CHARGE mode (position 1), factor 0.5
        genome = np.array([[1.0, 0.5]])
        device.apply_genome_batch(state, genome)
        req = device.build_device_request(state)
        assert req.port_requests[0].min_energy_wh[0, 0] > 0.0

    def test_discharge_steps_have_negative_min_energy(self):
        param = make_battery_param(min_discharge=0.1)
        device = HybridInverterDevice(param, 0, 0)
        device.setup_run(make_context(horizon=1))
        state = device.create_batch_state(1, 1)
        # DISCHARGE mode (position 2), factor 0.5
        genome = np.array([[2.0, 0.5]])
        device.apply_genome_batch(state, genome)
        req = device.build_device_request(state)
        assert req.port_requests[0].min_energy_wh[0, 0] < 0.0


# ============================================================
# TestApplyDeviceGrant
# ============================================================

class TestApplyDeviceGrant:
    def _make_grant(self, granted_wh: np.ndarray, port_index: int = 0) -> DeviceGrant:
        return DeviceGrant(
            device_index=0,
            port_grants=(
                PortGrant(port_index=port_index, granted_wh=granted_wh),
            ),
        )

    def test_ac_power_w_updated_from_grant(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
        granted_wh = np.full((POP, HORIZON), 1000.0)
        grant = self._make_grant(granted_wh)
        device.apply_device_grant(state, grant)
        step_h = STEP_INTERVAL / 3600.0
        np.testing.assert_allclose(state.ac_power_w, 1000.0 / step_h, rtol=1e-9)

    def test_grant_conversion_step_h(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(1, 1)
        # granted 500 Wh at 1-hour step → power should be 500 W
        granted_wh = np.array([[500.0]])
        grant = self._make_grant(granted_wh)
        device.apply_device_grant(state, grant)
        assert state.ac_power_w[0, 0] == pytest.approx(500.0 / 1.0, rel=1e-9)

    def test_other_state_arrays_unchanged(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
        # Populate modes with something non-zero
        state.modes[:] = InverterMode.CHARGE
        modes_before = state.modes.copy()
        grant = self._make_grant(np.zeros((POP, HORIZON)))
        device.apply_device_grant(state, grant)
        np.testing.assert_array_equal(state.modes, modes_before)


# ============================================================
# TestComputeCost
# ============================================================

class TestComputeCost:
    def test_shape_is_pop_zero(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        assert cost.shape == (POP, 0)

    def test_all_values_zero(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        np.testing.assert_array_equal(cost, 0.0)

    def test_solar_shape(self):
        device = make_device(make_solar_param())
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        assert cost.shape == (POP, 0)


# ============================================================
# TestExtractInstructions
# ============================================================

class TestExtractInstructions:
    def _run(self, param, modes_per_step: list, factors_per_step: list):
        """Run apply_genome_batch with a hand-crafted genome that produces
        the requested modes and factors (for a single individual)."""
        horizon = len(modes_per_step)
        device = make_device(param, horizon=horizon)
        state = device.create_batch_state(1, horizon)

        if param.has_factor_gene:
            # Build interleaved genome from desired position indices and factors.
            # We need position indices, not InverterMode values.
            mode_positions = [
                param.valid_modes.index(InverterMode(m)) for m in modes_per_step
            ]
            genome = np.empty((1, 2 * horizon))
            for t, (pos, f) in enumerate(zip(mode_positions, factors_per_step)):
                genome[0, 2 * t] = float(pos)
                genome[0, 2 * t + 1] = f
        else:
            mode_positions = [
                param.valid_modes.index(InverterMode(m)) for m in modes_per_step
            ]
            genome = np.array([[float(p) for p in mode_positions]])

        device.apply_genome_batch(state, genome)
        return device, state

    def test_returns_list_of_length_horizon(self):
        param = make_solar_param()
        device, state = self._run(
            param,
            [InverterMode.OFF, InverterMode.PV, InverterMode.OFF, InverterMode.PV],
            [0.0, 0.0, 0.0, 0.0],
        )
        result = device.extract_instructions(state, 0)
        assert len(result) == HORIZON

    def test_all_elements_are_ombc_instruction(self):
        param = make_battery_param()
        device, state = self._run(
            param,
            [InverterMode.OFF] * HORIZON,
            [0.0] * HORIZON,
        )
        result = device.extract_instructions(state, 0)
        assert all(isinstance(inst, OMBCInstruction) for inst in result)

    def test_resource_id_matches_device_id(self):
        param = make_battery_param(device_id="inv_001")
        device, state = self._run(param, [InverterMode.OFF] * HORIZON, [0.0] * HORIZON)
        result = device.extract_instructions(state, 0)
        assert all(inst.resource_id == "inv_001" for inst in result)

    def test_execution_times_match_step_times(self):
        param = make_solar_param()
        device, state = self._run(
            param,
            [InverterMode.PV] * HORIZON,
            [0.0] * HORIZON,
        )
        result = device.extract_instructions(state, 0)
        expected_times = make_step_times(HORIZON)
        for inst, expected_dt in zip(result, expected_times):
            assert inst.execution_time == expected_dt

    def test_operation_mode_id_matches_mode_name(self):
        param = make_battery_param()
        modes = [InverterMode.OFF, InverterMode.CHARGE, InverterMode.DISCHARGE, InverterMode.OFF]
        device, state = self._run(param, modes, [0.0, 0.5, 0.5, 0.0])
        result = device.extract_instructions(state, 0)
        expected_names = [InverterMode(m).name for m in modes]
        # Note: repair may zero some factors, but modes should be preserved for valid inputs
        for inst, name in zip(result, expected_names):
            assert inst.operation_mode_id == name

    def test_operation_mode_factor_matches_state(self):
        param = make_battery_param(min_charge=0.05, max_charge=1.0)
        device, state = self._run(
            param,
            [InverterMode.CHARGE] * HORIZON,
            [0.3] * HORIZON,
        )
        result = device.extract_instructions(state, 0)
        for inst in result:
            assert inst.operation_mode_factor == pytest.approx(
                state.factors[0, result.index(inst)], rel=1e-9
            )

    def test_individual_index_selects_correct_row(self):
        param = make_battery_param()
        horizon = 2
        device = make_device(param, horizon=horizon)
        state = device.create_batch_state(2, horizon)
        # Individual 0: all OFF; Individual 1: all CHARGE
        genome = np.array([
            [0.0, 0.0, 0.0, 0.0],   # pos 0=OFF, factor 0 × 2 steps
            [1.0, 0.3, 1.0, 0.3],   # pos 1=CHARGE, factor 0.3 × 2 steps
        ])
        device.apply_genome_batch(state, genome)

        instructions_0 = device.extract_instructions(state, 0)
        instructions_1 = device.extract_instructions(state, 1)

        assert all(inst.operation_mode_id == "OFF" for inst in instructions_0)
        assert all(inst.operation_mode_id == "CHARGE" for inst in instructions_1)
