"""Tests for the hybrid inverter energy device.

Module under test
-----------------
``akkudoktoreos.devices.genetic2.hybridinverter``

Test strategy
-------------
Each class exercises one cohesive responsibility. Internal helpers
(_decode_gene, _repair_factor, _compute_ac_power, _advance_soc) are
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
        - All types: size == horizon (single gene per step)
        - All types: lower_bound == 0.0 for all steps
        - BATTERY: upper_bound == n_modes == 3.0
        - SOLAR:   upper_bound == n_modes == 2.0
        - HYBRID:  upper_bound == n_modes == 4.0
        - Before setup_run raises RuntimeError

    TestCreateBatchState
        - Array shapes and dtypes
        - soc_wh pre-filled with resolved battery_initial_soc_wh from context
        - step_times reference matches setup_run argument
        - Before setup_run raises RuntimeError

    TestDecodeGene
        - Integer part selects mode position in valid_modes
        - Fractional part is returned as raw factor
        - BATTERY: positions 0/1/2 → OFF/CHARGE/DISCHARGE
        - SOLAR:   positions 0/1   → OFF/PV
        - HYBRID:  positions 0/1/2/3 → OFF/PV/CHARGE/DISCHARGE
        - Fractional part extracted correctly
        - Out-of-range high clipped to last mode, factor = 0 at integer boundary
        - Out-of-range low clipped to OFF
        - Mode dtype is int8

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
        - All types: genome shape is (population_size, horizon)
        - SOLAR: factors all zero; modes decoded from integer part of gene
        - BATTERY: modes and factors decoded from single-gene encoding
        - Lamarckian write-back: genome[:, t] = mode_position + repaired_factor
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
    ``resolve_prediction(key)`` returns a pre-configured PV array.
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


def make_genome(pop: int, horizon: int, mode_positions: list[int], factors: list[float]) -> np.ndarray:
    """Build a genome array from parallel lists of mode positions and factors.

    Each gene = mode_position + factor, broadcast across all individuals
    in the population.  ``mode_positions`` and ``factors`` must each have
    length ``horizon``.
    """
    gene_values = [float(p) + f for p, f in zip(mode_positions, factors)]
    return np.tile(gene_values, (pop, 1))


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
                device_id="s", port_id="p", bus_id="b",
                inverter_type=InverterType.SOLAR,
                off_state_power_consumption_w=0.0, on_state_power_consumption_w=0.0,
                pv_to_ac_efficiency=0.0,   # invalid
                pv_to_battery_efficiency=0.95,
                pv_max_power_w=8000.0, pv_min_power_w=0.0,
                pv_power_w_key=PV_KEY,
                ac_to_battery_efficiency=0.95, battery_to_ac_efficiency=0.95,
                battery_capacity_wh=0.0, battery_charge_rates=None,
                battery_min_charge_rate=0.0, battery_max_charge_rate=0.0,
                battery_min_discharge_rate=0.0, battery_max_discharge_rate=0.0,
                battery_min_soc_factor=0.0, battery_max_soc_factor=0.1,
                battery_initial_soc_factor_key=SOC_KEY,
            )

    def test_solar_pv_max_power_zero_raises(self):
        with pytest.raises(ValueError, match="pv_max_power_w"):
            HybridInverterParam(
                device_id="s", port_id="p", bus_id="b",
                inverter_type=InverterType.SOLAR,
                off_state_power_consumption_w=0.0, on_state_power_consumption_w=0.0,
                pv_to_ac_efficiency=0.97, pv_to_battery_efficiency=0.95,
                pv_max_power_w=0.0,   # invalid
                pv_min_power_w=0.0, pv_power_w_key=PV_KEY,
                ac_to_battery_efficiency=0.95, battery_to_ac_efficiency=0.95,
                battery_capacity_wh=0.0, battery_charge_rates=None,
                battery_min_charge_rate=0.0, battery_max_charge_rate=0.0,
                battery_min_discharge_rate=0.0, battery_max_discharge_rate=0.0,
                battery_min_soc_factor=0.0, battery_max_soc_factor=0.1,
                battery_initial_soc_factor_key=SOC_KEY,
            )

    def test_solar_pv_min_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="pv_min_power_w"):
            HybridInverterParam(
                device_id="s", port_id="p", bus_id="b",
                inverter_type=InverterType.SOLAR,
                off_state_power_consumption_w=0.0, on_state_power_consumption_w=0.0,
                pv_to_ac_efficiency=0.97, pv_to_battery_efficiency=0.95,
                pv_max_power_w=8000.0, pv_min_power_w=9000.0,   # > max
                pv_power_w_key=PV_KEY,
                ac_to_battery_efficiency=0.95, battery_to_ac_efficiency=0.95,
                battery_capacity_wh=0.0, battery_charge_rates=None,
                battery_min_charge_rate=0.0, battery_max_charge_rate=0.0,
                battery_min_discharge_rate=0.0, battery_max_discharge_rate=0.0,
                battery_min_soc_factor=0.0, battery_max_soc_factor=0.1,
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
            device_id="s", port_id="p", bus_id="b",
            inverter_type=InverterType.SOLAR,
            off_state_power_consumption_w=0.0, on_state_power_consumption_w=0.0,
            pv_to_ac_efficiency=0.97, pv_to_battery_efficiency=0.95,
            pv_max_power_w=8000.0, pv_min_power_w=0.0,
            pv_power_w_key=None,   # invalid for SOLAR
            ac_to_battery_efficiency=0.95, battery_to_ac_efficiency=0.95,
            battery_capacity_wh=0.0, battery_charge_rates=None,
            battery_min_charge_rate=0.0, battery_max_charge_rate=0.0,
            battery_min_discharge_rate=0.0, battery_max_discharge_rate=0.0,
            battery_min_soc_factor=0.0, battery_max_soc_factor=0.1,
            battery_initial_soc_factor_key=SOC_KEY,
        )
        device = HybridInverterDevice(param, 0, 0)
        with pytest.raises(ValueError, match="pv_power_w_key"):
            device.setup_run(make_context())

    def test_solar_pv_wrong_shape_raises(self):
        """SOLAR with resolved PV array of wrong length raises at setup_run."""
        param = make_solar_param()
        device = HybridInverterDevice(param, 0, 0)
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
        ctx = make_context(initial_soc_factor=1.1)
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
    """The genome has one gene per time step for all inverter types.

    Each gene encodes mode (integer part) + factor (fractional part).
    Lower bound is always 0.0; upper bound is n_modes (so the last valid
    mode is the integer n_modes − 1 and factor reaches up to 1.0).
    """

    def test_battery_genome_size_equals_horizon(self):
        device = make_device(make_battery_param(), horizon=4)
        slc = device.genome_requirements()
        assert slc.size == 4

    def test_solar_genome_size_equals_horizon(self):
        device = make_device(make_solar_param(), horizon=4)
        slc = device.genome_requirements()
        assert slc.size == 4

    def test_hybrid_genome_size_equals_horizon(self):
        device = make_device(make_hybrid_param(), horizon=4)
        slc = device.genome_requirements()
        assert slc.size == 4

    def test_battery_lower_bound_all_zero(self):
        device = make_device(make_battery_param(), horizon=4)
        slc = device.genome_requirements()
        np.testing.assert_array_equal(slc.lower_bound, np.zeros(4))

    def test_solar_lower_bound_all_zero(self):
        device = make_device(make_solar_param(), horizon=4)
        slc = device.genome_requirements()
        np.testing.assert_array_equal(slc.lower_bound, np.zeros(4))

    def test_hybrid_lower_bound_all_zero(self):
        device = make_device(make_hybrid_param(), horizon=4)
        slc = device.genome_requirements()
        np.testing.assert_array_equal(slc.lower_bound, np.zeros(4))

    def test_battery_upper_bound_equals_n_modes(self):
        # BATTERY has 3 modes → upper bound 3.0
        device = make_device(make_battery_param(), horizon=4)
        slc = device.genome_requirements()
        np.testing.assert_array_equal(slc.upper_bound, np.full(4, 3.0))

    def test_solar_upper_bound_equals_n_modes(self):
        # SOLAR has 2 modes → upper bound 2.0
        device = make_device(make_solar_param(), horizon=4)
        slc = device.genome_requirements()
        np.testing.assert_array_equal(slc.upper_bound, np.full(4, 2.0))

    def test_hybrid_upper_bound_equals_n_modes(self):
        # HYBRID has 4 modes → upper bound 4.0
        device = make_device(make_hybrid_param(), horizon=4)
        slc = device.genome_requirements()
        np.testing.assert_array_equal(slc.upper_bound, np.full(4, 4.0))

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
# TestDecodeGene
# ============================================================

class TestDecodeGene:
    """Tests for _decode_gene, which splits a gene into mode and raw factor.

    Gene encoding:
        gene = mode_position + factor    (factor ∈ [0, 1))

    For BATTERY valid_modes = (OFF, CHARGE, DISCHARGE):
        gene ∈ [0, 1) → position 0 → OFF
        gene ∈ [1, 2) → position 1 → CHARGE
        gene ∈ [2, 3) → position 2 → DISCHARGE

    For SOLAR valid_modes = (OFF, PV):
        gene ∈ [0, 1) → position 0 → OFF
        gene ∈ [1, 2) → position 1 → PV

    For HYBRID valid_modes = (OFF, PV, CHARGE, DISCHARGE):
        gene ∈ [0, 1) → position 0 → OFF
        gene ∈ [1, 2) → position 1 → PV
        gene ∈ [2, 3) → position 2 → CHARGE
        gene ∈ [3, 4) → position 3 → DISCHARGE
    """

    def _decode(self, device, gene_values):
        gene = np.array(gene_values, dtype=np.float64)
        return device._decode_gene(gene)

    # ---- BATTERY mode decoding ----

    def test_battery_gene_0_decodes_to_off(self):
        device = make_device(make_battery_param())
        mode, _ = self._decode(device, [0.0])
        assert mode[0] == InverterMode.OFF

    def test_battery_gene_1_decodes_to_charge(self):
        device = make_device(make_battery_param())
        mode, _ = self._decode(device, [1.0])
        assert mode[0] == InverterMode.CHARGE

    def test_battery_gene_2_decodes_to_discharge(self):
        device = make_device(make_battery_param())
        mode, _ = self._decode(device, [2.0])
        assert mode[0] == InverterMode.DISCHARGE

    def test_battery_never_decodes_to_pv(self):
        device = make_device(make_battery_param())
        mode, _ = self._decode(device, [0.0, 0.5, 1.0, 1.5, 2.0, 2.9])
        assert InverterMode.PV not in mode

    # ---- SOLAR mode decoding ----

    def test_solar_gene_0_decodes_to_off(self):
        device = make_device(make_solar_param())
        mode, _ = self._decode(device, [0.0])
        assert mode[0] == InverterMode.OFF

    def test_solar_gene_1_decodes_to_pv(self):
        device = make_device(make_solar_param())
        mode, _ = self._decode(device, [1.0])
        assert mode[0] == InverterMode.PV

    # ---- HYBRID mode decoding ----

    def test_hybrid_gene_0_decodes_to_off(self):
        device = make_device(make_hybrid_param())
        mode, _ = self._decode(device, [0.0])
        assert mode[0] == InverterMode.OFF

    def test_hybrid_gene_1_decodes_to_pv(self):
        device = make_device(make_hybrid_param())
        mode, _ = self._decode(device, [1.0])
        assert mode[0] == InverterMode.PV

    def test_hybrid_gene_2_decodes_to_charge(self):
        device = make_device(make_hybrid_param())
        mode, _ = self._decode(device, [2.0])
        assert mode[0] == InverterMode.CHARGE

    def test_hybrid_gene_3_decodes_to_discharge(self):
        device = make_device(make_hybrid_param())
        mode, _ = self._decode(device, [3.0])
        assert mode[0] == InverterMode.DISCHARGE

    # ---- Factor (fractional part) extraction ----

    def test_fractional_part_extracted_correctly(self):
        device = make_device(make_battery_param())
        _, factor = self._decode(device, [1.3, 2.7])
        assert factor[0] == pytest.approx(0.3, rel=1e-9)
        assert factor[1] == pytest.approx(0.7, rel=1e-9)

    def test_integer_gene_gives_zero_factor(self):
        device = make_device(make_battery_param())
        _, factor = self._decode(device, [0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(factor, 0.0)

    def test_mode_and_factor_independent(self):
        """A gene of 1.6 decodes to CHARGE (pos 1) with factor 0.6."""
        device = make_device(make_battery_param())
        mode, factor = self._decode(device, [1.6])
        assert mode[0] == InverterMode.CHARGE
        assert factor[0] == pytest.approx(0.6, rel=1e-9)

    # ---- Boundary and out-of-range handling ----

    def test_out_of_range_high_clipped_to_last_mode(self):
        # BATTERY n_modes=3; gene=99.0 → position clipped to 2 → DISCHARGE
        device = make_device(make_battery_param())
        mode, factor = self._decode(device, [99.0])
        assert mode[0] == InverterMode.DISCHARGE

    def test_out_of_range_low_clipped_to_off(self):
        device = make_device(make_battery_param())
        mode, _ = self._decode(device, [-5.0])
        assert mode[0] == InverterMode.OFF

    def test_mode_dtype_is_int8(self):
        device = make_device(make_battery_param())
        mode, _ = self._decode(device, [0.0, 1.0])
        assert mode.dtype == np.int8

    def test_population_vector_decoded_independently(self):
        """Multiple gene values (simulating a population) are decoded correctly."""
        device = make_device(make_hybrid_param())
        # genes: OFF(0.2), PV(1.5), CHARGE(2.8), DISCHARGE(3.1)
        mode, factor = self._decode(device, [0.2, 1.5, 2.8, 3.1])
        assert mode[0] == InverterMode.OFF
        assert mode[1] == InverterMode.PV
        assert mode[2] == InverterMode.CHARGE
        assert mode[3] == InverterMode.DISCHARGE
        assert factor[0] == pytest.approx(0.2, rel=1e-9)
        assert factor[1] == pytest.approx(0.5, rel=1e-9)
        assert factor[2] == pytest.approx(0.8, rel=1e-9)
        assert factor[3] == pytest.approx(0.1, rel=1e-9)


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
        assert result[0] == pytest.approx(0.7, rel=1e-6)

    def test_charge_below_min_charge_rate_clipped_to_min(self):
        param = make_battery_param(min_charge=0.1, max_charge=1.0, capacity_wh=10_000.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [0.05], [5000.0])
        assert result[0] == pytest.approx(0.1)

    # ---- CHARGE: discrete rate snapping ----

    def test_charge_discrete_rates_snap_to_nearest(self):
        param = make_battery_param(charge_rates=(0.25, 0.5, 0.75, 1.0))
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [0.6], [1000.0])
        assert result[0] == pytest.approx(0.5)

    def test_charge_discrete_rates_snap_rounds_up(self):
        param = make_battery_param(charge_rates=(0.25, 0.75, 1.0))
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [0.55], [1000.0])
        assert result[0] == pytest.approx(0.75)

    # ---- CHARGE: SoC cap ----

    def test_charge_soc_cap_reduces_factor(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=0.9, ac_to_bat_eff=0.95,
            min_charge=0.01, max_charge=1.0,
        )
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [1.0], [8500.0])
        expected_max = 500.0 / (10_000.0 * 1.0 * 0.95)
        assert result[0] == pytest.approx(expected_max, rel=1e-5)

    def test_charge_soc_at_max_zeroes_factor(self):
        param = make_battery_param(capacity_wh=10_000.0, max_soc=0.9, min_charge=0.05)
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [1.0], [9000.0])
        assert result[0] == pytest.approx(0.0)

    # ---- DISCHARGE: rate bounds ----

    def test_discharge_clips_to_max_discharge_rate(self):
        param = make_battery_param(max_discharge=0.5, capacity_wh=10_000.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [1.0], [9000.0])
        assert result[0] == pytest.approx(0.5)

    # ---- DISCHARGE: SoC floor ----

    def test_discharge_soc_at_min_zeroes_factor(self):
        param = make_battery_param(capacity_wh=10_000.0, min_soc=0.1, min_discharge=0.05)
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [1.0], [1000.0])
        assert result[0] == pytest.approx(0.0)

    def test_discharge_partial_soc_reduces_factor(self):
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

    def test_off_returns_off_state_power(self):
        param = make_battery_param(off_w=8.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.OFF], [0.0])
        assert result[0] == pytest.approx(8.0)

    def test_solar_pv_injects_correct_power(self):
        param = make_solar_param(on_w=10.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.PV], [0.0], pv_dc_w=4000.0)
        expected = -(4000.0 * 0.97) + 10.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_solar_pv_negative_power(self):
        param = make_solar_param(on_w=0.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.PV], [0.0], pv_dc_w=1000.0)
        assert result[0] < 0.0

    def test_battery_charge_active_consumes_from_bus(self):
        param = make_battery_param(capacity_wh=10_000.0, ac_to_bat_eff=0.95, on_w=10.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [0.5])
        expected = (0.5 * 10_000.0 / 0.95) + 10.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_charge_active_positive_power(self):
        device = make_device(make_battery_param())
        result = self._call(device, [InverterMode.CHARGE], [0.3])
        assert result[0] > 0.0

    def test_battery_charge_factor_zero_falls_back_to_off_state(self):
        param = make_battery_param(off_w=5.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.CHARGE], [0.0])
        assert result[0] == pytest.approx(5.0)

    def test_battery_discharge_active_injects_into_bus(self):
        param = make_battery_param(capacity_wh=10_000.0, bat_to_ac_eff=0.95, on_w=10.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [0.5])
        expected = -(0.5 * 10_000.0 * 0.95) + 10.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_discharge_negative_power(self):
        param = make_battery_param(on_w=1.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [1.0])
        assert result[0] < 0.0

    def test_battery_discharge_factor_zero_returns_on_state(self):
        param = make_battery_param(on_w=15.0)
        device = make_device(param)
        result = self._call(device, [InverterMode.DISCHARGE], [0.0])
        assert result[0] == pytest.approx(15.0)

    def test_hybrid_charge_pv_reduces_ac_demand(self):
        param = make_hybrid_param(
            capacity_wh=10_000.0, ac_to_bat_eff=0.95,
            pv_to_bat_eff=0.95, pv_to_ac_eff=0.97, on_w=10.0,
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
        param = make_hybrid_param(capacity_wh=1_000.0, on_w=0.0)
        device = make_device(param, pv_w=10_000.0)
        result = self._call(device, [InverterMode.CHARGE], [0.1], pv_dc_w=10_000.0)
        assert result[0] < 0.0

    def test_hybrid_discharge_combines_battery_and_pv(self):
        param = make_hybrid_param(
            capacity_wh=10_000.0, bat_to_ac_eff=0.95, pv_to_ac_eff=0.97, on_w=5.0,
        )
        device = make_device(param, pv_w=2000.0)
        result = self._call(device, [InverterMode.DISCHARGE], [0.5], pv_dc_w=2000.0)
        expected = -(0.5 * 10_000.0 * 0.95 + 2000.0 * 0.97) + 5.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_mixed_modes_independent(self):
        param = make_battery_param(off_w=5.0, on_w=10.0, capacity_wh=10_000.0)
        device = make_device(param)
        modes = np.array([InverterMode.OFF, InverterMode.CHARGE, InverterMode.DISCHARGE], dtype=np.int8)
        factors = np.array([0.0, 0.5, 0.5])
        result = device._compute_ac_power(modes, factors, 0.0)
        assert result[0] == pytest.approx(5.0)
        assert result[1] > 0.0
        assert result[2] < 0.0


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
        assert result[0] == pytest.approx(0.0)

    def test_battery_charge_increases_soc(self):
        param = make_battery_param(
            capacity_wh=10_000.0, ac_to_bat_eff=0.95, min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        result = self._call(device, [2000.0], [InverterMode.CHARGE], [0.5])
        expected = 2000.0 + 0.5 * 10_000.0 * 1.0 * 0.95
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_discharge_decreases_soc(self):
        param = make_battery_param(capacity_wh=10_000.0, min_soc=0.0, max_soc=1.0)
        device = make_device(param, initial_soc_factor=0.5)
        result = self._call(device, [8000.0], [InverterMode.DISCHARGE], [0.3])
        expected = 8000.0 - 0.3 * 10_000.0 * 1.0
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_soc_clamped_at_max_on_overcharge(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=0.9, min_soc=0.0, ac_to_bat_eff=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        result = self._call(device, [8900.0], [InverterMode.CHARGE], [1.0])
        assert result[0] == pytest.approx(9000.0)

    def test_soc_clamped_at_min_on_over_discharge(self):
        param = make_battery_param(capacity_wh=10_000.0, min_soc=0.1, max_soc=1.0)
        device = make_device(param, initial_soc_factor=0.5)
        result = self._call(device, [1100.0], [InverterMode.DISCHARGE], [1.0])
        assert result[0] == pytest.approx(1000.0)

    def test_hybrid_charge_pv_contributes_first(self):
        param = make_hybrid_param(
            capacity_wh=10_000.0, ac_to_bat_eff=0.95, pv_to_bat_eff=0.95,
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
        param = make_hybrid_param(
            capacity_wh=1_000.0, ac_to_bat_eff=0.95, pv_to_bat_eff=0.95,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, pv_w=5000.0, initial_soc_factor=0.5)
        result = self._call(device, [200.0], [InverterMode.CHARGE], [0.1], pv_dc_w=5000.0)
        stored_wh = 0.1 * 1_000.0
        expected = 200.0 + stored_wh
        assert result[0] == pytest.approx(expected, rel=1e-9)


# ============================================================
# TestApplyGenomeBatch
# ============================================================

class TestApplyGenomeBatch:
    """Integration tests for the full apply_genome_batch pipeline.

    Genome shape is always (population_size, horizon) — one gene per step.
    """

    # ---- SOLAR ----

    def test_solar_genome_shape_is_pop_by_horizon(self):
        """SOLAR genome has shape (pop, horizon), not (pop, 2*horizon)."""
        device = make_device(make_solar_param(), horizon=4)
        state = device.create_batch_state(POP, 4)
        genome = np.zeros((POP, 4))
        device.apply_genome_batch(state, genome)
        assert genome.shape == (POP, 4)

    def test_solar_all_factors_zero(self):
        device = make_device(make_solar_param(), horizon=4)
        state = device.create_batch_state(POP, 4)
        genome = np.zeros((POP, 4))   # all OFF (position 0)
        device.apply_genome_batch(state, genome)
        np.testing.assert_array_equal(state.factors, 0.0)

    def test_solar_pv_mode_decoded_from_gene_1(self):
        """Gene value 1.0 → position 1 → InverterMode.PV for SOLAR."""
        device = make_device(make_solar_param(), horizon=4)
        state = device.create_batch_state(1, 4)
        genome = np.ones((1, 4))   # gene=1.0 → PV
        device.apply_genome_batch(state, genome)
        assert all(state.modes[0, :] == InverterMode.PV)

    # ---- BATTERY ----

    def test_battery_genome_shape_is_pop_by_horizon(self):
        """BATTERY genome shape (pop, horizon), same as all other types."""
        device = make_device(make_battery_param(), horizon=2)
        state = device.create_batch_state(1, 2)
        genome = np.array([[1.5, 2.3]])   # shape (1, 2)
        device.apply_genome_batch(state, genome)
        assert genome.shape == (1, 2)

    def test_battery_charge_decoded_from_gene_in_range_1_to_2(self):
        """Gene ∈ [1, 2) → position 1 → CHARGE for BATTERY."""
        device = make_device(make_battery_param(), horizon=1)
        state = device.create_batch_state(1, 1)
        genome = np.array([[1.5]])   # position=1 → CHARGE, raw_factor=0.5
        device.apply_genome_batch(state, genome)
        assert state.modes[0, 0] == InverterMode.CHARGE

    def test_battery_discharge_decoded_from_gene_in_range_2_to_3(self):
        """Gene ∈ [2, 3) → position 2 → DISCHARGE for BATTERY."""
        device = make_device(make_battery_param(), horizon=1)
        state = device.create_batch_state(1, 1)
        genome = np.array([[2.3]])   # position=2 → DISCHARGE, raw_factor=0.3
        device.apply_genome_batch(state, genome)
        assert state.modes[0, 0] == InverterMode.DISCHARGE

    def test_battery_factor_decoded_from_fractional_part(self):
        """After repair the factor equals the (clamped) fractional part of the gene."""
        param = make_battery_param(min_charge=0.0, max_charge=1.0, capacity_wh=10_000.0)
        device = make_device(param, initial_soc_factor=0.1)
        state = device.create_batch_state(1, 1)
        genome = np.array([[1.4]])   # CHARGE with raw_factor=0.4
        device.apply_genome_batch(state, genome)
        assert state.factors[0, 0] == pytest.approx(0.4, rel=1e-6)

    # ---- Lamarckian write-back ----

    def test_lamarckian_writeback_encodes_position_plus_factor(self):
        """After apply_genome_batch, genome[:, t] = mode_position + repaired_factor."""
        param = make_battery_param(max_charge=0.6, min_charge=0.0)
        device = make_device(param, horizon=1)
        state = device.create_batch_state(1, 1)
        # CHARGE (position 1) with factor 1.0 → capped to 0.6
        genome = np.array([[1.99]])
        device.apply_genome_batch(state, genome)
        # Written-back gene: position=1, repaired_factor <= 0.6 → gene in [1.0, 1.6]
        assert 1.0 <= genome[0, 0] <= 1.6 + 1e-9

    def test_lamarckian_writeback_off_mode_has_zero_factor(self):
        """OFF mode genes write back as pure integers (factor zeroed)."""
        device = make_device(make_battery_param(), horizon=1)
        state = device.create_batch_state(1, 1)
        genome = np.array([[0.7]])   # OFF with factor=0.7; factor zeroed by repair
        device.apply_genome_batch(state, genome)
        assert genome[0, 0] == pytest.approx(0.0, abs=1e-9)  # position=0, factor=0

    # ---- SoC evolution ----

    def test_soc_evolves_across_steps(self):
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.0, max_soc=1.0, ac_to_bat_eff=1.0,
        )
        device = make_device(param, horizon=3, initial_soc_factor=0.5)
        state = device.create_batch_state(1, 3)
        # Gene 1.1: position=1 (CHARGE), factor=0.1 each step
        genome = np.array([[1.1, 1.1, 1.1]])
        device.apply_genome_batch(state, genome)
        expected_t0 = 5000.0 + 0.1 * 10_000.0 * 1.0 * 1.0
        expected_t1 = expected_t0 + 0.1 * 10_000.0
        expected_t2 = expected_t1 + 0.1 * 10_000.0
        assert state.soc_wh[0, 0] == pytest.approx(expected_t0, rel=1e-6)
        assert state.soc_wh[0, 1] == pytest.approx(expected_t1, rel=1e-6)
        assert state.soc_wh[0, 2] == pytest.approx(expected_t2, rel=1e-6)

    # ---- Population axis independence ----

    def test_population_axis_independent(self):
        device = make_device(make_battery_param(capacity_wh=10_000.0), horizon=1)
        state = device.create_batch_state(2, 1)
        # Individual 0: gene=1.5 → CHARGE, factor=0.5
        # Individual 1: gene=2.3 → DISCHARGE, factor=0.3
        genome = np.array([[1.5], [2.3]])
        device.apply_genome_batch(state, genome)
        assert state.modes[0, 0] == InverterMode.CHARGE
        assert state.modes[1, 0] == InverterMode.DISCHARGE
        assert state.ac_power_w[0, 0] > 0.0   # CHARGE consumes
        assert state.ac_power_w[1, 0] < 0.0   # DISCHARGE injects


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
        # Gene 1.2: position=1 (CHARGE), factor=0.2 — predictable ac_power_w
        genome = np.full((pop, horizon), 1.2)
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
        # Gene 0.0: position=0 (OFF), factor=0.0
        genome = np.zeros((1, HORIZON))
        device.apply_genome_batch(state, genome)
        req = device.build_device_request(state)
        np.testing.assert_array_equal(req.port_requests[0].min_energy_wh, 0.0)

    def test_charge_steps_have_positive_min_energy(self):
        param = make_battery_param(min_charge=0.1)
        device = HybridInverterDevice(param, 0, 0)
        device.setup_run(make_context(horizon=1))
        state = device.create_batch_state(1, 1)
        # Gene 1.5: CHARGE with factor=0.5
        genome = np.array([[1.5]])
        device.apply_genome_batch(state, genome)
        req = device.build_device_request(state)
        assert req.port_requests[0].min_energy_wh[0, 0] > 0.0

    def test_discharge_steps_have_negative_min_energy(self):
        param = make_battery_param(min_discharge=0.1)
        device = HybridInverterDevice(param, 0, 0)
        device.setup_run(make_context(horizon=1))
        state = device.create_batch_state(1, 1)
        # Gene 2.5: DISCHARGE with factor=0.5
        genome = np.array([[2.5]])
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
        granted_wh = np.array([[500.0]])
        grant = self._make_grant(granted_wh)
        device.apply_device_grant(state, grant)
        assert state.ac_power_w[0, 0] == pytest.approx(500.0 / 1.0, rel=1e-9)

    def test_other_state_arrays_unchanged(self):
        device = make_device(make_battery_param())
        state = device.create_batch_state(POP, HORIZON)
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
        the requested modes and factors (for a single individual).

        Genes are constructed as ``mode_position + factor`` so that
        _decode_gene produces the desired values before repair.
        """
        horizon = len(modes_per_step)
        device = make_device(param, horizon=horizon)
        state = device.create_batch_state(1, horizon)

        # Build position index for each requested mode.
        mode_positions = [
            param.valid_modes.index(InverterMode(m)) for m in modes_per_step
        ]
        genes = [float(pos) + f for pos, f in zip(mode_positions, factors_per_step)]
        genome = np.array([genes])   # shape (1, horizon)

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
        device, state = self._run(param, [InverterMode.OFF] * HORIZON, [0.0] * HORIZON)
        result = device.extract_instructions(state, 0)
        assert all(isinstance(inst, OMBCInstruction) for inst in result)

    def test_resource_id_matches_device_id(self):
        param = make_battery_param(device_id="inv_001")
        device, state = self._run(param, [InverterMode.OFF] * HORIZON, [0.0] * HORIZON)
        result = device.extract_instructions(state, 0)
        assert all(inst.resource_id == "inv_001" for inst in result)

    def test_execution_times_match_step_times(self):
        param = make_solar_param()
        device, state = self._run(param, [InverterMode.PV] * HORIZON, [0.0] * HORIZON)
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
        for inst, name in zip(result, expected_names):
            assert inst.operation_mode_id == name

    def test_operation_mode_factor_matches_state(self):
        param = make_battery_param(min_charge=0.05, max_charge=1.0)
        device, state = self._run(
            param, [InverterMode.CHARGE] * HORIZON, [0.3] * HORIZON,
        )
        result = device.extract_instructions(state, 0)
        for i, inst in enumerate(result):
            assert inst.operation_mode_factor == pytest.approx(state.factors[0, i], rel=1e-9)

    def test_individual_index_selects_correct_row(self):
        param = make_battery_param()
        horizon = 2
        device = make_device(param, horizon=horizon)
        state = device.create_batch_state(2, horizon)
        # Individual 0: OFF (gene=0.0); Individual 1: CHARGE (gene=1.3)
        genome = np.array([
            [0.0, 0.0],    # individual 0: OFF both steps
            [1.3, 1.3],    # individual 1: CHARGE both steps, factor=0.3
        ])
        device.apply_genome_batch(state, genome)

        instructions_0 = device.extract_instructions(state, 0)
        instructions_1 = device.extract_instructions(state, 1)

        assert all(inst.operation_mode_id == "OFF" for inst in instructions_0)
        assert all(inst.operation_mode_id == "CHARGE" for inst in instructions_1)
