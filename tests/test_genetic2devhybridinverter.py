"""Tests for the continuous single-gene hybrid inverter device.

Module under test
-----------------
``akkudoktoreos.devices.genetic2.hybridinverter``

Design under test
-----------------
One gene per time step:

* ``bat_factor ∈ [−1, +1]``:
  - ``> 0`` charge at ``bat_factor × max_charge_rate × capacity`` W (bat side)
  - ``< 0`` discharge at ``|bat_factor| × max_discharge_rate × capacity`` W (bat side)
  - ``= 0`` battery idle

PV utilisation is no longer a genome gene. It is hardcoded to ``1.0`` (fully
utilise all available PV) because curtailment is never optimal in a home
energy system — available PV always offsets import cost or earns export
revenue. Making it a gene would only add noise and halve the effective
search resolution on the battery.

Param construction
------------------
``HybridInverterParam`` inherits ``device_id`` and ``ports`` from
``DeviceParam``.  There are no separate ``port_id`` / ``bus_id`` kwargs
on the param itself — the port identity lives inside ``EnergyPort``
objects passed as ``ports=(EnergyPort(...),)``.

Internal API changes
--------------------
The three separate helpers ``_repair_genes``, ``_compute_ac_power``, and
``_advance_soc`` have been replaced by the single combined method
``_compute_ac_power_and_soc_and_repair``.  Tests that previously tested
those helpers in isolation now test the combined method directly.

Test strategy
-------------
The combined helper ``_compute_ac_power_and_soc_and_repair`` is tested in
isolation across all relevant physics cases (repair, AC power, SoC advance).
Integration tests on ``apply_genome_batch`` confirm composition and
Lamarckian write-back.

Tolerances: rtol=1e-9 throughout; zero checks use approx(0.0, abs=1e-9).
"""

from __future__ import annotations

from typing import cast

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
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
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
    """Minimal SimulationContext stand-in for unit tests.

    Not a real SimulationContext (which is a frozen dataclass with no
    resolve_* methods in the base).  Call sites that pass this to
    setup_run() use cast(SimulationContext, ctx) so mypy is satisfied
    while the duck-typed runtime behaviour is preserved.
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


def make_hybrid_param(
    device_id: str = "hybrid",
    capacity_wh: float = 10000.0,
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
    device.setup_run(cast(SimulationContext, context))
    return device


def make_genome(
    pop: int,
    horizon: int,
    bat_factors: list[float],
) -> np.ndarray:
    """Build genome array from per-step bat_factor list.

    One gene per step — pv_util is no longer part of the genome.
    """
    genes = [float(b) for b in bat_factors]
    return np.tile(genes, (pop, 1))


# ---------------------------------------------------------------------------
# Helper: call _compute_ac_power_and_soc_and_repair for one step
# ---------------------------------------------------------------------------

def _call_combined(
    device: HybridInverterDevice,
    bat_list: list[float],
    soc_list: list[float],
    pv_dc_avail: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invoke ``_compute_ac_power_and_soc_and_repair`` for one time step.

    Returns ``(ac_w, new_soc_wh, bat_repaired)``.
    """
    raw_bat = np.array(bat_list, dtype=np.float64)
    soc_wh  = np.array(soc_list, dtype=np.float64)
    return device._compute_ac_power_and_soc_and_repair(raw_bat, soc_wh, pv_dc_avail)


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

    def test_objective_names_battery_hybrid(self):
        # BATTERY and HYBRID inverters contribute SoC value accounting via
        # the energy_cost_amt objective column.
        assert make_device(make_battery_param()).objective_names == ["energy_cost_amt"]

    def test_objective_names_solar_empty(self):
        # SOLAR inverters have no battery and therefore no SoC objective.
        assert make_device(make_solar_param()).objective_names == []

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
        dev.setup_run(cast(SimulationContext, ctx))
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
            HybridInverterDevice(param, 0, 0).setup_run(cast(SimulationContext, make_context()))

    def test_pv_wrong_shape_raises(self):
        dev = HybridInverterDevice(make_solar_param(), 0, 0)
        ctx = FakeContext(make_step_times(4), pv_w=np.full(2, 3000.0))
        with pytest.raises(ValueError):
            dev.setup_run(cast(SimulationContext, ctx))

    def test_initial_soc_out_of_bounds_raises(self):
        dev = HybridInverterDevice(make_battery_param(min_soc=0.2, max_soc=0.8), 0, 0)
        with pytest.raises(ValueError, match="initial SoC factor"):
            dev.setup_run(cast(SimulationContext, make_context(initial_soc_factor=-0.1)))
        with pytest.raises(ValueError, match="initial SoC factor"):
            dev.setup_run(cast(SimulationContext, make_context(initial_soc_factor=1.1)))

    def test_battery_resolves_initial_soc_wh(self):
        dev = HybridInverterDevice(
            make_battery_param(capacity_wh=10_000.0, min_soc=0.1, max_soc=0.9), 0, 0
        )
        dev.setup_run(cast(SimulationContext, make_context(initial_soc_factor=0.6)))
        assert dev._battery_initial_soc_wh == pytest.approx(6_000.0)

    def test_solar_resolves_pv_array(self):
        dev = HybridInverterDevice(make_solar_param(), 0, 0)
        dev.setup_run(cast(SimulationContext, make_context(pv_w=np.full(HORIZON, 5_000.0))))
        assert dev._pv_power_w is not None
        np.testing.assert_allclose(dev._pv_power_w, 5_000.0)


# ============================================================
# TestGenomeRequirements
# ============================================================

class TestGenomeRequirements:
    """Genome has 1 gene per step (bat_factor only) for all inverter types.

    pv_util is no longer a genome gene — it is hardcoded to 1.0 (fully
    utilise all available PV) because curtailment is never optimal.
    Genome layout: ``[bat_factor₀, bat_factor₁, …, bat_factor_{n-1}]``
    """

    def test_battery_genome_size(self):
        # One gene per step, so horizon=4 → size=4.
        assert make_device(make_battery_param(), horizon=4).genome_requirements().size == 4

    def test_solar_genome_size(self):
        # SOLAR still allocates one bat_factor gene per step (always repaired
        # to 0.0), keeping the genome layout uniform across inverter types.
        assert make_device(make_solar_param(), horizon=4).genome_requirements().size == 4

    def test_hybrid_genome_size(self):
        assert make_device(make_hybrid_param(), horizon=4).genome_requirements().size == 4

    def test_bat_factor_lower_bound_is_minus_one(self):
        slc = make_device(make_battery_param(), horizon=4).genome_requirements()
        assert slc.lower_bound is not None
        np.testing.assert_array_equal(slc.lower_bound, np.full(4, -1.0))

    def test_bat_factor_upper_bound_is_plus_one(self):
        slc = make_device(make_battery_param(), horizon=4).genome_requirements()
        assert slc.upper_bound is not None
        np.testing.assert_array_equal(slc.upper_bound, np.full(4, +1.0))

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
# TestComputeAcPowerAndSocAndRepair
# ============================================================

class TestComputeAcPowerAndSocAndRepair:
    """Tests for ``_compute_ac_power_and_soc_and_repair`` — the single combined
    method that replaced the three separate helpers ``_repair_genes``,
    ``_compute_ac_power``, and ``_advance_soc``.

    The method signature is::

        ac_w, new_soc_wh, bat_repaired = device._compute_ac_power_and_soc_and_repair(
            raw_bat,        # (pop,) float64
            soc_wh,         # (pop,) float64
            pv_dc_avail_w,  # float scalar
        )

    ``pv_util`` is always 1.0 internally — all available PV is always used.
    """

    # ------------------------------------------------------------------
    # SOLAR: no battery at all
    # ------------------------------------------------------------------

    def test_solar_bat_repaired_to_zero(self):
        """SOLAR type: bat_factor gene must always be repaired to 0.0."""
        device = make_device(make_solar_param())
        _, _, bat = _call_combined(device, [0.8], [0.0], pv_dc_avail=3000.0)
        assert bat[0] == pytest.approx(0.0)

    def test_solar_ac_power_is_pv_injection(self):
        """With PV available and no battery, net AC is pure PV injection."""
        param = make_solar_param(on_w=10.0)
        device = make_device(param)
        pv_dc = 4000.0
        expected_ac = -(pv_dc * 0.97) + 10.0
        ac_w, _, _ = _call_combined(device, [0.0], [0.0], pv_dc_avail=pv_dc)
        assert ac_w[0] == pytest.approx(expected_ac, rel=1e-9)

    def test_solar_no_pv_returns_off_state(self):
        """With zero PV, SOLAR device draws off-state power only."""
        param = make_solar_param(off_w=5.0)
        device = make_device(param, pv_w=0.0)
        ac_w, _, _ = _call_combined(device, [0.0], [0.0], pv_dc_avail=0.0)
        assert ac_w[0] == pytest.approx(5.0)

    def test_solar_soc_unchanged(self):
        """SOLAR type: SoC stays at 0.0 (no battery)."""
        device = make_device(make_solar_param())
        _, new_soc, _ = _call_combined(device, [0.0], [0.0], pv_dc_avail=4000.0)
        assert new_soc[0] == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # BATTERY: idle (bat_factor = 0)
    # ------------------------------------------------------------------

    def test_battery_idle_no_pv_returns_off_state(self):
        param = make_battery_param(off_w=8.0)
        device = make_device(param)
        ac_w, _, _ = _call_combined(device, [0.0], [5000.0], pv_dc_avail=0.0)
        assert ac_w[0] == pytest.approx(8.0)

    def test_battery_idle_soc_unchanged(self):
        device = make_device(make_battery_param())
        _, new_soc, bat = _call_combined(device, [0.0], [5000.0])
        assert new_soc[0] == pytest.approx(5000.0)
        assert bat[0] == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # BATTERY: charging (bat_factor > 0)
    # ------------------------------------------------------------------

    def test_battery_charge_consumes_correct_ac(self):
        """Charging AC draw = (bat_factor × max_charge × capacity / ac_eff) + on_w.

        Use min_soc=0.0, max_soc=1.0 to disable SoC-headroom repair so the
        gene reaches the physics unchanged.
        """
        param = make_battery_param(
            capacity_wh=10_000.0, max_charge=1.0, ac_to_bat_eff=0.95, on_w=10.0,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param)
        ac_w, _, _ = _call_combined(device, [0.5], [5000.0], pv_dc_avail=0.0)
        expected = (0.5 * 1.0 * 10_000.0 / 0.95) + 10.0
        assert ac_w[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_charge_positive_ac_power(self):
        ac_w, _, _ = _call_combined(make_device(make_battery_param()), [0.3], [5000.0])
        assert ac_w[0] > 0.0

    def test_battery_charge_increases_soc(self):
        """Stored energy (bat side) = bat_factor × max_charge × capacity × step_h.

        Note: the code stores ``charge_bat_w * step_h`` (battery-side watts, not
        AC-side), so ``ac_to_bat_eff`` does NOT appear in the SoC advance formula.
        AC efficiency only affects how much grid power is drawn, not how much
        energy lands in the battery given a requested battery-side charge rate.
        """
        param = make_battery_param(
            capacity_wh=10_000.0, max_charge=1.0, ac_to_bat_eff=0.95,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, initial_soc_factor=0.2)
        _, new_soc, _ = _call_combined(device, [0.5], [2000.0], pv_dc_avail=0.0)
        # stored_bat_wh = charge_bat_w * step_h = (0.5 * 1.0 * 10000) * 1.0
        stored = 0.5 * 1.0 * 10_000.0 * (STEP_INTERVAL / 3600.0)
        assert new_soc[0] == pytest.approx(2000.0 + stored, rel=1e-9)

    def test_battery_charge_gene_maps_through_max_charge_rate(self):
        """bat_factor=1.0 maps to max_charge_rate."""
        param = make_battery_param(max_charge=0.8, min_charge=0.0, min_soc=0.0, max_soc=1.0)
        device = make_device(param)
        _, _, bat = _call_combined(device, [1.0], [500.0])
        assert bat[0] == pytest.approx(1.0)

    def test_battery_charge_below_min_deadband_zeroed(self):
        """Charge gene below min_charge_rate is driven to zero."""
        param = make_battery_param(min_charge=0.05, max_charge=1.0, min_soc=0.0, max_soc=1.0)
        device = make_device(param)
        _, _, bat = _call_combined(device, [0.02], [500.0])
        assert bat[0] == pytest.approx(0.0)

    def test_battery_charge_discrete_snap_to_nearest(self):
        """With discrete rates, gene snaps to the nearest allowed rate."""
        param = make_battery_param(charge_rates=(0.25, 0.5, 0.75))
        device = make_device(param)
        # 0.6 is closer to 0.5 than to 0.75, but snap is by abs_rate proximity.
        # abs_rate = 0.6 × max_charge_rate(=1.0) = 0.6; nearest rate = 0.75? No:
        # rates are [0.25, 0.5, 0.75] — |0.6-0.5|=0.1, |0.6-0.75|=0.15 → snaps to 0.5.
        _, _, bat = _call_combined(device, [0.6], [1000.0])
        # The repaired gene = nearest_rate / max_charge_rate = 0.5 / 1.0 = 0.5
        assert bat[0] == pytest.approx(0.5)

    def test_battery_charge_soc_headroom_reduces_gene(self):
        """Charge gene is capped when SoC is close to max."""
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=0.9, ac_to_bat_eff=0.95,
            min_charge=0.05, max_charge=1.0,
        )
        device = make_device(param)
        # SoC=8800 → headroom=200 Wh → max rate ≈ 0.02 < min_charge → zeroed
        _, _, bat = _call_combined(device, [1.0], [8800.0])
        assert bat[0] == pytest.approx(0.0)

    def test_battery_charge_soc_at_max_zeroed(self):
        """Charging at max SoC must be repaired to zero."""
        param = make_battery_param(capacity_wh=10_000.0, max_soc=0.9, min_charge=0.05)
        device = make_device(param)
        _, _, bat = _call_combined(device, [1.0], [9000.0])
        assert bat[0] == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # BATTERY: discharging (bat_factor < 0)
    # ------------------------------------------------------------------

    def test_battery_discharge_injects_correct_ac(self):
        """Discharge AC injection = -(bat_factor × max_dis × capacity × bat_eff) + on_w.

        Use min_soc=0.0, max_soc=1.0 so SoC-floor repair does not cap the gene
        and the full gene value reaches the physics unchanged.
        """
        param = make_battery_param(
            capacity_wh=10_000.0, max_discharge=1.0, bat_to_ac_eff=0.95, on_w=10.0,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param)
        ac_w, _, _ = _call_combined(device, [-0.5], [5000.0], pv_dc_avail=0.0)
        expected = -(0.5 * 1.0 * 10_000.0 * 0.95) + 10.0
        assert ac_w[0] == pytest.approx(expected, rel=1e-9)

    def test_battery_discharge_negative_ac_power(self):
        ac_w, _, _ = _call_combined(
            make_device(make_battery_param(on_w=1.0)), [-1.0], [9000.0]
        )
        assert ac_w[0] < 0.0

    def test_battery_discharge_decreases_soc(self):
        """Drawn energy = abs_bat × max_dis × capacity × step_h."""
        param = make_battery_param(
            capacity_wh=10_000.0, max_discharge=1.0, min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        _, new_soc, _ = _call_combined(device, [-0.3], [8000.0], pv_dc_avail=0.0)
        drawn = 0.3 * 1.0 * 10_000.0 * (STEP_INTERVAL / 3600.0)
        assert new_soc[0] == pytest.approx(8000.0 - drawn, rel=1e-9)

    def test_battery_discharge_half_gene_gives_half_rate(self):
        """bat_factor=-0.5 uses half the max discharge rate."""
        param = make_battery_param(max_discharge=1.0, min_discharge=0.0,
                                   min_soc=0.0, max_soc=1.0)
        device = make_device(param)
        _, _, bat = _call_combined(device, [-0.5], [9000.0])
        assert bat[0] == pytest.approx(-0.5)

    def test_battery_discharge_below_min_deadband_zeroed(self):
        """Discharge gene below min_discharge_rate is driven to zero."""
        param = make_battery_param(min_discharge=0.05, max_discharge=1.0,
                                   min_soc=0.0, max_soc=1.0)
        device = make_device(param)
        _, _, bat = _call_combined(device, [-0.02], [9000.0])
        assert bat[0] == pytest.approx(0.0)

    def test_battery_discharge_soc_at_min_zeroed(self):
        """Discharging at min SoC must be repaired to zero."""
        param = make_battery_param(capacity_wh=10_000.0, min_soc=0.1, min_discharge=0.05)
        device = make_device(param)
        _, _, bat = _call_combined(device, [-1.0], [1000.0])
        assert bat[0] == pytest.approx(0.0)

    def test_battery_discharge_soc_floor_reduces_gene(self):
        """Discharge gene is capped when SoC is close to min."""
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.1, min_discharge=0.0, max_discharge=1.0,
        )
        device = make_device(param)
        # SoC=2000, min=1000 → available=1000 Wh → max rate = 1000/10000 = 0.1
        _, _, bat = _call_combined(device, [-1.0], [2000.0])
        assert bat[0] == pytest.approx(-0.1, rel=1e-5)

    def test_battery_soc_clamped_at_max(self):
        """SoC cannot exceed battery_capacity_wh after charging (physical clamp).

        The code clamps new_soc_wh to [0, capacity_wh] unconditionally.
        Use min_soc=0.0, max_soc=1.0, ac_eff=1.0 so repair does not interfere
        and the raw physics drives SoC beyond capacity before clamping.
        """
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=1.0, min_soc=0.0,
            ac_to_bat_eff=1.0, max_charge=1.0, min_charge=0.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        # SoC=9900 Wh, bat_factor=1.0 → would store 10000 Wh → clamped to 10000
        _, new_soc, _ = _call_combined(device, [1.0], [9900.0])
        assert float(new_soc[0]) <= 10_000.0

    def test_battery_soc_clamped_at_min(self):
        """SoC cannot drop below 0 after discharging (physical clamp).

        Repair limits discharge to available SoC, so the clamp is a backstop.
        Use min_soc=0.0 so the repair allows drawing down to 0, then verify.
        """
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.0, max_soc=1.0,
            max_discharge=1.0, min_discharge=0.0,
        )
        device = make_device(param, initial_soc_factor=0.5)
        # Repair caps discharge by available SoC; physical clamp keeps SoC >= 0.
        _, new_soc, _ = _call_combined(device, [-1.0], [100.0])
        assert float(new_soc[0]) >= 0.0

    # ------------------------------------------------------------------
    # HYBRID: charging with PV assist
    # ------------------------------------------------------------------

    def test_hybrid_charge_pv_reduces_ac_draw(self):
        """PV covers part of the battery charge demand, reducing AC import.

        Use min_soc=0.0, max_soc=1.0, min_charge=0.0 to disable SoC-headroom
        repair so the full gene value reaches the physics unchanged.
        """
        param = make_hybrid_param(
            capacity_wh=10000.0, max_charge=1.0, min_charge=0.0,
            min_soc=0.0, max_soc=1.0,
            ac_to_bat_eff=0.95, pv_to_bat_eff=0.95, pv_to_ac_eff=0.97, on_w=10.0,
        )
        device = make_device(param, pv_w=6000.0)
        pv_dc = 6000.0  # pv_util hardcoded to 1.0 internally
        charge_bat = 0.5 * 1.0 * 10_000.0
        pv_for_bat = min(pv_dc * 0.95, charge_bat)
        pv_surplus_dc = pv_dc - pv_for_bat / 0.95
        pv_surplus_ac = pv_surplus_dc * 0.97
        ac_for_bat = max(0.0, charge_bat - pv_for_bat) / 0.95
        expected = ac_for_bat - pv_surplus_ac + 10.0
        ac_w, _, _ = _call_combined(device, [0.5], [5000.0], pv_dc_avail=pv_dc)
        assert ac_w[0] == pytest.approx(expected, rel=1e-9)

    def test_hybrid_charge_pv_surplus_makes_net_negative(self):
        """Large PV with small charge demand → net AC injection (negative)."""
        param = make_hybrid_param(capacity_wh=500.0, max_charge=1.0, on_w=0.0)
        device = make_device(param, pv_w=10_000.0)
        ac_w, _, _ = _call_combined(device, [0.1], [200.0], pv_dc_avail=10_000.0)
        assert ac_w[0] < 0.0

    def test_hybrid_zero_pv_same_as_battery_charge(self):
        """Hybrid with zero PV behaves identically to pure battery for charging."""
        bat_param  = make_battery_param(capacity_wh=10_000.0, max_charge=1.0,
                                        ac_to_bat_eff=0.95, on_w=10.0)
        hybr_param = make_hybrid_param(capacity_wh=10_000.0, max_charge=1.0,
                                       ac_to_bat_eff=0.95, on_w=10.0)
        bat_device  = make_device(bat_param)
        hybr_device = make_device(hybr_param, pv_w=0.0)
        bat_ac, bat_soc, _ = _call_combined(bat_device,  [0.4], [5000.0], pv_dc_avail=0.0)
        hyb_ac, hyb_soc, _ = _call_combined(hybr_device, [0.4], [5000.0], pv_dc_avail=0.0)
        assert bat_ac[0]  == pytest.approx(hyb_ac[0],  rel=1e-9)
        assert bat_soc[0] == pytest.approx(hyb_soc[0], rel=1e-9)

    # ------------------------------------------------------------------
    # HYBRID: discharging with PV
    # ------------------------------------------------------------------

    def test_hybrid_discharge_battery_and_pv_combine(self):
        """Discharge and PV injection are additive on the AC bus.

        Use min_soc=0.0, min_discharge=0.0 so the SoC-floor repair does not
        cap the gene and the full gene value reaches the physics unchanged.
        """
        param = make_hybrid_param(
            capacity_wh=10_000.0, max_discharge=1.0, min_discharge=0.0,
            min_soc=0.0, max_soc=1.0,
            bat_to_ac_eff=0.95, pv_to_ac_eff=0.97, on_w=5.0,
        )
        device = make_device(param, pv_w=2000.0)
        pv_dc = 2000.0
        dis_bat_ac = 0.5 * 1.0 * 10_000.0 * 0.95
        pv_ac      = pv_dc * 0.97  # pv_util = 1.0
        expected   = -(dis_bat_ac + pv_ac) + 5.0
        ac_w, _, _ = _call_combined(device, [-0.5], [5000.0], pv_dc_avail=pv_dc)
        assert ac_w[0] == pytest.approx(expected, rel=1e-9)

    # ------------------------------------------------------------------
    # HYBRID: idle battery, PV only
    # ------------------------------------------------------------------

    def test_hybrid_idle_battery_pv_injects(self):
        """Idle battery + PV: net AC is pure PV injection."""
        param = make_hybrid_param(on_w=5.0)
        device = make_device(param, pv_w=3000.0)
        expected = -(3000.0 * 0.97) + 5.0
        ac_w, _, _ = _call_combined(device, [0.0], [5000.0], pv_dc_avail=3000.0)
        assert ac_w[0] == pytest.approx(expected, rel=1e-9)

    def test_hybrid_idle_battery_zero_pv_returns_off_state(self):
        """Idle battery + zero PV → off-state power draw only."""
        param = make_hybrid_param(off_w=7.0)
        device = make_device(param, pv_w=0.0)
        ac_w, _, _ = _call_combined(device, [0.0], [5000.0], pv_dc_avail=0.0)
        assert ac_w[0] == pytest.approx(7.0)

    # ------------------------------------------------------------------
    # HYBRID: SoC advance with PV assist
    # ------------------------------------------------------------------

    def test_hybrid_charge_pv_contributes_first(self):
        """PV covers battery charge demand; remaining PV goes to AC.

        Use min_soc=0.0, max_soc=1.0, min_charge=0.0 so no repair clips the
        gene before the physics runs.  SoC formula (code):
            stored = (pv_for_bat_w + ac_for_bat_w × ac_eff) × step_h
        where ac_for_bat_w = max(0, charge_bat - pv_for_bat) / ac_eff,
        so stored = (pv_for_bat + max(0, charge_bat - pv_for_bat)) × step_h
                  = charge_bat × step_h  when pv_for_bat ≤ charge_bat.
        """
        param = make_hybrid_param(
            capacity_wh=10_000.0, max_charge=1.0, min_charge=0.0,
            min_soc=0.0, max_soc=1.0,
            ac_to_bat_eff=0.95, pv_to_bat_eff=0.95,
        )
        device = make_device(param, pv_w=4000.0, initial_soc_factor=0.5)
        pv_dc_used = 4000.0  # pv_util = 1.0
        charge_bat = 0.5 * 1.0 * 10_000.0
        pv_for_bat = min(pv_dc_used * 0.95, charge_bat)
        ac_for_bat_w = max(0.0, charge_bat - pv_for_bat) / 0.95
        stored_wh  = (pv_for_bat + ac_for_bat_w * 0.95) * (STEP_INTERVAL / 3600.0)
        expected_soc = 3000.0 + stored_wh
        _, new_soc, _ = _call_combined(device, [0.5], [3000.0], pv_dc_avail=pv_dc_used)
        assert new_soc[0] == pytest.approx(expected_soc, rel=1e-9)

    # ------------------------------------------------------------------
    # Population independence
    # ------------------------------------------------------------------

    def test_mixed_population_independent(self):
        """Different individuals in a population must be processed independently."""
        param = make_battery_param(capacity_wh=10_000.0, off_w=5.0, on_w=10.0)
        device = make_device(param)
        raw_bat = np.array([0.0, 0.5, -0.5])
        soc_wh  = np.array([5000.0, 5000.0, 5000.0])
        ac_w, _, _ = device._compute_ac_power_and_soc_and_repair(raw_bat, soc_wh, 0.0)
        assert ac_w[0] == pytest.approx(5.0)   # idle → off-state
        assert ac_w[1] > 0.0                    # charging
        assert ac_w[2] < 0.0                    # discharging


# ============================================================
# TestApplyGenomeBatch
# ============================================================

class TestApplyGenomeBatch:
    """Integration tests for the full apply_genome_batch pipeline.

    Genome layout: one bat_factor gene per step.
    ``[bat_factor₀, bat_factor₁, …, bat_factor_{n-1}]``

    pv_util is always hardcoded to 1.0 inside apply_genome_batch and is
    never read from the genome, so pv_util_factors in state will always
    reflect the available PV ratio for HYBRID/SOLAR (1.0 when PV > 0)
    or 0.0 for BATTERY (no PV).
    """

    def test_genome_shape_is_pop_by_horizon(self):
        # One gene per step → genome shape is (pop, horizon), not (pop, 2*horizon).
        device = make_device(make_battery_param(), horizon=4)
        state  = device.create_batch_state(POP, 4)
        genome = np.zeros((POP, 4))
        device.apply_genome_batch(state, genome)
        assert genome.shape == (POP, 4)

    def test_bat_factors_stored_in_state(self):
        param = make_battery_param(min_soc=0.0, max_soc=1.0)
        device = make_device(param, initial_soc_factor=0.1)
        state  = device.create_batch_state(1, 2)
        # One gene per step: [bat_factor_step0, bat_factor_step1]
        genome = np.array([[0.5, -0.4]])
        device.apply_genome_batch(state, genome)
        assert state.bat_factors[0, 0] == pytest.approx(0.5, rel=1e-6)

    def test_pv_util_factors_reflect_available_pv_fraction(self):
        # pv_util is hardcoded to 1.0 internally (all available PV is used),
        # but pv_util_factors stores pv_dc_avail_w / pv_max_power_w — the
        # fraction of rated PV capacity that was available, not a controlled gene.
        # With pv_w=4000 and pv_max_w=8000 (make_hybrid_param default):
        #   pv_util_factors = 4000 / 8000 = 0.5
        param = make_hybrid_param()  # pv_max_w=8000 by default
        device = make_device(param, pv_w=4000.0, horizon=2)
        state  = device.create_batch_state(1, 2)
        genome = np.array([[0.0, 0.0]])
        device.apply_genome_batch(state, genome)
        expected_ratio = 4000.0 / 8000.0  # 0.5
        np.testing.assert_allclose(state.pv_util_factors[0, :], expected_ratio)

    def test_solar_bat_factors_zeroed(self):
        device = make_device(make_solar_param(), horizon=2)
        state  = device.create_batch_state(1, 2)
        # SOLAR: bat_factor genes are repaired to 0.0
        genome = np.array([[0.9, -0.7]])
        device.apply_genome_batch(state, genome)
        np.testing.assert_array_equal(state.bat_factors, 0.0)

    def test_battery_pv_util_zeroed(self):
        device = make_device(make_battery_param(), horizon=2)
        state  = device.create_batch_state(1, 2)
        genome = np.array([[0.5, -0.3]])
        device.apply_genome_batch(state, genome)
        # BATTERY: no PV, pv_util repaired to 0.0
        np.testing.assert_array_equal(state.pv_util_factors, 0.0)

    def test_lamarckian_writeback_bat_factor(self):
        param = make_battery_param(
            capacity_wh=10_000.0, min_soc=0.0, max_soc=1.0,
            min_charge=0.0, max_charge=1.0, ac_to_bat_eff=0.95,
        )
        device = make_device(param, horizon=1, initial_soc_factor=0.5)
        state  = device.create_batch_state(1, 1)
        genome = np.array([[0.3]])
        device.apply_genome_batch(state, genome)
        # The repaired bat_factor is written back to the genome (Lamarckian repair).
        assert genome[0, 0] == pytest.approx(0.3, rel=1e-6)

    def test_lamarckian_writeback_clamps_soc_capped_gene(self):
        # When a large charge gene is capped by SoC headroom, the write-back
        # gene should reflect the capped value, not the original raw gene.
        param = make_battery_param(
            capacity_wh=10_000.0, max_soc=0.9, min_charge=0.0, max_charge=1.0,
            ac_to_bat_eff=0.95,
        )
        device = make_device(param, horizon=1, initial_soc_factor=0.9)
        state  = device.create_batch_state(1, 1)
        # SoC already at max → gene should be repaired to 0.0 and written back.
        genome = np.array([[1.0]])
        device.apply_genome_batch(state, genome)
        assert genome[0, 0] == pytest.approx(0.0)

    def test_soc_evolves_over_steps(self):
        param = make_battery_param(
            capacity_wh=10_000.0, max_charge=1.0, ac_to_bat_eff=1.0,
            min_soc=0.0, max_soc=1.0,
        )
        device = make_device(param, horizon=3, initial_soc_factor=0.5)
        state  = device.create_batch_state(1, 3)
        # One gene per step: bat_factor=0.1 at each of the 3 steps.
        genome = np.array([[0.1, 0.1, 0.1]])
        device.apply_genome_batch(state, genome)
        assert state.soc_wh[0, 0] == pytest.approx(5000.0 + 1000.0, rel=1e-6)
        assert state.soc_wh[0, 1] == pytest.approx(5000.0 + 2000.0, rel=1e-6)
        assert state.soc_wh[0, 2] == pytest.approx(5000.0 + 3000.0, rel=1e-6)

    def test_population_axis_independent(self):
        device = make_device(make_battery_param(capacity_wh=10_000.0), horizon=1)
        state  = device.create_batch_state(2, 1)
        # Individual 0: charge (bat_factor=+0.5); individual 1: discharge (bat_factor=-0.5).
        genome = np.array([
            [0.5],
            [-0.5],
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
        device.setup_run(cast(SimulationContext, make_context(horizon=horizon)))
        state  = device.create_batch_state(pop, horizon)
        # One gene per step.
        genome = np.full((pop, horizon), 0.3)
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
        device.setup_run(cast(SimulationContext, make_context(horizon=HORIZON)))
        state  = device.create_batch_state(1, HORIZON)
        # All-zero genome → all steps idle.
        genome = np.zeros((1, HORIZON))
        device.apply_genome_batch(state, genome)
        np.testing.assert_array_equal(
            device.build_device_request(state).port_requests[0].min_energy_wh, 0.0,
        )

    def test_charge_steps_positive_min_energy(self):
        device = HybridInverterDevice(make_battery_param(min_charge=0.1), 0, 0)
        device.setup_run(cast(SimulationContext, make_context(horizon=1)))
        state  = device.create_batch_state(1, 1)
        genome = np.array([[0.5]])
        device.apply_genome_batch(state, genome)
        assert device.build_device_request(state).port_requests[0].min_energy_wh[0, 0] > 0.0

    def test_discharge_steps_negative_min_energy(self):
        device = HybridInverterDevice(make_battery_param(min_discharge=0.1), 0, 0)
        device.setup_run(cast(SimulationContext, make_context(horizon=1)))
        state  = device.create_batch_state(1, 1)
        genome = np.array([[-0.5]])
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
    def test_shape_pop_one_for_battery(self):
        # BATTERY/HYBRID: returns (pop, 1) — one energy_cost_amt objective
        # column for SoC value accounting.
        device = make_device(make_battery_param())
        state  = device.create_batch_state(POP, HORIZON)
        assert device.compute_cost(state).shape == (POP, 1)

    def test_shape_pop_zero_for_solar(self):
        # SOLAR: no battery, no SoC objective → returns (pop, 0).
        device = make_device(make_solar_param())
        state  = device.create_batch_state(POP, HORIZON)
        assert device.compute_cost(state).shape == (POP, 0)

    def test_all_zeros_when_soc_unchanged(self):
        # When no genome is applied (all idle, no charge/discharge), the
        # terminal SoC equals the initial SoC and soc_cost should be zero.
        device = make_device(make_battery_param())
        state  = device.create_batch_state(POP, HORIZON)
        np.testing.assert_array_equal(device.compute_cost(state), 0.0)

    def test_discharge_reward_shape_pop_two(self):
        """When battery_discharge_reward_amt_kwh > 0, cost matrix has 2 columns."""
        param = HybridInverterParam(
            device_id="bat",
            ports=(_AC_PORT,),
            inverter_type=InverterType.BATTERY,
            off_state_power_consumption_w=0.0,
            on_state_power_consumption_w=0.0,
            pv_to_ac_efficiency=0.97,
            pv_to_battery_efficiency=0.95,
            pv_max_power_w=10_000.0,
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
            battery_min_soc_factor=0.0,
            battery_max_soc_factor=1.0,
            battery_initial_soc_factor_key=SOC_KEY,
            battery_discharge_reward_amt_kwh=0.17,
        )
        device = make_device(param)
        state  = device.create_batch_state(POP, HORIZON)
        assert device.compute_cost(state).shape == (POP, 2)


# ============================================================
# TestExtractInstructions
# ============================================================

class TestExtractInstructions:
    """extract_instructions emits exactly ONE OMBCInstruction per time step.

    In the new single-gene design pv_util is hardcoded to 1.0 and is not
    a controllable gene, so no separate PV_UTILISE instruction is emitted.

    Instruction layout per step (index = step number):
    - One battery-mode instruction: CHARGE / DISCHARGE / IDLE / SELF_CONSUMPTION
    """

    def _run(self, param, bat_per_step: list):
        """Run genome for one individual with the given per-step bat_factors.

        pv_util is no longer a genome gene, so only bat_factors are passed.
        """
        horizon = len(bat_per_step)
        device  = make_device(param, horizon=horizon)
        state   = device.create_batch_state(1, horizon)
        # One gene per step.
        genome = np.array([[float(b) for b in bat_per_step]])
        device.apply_genome_batch(state, genome)
        return device, state

    def test_one_instruction_per_step(self):
        """Exactly one OMBCInstruction is emitted per time step."""
        param = make_battery_param()
        device, state = self._run(param, [0.3] * HORIZON)
        assert len(device.extract_instructions(state, 0)) == HORIZON

    def test_all_are_ombc_instructions(self):
        param = make_battery_param()
        device, state = self._run(param, [0.3] * HORIZON)
        assert all(
            isinstance(i, OMBCInstruction)
            for i in device.extract_instructions(state, 0)
        )

    def test_battery_instruction_mode_id_charge(self):
        param = make_battery_param()
        device, state = self._run(param, [0.5])
        assert device.extract_instructions(state, 0)[0].operation_mode_id == "CHARGE"

    def test_battery_instruction_mode_id_discharge(self):
        param = make_battery_param()
        device, state = self._run(param, [-0.5])
        assert device.extract_instructions(state, 0)[0].operation_mode_id == "DISCHARGE"

    def test_battery_instruction_mode_id_idle(self):
        param = make_battery_param()
        device, state = self._run(param, [0.0])
        assert device.extract_instructions(state, 0)[0].operation_mode_id == "IDLE"

    def test_battery_instruction_factor_is_abs_bat_factor(self):
        """operation_mode_factor == abs(bat_factor) — direction is encoded in mode_id."""
        param = make_battery_param(min_charge=0.0, max_charge=1.0)
        device, state = self._run(param, [0.3])
        instrs = device.extract_instructions(state, 0)
        assert instrs[0].operation_mode_factor == pytest.approx(
            abs(state.bat_factors[0, 0]), rel=1e-9,
        )

    def test_no_pv_utilise_instruction_emitted(self):
        """PV_UTILISE is no longer emitted because pv_util is not a genome gene."""
        param = make_hybrid_param()
        device, state = self._run(param, [0.0])
        instrs = device.extract_instructions(state, 0)
        # Only one instruction per step, and it must not be PV_UTILISE.
        assert len(instrs) == 1
        assert instrs[0].operation_mode_id != "PV_UTILISE"

    def test_resource_id_matches(self):
        param = make_battery_param(device_id="inv_X")
        device, state = self._run(param, [0.3] * HORIZON)
        assert all(
            i.resource_id == "inv_X"
            for i in device.extract_instructions(state, 0)
        )

    def test_execution_times_match_step_times(self):
        """One instruction per step, execution_time must match step timestamp."""
        param = make_battery_param()
        device, state = self._run(param, [0.3] * HORIZON)
        instrs = device.extract_instructions(state, 0)
        expected_times = make_step_times(HORIZON)
        for step_idx, dt in enumerate(expected_times):
            assert instrs[step_idx].execution_time == dt

    def test_individual_index_selects_correct_row(self):
        param  = make_battery_param()
        device = make_device(param, horizon=1)
        state  = device.create_batch_state(2, 1)
        # Individual 0: charge; individual 1: discharge.
        genome = np.array([
            [0.4],
            [-0.4],
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
        device, state = self._run(param, [-0.5])
        instr = device.extract_instructions(state, 0, instruction_context=None)
        assert instr[0].operation_mode_id == "DISCHARGE"
        assert instr[0].operation_mode_factor > 0.0

    def test_self_consumption_emitted_when_grid_near_zero_and_battery_discharging(self):
        param = make_battery_param()
        device, state = self._run(param, [-0.5])
        ctx = self._make_context_with_grid([0.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "SELF_CONSUMPTION"
        # The implementation emits factor=1.0 in SELF_CONSUMPTION mode so
        # that the optimised bat_factor magnitude is visible in the solution
        # DataFrame even though the inverter firmware ignores the setpoint.
        assert instr[0].operation_mode_factor == pytest.approx(1.0)

    def test_self_consumption_emitted_when_grid_near_zero_and_battery_charging(self):
        param = make_battery_param()
        device, state = self._run(param, [0.5])
        ctx = self._make_context_with_grid([0.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "SELF_CONSUMPTION"
        # Same as discharging case: factor is always 1.0 in SELF_CONSUMPTION mode.
        assert instr[0].operation_mode_factor == pytest.approx(1.0)

    def test_self_consumption_within_threshold_wh(self):
        """40 Wh < 50 Wh threshold → SELF_CONSUMPTION."""
        param = make_battery_param()
        device, state = self._run(param, [-0.5])
        ctx = self._make_context_with_grid([40.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "SELF_CONSUMPTION"

    def test_explicit_discharge_above_threshold(self):
        """200 Wh > 50 Wh threshold → explicit DISCHARGE."""
        param = make_battery_param()
        device, state = self._run(param, [-0.5])
        ctx = self._make_context_with_grid([200.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "DISCHARGE"
        assert instr[0].operation_mode_factor > 0.0

    def test_idle_not_promoted_to_self_consumption(self):
        """bat_factor == 0 stays IDLE even when grid exchange is zero."""
        param = make_battery_param()
        device, state = self._run(param, [0.0])
        ctx = self._make_context_with_grid([0.0])
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "IDLE"

    def test_context_with_none_grid_falls_back_to_explicit(self):
        """InstructionContext present but grid_granted_wh=None → explicit modes."""
        param = make_battery_param()
        device, state = self._run(param, [-0.5])
        ctx = InstructionContext(grid_granted_wh=None, step_interval_sec=STEP_INTERVAL)
        instr = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instr[0].operation_mode_id == "DISCHARGE"

    def test_mixed_steps_per_horizon(self):
        """Step 0: grid≈0 → SELF_CONSUMPTION; step 1: large export → DISCHARGE.

        Use initial_soc_factor=0.8 so both steps have enough SoC to discharge
        without the floor repair zeroing the second step's bat_factor.
        """
        param   = make_battery_param()
        device  = make_device(param, horizon=2, initial_soc_factor=0.8)
        state   = device.create_batch_state(1, 2)
        # Two discharge steps, one gene each.
        genome  = np.array([[-0.1, -0.1]])
        device.apply_genome_batch(state, genome)
        ctx = self._make_context_with_grid([0.0, 500.0])
        instrs = device.extract_instructions(state, 0, instruction_context=ctx)
        assert instrs[0].operation_mode_id == "SELF_CONSUMPTION"
        assert instrs[1].operation_mode_id == "DISCHARGE"


# ============================================================
# TestNoPvDischargeBehaviour
# ============================================================

class TestNoPvDischargeBehaviour:
    """Strong validation of discharge behaviour when NO PV is available.

    Focus:
    - Negative bat_factors must result in real discharge
    - Repair must respect SoC energy limits (not just pass-through)
    - Hybrid == Battery when pv_dc_avail_w == 0

    All tests use ``_compute_ac_power_and_soc_and_repair`` — the single
    combined method that supersedes the old separate helpers.
    """

    # ------------------------------------------------------------------
    # Repair via combined method
    # ------------------------------------------------------------------

    def test_negative_bat_factor_preserved_if_within_energy_limits(self):
        """A valid negative gene must remain unchanged if energy allows it."""
        device = make_device(make_battery_param(min_soc=0.0, max_soc=1.0))
        _, _, bat = _call_combined(device, [-0.5], [5000.0], pv_dc_avail=0.0)
        assert bat[0] == pytest.approx(-0.5)

    def test_discharge_limited_by_available_energy(self):
        """Discharge must be capped by (SoC - min SoC)."""
        param = make_battery_param(
            capacity_wh=10_000.0,
            min_soc=0.1,   # → 1000 Wh minimum
            min_discharge=0.0,
            max_discharge=1.0,
        )
        device = make_device(param)
        _, _, bat = _call_combined(device, [-1.0], [5000.0], pv_dc_avail=0.0)
        # max possible = (5000 - 1000) / 10000 = 0.4
        assert bat[0] == pytest.approx(-0.4)

    def test_discharge_zeroed_when_soc_at_min(self):
        """Discharge must be blocked at SoC floor."""
        param = make_battery_param(
            capacity_wh=10_000.0,
            min_soc=0.1,
            min_discharge=0.0,
        )
        device = make_device(param)
        _, _, bat = _call_combined(device, [-1.0], [1000.0], pv_dc_avail=0.0)
        assert bat[0] == pytest.approx(0.0)

    def test_discharge_respects_min_rate(self):
        """Gene below min_discharge_rate is zeroed by the deadband (not snapped up).

        The repair code applies: ``abs_rate[abs_rate < min_discharge_rate] = 0.0``
        so a gene that is too small to meet the minimum rate is treated as idle,
        not rounded up.
        """
        param = make_battery_param(
            min_discharge=0.5,
            max_discharge=1.0,
        )
        device = make_device(param)
        _, _, bat = _call_combined(device, [-0.3], [8000.0], pv_dc_avail=0.0)
        # abs_rate = 0.3 < min_discharge=0.5 → zeroed by deadband
        assert bat[0] == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # AC power via combined method
    # ------------------------------------------------------------------

    def test_negative_bat_factor_produces_negative_ac_power(self):
        """Discharge must inject power (negative AC)."""
        device = make_device(make_battery_param(on_w=0.0))
        ac_w, _, _ = _call_combined(device, [-0.5], [5000.0], pv_dc_avail=0.0)
        assert ac_w[0] < 0.0

    def test_stronger_negative_gene_means_more_power(self):
        """More negative gene → larger discharge."""
        device = make_device(make_battery_param(on_w=0.0))
        weak_ac, _, _ = _call_combined(device, [-0.2], [5000.0], pv_dc_avail=0.0)
        strong_ac, _, _ = _call_combined(device, [-0.8], [5000.0], pv_dc_avail=0.0)
        assert abs(strong_ac[0]) > abs(weak_ac[0])

    # ------------------------------------------------------------------
    # SoC advance via combined method
    # ------------------------------------------------------------------

    def test_negative_bat_factor_reduces_soc(self):
        """Core invariant: discharge must reduce SoC."""
        device = make_device(make_battery_param(), initial_soc_factor=0.6)
        _, new_soc, _ = _call_combined(device, [-0.5], [6000.0], pv_dc_avail=0.0)
        assert new_soc[0] < 6000.0

    def test_discharge_amount_matches_expected(self):
        """Energy removed must match bat_factor scaling."""
        param = make_battery_param(
            capacity_wh=10_000.0,
            max_discharge=1.0,
            min_soc=0.0,
            max_soc=1.0,
        )
        device = make_device(param, initial_soc_factor=0.6)
        _, new_soc, _ = _call_combined(device, [-0.5], [6000.0], pv_dc_avail=0.0)
        # drawn_wh = 0.5 * max_discharge(1.0) * capacity * step_h
        drawn = 0.5 * 1.0 * 10_000.0 * (STEP_INTERVAL / 3600.0)
        assert new_soc[0] == pytest.approx(6000.0 - drawn, rel=1e-9)

    # ------------------------------------------------------------------
    # Hybrid = Battery when no PV
    # ------------------------------------------------------------------

    def test_hybrid_behaves_like_battery_without_pv(self):
        """PV absence collapses hybrid → battery for SoC advance."""
        bat = make_device(make_battery_param(), initial_soc_factor=0.6)
        hyb = make_device(make_hybrid_param(), pv_w=0.0, initial_soc_factor=0.6)
        bat_ac, bat_soc, _ = _call_combined(bat, [-0.5], [6000.0], pv_dc_avail=0.0)
        hyb_ac, hyb_soc, _ = _call_combined(hyb, [-0.5], [6000.0], pv_dc_avail=0.0)
        np.testing.assert_allclose(bat_ac,  hyb_ac,  rtol=1e-9)
        np.testing.assert_allclose(bat_soc, hyb_soc, rtol=1e-9)

    # ------------------------------------------------------------------
    # Integration (apply_genome_batch)
    # ------------------------------------------------------------------

    def test_apply_genome_negative_gene_causes_discharge(self):
        """End-to-end: negative gene must cause discharge."""
        device = make_device(make_battery_param(), horizon=1, initial_soc_factor=0.6)
        state  = device.create_batch_state(1, 1)

        genome = np.array([[-0.5]])
        device.apply_genome_batch(state, genome)

        assert state.bat_factors[0, 0] < 0.0
        assert state.soc_wh[0, 0] < 6000.0
        assert state.ac_power_w[0, 0] < 0.0

    def test_apply_genome_stronger_negative_means_more_discharge(self):
        """Population comparison for scaling correctness."""
        device = make_device(make_battery_param(), horizon=1, initial_soc_factor=0.6)
        state  = device.create_batch_state(2, 1)

        genome = np.array([
            [-0.2],
            [-0.8],
        ])
        device.apply_genome_batch(state, genome)

        assert state.soc_wh[1, 0] < state.soc_wh[0, 0]
        assert abs(state.ac_power_w[1, 0]) > abs(state.ac_power_w[0, 0])
