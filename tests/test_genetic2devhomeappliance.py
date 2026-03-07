"""Tests for the vectorized home appliance device.

Module under test
-----------------
``akkudoktoreos.devices.genetic2.homeappliance``

Design under test
-----------------
One integer-valued gene per *remaining* cycle::

    genome shape: (population_size, num_remaining_cycles)
    genome[i, k] ∈ [0, horizon - 1]    start step for remaining cycle k

The repair pipeline:
    1. Round and clip to [0, max_start].
    2. Per-cycle window snap (nearest allowed step).
    3. Sort across cycles (time-ordering, removes genome symmetry).
    4. Gap enforcement (push forward to satisfy min_cycle_gap_h).
    5. Reconstruct flat-block power schedule.
    6. Lamarckian write-back.

Test strategy
-------------
``HomeApplianceParam`` and ``CycleTimeWindow`` validation are tested in
isolation. The repair helpers and lifecycle methods are tested with a
``FakeContext`` stand-in; no real simulation context is needed.

Tolerances: rtol=1e-9 throughout; zero checks use approx(0.0, abs=1e-9).

Covers
------
    CycleTimeWindow — construction and round-trip
        - from_time_window / to_time_window round-trip preserves data
        - Hashable and comparable as a value type
        - Two instances with same data are equal and share hash

    cycle_windows_from_sequence
        - None input returns None
        - Non-None sequence converted to tuple of CycleTimeWindow

    HomeApplianceParam — validation
        - consumption_wh <= 0 raises
        - duration_h < 1 raises
        - num_cycles < 1 raises
        - min_cycle_gap_h < 0 raises
        - No ports raises
        - cycle_time_windows length mismatch raises
        - Valid construction succeeds

    HomeApplianceParam — derived properties
        - effective_cycles_completed_key defaults to device-scoped name
        - effective_cycles_completed_key uses explicit key when set
        - resolved_cycle_time_windows returns (None,)*num_cycles when empty
        - resolved_cycle_time_windows returns stored windows when set

    HomeApplianceDevice — topology and identity
        - device_id matches param
        - ports matches param.ports
        - objective_names == ["energy_cost_eur"]
        - device_index and port_index stored correctly

    TestSetupRun
        - horizon, step_interval_h, step_times stored
        - power_per_step_w = consumption_wh / duration_h
        - completed=0 when measurement missing (KeyError)
        - completed=0 when measurement returns None
        - completed clamped to [0, num_cycles]
        - num_remaining_cycles = num_cycles - completed
        - allowed_steps shape = (num_remaining, horizon)
        - allowed_steps with window: steps outside window are False,
          steps inside window are True (UTC+1 offset accounted for)
        - Before setup_run raises RuntimeError

    TestGenomeRequirements
        - None returned when all cycles completed
        - size == num_remaining_cycles
        - lower_bound all zeros
        - upper_bound all == max_start
        - Before setup_run raises RuntimeError

    TestCreateBatchState
        - start_steps, schedule, granted_energy_wh shapes correct
        - All arrays initialised to zero
        - step_times forwarded from setup_run
        - Before setup_run raises RuntimeError

    TestRepairPipeline
        - Genes rounded to nearest integer step
        - Genes clipped to [0, max_start]
        - Steps outside allowed window snapped to nearest allowed step
        - Cycles sorted in time order
        - Gap enforcement pushes later cycle forward
        - Lamarckian write-back: genome_batch updated in-place

    TestScheduleReconstruction
        - schedule is zero outside run blocks
        - schedule equals power_per_step_w inside run blocks
        - Multi-cycle: both blocks present

    TestBuildDeviceRequest
        - Returns None when no remaining cycles (all done)
        - DeviceRequest carries correct device_index
        - PortRequest carries correct port_index
        - energy_wh == schedule * step_interval_h
        - min_energy_wh all zeros
        - is_slack False
        - Shapes (pop, horizon) correct
        - Before setup_run raises RuntimeError

    TestApplyDeviceGrant
        - granted_energy_wh updated from grant.port_grants[0].granted_wh
        - granted_energy_wh zeroed when port_grants is empty
        - schedule unchanged after grant

    TestComputeCost
        - Shape (pop, 1)
        - cost = sum_t(granted_wh[t] * price[t])
        - Zero grant → zero cost
        - price_forecast_key=None uses flat fallback

    TestExtractInstructions
        - One instruction per step
        - Steps inside a run block → RUN mode
        - Steps outside run blocks → OFF mode
        - execution_time matches step_times
        - resource_id matches param device_id
        - individual_index selects correct row
        - All cycles done (zero schedule) → all OFF
"""

from __future__ import annotations

import numpy as np
import pytest
from pendulum import Duration

from akkudoktoreos.core.emplan import OMBCInstruction
from akkudoktoreos.devices.devicesabc import ApplianceOperationMode, EnergyPort, PortDirection
from akkudoktoreos.devices.genetic2.homeappliance import (
    CycleTimeWindow,
    HomeApplianceDevice,
    HomeApplianceParam,
    cycle_windows_from_sequence,
)
from akkudoktoreos.simulation.genetic2.arbitrator import DeviceGrant, PortGrant
from akkudoktoreos.utils.datetimeutil import TimeWindow, TimeWindowSequence, Time, to_duration, to_datetime

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STEP_INTERVAL = 3600.0   # 1-hour steps [s]
HORIZON = 8
POP = 3
PRICE_KEY = "price_forecast"
CYCLES_KEY = "appliance.cycles_completed"

CONSUMPTION_WH = 2_000.0
DURATION_H = 2

# step_times are UTC epoch seconds; to_datetime(i * 3600) produces UTC datetimes.
# TimeWindow uses naive Time objects, which TimeWindow.contains() resolves into the
# reference datetime's timezone (Europe/Berlin = UTC+1 in winter).
# Therefore: UTC step t corresponds to local hour t+1.
# A naive Time(h, 0, 0) window is satisfied starting at UTC step h-1.
# All window-dependent tests account for this +1h offset.
UTC_TO_LOCAL_OFFSET_H = 1  # Berlin winter (UTC+1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_step_times(n: int = HORIZON) -> tuple:
    return tuple(to_datetime(i * STEP_INTERVAL) for i in range(n))


def make_port() -> tuple[EnergyPort, ...]:
    return (EnergyPort(port_id="p_ac", bus_id="bus_ac", direction=PortDirection.SINK),)


# ---------------------------------------------------------------------------
# FakeContext
# ---------------------------------------------------------------------------

class FakeContext:
    """Minimal SimulationContext stand-in for unit tests."""

    def __init__(
        self,
        step_times: tuple,
        step_interval_sec: float = STEP_INTERVAL,
        price_wh: np.ndarray | None = None,
        completed_cycles: int | None = 0,
    ) -> None:
        self.step_times = step_times
        self.step_interval = Duration(seconds=step_interval_sec)
        self.horizon = len(step_times)
        self._price_wh = price_wh if price_wh is not None else np.full(len(step_times), 1e-4)
        self._completed_cycles = completed_cycles

    def resolve_prediction(self, key: str) -> np.ndarray:
        return self._price_wh.copy()

    def resolve_measurement(self, key: str) -> float | None:
        if self._completed_cycles is None:
            return None
        return float(self._completed_cycles)


class FakeContextMissingKey(FakeContext):
    """FakeContext that raises KeyError on resolve_measurement."""

    def resolve_measurement(self, key: str) -> float:
        raise KeyError(key)


def make_context(
    horizon: int = HORIZON,
    step_interval_sec: float = STEP_INTERVAL,
    price_wh: float | np.ndarray = 1e-4,
    completed_cycles: int | None = 0,
) -> FakeContext:
    times = make_step_times(horizon)
    if np.ndim(price_wh) == 0:
        price_array = np.full(horizon, float(price_wh))
    else:
        price_array = np.asarray(price_wh, dtype=np.float64)
    return FakeContext(times, step_interval_sec, price_array, completed_cycles)


# ---------------------------------------------------------------------------
# Param factories
# ---------------------------------------------------------------------------

def make_param(
    device_id: str = "dishwasher",
    consumption_wh: float = CONSUMPTION_WH,
    duration_h: int = DURATION_H,
    num_cycles: int = 1,
    min_cycle_gap_h: int = 0,
    price_forecast_key: str | None = None,
    cycles_completed_measurement_key: str | None = None,
    cycle_time_windows: tuple = (),
) -> HomeApplianceParam:
    return HomeApplianceParam(
        device_id=device_id,
        ports=make_port(),
        consumption_wh=consumption_wh,
        duration_h=duration_h,
        num_cycles=num_cycles,
        min_cycle_gap_h=min_cycle_gap_h,
        price_forecast_key=price_forecast_key,
        cycles_completed_measurement_key=cycles_completed_measurement_key,
        cycle_time_windows=cycle_time_windows,
    )


def make_cycle_window(
    start_h: int = 8,
    duration_h: int = 4,
) -> CycleTimeWindow:
    """Return a CycleTimeWindow starting at start_h:00 for duration_h hours."""
    return CycleTimeWindow(
        start_time=(start_h, 0, 0),
        duration_seconds=duration_h * 3600,
    )


def make_time_window_sequence(start_h: int = 8, duration_h: int = 4) -> TimeWindowSequence:
    tw = TimeWindow(
        start_time=Time(start_h, 0, 0),
        duration=to_duration(f"{duration_h} hours"),
    )
    return TimeWindowSequence(windows=[tw])


# ---------------------------------------------------------------------------
# Device factory
# ---------------------------------------------------------------------------

def make_device(
    param: HomeApplianceParam | None = None,
    horizon: int = HORIZON,
    device_index: int = 0,
    port_index: int = 0,
    completed_cycles: int | None = 0,
    price_wh: float | np.ndarray = 1e-4,
) -> HomeApplianceDevice:
    if param is None:
        param = make_param()
    context = make_context(horizon, STEP_INTERVAL, price_wh, completed_cycles)
    device = HomeApplianceDevice(param, device_index, port_index)
    device.setup_run(context)
    return device


def make_device_with_price(
    price_wh: float | np.ndarray,
    horizon: int = HORIZON,
    completed_cycles: int = 0,
) -> HomeApplianceDevice:
    """Make a device with an active price forecast key."""
    param = make_param(price_forecast_key=PRICE_KEY)
    context = make_context(horizon, STEP_INTERVAL, price_wh, completed_cycles)
    device = HomeApplianceDevice(param, 0, 0)
    device.setup_run(context)
    return device


def make_genome(pop: int, starts: list[list[int]]) -> np.ndarray:
    """Build a genome array from per-individual start steps."""
    if len(starts) == 1:
        starts = starts * pop
    return np.array([[float(s) for s in row] for row in starts])


# ============================================================
# TestCycleTimeWindow
# ============================================================

class TestCycleTimeWindow:
    def test_round_trip_start_time(self):
        cw = make_cycle_window(start_h=10, duration_h=3)
        tw = cw.to_time_window()
        cw2 = CycleTimeWindow.from_time_window(tw)
        assert cw2.start_time == (10, 0, 0)

    def test_round_trip_duration_seconds(self):
        cw = make_cycle_window(start_h=8, duration_h=5)
        tw = cw.to_time_window()
        cw2 = CycleTimeWindow.from_time_window(tw)
        assert cw2.duration_seconds == 5 * 3600

    def test_round_trip_day_of_week(self):
        cw = CycleTimeWindow(start_time=(9, 0, 0), duration_seconds=3600, day_of_week=2)
        tw = cw.to_time_window()
        cw2 = CycleTimeWindow.from_time_window(tw)
        assert cw2.day_of_week == 2

    def test_round_trip_date(self):
        cw = CycleTimeWindow(start_time=(9, 0, 0), duration_seconds=3600, date=(2025, 6, 15))
        tw = cw.to_time_window()
        cw2 = CycleTimeWindow.from_time_window(tw)
        assert cw2.date == (2025, 6, 15)

    def test_hashable(self):
        cw = make_cycle_window()
        assert isinstance(hash(cw), int)

    def test_equal_instances_same_hash(self):
        cw1 = make_cycle_window(start_h=8, duration_h=4)
        cw2 = make_cycle_window(start_h=8, duration_h=4)
        assert cw1 == cw2
        assert hash(cw1) == hash(cw2)

    def test_different_instances_not_equal(self):
        cw1 = make_cycle_window(start_h=8, duration_h=4)
        cw2 = make_cycle_window(start_h=10, duration_h=4)
        assert cw1 != cw2

    def test_usable_as_dict_key(self):
        cw = make_cycle_window()
        d = {cw: "morning"}
        assert d[cw] == "morning"


# ============================================================
# TestCycleWindowsFromSequence
# ============================================================

class TestCycleWindowsFromSequence:
    def test_none_input_returns_none(self):
        assert cycle_windows_from_sequence(None) is None

    def test_sequence_converted_to_tuple(self):
        seq = make_time_window_sequence(start_h=8, duration_h=4)
        result = cycle_windows_from_sequence(seq)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_converted_value_matches_original(self):
        seq = make_time_window_sequence(start_h=10, duration_h=2)
        result = cycle_windows_from_sequence(seq)
        assert result[0].start_time == (10, 0, 0)
        assert result[0].duration_seconds == 2 * 3600

    def test_empty_sequence_returns_empty_tuple(self):
        seq = TimeWindowSequence(windows=[])
        result = cycle_windows_from_sequence(seq)
        assert result == ()


# ============================================================
# TestHomeApplianceParamValidation
# ============================================================

class TestHomeApplianceParamValidation:
    def test_zero_consumption_raises(self):
        with pytest.raises(ValueError, match="consumption_wh"):
            make_param(consumption_wh=0.0)

    def test_negative_consumption_raises(self):
        with pytest.raises(ValueError, match="consumption_wh"):
            make_param(consumption_wh=-100.0)

    def test_duration_zero_raises(self):
        with pytest.raises(ValueError, match="duration_h"):
            make_param(duration_h=0)

    def test_num_cycles_zero_raises(self):
        with pytest.raises(ValueError, match="num_cycles"):
            make_param(num_cycles=0)

    def test_negative_min_cycle_gap_raises(self):
        with pytest.raises(ValueError, match="min_cycle_gap_h"):
            make_param(min_cycle_gap_h=-1)

    def test_no_ports_raises(self):
        with pytest.raises(ValueError, match="port"):
            HomeApplianceParam(
                device_id="x",
                ports=(),
                consumption_wh=1000.0,
                duration_h=1,
            )

    def test_cycle_windows_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="cycle_time_windows"):
            make_param(
                num_cycles=2,
                cycle_time_windows=(make_cycle_window(),),  # only 1, needs 2
            )

    def test_valid_single_cycle_constructs(self):
        p = make_param(num_cycles=1)
        assert p.num_cycles == 1

    def test_valid_multi_cycle_constructs(self):
        p = make_param(
            num_cycles=2,
            cycle_time_windows=(make_cycle_window(8), make_cycle_window(18)),
        )
        assert p.num_cycles == 2

    def test_valid_no_windows_constructs(self):
        p = make_param(num_cycles=1, cycle_time_windows=())
        assert p.cycle_time_windows == ()


# ============================================================
# TestHomeApplianceParamDerivedProperties
# ============================================================

class TestHomeApplianceParamDerivedProperties:
    def test_effective_key_defaults_to_device_id(self):
        p = make_param(device_id="washer", cycles_completed_measurement_key=None)
        assert p.effective_cycles_completed_key == "washer.cycles_completed"

    def test_effective_key_uses_explicit_key(self):
        p = make_param(cycles_completed_measurement_key="my.key")
        assert p.effective_cycles_completed_key == "my.key"

    def test_resolved_windows_defaults_to_none_tuple(self):
        p = make_param(num_cycles=3, cycle_time_windows=())
        assert p.resolved_cycle_time_windows == (None, None, None)

    def test_resolved_windows_returns_stored_windows(self):
        cw = make_cycle_window()
        p = make_param(num_cycles=1, cycle_time_windows=((cw,),))
        assert p.resolved_cycle_time_windows == ((cw,),)

    def test_param_is_hashable(self):
        p = make_param()
        assert isinstance(hash(p), int)

    def test_equal_params_same_hash(self):
        p1 = make_param(consumption_wh=1000.0, duration_h=2)
        p2 = make_param(consumption_wh=1000.0, duration_h=2)
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_params_with_different_windows_not_equal(self):
        cw_morning = make_cycle_window(start_h=8, duration_h=4)
        cw_evening = make_cycle_window(start_h=18, duration_h=4)
        p1 = make_param(num_cycles=1, cycle_time_windows=((cw_morning,),))
        p2 = make_param(num_cycles=1, cycle_time_windows=((cw_evening,),))
        assert p1 != p2


# ============================================================
# TestHomeApplianceDeviceTopology
# ============================================================

class TestHomeApplianceDeviceTopology:
    def test_device_id_matches_param(self):
        param = make_param(device_id="dryer_42")
        device = HomeApplianceDevice(param, 0, 0)
        assert device.device_id == "dryer_42"

    def test_ports_match_param(self):
        param = make_param()
        device = HomeApplianceDevice(param, 0, 0)
        assert device.ports == param.ports

    def test_objective_names(self):
        device = HomeApplianceDevice(make_param(), 0, 0)
        assert device.objective_names == ["energy_cost_eur"]

    def test_device_index_stored(self):
        device = HomeApplianceDevice(make_param(), device_index=7, port_index=0)
        assert device._device_index == 7

    def test_port_index_stored(self):
        device = HomeApplianceDevice(make_param(), device_index=0, port_index=3)
        assert device._port_index == 3


# ============================================================
# TestSetupRun
# ============================================================

class TestSetupRun:
    def test_stores_horizon(self):
        device = make_device(horizon=6)
        assert device._horizon == 6

    def test_stores_step_interval_h(self):
        device = make_device()
        assert device._step_interval_h == pytest.approx(STEP_INTERVAL / 3600.0)

    def test_stores_step_times(self):
        ctx = make_context(horizon=4)
        device = HomeApplianceDevice(make_param(), 0, 0)
        device.setup_run(ctx)
        assert device._step_times == ctx.step_times

    def test_power_per_step_w(self):
        param = make_param(consumption_wh=2000.0, duration_h=2)
        device = make_device(param=param)
        assert device._power_per_step_w == pytest.approx(1000.0)

    def test_completed_zero_when_key_missing(self):
        ctx = FakeContextMissingKey(make_step_times())
        device = HomeApplianceDevice(make_param(), 0, 0)
        device.setup_run(ctx)
        assert device._num_remaining_cycles == 1

    def test_completed_zero_when_measurement_none(self):
        device = make_device(completed_cycles=None)
        assert device._num_remaining_cycles == 1

    def test_completed_reduces_remaining(self):
        param = make_param(num_cycles=3)
        device = make_device(param=param, completed_cycles=1)
        assert device._num_remaining_cycles == 2

    def test_completed_clamped_to_num_cycles(self):
        param = make_param(num_cycles=2)
        device = make_device(param=param, completed_cycles=5)
        assert device._num_remaining_cycles == 0

    def test_allowed_steps_shape(self):
        param = make_param(num_cycles=2)
        device = make_device(param=param, horizon=HORIZON)
        assert device._allowed_steps.shape == (2, HORIZON)

    def test_allowed_steps_none_windows_all_true_up_to_max_start(self):
        param = make_param(num_cycles=1, duration_h=2)
        device = make_device(param=param, horizon=6)
        # max_start = horizon - duration_h = 4; steps 0..4 allowed, 5 not
        assert device._allowed_steps[0, :5].all()
        assert not device._allowed_steps[0, 5]

    def test_allowed_steps_with_window_restricts_steps(self):
        # step_times are UTC epoch seconds; Berlin (UTC+1 winter) shifts by +1h.
        # A naive Time(h, 0, 0) window is reached at UTC step h-1.
        # Window wall-clock 07:00–09:00 → UTC steps 6 and 7 are the allowed
        # start steps for a 1h appliance.  UTC step 5 = local 06:00 is before
        # the window; UTC step 8 = local 09:00 is at the window end with no
        # room left for a 1h block.
        local_start_h = 7  # wall-clock window start
        utc_first_allowed = local_start_h - UTC_TO_LOCAL_OFFSET_H  # step 6
        cw = CycleTimeWindow(
            start_time=(local_start_h, 0, 0),
            duration_seconds=2 * 3600,
        )
        param = make_param(num_cycles=1, duration_h=1, cycle_time_windows=((cw,),))
        device = make_device(param=param, horizon=HORIZON)
        assert not device._allowed_steps[0, 0], "step 0 should be outside window"
        assert not device._allowed_steps[0, utc_first_allowed - 1], "step before window should be False"
        assert device._allowed_steps[0, utc_first_allowed], "first UTC step inside window should be True"
        assert device._allowed_steps[0, utc_first_allowed + 1], "second UTC step inside window should be True"

    def test_before_setup_run_raises(self):
        device = HomeApplianceDevice(make_param(), 0, 0)
        with pytest.raises(RuntimeError, match="setup_run"):
            device.genome_requirements()


# ============================================================
# TestGenomeRequirements
# ============================================================

class TestGenomeRequirements:
    def test_returns_none_when_all_cycles_done(self):
        param = make_param(num_cycles=2)
        device = make_device(param=param, completed_cycles=2)
        assert device.genome_requirements() is None

    def test_size_equals_num_remaining(self):
        param = make_param(num_cycles=3)
        device = make_device(param=param, completed_cycles=1)
        req = device.genome_requirements()
        assert req.size == 2

    def test_lower_bound_all_zero(self):
        param = make_param(num_cycles=2)
        device = make_device(param=param)
        req = device.genome_requirements()
        assert (req.lower_bound == 0.0).all()

    def test_upper_bound_equals_max_start(self):
        param = make_param(num_cycles=1, duration_h=2)
        device = make_device(param=param, horizon=8)
        req = device.genome_requirements()
        expected_max_start = float(8 - 2)
        assert req.upper_bound[0] == pytest.approx(expected_max_start)

    def test_before_setup_run_raises(self):
        device = HomeApplianceDevice(make_param(), 0, 0)
        with pytest.raises(RuntimeError, match="setup_run"):
            device.genome_requirements()


# ============================================================
# TestCreateBatchState
# ============================================================

class TestCreateBatchState:
    def test_start_steps_shape(self):
        param = make_param(num_cycles=2)
        device = make_device(param=param, horizon=HORIZON)
        state = device.create_batch_state(POP, HORIZON)
        assert state.start_steps.shape == (POP, 2)

    def test_schedule_shape(self):
        device = make_device(horizon=HORIZON)
        state = device.create_batch_state(POP, HORIZON)
        assert state.schedule.shape == (POP, HORIZON)

    def test_granted_energy_wh_shape(self):
        device = make_device(horizon=HORIZON)
        state = device.create_batch_state(POP, HORIZON)
        assert state.granted_energy_wh.shape == (POP, HORIZON)

    def test_all_arrays_zero_initialised(self):
        device = make_device(horizon=HORIZON)
        state = device.create_batch_state(POP, HORIZON)
        assert (state.start_steps == 0.0).all()
        assert (state.schedule == 0.0).all()
        assert (state.granted_energy_wh == 0.0).all()

    def test_step_times_forwarded(self):
        ctx = make_context(horizon=4)
        device = HomeApplianceDevice(make_param(), 0, 0)
        device.setup_run(ctx)
        state = device.create_batch_state(POP, 4)
        assert state.step_times == ctx.step_times

    def test_before_setup_run_raises(self):
        device = HomeApplianceDevice(make_param(), 0, 0)
        with pytest.raises(RuntimeError, match="setup_run"):
            device.create_batch_state(POP, HORIZON)


# ============================================================
# TestRepairPipeline
# ============================================================

class TestRepairPipeline:
    """Tests for apply_genome_batch repair steps (single individual for clarity)."""

    def _run(
        self,
        param: HomeApplianceParam,
        starts: list[list[int]],
        horizon: int = HORIZON,
        completed: int = 0,
    ):
        device = make_device(param=param, horizon=horizon, completed_cycles=completed)
        state = device.create_batch_state(len(starts), horizon)
        genome = make_genome(len(starts), starts)
        repaired = device.apply_genome_batch(state, genome)
        return device, state, repaired

    def test_float_genes_rounded(self):
        param = make_param(num_cycles=1, duration_h=1)
        _, state, repaired = self._run(param, [[2.7]])
        assert state.start_steps[0, 0] == pytest.approx(3.0)

    def test_genes_clipped_to_max_start(self):
        param = make_param(num_cycles=1, duration_h=2)
        # max_start = 8 - 2 = 6; gene of 99 should be clipped to 6
        _, state, _ = self._run(param, [[99]])
        assert state.start_steps[0, 0] == pytest.approx(6.0)

    def test_genes_clipped_to_zero(self):
        param = make_param(num_cycles=1, duration_h=1)
        _, state, _ = self._run(param, [[-5]])
        assert state.start_steps[0, 0] == pytest.approx(0.0)

    def test_cycles_sorted_in_time_order(self):
        # Two cycles: genome gives reversed order [5, 1], repair must sort to [1, 5]
        param = make_param(num_cycles=2, duration_h=1, min_cycle_gap_h=0)
        _, state, _ = self._run(param, [[5, 1]])
        assert state.start_steps[0, 0] <= state.start_steps[0, 1]

    def test_gap_enforcement_pushes_second_cycle(self):
        # duration_h=2, min_cycle_gap_h=1 → second cycle must start >= first + 3
        param = make_param(num_cycles=2, duration_h=2, min_cycle_gap_h=1)
        _, state, _ = self._run(param, [[0, 0]])  # both start at 0 initially
        gap = state.start_steps[0, 1] - state.start_steps[0, 0]
        assert gap >= 3.0

    def test_window_snap_moves_disallowed_step(self):
        # Wall-clock 07:00–09:00 → UTC steps 6 and 7 allowed for a 1h appliance.
        # Gene at step 0 (disallowed) must snap to step 6 (nearest allowed).
        local_start_h = 7
        utc_first_allowed = local_start_h - UTC_TO_LOCAL_OFFSET_H  # step 6
        cw = CycleTimeWindow(
            start_time=(local_start_h, 0, 0),
            duration_seconds=2 * 3600,
        )
        param = make_param(num_cycles=1, duration_h=1, cycle_time_windows=((cw,),))
        _, state, _ = self._run(param, [[0]])
        assert state.start_steps[0, 0] == pytest.approx(float(utc_first_allowed))

    def test_lamarckian_writeback_updates_genome(self):
        param = make_param(num_cycles=2, duration_h=1, min_cycle_gap_h=0)
        genome_in = make_genome(1, [[5, 1]])  # unsorted
        device = make_device(param=param)
        state = device.create_batch_state(1, HORIZON)
        repaired = device.apply_genome_batch(state, genome_in)
        # After repair, genome should reflect sorted starts
        assert repaired[0, 0] <= repaired[0, 1]

    def test_all_cycles_done_leaves_schedule_zero(self):
        param = make_param(num_cycles=2)
        device = make_device(param=param, completed_cycles=2)
        state = device.create_batch_state(POP, HORIZON)
        genome = np.zeros((POP, 0))
        device.apply_genome_batch(state, genome)
        assert (state.schedule == 0.0).all()

    def test_population_axis_independent(self):
        param = make_param(num_cycles=1, duration_h=1)
        device = make_device(param=param, horizon=HORIZON)
        state = device.create_batch_state(2, HORIZON)
        genome = make_genome(2, [[1], [5]])  # different starts per individual
        device.apply_genome_batch(state, genome)
        assert state.start_steps[0, 0] != state.start_steps[1, 0]


# ============================================================
# TestScheduleReconstruction
# ============================================================

class TestScheduleReconstruction:
    """Tests that the power schedule is correctly filled from start steps."""

    def _run(self, param, starts, horizon=HORIZON):
        device = make_device(param=param, horizon=horizon)
        state = device.create_batch_state(1, horizon)
        genome = make_genome(1, [starts])
        device.apply_genome_batch(state, genome)
        return state.schedule[0]

    def test_schedule_zero_before_run_block(self):
        param = make_param(consumption_wh=1000.0, duration_h=2)
        schedule = self._run(param, [4])
        assert schedule[:4].sum() == pytest.approx(0.0, abs=1e-9)

    def test_schedule_zero_after_run_block(self):
        param = make_param(consumption_wh=1000.0, duration_h=2)
        schedule = self._run(param, [4])
        assert schedule[6:].sum() == pytest.approx(0.0, abs=1e-9)

    def test_schedule_equals_power_inside_block(self):
        param = make_param(consumption_wh=1000.0, duration_h=2)
        expected_power_w = 1000.0 / 2.0  # 500 W per step
        schedule = self._run(param, [4])
        assert schedule[4] == pytest.approx(expected_power_w)
        assert schedule[5] == pytest.approx(expected_power_w)

    def test_schedule_two_cycles_both_blocks_filled(self):
        param = make_param(consumption_wh=1000.0, duration_h=1, num_cycles=2)
        schedule = self._run(param, [1, 5])
        assert schedule[1] > 0.0
        assert schedule[5] > 0.0
        # Steps not in either block are zero
        assert schedule[0] == pytest.approx(0.0, abs=1e-9)
        assert schedule[3] == pytest.approx(0.0, abs=1e-9)


# ============================================================
# TestBuildDeviceRequest
# ============================================================

class TestBuildDeviceRequest:
    def test_returns_request_when_all_cycles_done(self):
        # All cycles done: schedule is zero, but a request is still issued
        # so the arbitrator can allocate zero energy (no special-casing needed).
        param = make_param(num_cycles=1)
        device = make_device(param=param, completed_cycles=1)
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req is not None

    def test_device_index_in_request(self):
        device = make_device(device_index=5)
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.device_index == 5

    def test_port_index_in_request(self):
        device = make_device(port_index=3)
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].port_index == 3

    def test_energy_wh_equals_schedule_times_step_h(self):
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        genome = np.zeros((POP, 1))
        genome[:, 0] = 2.0  # all individuals start at step 2
        device.apply_genome_batch(state, genome)
        req = device.build_device_request(state)
        step_h = STEP_INTERVAL / 3600.0
        expected = state.schedule * step_h
        np.testing.assert_allclose(req.port_requests[0].energy_wh, expected, rtol=1e-9)

    def test_min_energy_wh_all_zeros(self):
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert (req.port_requests[0].min_energy_wh == 0.0).all()

    def test_is_slack_false(self):
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].is_slack is False

    def test_energy_wh_shape(self):
        device = make_device(horizon=HORIZON)
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].energy_wh.shape == (POP, HORIZON)

    def test_before_setup_run_raises(self):
        device = HomeApplianceDevice(make_param(), 0, 0)
        state_dummy = object()
        with pytest.raises(RuntimeError, match="setup_run"):
            device.build_device_request(state_dummy)


# ============================================================
# TestApplyDeviceGrant
# ============================================================

class TestApplyDeviceGrant:
    def _make_grant(self, granted: np.ndarray, device_index: int = 0) -> DeviceGrant:
        return DeviceGrant(
            device_index=device_index,
            port_grants=(PortGrant(port_index=0, granted_wh=granted),),
        )

    def test_granted_energy_wh_updated(self):
        device = make_device(horizon=HORIZON)
        state = device.create_batch_state(POP, HORIZON)
        awarded = np.ones((POP, HORIZON)) * 123.0
        device.apply_device_grant(state, self._make_grant(awarded))
        np.testing.assert_allclose(state.granted_energy_wh, awarded, rtol=1e-9)

    def test_empty_port_grants_zeroes_granted(self):
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        state.granted_energy_wh[:] = 99.0  # pre-fill
        grant = DeviceGrant(device_index=0, port_grants=())
        device.apply_device_grant(state, grant)
        assert (state.granted_energy_wh == 0.0).all()

    def test_schedule_unchanged_after_grant(self):
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        genome = np.full((POP, 1), 2.0)
        device.apply_genome_batch(state, genome)
        schedule_before = state.schedule.copy()
        awarded = np.ones((POP, HORIZON)) * 50.0
        device.apply_device_grant(state, self._make_grant(awarded))
        np.testing.assert_array_equal(state.schedule, schedule_before)


# ============================================================
# TestComputeCost
# ============================================================

class TestComputeCost:
    def test_cost_shape(self):
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        assert cost.shape == (POP, 1)

    def test_zero_grant_zero_cost(self):
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        np.testing.assert_allclose(cost, 0.0, atol=1e-9)

    def test_cost_equals_granted_times_price(self):
        uniform_price = 2e-4  # EUR/Wh
        device = make_device_with_price(price_wh=uniform_price)
        state = device.create_batch_state(1, HORIZON)
        state.granted_energy_wh[0, 3] = 500.0
        state.granted_energy_wh[0, 5] = 300.0
        cost = device.compute_cost(state)
        expected = (500.0 + 300.0) * uniform_price
        assert cost[0, 0] == pytest.approx(expected, rel=1e-9)

    def test_cost_uses_flat_fallback_when_no_price_key(self):
        # price_forecast_key=None → fallback 1e-4 EUR/Wh regardless of context price
        device = make_device(price_wh=99.0)  # context price ignored
        state = device.create_batch_state(1, HORIZON)
        state.granted_energy_wh[0, 0] = 1000.0
        cost = device.compute_cost(state)
        assert cost[0, 0] == pytest.approx(1000.0 * 1e-4, rel=1e-9)

    def test_cost_varies_with_time_of_use_price(self):
        prices = np.zeros(HORIZON)
        prices[2] = 3e-4
        prices[6] = 1e-4
        device = make_device_with_price(price_wh=prices)
        state = device.create_batch_state(1, HORIZON)
        state.granted_energy_wh[0, 2] = 100.0
        state.granted_energy_wh[0, 6] = 100.0
        cost = device.compute_cost(state)
        expected = 100.0 * 3e-4 + 100.0 * 1e-4
        assert cost[0, 0] == pytest.approx(expected, rel=1e-9)

    def test_before_setup_run_raises(self):
        device = HomeApplianceDevice(make_param(), 0, 0)
        state_dummy = object()
        with pytest.raises(RuntimeError, match="setup_run"):
            device.compute_cost(state_dummy)


# ============================================================
# TestExtractInstructions
# ============================================================

class TestExtractInstructions:
    def _run(
        self,
        starts: list[int],
        param: HomeApplianceParam | None = None,
        horizon: int = HORIZON,
        pop: int = 1,
    ):
        if param is None:
            param = make_param(duration_h=DURATION_H)
        device = make_device(param=param, horizon=horizon)
        state = device.create_batch_state(pop, horizon)
        genome = np.array([[float(s) for s in starts]] * pop)
        device.apply_genome_batch(state, genome)
        return device, state

    def test_one_instruction_per_step(self):
        device, state = self._run([0])
        instrs = device.extract_instructions(state, 0)
        assert len(instrs) == HORIZON

    def test_steps_in_run_block_are_run(self):
        device, state = self._run([2])  # starts at step 2, duration=2 → steps 2,3
        instrs = device.extract_instructions(state, 0)
        assert instrs[2].operation_mode_id == str(ApplianceOperationMode.RUN)
        assert instrs[3].operation_mode_id == str(ApplianceOperationMode.RUN)

    def test_steps_outside_run_block_are_off(self):
        device, state = self._run([2])
        instrs = device.extract_instructions(state, 0)
        assert instrs[0].operation_mode_id == str(ApplianceOperationMode.OFF)
        assert instrs[1].operation_mode_id == str(ApplianceOperationMode.OFF)
        assert instrs[4].operation_mode_id == str(ApplianceOperationMode.OFF)

    def test_execution_times_match_step_times(self):
        device, state = self._run([0])
        instrs = device.extract_instructions(state, 0)
        expected_times = make_step_times(HORIZON)
        for t, dt in enumerate(expected_times):
            assert instrs[t].execution_time == dt

    def test_resource_id_in_instructions(self):
        param = make_param(device_id="washer_01", duration_h=1)
        device, state = self._run([0], param=param)
        instrs = device.extract_instructions(state, 0)
        assert all(i.resource_id == "washer_01" for i in instrs)

    def test_individual_index_selects_correct_row(self):
        param = make_param(duration_h=1)
        device = make_device(param=param, horizon=HORIZON)
        state = device.create_batch_state(2, HORIZON)
        genome = np.array([
            [1.0],  # individual 0 runs at step 1
            [5.0],  # individual 1 runs at step 5
        ])
        device.apply_genome_batch(state, genome)
        instrs_0 = device.extract_instructions(state, 0)
        instrs_1 = device.extract_instructions(state, 1)
        assert instrs_0[1].operation_mode_id == str(ApplianceOperationMode.RUN)
        assert instrs_0[5].operation_mode_id == str(ApplianceOperationMode.OFF)
        assert instrs_1[1].operation_mode_id == str(ApplianceOperationMode.OFF)
        assert instrs_1[5].operation_mode_id == str(ApplianceOperationMode.RUN)

    def test_all_off_when_no_remaining_cycles(self):
        param = make_param(num_cycles=1)
        device = make_device(param=param, completed_cycles=1, horizon=HORIZON)
        state = device.create_batch_state(1, HORIZON)
        genome = np.zeros((1, 0))
        device.apply_genome_batch(state, genome)
        instrs = device.extract_instructions(state, 0)
        assert all(i.operation_mode_id == str(ApplianceOperationMode.OFF) for i in instrs)

    def test_multi_cycle_both_blocks_are_run(self):
        param = make_param(num_cycles=2, duration_h=1, min_cycle_gap_h=0)
        device = make_device(param=param, horizon=HORIZON)
        state = device.create_batch_state(1, HORIZON)
        genome = np.array([[1.0, 5.0]])
        device.apply_genome_batch(state, genome)
        instrs = device.extract_instructions(state, 0)
        assert instrs[1].operation_mode_id == str(ApplianceOperationMode.RUN)
        assert instrs[5].operation_mode_id == str(ApplianceOperationMode.RUN)
        assert instrs[3].operation_mode_id == str(ApplianceOperationMode.OFF)
