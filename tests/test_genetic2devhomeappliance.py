"""Tests for the vectorized home appliance device.

Module under test
-----------------
``akkudoktoreos.devices.genetic2.homeappliance``

Design under test
-----------------
One integer-valued gene per *remaining* cycle::

    genome shape: (population_size, num_remaining_cycles)
    genome[i, k] in [0, horizon - 1]    start step for remaining cycle k

The repair pipeline:
    1. Round and clip to [0, max_start].
    2. Per-cycle window snap (nearest allowed step for that cycle).
    3. Sort across cycles (time-ordering, removes genome symmetry).
    4. Gap enforcement (push forward to satisfy min_cycle_gap_h).
    5. Reconstruct flat-block power schedule.
    6. Lamarckian write-back.

Window scheduling
-----------------
``HomeApplianceParam.time_window_key`` is a ``/``-separated config path
resolved during ``setup_run`` via ``context.resolve_config_cycle_time_windows``,
which returns ``(cycle_indices, matrix)`` — a sorted list of cycle numbers and
a ``(num_cycles, horizon)`` float matrix.

``FakeContext.resolve_config_cycle_time_windows`` registers
``CycleTimeWindowSequence`` objects by key and delegates to
``cycles_to_matrix`` to produce the same output as the real context.

Constructor signature
---------------------
``HomeApplianceDevice(param, device_index, port_index)`` — identical to
``GridConnectionDevice`` and ``HybridInverterDevice``.

Tolerances: rtol=1e-9 throughout; zero checks use approx(0.0, abs=1e-9).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pendulum import Duration

from akkudoktoreos.config.configabc import CycleTimeWindowSequence, ValueTimeWindow
from akkudoktoreos.devices.devicesabc import (
    ApplianceOperationMode,
    EnergyPort,
    PortDirection,
)
from akkudoktoreos.devices.genetic2.homeappliance import (
    HomeApplianceBatchState,
    HomeApplianceDevice,
    HomeApplianceParam,
)
from akkudoktoreos.simulation.genetic2.arbitrator import DeviceGrant, PortGrant
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import Time, to_datetime, to_duration

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STEP_INTERVAL = 3600.0   # 1-hour steps [s]
HORIZON = 8
POP = 3
PRICE_KEY = "price_forecast"
WINDOW_KEY = "devices/home_appliances/dishwasher/cycle_time_windows"

CONSUMPTION_WH = 2_000.0
DURATION_H = 2

# step_times are built from UTC epoch seconds: to_datetime(i * 3600) is UTC.
# Berlin (UTC+1 winter): a naive Time(h, 0, 0) window is first reached at
# UTC step h-1.
UTC_TO_LOCAL_OFFSET_H = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_step_times(n: int = HORIZON) -> tuple:
    return tuple(to_datetime(i * STEP_INTERVAL) for i in range(n))


def make_port() -> tuple[EnergyPort, ...]:
    return (EnergyPort(port_id="p_ac", bus_id="bus_ac", direction=PortDirection.SINK),)


def make_cycle_time_windows(
    windows: list[tuple[int, int, int]],  # (start_h, duration_h, cycle_index)
) -> CycleTimeWindowSequence:
    """Build a CycleTimeWindowSequence from (start_h, duration_h, cycle_index) tuples."""
    vws = [
        ValueTimeWindow(
            start_time=Time(start_h, 0, 0),
            duration=to_duration(f"{dur_h} hours"),
            value=float(cycle_idx),
        )
        for start_h, dur_h, cycle_idx in windows
    ]
    return CycleTimeWindowSequence(windows=vws)


# ---------------------------------------------------------------------------
# FakeContext
# ---------------------------------------------------------------------------

class FakeContext:
    """Minimal SimulationContext stand-in for unit tests.

    ``resolve_config_cycle_time_windows`` delegates to
    ``CycleTimeWindowSequence.cycles_to_matrix``, reproducing the same
    ``(cycle_indices, matrix)`` output as the real SimulationContext without
    touching the config singleton.

    Call sites pass this to setup_run() via cast(SimulationContext, ctx)
    so mypy is satisfied while the duck-typed runtime behaviour is preserved.
    """

    def __init__(
        self,
        step_times: tuple,
        step_interval_sec: float = STEP_INTERVAL,
        price_wh: np.ndarray | None = None,
        completed_cycles: int | None = 0,
        cycle_time_windows: dict[str, CycleTimeWindowSequence] | None = None,
    ) -> None:
        self.step_times = step_times
        self.step_interval = Duration(seconds=step_interval_sec)
        self.horizon = len(step_times)
        self._price_wh = price_wh if price_wh is not None else np.full(len(step_times), 1e-4)
        self._completed_cycles = completed_cycles
        self._cycle_time_windows: dict[str, CycleTimeWindowSequence] = cycle_time_windows or {}

    def resolve_prediction(self, key: str) -> np.ndarray:
        return self._price_wh.copy()

    def resolve_measurement(self, key: str) -> float | None:
        if self._completed_cycles is None:
            return None
        return float(self._completed_cycles)

    def resolve_config_cycle_time_windows(
        self, config_path: str
    ) -> tuple[list[int], np.ndarray]:
        if config_path not in self._cycle_time_windows:
            raise KeyError(f"No CycleTimeWindowSequence registered for path '{config_path}'")
        seq = self._cycle_time_windows[config_path]
        cycle_indices, matrix = seq.cycles_to_matrix(
            start_datetime=self.step_times[0],
            end_datetime=self.step_times[-1] + self.step_interval,
            interval=self.step_interval,
        )
        return cycle_indices, matrix


class FakeContextMissingKey(FakeContext):
    def resolve_measurement(self, key: str) -> float:
        raise KeyError(key)


def make_context(
    horizon: int = HORIZON,
    step_interval_sec: float = STEP_INTERVAL,
    price_wh: float | np.ndarray = 1e-4,
    completed_cycles: int | None = 0,
    cycle_time_windows: dict[str, CycleTimeWindowSequence] | None = None,
) -> FakeContext:
    times = make_step_times(horizon)
    price_array = (
        np.full(horizon, float(price_wh)) if np.ndim(price_wh) == 0
        else np.asarray(price_wh, dtype=np.float64)
    )
    return FakeContext(times, step_interval_sec, price_array, completed_cycles,
                       cycle_time_windows)


# ---------------------------------------------------------------------------
# Param / device factories
# ---------------------------------------------------------------------------

def make_param(
    device_id: str = "dishwasher",
    consumption_wh: float = CONSUMPTION_WH,
    duration_h: int = DURATION_H,
    num_cycles: int = 1,
    min_cycle_gap_h: int = 0,
    price_forecast_key: str | None = None,
    cycles_completed_measurement_key: str | None = None,
    time_window_key: str | None = None,
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
        time_window_key=time_window_key,
    )


def make_device(
    param: HomeApplianceParam | None = None,
    device_index: int = 0,
    port_index: int = 0,
    horizon: int = HORIZON,
    completed_cycles: int | None = 0,
    price_wh: float | np.ndarray = 1e-4,
    cycle_time_windows: dict[str, CycleTimeWindowSequence] | None = None,
) -> HomeApplianceDevice:
    if param is None:
        param = make_param()
    context = make_context(horizon, STEP_INTERVAL, price_wh, completed_cycles,
                           cycle_time_windows)
    device = HomeApplianceDevice(param, device_index, port_index)
    device.setup_run(cast(SimulationContext, context))
    return device


def make_device_with_price(
    price_wh: float | np.ndarray,
    horizon: int = HORIZON,
    completed_cycles: int = 0,
) -> HomeApplianceDevice:
    param = make_param(price_forecast_key=PRICE_KEY)
    context = make_context(horizon, STEP_INTERVAL, price_wh, completed_cycles)
    device = HomeApplianceDevice(param, 0, 0)
    device.setup_run(cast(SimulationContext, context))
    return device


def make_genome(pop: int, starts: list[list[int]]) -> np.ndarray:
    if len(starts) == 1:
        starts = starts * pop
    return np.array([[float(s) for s in row] for row in starts])


# ============================================================
# TestHomeApplianceParamValidation
# ============================================================

class TestHomeApplianceParamValidation:
    def test_zero_consumption_raises(self) -> None:
        with pytest.raises(ValueError, match="consumption_wh"):
            make_param(consumption_wh=0.0)

    def test_negative_consumption_raises(self) -> None:
        with pytest.raises(ValueError, match="consumption_wh"):
            make_param(consumption_wh=-100.0)

    def test_duration_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_h"):
            make_param(duration_h=0)

    def test_num_cycles_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_cycles"):
            make_param(num_cycles=0)

    def test_negative_min_cycle_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="min_cycle_gap_h"):
            make_param(min_cycle_gap_h=-1)

    def test_no_ports_raises(self) -> None:
        with pytest.raises(ValueError, match="port"):
            HomeApplianceParam(
                device_id="x", ports=(), consumption_wh=1000.0, duration_h=1,
            )

    def test_valid_single_cycle_constructs(self) -> None:
        assert make_param(num_cycles=1).num_cycles == 1

    def test_valid_multi_cycle_constructs(self) -> None:
        assert make_param(num_cycles=3).num_cycles == 3

    def test_valid_with_time_window_key_constructs(self) -> None:
        assert make_param(time_window_key=WINDOW_KEY).time_window_key == WINDOW_KEY


# ============================================================
# TestHomeApplianceParamDerivedProperties
# ============================================================

class TestHomeApplianceParamDerivedProperties:
    def test_effective_key_defaults_to_device_id(self) -> None:
        p = make_param(device_id="washer", cycles_completed_measurement_key=None)
        assert p.effective_cycles_completed_key == "washer.cycles_completed"

    def test_effective_key_uses_explicit_key(self) -> None:
        assert make_param(cycles_completed_measurement_key="my.key").effective_cycles_completed_key == "my.key"

    def test_time_window_key_none_by_default(self) -> None:
        assert make_param().time_window_key is None

    def test_time_window_key_stored_as_string(self) -> None:
        assert make_param(time_window_key=WINDOW_KEY).time_window_key == WINDOW_KEY

    def test_param_is_hashable(self) -> None:
        assert isinstance(hash(make_param()), int)

    def test_equal_params_same_hash(self) -> None:
        p1 = make_param(consumption_wh=1000.0, duration_h=2)
        p2 = make_param(consumption_wh=1000.0, duration_h=2)
        assert p1 == p2 and hash(p1) == hash(p2)

    def test_params_with_different_keys_not_equal(self) -> None:
        p1 = make_param(time_window_key="devices/home_appliances/dishwasher/cycle_time_windows")
        p2 = make_param(time_window_key="devices/home_appliances/washing_machine/cycle_time_windows")
        assert p1 != p2


# ============================================================
# TestHomeApplianceDeviceTopology
# ============================================================

class TestHomeApplianceDeviceTopology:
    def test_device_id_matches_param(self) -> None:
        assert HomeApplianceDevice(make_param(device_id="dryer"), 0, 0).device_id == "dryer"

    def test_ports_match_param(self) -> None:
        param = make_param()
        assert HomeApplianceDevice(param, 0, 0).ports == param.ports

    def test_objective_names(self) -> None:
        assert HomeApplianceDevice(make_param(), 0, 0).objective_names == ["energy_cost_amt"]

    def test_device_index_stored(self) -> None:
        assert HomeApplianceDevice(make_param(), device_index=7, port_index=0)._device_index == 7

    def test_port_index_stored(self) -> None:
        assert HomeApplianceDevice(make_param(), device_index=0, port_index=3)._port_index == 3


# ============================================================
# TestSetupRun
# ============================================================

class TestSetupRun:
    def test_stores_horizon(self) -> None:
        assert make_device(horizon=6)._horizon == 6

    def test_stores_step_interval_h(self) -> None:
        assert make_device()._step_interval_h == pytest.approx(STEP_INTERVAL / 3600.0)

    def test_stores_step_times(self) -> None:
        ctx = make_context(horizon=4)
        device = HomeApplianceDevice(make_param(), 0, 0)
        device.setup_run(cast(SimulationContext, ctx))
        assert device._step_times == ctx.step_times

    def test_power_per_step_w(self) -> None:
        device = make_device(param=make_param(consumption_wh=2000.0, duration_h=2))
        assert device._power_per_step_w == pytest.approx(1000.0)

    def test_completed_zero_when_key_missing(self) -> None:
        ctx = FakeContextMissingKey(make_step_times())
        device = HomeApplianceDevice(make_param(), 0, 0)
        device.setup_run(cast(SimulationContext, ctx))
        assert device._num_remaining_cycles == 1

    def test_completed_zero_when_measurement_none(self) -> None:
        assert make_device(completed_cycles=None)._num_remaining_cycles == 1

    def test_completed_reduces_remaining(self) -> None:
        assert make_device(param=make_param(num_cycles=3), completed_cycles=1)._num_remaining_cycles == 2

    def test_completed_clamped_to_num_cycles(self) -> None:
        assert make_device(param=make_param(num_cycles=2), completed_cycles=5)._num_remaining_cycles == 0

    def test_allowed_steps_shape(self) -> None:
        device = make_device(param=make_param(num_cycles=2), horizon=HORIZON)
        assert device._allowed_steps is not None
        assert device._allowed_steps.shape == (2, HORIZON)

    def test_allowed_steps_no_key_all_true_up_to_max_start(self) -> None:
        # horizon=6, duration_h=2 → max_start=4; steps 0..4 True, step 5 False
        device = make_device(param=make_param(num_cycles=1, duration_h=2), horizon=6)
        assert device._allowed_steps is not None
        assert device._allowed_steps[0, :5].all()
        assert not device._allowed_steps[0, 5]

    def test_allowed_steps_single_cycle_with_window(self) -> None:
        # Window wall-clock 07:00-09:00; Berlin UTC+1 -> UTC steps 6,7 allowed
        # for a 1h appliance (duration=1).
        local_start_h = 7
        utc_first = local_start_h - UTC_TO_LOCAL_OFFSET_H  # 6
        seq = make_cycle_time_windows([(local_start_h, 2, 0)])
        param = make_param(num_cycles=1, duration_h=1, time_window_key=WINDOW_KEY)
        device = make_device(param=param, horizon=HORIZON,
                             cycle_time_windows={WINDOW_KEY: seq})
        assert device._allowed_steps is not None
        assert not device._allowed_steps[0, 0]
        assert not device._allowed_steps[0, utc_first - 1]
        assert device._allowed_steps[0, utc_first]
        assert device._allowed_steps[0, utc_first + 1]

    def test_allowed_steps_two_cycles_distinct_windows(self) -> None:
        # Cycle 0: wall-clock 04:00-06:00 -> UTC steps 3,4
        # Cycle 1: wall-clock 07:00-09:00 -> UTC steps 6,7
        seq = make_cycle_time_windows([
            (4, 2, 0),   # cycle 0: 04:00 local, 2h
            (7, 2, 1),   # cycle 1: 07:00 local, 2h
        ])
        param = make_param(num_cycles=2, duration_h=1, time_window_key=WINDOW_KEY)
        device = make_device(param=param, horizon=HORIZON,
                             cycle_time_windows={WINDOW_KEY: seq})
        assert device._allowed_steps is not None
        # Row 0 (cycle 0): UTC steps 3 and 4 allowed
        assert device._allowed_steps[0, 3]
        assert device._allowed_steps[0, 4]
        assert not device._allowed_steps[0, 6]
        # Row 1 (cycle 1): UTC steps 6 and 7 allowed
        assert not device._allowed_steps[1, 3]
        assert device._allowed_steps[1, 6]
        assert device._allowed_steps[1, 7]

    def test_allowed_steps_completed_cycle_skipped(self) -> None:
        # 2 cycles defined; 1 already completed.
        # Only cycle 1 row should remain.
        seq = make_cycle_time_windows([
            (4, 2, 0),
            (7, 2, 1),
        ])
        param = make_param(num_cycles=2, duration_h=1, time_window_key=WINDOW_KEY)
        device = make_device(param=param, horizon=HORIZON, completed_cycles=1,
                             cycle_time_windows={WINDOW_KEY: seq})
        assert device._allowed_steps is not None
        # Only 1 remaining row; it should correspond to cycle 1's window
        assert device._allowed_steps.shape == (1, HORIZON)
        assert device._allowed_steps[0, 6]
        assert device._allowed_steps[0, 7]
        assert not device._allowed_steps[0, 3]

    def test_before_setup_run_raises(self) -> None:
        with pytest.raises(RuntimeError, match="setup_run"):
            HomeApplianceDevice(make_param(), 0, 0).genome_requirements()


# ============================================================
# TestGenomeRequirements
# ============================================================

class TestGenomeRequirements:
    def test_returns_none_when_all_cycles_done(self) -> None:
        assert make_device(param=make_param(num_cycles=2), completed_cycles=2).genome_requirements() is None

    def test_size_equals_num_remaining(self) -> None:
        req = make_device(param=make_param(num_cycles=3), completed_cycles=1).genome_requirements()
        assert req is not None
        assert req.size == 2

    def test_lower_bound_all_zero(self) -> None:
        req = make_device(param=make_param(num_cycles=2)).genome_requirements()
        assert req is not None
        assert req.lower_bound is not None
        assert (req.lower_bound == 0.0).all()

    def test_upper_bound_equals_max_start(self) -> None:
        req = make_device(param=make_param(num_cycles=1, duration_h=2), horizon=8).genome_requirements()
        assert req is not None
        assert req.upper_bound is not None
        assert req.upper_bound[0] == pytest.approx(6.0)

    def test_before_setup_run_raises(self) -> None:
        with pytest.raises(RuntimeError, match="setup_run"):
            HomeApplianceDevice(make_param(), 0, 0).genome_requirements()


# ============================================================
# TestCreateBatchState
# ============================================================

class TestCreateBatchState:
    def test_start_steps_shape(self) -> None:
        assert make_device(param=make_param(num_cycles=2)).create_batch_state(POP, HORIZON).start_steps.shape == (POP, 2)

    def test_schedule_shape(self) -> None:
        assert make_device().create_batch_state(POP, HORIZON).schedule.shape == (POP, HORIZON)

    def test_granted_energy_wh_shape(self) -> None:
        assert make_device().create_batch_state(POP, HORIZON).granted_energy_wh.shape == (POP, HORIZON)

    def test_all_arrays_zero_initialised(self) -> None:
        state = make_device().create_batch_state(POP, HORIZON)
        assert (state.start_steps == 0.0).all()
        assert (state.schedule == 0.0).all()
        assert (state.granted_energy_wh == 0.0).all()

    def test_step_times_forwarded(self) -> None:
        ctx = make_context(horizon=4)
        device = HomeApplianceDevice(make_param(), 0, 0)
        device.setup_run(cast(SimulationContext, ctx))
        assert device.create_batch_state(POP, 4).step_times == ctx.step_times

    def test_before_setup_run_raises(self) -> None:
        with pytest.raises(RuntimeError, match="setup_run"):
            HomeApplianceDevice(make_param(), 0, 0).create_batch_state(POP, HORIZON)


# ============================================================
# TestRepairPipeline
# ============================================================

class TestRepairPipeline:
    def _run(self, param, starts, horizon=HORIZON, completed=0,
             cycle_time_windows=None):
        ctx = make_context(horizon, STEP_INTERVAL, 1e-4, completed, cycle_time_windows)
        device = HomeApplianceDevice(param, 0, 0)
        device.setup_run(cast(SimulationContext, ctx))
        state = device.create_batch_state(len(starts), horizon)
        genome = make_genome(len(starts), starts)
        repaired = device.apply_genome_batch(state, genome)
        return device, state, repaired

    def test_float_genes_rounded(self) -> None:
        _, state, _ = self._run(make_param(num_cycles=1, duration_h=1), [[2.7]])
        assert state.start_steps[0, 0] == pytest.approx(3.0)

    def test_genes_clipped_to_max_start(self) -> None:
        _, state, _ = self._run(make_param(num_cycles=1, duration_h=2), [[99]])
        assert state.start_steps[0, 0] == pytest.approx(6.0)

    def test_genes_clipped_to_zero(self) -> None:
        _, state, _ = self._run(make_param(num_cycles=1, duration_h=1), [[-5]])
        assert state.start_steps[0, 0] == pytest.approx(0.0)

    def test_cycles_sorted_in_time_order(self) -> None:
        _, state, _ = self._run(make_param(num_cycles=2, duration_h=1), [[5, 1]])
        assert state.start_steps[0, 0] <= state.start_steps[0, 1]

    def test_gap_enforcement_pushes_second_cycle(self) -> None:
        _, state, _ = self._run(make_param(num_cycles=2, duration_h=2, min_cycle_gap_h=1), [[0, 0]])
        assert state.start_steps[0, 1] - state.start_steps[0, 0] >= 3.0

    def test_window_snap_moves_disallowed_step_cycle0(self) -> None:
        # Cycle 0: wall-clock 07:00-09:00 -> UTC steps 6,7 allowed for 1h.
        # Gene at step 0 (disallowed) must snap to step 6.
        local_start_h = 7
        utc_first = local_start_h - UTC_TO_LOCAL_OFFSET_H  # 6
        seq = make_cycle_time_windows([(local_start_h, 2, 0)])
        param = make_param(num_cycles=1, duration_h=1, time_window_key=WINDOW_KEY)
        _, state, _ = self._run(param, [[0]], cycle_time_windows={WINDOW_KEY: seq})
        assert state.start_steps[0, 0] == pytest.approx(float(utc_first))

    def test_per_cycle_windows_snap_independently(self) -> None:
        # Cycle 0: UTC steps 3,4; Cycle 1: UTC steps 6,7.
        # Gene [0, 0] should snap to [3, 6].
        seq = make_cycle_time_windows([(4, 2, 0), (7, 2, 1)])
        param = make_param(num_cycles=2, duration_h=1, time_window_key=WINDOW_KEY)
        _, state, _ = self._run(param, [[0, 0]], cycle_time_windows={WINDOW_KEY: seq})
        assert state.start_steps[0, 0] == pytest.approx(3.0)
        assert state.start_steps[0, 1] == pytest.approx(6.0)

    def test_lamarckian_writeback_updates_genome(self) -> None:
        ctx = make_context()
        device = HomeApplianceDevice(make_param(num_cycles=2, duration_h=1), 0, 0)
        device.setup_run(cast(SimulationContext, ctx))
        state = device.create_batch_state(1, HORIZON)
        repaired = device.apply_genome_batch(state, make_genome(1, [[5, 1]]))
        assert repaired[0, 0] <= repaired[0, 1]

    def test_all_cycles_done_leaves_schedule_zero(self) -> None:
        device = make_device(param=make_param(num_cycles=2), completed_cycles=2)
        state = device.create_batch_state(POP, HORIZON)
        device.apply_genome_batch(state, np.zeros((POP, 0)))
        assert (state.schedule == 0.0).all()

    def test_population_axis_independent(self) -> None:
        device = make_device(param=make_param(num_cycles=1, duration_h=1))
        state = device.create_batch_state(2, HORIZON)
        device.apply_genome_batch(state, make_genome(2, [[1], [5]]))
        assert state.start_steps[0, 0] != state.start_steps[1, 0]


# ============================================================
# TestScheduleReconstruction
# ============================================================

class TestScheduleReconstruction:
    def _run(self, param, starts, horizon=HORIZON):
        device = make_device(param=param, horizon=horizon)
        state = device.create_batch_state(1, horizon)
        device.apply_genome_batch(state, make_genome(1, [starts]))
        return state.schedule[0]

    def test_schedule_zero_before_run_block(self) -> None:
        assert self._run(make_param(consumption_wh=1000.0, duration_h=2), [4])[:4].sum() == pytest.approx(0.0, abs=1e-9)

    def test_schedule_zero_after_run_block(self) -> None:
        assert self._run(make_param(consumption_wh=1000.0, duration_h=2), [4])[6:].sum() == pytest.approx(0.0, abs=1e-9)

    def test_schedule_equals_power_inside_block(self) -> None:
        schedule = self._run(make_param(consumption_wh=1000.0, duration_h=2), [4])
        assert schedule[4] == pytest.approx(500.0)
        assert schedule[5] == pytest.approx(500.0)

    def test_schedule_two_cycles_both_blocks_filled(self) -> None:
        schedule = self._run(make_param(consumption_wh=1000.0, duration_h=1, num_cycles=2), [1, 5])
        assert schedule[1] > 0.0
        assert schedule[5] > 0.0
        assert schedule[0] == pytest.approx(0.0, abs=1e-9)
        assert schedule[3] == pytest.approx(0.0, abs=1e-9)


# ============================================================
# TestBuildDeviceRequest
# ============================================================

class TestBuildDeviceRequest:
    def test_returns_request_when_all_cycles_done(self) -> None:
        device = make_device(param=make_param(num_cycles=1), completed_cycles=1)
        assert device.build_device_request(device.create_batch_state(POP, HORIZON)) is not None

    def test_device_index_in_request(self) -> None:
        device = make_device(device_index=5)
        req = device.build_device_request(device.create_batch_state(POP, HORIZON))
        assert req is not None
        assert req.device_index == 5

    def test_port_index_in_request(self) -> None:
        device = make_device(port_index=2)
        req = device.build_device_request(device.create_batch_state(POP, HORIZON))
        assert req is not None
        assert req.port_requests[0].port_index == 2

    def test_energy_wh_equals_schedule_times_step_h(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        device.apply_genome_batch(state, np.full((POP, 1), 2.0))
        req = device.build_device_request(state)
        assert req is not None
        np.testing.assert_allclose(req.port_requests[0].energy_wh,
                                   state.schedule * (STEP_INTERVAL / 3600.0), rtol=1e-9)

    def test_min_energy_wh_all_zeros(self) -> None:
        device = make_device()
        req = device.build_device_request(device.create_batch_state(POP, HORIZON))
        assert req is not None
        assert (req.port_requests[0].min_energy_wh == 0.0).all()

    def test_is_slack_false(self) -> None:
        device = make_device()
        req = device.build_device_request(device.create_batch_state(POP, HORIZON))
        assert req is not None
        assert req.port_requests[0].is_slack is False

    def test_energy_wh_shape(self) -> None:
        device = make_device()
        req = device.build_device_request(device.create_batch_state(POP, HORIZON))
        assert req is not None
        assert req.port_requests[0].energy_wh.shape == (POP, HORIZON)

    def test_before_setup_run_raises(self) -> None:
        state = HomeApplianceBatchState(
            start_steps=np.zeros((1, 1)),
            schedule=np.zeros((1, HORIZON)),
            granted_energy_wh=np.zeros((1, HORIZON)),
            population_size=1,
            horizon=HORIZON,
            step_times=make_step_times(),
            num_remaining_cycles=1,
        )
        with pytest.raises(RuntimeError, match="setup_run"):
            HomeApplianceDevice(make_param(), 0, 0).build_device_request(state)


# ============================================================
# TestApplyDeviceGrant
# ============================================================

class TestApplyDeviceGrant:
    def _grant(self, awarded: np.ndarray) -> DeviceGrant:
        return DeviceGrant(device_index=0,
                           port_grants=(PortGrant(port_index=0, granted_wh=awarded),))

    def test_granted_energy_wh_updated(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        awarded = np.ones((POP, HORIZON)) * 123.0
        device.apply_device_grant(state, self._grant(awarded))
        np.testing.assert_allclose(state.granted_energy_wh, awarded, rtol=1e-9)

    def test_empty_port_grants_zeroes_granted(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        state.granted_energy_wh[:] = 99.0
        device.apply_device_grant(state, DeviceGrant(device_index=0, port_grants=()))
        assert (state.granted_energy_wh == 0.0).all()

    def test_schedule_unchanged_after_grant(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        device.apply_genome_batch(state, np.full((POP, 1), 2.0))
        before = state.schedule.copy()
        device.apply_device_grant(state, self._grant(np.ones((POP, HORIZON)) * 50.0))
        np.testing.assert_array_equal(state.schedule, before)


# ============================================================
# TestComputeCost
# ============================================================

class TestComputeCost:
    def test_cost_shape(self) -> None:
        device = make_device()
        assert device.compute_cost(device.create_batch_state(POP, HORIZON)).shape == (POP, 1)

    def test_zero_grant_zero_cost(self) -> None:
        device = make_device()
        np.testing.assert_allclose(device.compute_cost(device.create_batch_state(POP, HORIZON)), 0.0, atol=1e-9)

    def test_cost_equals_granted_times_price(self) -> None:
        device = make_device_with_price(price_wh=2e-4)
        state = device.create_batch_state(1, HORIZON)
        state.granted_energy_wh[0, 3] = 500.0
        state.granted_energy_wh[0, 5] = 300.0
        assert device.compute_cost(state)[0, 0] == pytest.approx(800.0 * 2e-4, rel=1e-9)

    def test_cost_uses_flat_fallback_when_no_price_key(self) -> None:
        device = make_device(price_wh=99.0)
        state = device.create_batch_state(1, HORIZON)
        state.granted_energy_wh[0, 0] = 1000.0
        assert device.compute_cost(state)[0, 0] == pytest.approx(1000.0 * 1e-4, rel=1e-9)

    def test_cost_varies_with_time_of_use_price(self) -> None:
        prices = np.zeros(HORIZON)
        prices[2] = 3e-4
        prices[6] = 1e-4
        device = make_device_with_price(price_wh=prices)
        state = device.create_batch_state(1, HORIZON)
        state.granted_energy_wh[0, 2] = 100.0
        state.granted_energy_wh[0, 6] = 100.0
        assert device.compute_cost(state)[0, 0] == pytest.approx(100.0 * 3e-4 + 100.0 * 1e-4, rel=1e-9)

    def test_before_setup_run_raises(self) -> None:
        state = HomeApplianceBatchState(
            start_steps=np.zeros((1, 1)),
            schedule=np.zeros((1, HORIZON)),
            granted_energy_wh=np.zeros((1, HORIZON)),
            population_size=1,
            horizon=HORIZON,
            step_times=make_step_times(),
            num_remaining_cycles=1,
        )
        with pytest.raises(RuntimeError, match="setup_run"):
            HomeApplianceDevice(make_param(), 0, 0).compute_cost(state)


# ============================================================
# TestExtractInstructions
# ============================================================

class TestExtractInstructions:
    def _run(self, starts, param=None, horizon=HORIZON, pop=1):
        if param is None:
            param = make_param(duration_h=DURATION_H)
        device = make_device(param=param, horizon=horizon)
        state = device.create_batch_state(pop, horizon)
        genome = np.array([[float(s) for s in starts]] * pop)
        device.apply_genome_batch(state, genome)
        return device, state

    def test_one_instruction_per_step(self) -> None:
        device, state = self._run([0])
        assert len(device.extract_instructions(state, 0)) == HORIZON

    def test_steps_in_run_block_are_run(self) -> None:
        device, state = self._run([2])  # duration=2 -> steps 2,3
        instrs = device.extract_instructions(state, 0)
        assert instrs[2].operation_mode_id == ApplianceOperationMode.RUN.value
        assert instrs[3].operation_mode_id == ApplianceOperationMode.RUN.value

    def test_steps_outside_run_block_are_off(self) -> None:
        device, state = self._run([2])
        instrs = device.extract_instructions(state, 0)
        assert instrs[0].operation_mode_id == ApplianceOperationMode.OFF.value
        assert instrs[1].operation_mode_id == ApplianceOperationMode.OFF.value
        assert instrs[4].operation_mode_id == ApplianceOperationMode.OFF.value

    def test_execution_times_match_step_times(self) -> None:
        device, state = self._run([0])
        instrs = device.extract_instructions(state, 0)
        for t, dt in enumerate(make_step_times(HORIZON)):
            assert instrs[t].execution_time == dt

    def test_device_id_in_instructions(self) -> None:
        param = make_param(device_id="washer_01", duration_h=1)
        device, state = self._run([0], param=param)
        assert all(i.resource_id == "washer_01" for i in device.extract_instructions(state, 0))

    def test_individual_index_selects_correct_row(self) -> None:
        param = make_param(duration_h=1)
        device = make_device(param=param, horizon=HORIZON)
        state = device.create_batch_state(2, HORIZON)
        device.apply_genome_batch(state, np.array([[1.0], [5.0]]))
        i0 = device.extract_instructions(state, 0)
        i1 = device.extract_instructions(state, 1)
        assert i0[1].operation_mode_id == ApplianceOperationMode.RUN.value
        assert i0[5].operation_mode_id == ApplianceOperationMode.OFF.value
        assert i1[1].operation_mode_id == ApplianceOperationMode.OFF.value
        assert i1[5].operation_mode_id == ApplianceOperationMode.RUN.value

    def test_all_off_when_no_remaining_cycles(self) -> None:
        device = make_device(param=make_param(num_cycles=1), completed_cycles=1)
        state = device.create_batch_state(1, HORIZON)
        device.apply_genome_batch(state, np.zeros((1, 0)))
        assert all(i.operation_mode_id == ApplianceOperationMode.OFF.value for i in device.extract_instructions(state, 0))

    def test_multi_cycle_both_blocks_are_run(self) -> None:
        param = make_param(num_cycles=2, duration_h=1, min_cycle_gap_h=0)
        device = make_device(param=param, horizon=HORIZON)
        state = device.create_batch_state(1, HORIZON)
        device.apply_genome_batch(state, np.array([[1.0, 5.0]]))
        instrs = device.extract_instructions(state, 0)
        assert instrs[1].operation_mode_id == ApplianceOperationMode.RUN.value
        assert instrs[5].operation_mode_id == ApplianceOperationMode.RUN.value
        assert instrs[3].operation_mode_id == ApplianceOperationMode.OFF.value
