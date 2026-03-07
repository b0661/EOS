"""Vectorized home appliance device for the genetic2 simulation framework.

A ``HomeApplianceDevice`` models a shiftable, fixed-duration load (e.g.
dishwasher, washing machine, EV charging session) that may run one or
more times within the simulation horizon.

Genome encoding
---------------
One integer-valued gene per *remaining* cycle::

    genome shape: (population_size, num_remaining_cycles)
    genome[i, k] ∈ [0, horizon - 1]    start step for remaining cycle k

``num_remaining_cycles = num_cycles - completed_cycles`` is determined
during ``setup_run`` by reading the ``cycles_completed_measurement_key``
from the measurement store.  When all cycles are already done the device
returns ``None`` from ``genome_requirements`` and produces a zero
schedule for the rest of the horizon.

Completed-cycles measurement
-----------------------------
The measurement store is queried during ``setup_run`` via
``context.resolve_measurement(cycles_completed_measurement_key)``.  The
call uses ``key_to_value`` internally, which searches for the nearest
record to the simulation start time within the full horizon window —
equivalent to ``get_nearest_by_datetime`` without an explicit time
window constraint.

The returned value is cast to ``int`` and clamped to
``[0, num_cycles]``.  If no measurement exists yet (first run of the
day, device not yet started) the call returns ``None`` and the device
defaults to ``completed = 0``, planning all cycles as normal.

The ``completed`` count is consumed from the *front* of
``cycle_time_windows``: if 1 of 2 cycles is done, the first (earlier)
window is dropped and only the second is planned.  This is consistent
with the ordering guarantee enforced during repair — the earliest cycle
always runs first.

Repair pipeline
---------------
1. Round each gene to the nearest integer step.
2. Clip to ``[0, max_start]``.
3. Per-cycle window repair: snap to nearest allowed step for that cycle.
4. Sort starts so ``start[k] <= start[k+1]`` (ordering, removes genome
   symmetry).
5. Gap enforcement: push ``start[k+1]`` forward if needed so the gap
   between consecutive cycles satisfies ``min_cycle_gap_h``.

Steps 4–5 iterate over the (small) cycle axis but remain fully
vectorised over the population axis.

Sign convention
---------------
Positive power/energy → consuming from bus (load).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from akkudoktoreos.core.emplan import OMBCInstruction
from akkudoktoreos.devices.devicesabc import (
    ApplianceOperationMode,
    DeviceParam,
    EnergyDevice,
    EnergyPort,
)
from akkudoktoreos.optimization.genetic2.genome import GenomeSlice
from akkudoktoreos.simulation.genetic2.arbitrator import (
    DeviceGrant,
    DeviceRequest,
    PortRequest,
)
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import Date, DateTime, Duration, Time, TimeWindow, TimeWindowSequence, to_duration


# ============================================================
# Hashable time-window value type
# ============================================================


@dataclass(frozen=True, slots=True)
class CycleTimeWindow:
    """Hashable, comparable representation of a single ``TimeWindow``.

    ``TimeWindow`` is a mutable Pydantic ``BaseModel`` and cannot be stored
    in a ``frozen=True`` dataclass or used as a dictionary/set key.
    ``CycleTimeWindow`` captures the same scheduling data as plain Python
    primitives so that ``HomeApplianceParam`` remains a proper value object
    — two params with identical scheduling constraints compare as equal and
    hash identically.

    Attributes:
        start_time: Wall-clock start of the window as ``(hour, minute, second)``.
        duration_seconds: Window length in whole seconds.
        day_of_week: Optional weekday restriction (0 = Monday … 6 = Sunday).
        date: Optional specific calendar date restriction as ``(year, month, day)``.
    """

    start_time: tuple[int, int, int]        # (hour, minute, second)
    duration_seconds: int
    day_of_week: int | None = None
    date: tuple[int, int, int] | None = None  # (year, month, day)

    def to_time_window(self) -> TimeWindow:
        """Reconstruct a ``TimeWindow`` for use in ``_build_allowed_steps``."""
        import pendulum
        h, m, s = self.start_time
        start = Time(h, m, s)
        duration = to_duration(f"{self.duration_seconds} seconds")
        date = (
            pendulum.Date(*self.date) if self.date is not None else None
        )
        return TimeWindow(
            start_time=start,
            duration=duration,
            day_of_week=self.day_of_week,
            date=date,
        )

    @classmethod
    def from_time_window(cls, tw: TimeWindow) -> "CycleTimeWindow":
        """Convert a ``TimeWindow`` to a ``CycleTimeWindow``."""
        st = tw.start_time
        date = (tw.date.year, tw.date.month, tw.date.day) if tw.date is not None else None
        return cls(
            start_time=(st.hour, st.minute, st.second),
            duration_seconds=int(tw.duration.total_seconds()),
            day_of_week=tw.day_of_week if isinstance(tw.day_of_week, int) else None,
            date=date,
        )


def cycle_windows_from_sequence(
    seq: TimeWindowSequence | None,
) -> tuple[CycleTimeWindow, ...] | None:
    """Convert a ``TimeWindowSequence`` to a tuple of ``CycleTimeWindow``.

    Returns ``None`` when ``seq`` is ``None`` (meaning: any step is valid).
    Returns an empty tuple when the sequence has no windows (treated the same
    as ``None`` by ``_build_allowed_steps``).
    """
    if seq is None:
        return None
    return tuple(CycleTimeWindow.from_time_window(tw) for tw in seq)


# ============================================================
# Immutable parameter dataclass
# ============================================================


@dataclass(frozen=True, slots=True)
class HomeApplianceParam(DeviceParam):
    """Immutable parameters for a shiftable home appliance.

    Attributes:
        device_id: Unique identifier for this appliance instance.
        ports: Ports connecting the appliance to energy buses.
        consumption_wh: Total energy consumed per run cycle [Wh].
        duration_h: Run duration per cycle [h], must be >= 1.
        num_cycles: Total number of times the appliance must run within
            the horizon.  Must be >= 1.  Defaults to 1.
        min_cycle_gap_h: Minimum idle hours between the end of one cycle
            and the start of the next.  Must be >= 0.  Defaults to 0.
        price_forecast_key: Prediction store key for the electricity
            price series [EUR/Wh].  ``None`` → flat fallback of
            1e-4 EUR/Wh (≈ 0.10 EUR/kWh).
        cycles_completed_measurement_key: Measurement store key for the
            number of cycles already completed in the current day.
            Resolved via ``context.resolve_measurement`` in
            ``setup_run``.  Defaults to
            ``"{device_id}.cycles_completed"`` when ``None``.
            If no measurement is found the device assumes 0 completed
            cycles and plans all ``num_cycles`` runs normally.
        cycle_time_windows: Per-cycle allowed scheduling windows, length
            must equal ``num_cycles``.  Each entry is either ``None``
            (any step valid) or a tuple of ``CycleTimeWindow`` objects
            representing the allowed windows for that cycle.
            ``CycleTimeWindow`` is a hashable value type derived from
            ``TimeWindowSequence`` via ``cycle_windows_from_sequence``.
    """

    consumption_wh: float
    duration_h: int
    num_cycles: int = 1
    min_cycle_gap_h: int = 0
    price_forecast_key: str | None = None
    cycles_completed_measurement_key: str | None = None
    cycle_time_windows: tuple[tuple[CycleTimeWindow, ...] | None, ...] = ()

    def __post_init__(self) -> None:
        if self.consumption_wh <= 0:
            raise ValueError("consumption_wh must be > 0")
        if self.duration_h < 1:
            raise ValueError("duration_h must be >= 1")
        if self.num_cycles < 1:
            raise ValueError("num_cycles must be >= 1")
        if self.min_cycle_gap_h < 0:
            raise ValueError("min_cycle_gap_h must be >= 0")
        if not self.ports:
            raise ValueError("HomeApplianceParam requires at least one port")
        if self.cycle_time_windows and len(self.cycle_time_windows) != self.num_cycles:
            raise ValueError(
                f"cycle_time_windows has {len(self.cycle_time_windows)} entries "
                f"but num_cycles is {self.num_cycles}."
            )

    @property
    def effective_cycles_completed_key(self) -> str:
        """Measurement key to use, falling back to the device-scoped default."""
        return self.cycles_completed_measurement_key or f"{self.device_id}.cycles_completed"

    @property
    def resolved_cycle_time_windows(self) -> tuple[tuple[CycleTimeWindow, ...] | None, ...]:
        """Cycle windows, defaulting to ``(None,) * num_cycles`` when not set."""
        if self.cycle_time_windows:
            return self.cycle_time_windows
        return (None,) * self.num_cycles


# ============================================================
# Batch state
# ============================================================


@dataclass
class HomeApplianceBatchState:
    """Mutable batch state for one generation of ``HomeApplianceDevice``.

    Attributes:
        start_steps: Repaired start step indices for remaining cycles,
            shape ``(population_size, num_remaining_cycles)``.
            Integer-valued but stored as float for genome compatibility.
        schedule: Reconstructed power schedule,
            shape ``(population_size, horizon)`` [W].
        granted_energy_wh: Energy granted by the arbitrator,
            shape ``(population_size, horizon)`` [Wh].
        population_size: Number of individuals.
        horizon: Number of simulation time steps.
        step_times: Ordered ``DateTime`` timestamps, length ``horizon``.
        num_remaining_cycles: Number of cycles still to be planned.
            May be less than ``HomeApplianceParam.num_cycles`` when the
            measurement store reports completed cycles.
    """

    start_steps: np.ndarray        # (population_size, num_remaining_cycles)
    schedule: np.ndarray           # (population_size, horizon)  [W]
    granted_energy_wh: np.ndarray  # (population_size, horizon)  [Wh]
    population_size: int
    horizon: int
    step_times: tuple[DateTime, ...]
    num_remaining_cycles: int


# ============================================================
# Device implementation
# ============================================================


class HomeApplianceDevice(EnergyDevice):
    """Vectorized multi-cycle shiftable home appliance for the genetic2 framework.

    Parameters
    ----------
    param:
        Immutable device parameters, including ``cycle_time_windows``.
    device_index:
        Index used by the bus arbitrator to identify this device's port
        request and match it to its grant. Must be unique across all
        devices registered with the same engine.
    port_index:
        Index of this device's AC port in the arbitrator's port-to-bus
        topology array.
    """

    def __init__(
        self,
        param: HomeApplianceParam,
        device_index: int,
        port_index: int,
    ) -> None:
        super().__init__()
        self._param = param
        self.device_id: str = param.device_id
        self._device_index = device_index
        self._port_index = port_index

        # Set during setup_run
        self._step_times: tuple[DateTime, ...] | None = None
        self._horizon: int | None = None
        self._step_interval_h: float | None = None
        self._power_per_step_w: float | None = None
        self._num_remaining_cycles: int | None = None
        # Shape: (num_remaining_cycles, horizon)
        self._allowed_steps: np.ndarray | None = None
        self._price_eur_per_wh: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Structure Phase
    # ------------------------------------------------------------------

    def setup_run(self, context: SimulationContext) -> None:
        """Cache run-scoped quantities, read completed-cycle measurement,
        and build per-cycle feasibility masks for remaining cycles only.

        The completed-cycle count is read from the measurement store via
        ``context.resolve_measurement``.  This calls ``key_to_value``
        on the measurement store, which uses ``get_nearest_by_datetime``
        internally to find the record nearest to the simulation start
        within the full horizon window.  If no record exists yet
        (``None`` returned) the device defaults to 0 completed cycles.

        Args:
            context: Immutable run context.
        """
        self._step_times = context.step_times
        self._horizon = context.horizon
        self._step_interval_h = context.step_interval.total_seconds() / 3600.0
        self._power_per_step_w = self._param.consumption_wh / float(self._param.duration_h)

        # --- Resolve completed cycles from measurement store ---
        completed = self._resolve_completed_cycles(context)
        self._num_remaining_cycles = self._param.num_cycles - completed

        # --- Build feasibility masks for remaining cycles only ---
        # Drop the first `completed` windows — they correspond to the
        # cycles that have already run (cycles are always time-ordered).
        remaining_windows = list(self._param.resolved_cycle_time_windows[completed:])
        self._allowed_steps = self._build_allowed_steps(context, remaining_windows)

        self._price_eur_per_wh = self._resolve_price(context)

    def _resolve_completed_cycles(self, context: SimulationContext) -> int:
        """Read the completed-cycle count from the measurement store.

        Uses ``context.resolve_measurement``, which calls
        ``key_to_value(key, target_datetime=step_times[0],
        time_window=step_interval * horizon)`` on the measurement store.
        This is equivalent to ``get_nearest_by_datetime`` bounded to the
        simulation horizon window.

        Returns:
            Number of completed cycles, clamped to ``[0, num_cycles]``.
            Defaults to 0 if no measurement is found.
        """
        key = self._param.effective_cycles_completed_key
        try:
            raw = context.resolve_measurement(key)
        except KeyError:
            # Key not registered in the measurement store at all
            logger.debug(
                "HomeApplianceDevice '{}': measurement key '{}' not found, "
                "assuming 0 completed cycles.",
                self.device_id,
                key,
            )
            return 0

        if raw is None:
            # Key exists but no record near the simulation start yet
            logger.debug(
                "HomeApplianceDevice '{}': no measurement for '{}' near {}, "
                "assuming 0 completed cycles.",
                self.device_id,
                key,
                context.step_times[0],
            )
            return 0

        completed = int(raw)
        clamped = max(0, min(completed, self._param.num_cycles))
        if clamped != completed:
            logger.warning(
                "HomeApplianceDevice '{}': completed cycles {} out of range "
                "[0, {}], clamped to {}.",
                self.device_id,
                completed,
                self._param.num_cycles,
                clamped,
            )
        return clamped

    def _build_allowed_steps(
        self,
        context: SimulationContext,
        windows: list[tuple[CycleTimeWindow, ...] | None],
    ) -> np.ndarray:
        """Build a ``(num_remaining_cycles, horizon)`` boolean feasibility array.

        ``allowed[k, t]`` is ``True`` when remaining cycle ``k`` may
        start at step ``t`` — i.e. the full block ``[t, t+duration_h)``
        lies within at least one of cycle ``k``'s time windows.

        Each entry in ``windows`` is either ``None`` (any step valid) or a
        tuple of ``CycleTimeWindow`` objects that are reconstructed into
        ``TimeWindow`` instances for the ``contains`` check.

        Args:
            context: Run context providing step times.
            windows: Per-remaining-cycle time windows.

        Returns:
            Boolean array of shape ``(len(windows), horizon)``.
        """
        horizon = context.horizon
        num_remaining = len(windows)
        duration_steps = self._param.duration_h
        duration = to_duration(f"{duration_steps} hours")

        allowed = np.zeros((num_remaining, horizon), dtype=bool)
        for k, cycle_windows in enumerate(windows):
            if cycle_windows is None:
                allowed[k, : horizon - duration_steps + 1] = True
            else:
                time_windows = [cw.to_time_window() for cw in cycle_windows]
                for t in range(horizon - duration_steps + 1):
                    allowed[k, t] = any(
                        tw.contains(context.step_times[t], duration=duration)
                        for tw in time_windows
                    )
        return allowed

    def _resolve_price(self, context: SimulationContext) -> np.ndarray:
        """Resolve the electricity price forecast or return a flat fallback.

        Returns:
            Price array of shape ``(horizon,)`` [EUR/Wh].
        """
        if self._param.price_forecast_key is not None:
            try:
                raw = context.resolve_prediction(self._param.price_forecast_key)
                return raw[: context.horizon].astype(float)
            except (KeyError, ValueError):
                pass
        return np.full(context.horizon, 1e-4)

    def genome_requirements(self) -> GenomeSlice | None:
        """Declare one start-step gene per *remaining* cycle.

        Returns ``None`` when all cycles are already completed so the
        genome assembler allocates no genes for this device.

        Returns:
            ``GenomeSlice`` of size ``num_remaining_cycles``, or ``None``
            if ``num_remaining_cycles == 0``.

        Raises:
            RuntimeError: If called before ``setup_run``.
        """
        if self._num_remaining_cycles is None or self._horizon is None:
            raise RuntimeError("Call setup_run() before genome_requirements().")

        if self._num_remaining_cycles == 0:
            return None

        max_start = float(max(0, self._horizon - self._param.duration_h))
        n = self._num_remaining_cycles

        return GenomeSlice(
            start=0,  # re-indexed by GenomeAssembler
            size=n,
            lower_bound=np.zeros(n),
            upper_bound=np.full(n, max_start),
        )

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return self._param.ports

    @property
    def objective_names(self) -> list[str]:
        return ["energy_cost_eur"]

    # ------------------------------------------------------------------
    # Batch Lifecycle
    # ------------------------------------------------------------------

    def create_batch_state(
        self,
        population_size: int,
        horizon: int,
    ) -> HomeApplianceBatchState:
        """Allocate zeroed batch state for one generation."""
        if self._step_times is None or self._num_remaining_cycles is None:
            raise RuntimeError("Call setup_run() before create_batch_state().")
        return HomeApplianceBatchState(
            start_steps=np.zeros((population_size, self._num_remaining_cycles)),
            schedule=np.zeros((population_size, horizon)),
            granted_energy_wh=np.zeros((population_size, horizon)),
            population_size=population_size,
            horizon=horizon,
            step_times=self._step_times,
            num_remaining_cycles=self._num_remaining_cycles,
        )

    def apply_genome_batch(
        self,
        state: HomeApplianceBatchState,
        genome_batch: np.ndarray,
    ) -> np.ndarray:
        """Decode, repair, and expand start-step genes into a schedule.

        No-ops immediately when ``num_remaining_cycles == 0`` (all
        cycles done), leaving the schedule as all zeros.

        Repair pipeline:
            1. Round and clip to ``[0, max_start]``.
            2. Per-cycle window repair (snap to nearest allowed step).
            3. Sort starts across cycles (ordering, removes symmetry).
            4. Gap enforcement (push forward to satisfy min_cycle_gap_h).
            5. Reconstruct flat-block power schedule.
            6. Lamarckian write-back into ``genome_batch``.

        Args:
            state: Device batch state (modified in-place).
            genome_batch: Shape ``(population_size, num_remaining_cycles)``.

        Returns:
            Repaired ``genome_batch``, same shape.
        """
        if self._allowed_steps is None or self._horizon is None:
            raise RuntimeError("Call setup_run() before apply_genome_batch().")

        num_remaining = state.num_remaining_cycles
        if num_remaining == 0:
            state.schedule[:] = 0.0
            return genome_batch

        population_size = state.population_size
        horizon = state.horizon
        duration_steps = self._param.duration_h
        gap_steps = self._param.min_cycle_gap_h
        power_w = self._power_per_step_w
        max_start = max(0, horizon - duration_steps)

        # --- Steps 1 & 2: round, clip, per-cycle window repair ---
        starts = np.clip(np.rint(genome_batch).astype(int), 0, max_start)
        for k in range(num_remaining):
            starts[:, k] = self._repair_cycle_to_allowed(starts[:, k], k, max_start)

        # --- Step 3: sort so cycles are time-ordered ---
        starts = np.sort(starts, axis=1)

        # --- Step 4: gap enforcement (causal, sequential over cycle axis) ---
        min_next_start = duration_steps + gap_steps
        for k in range(1, num_remaining):
            earliest = np.minimum(starts[:, k - 1] + min_next_start, max_start)
            starts[:, k] = np.maximum(starts[:, k], earliest)

        state.start_steps[:] = starts.astype(float)

        # --- Step 5: reconstruct schedule ---
        state.schedule[:] = 0.0
        if power_w is not None and power_w > 0:
            d_idx = np.arange(duration_steps)
            row_idx = np.arange(population_size)[:, np.newaxis]
            for k in range(num_remaining):
                col_idx = starts[:, k, np.newaxis] + d_idx[np.newaxis, :]
                col_idx = np.clip(col_idx, 0, horizon - 1)
                state.schedule[row_idx, col_idx] += power_w

        # --- Step 6: Lamarckian write-back ---
        genome_batch[:] = starts.astype(float)
        return genome_batch

    def _repair_cycle_to_allowed(
        self,
        starts_k: np.ndarray,
        cycle_index: int,
        max_start: int,
    ) -> np.ndarray:
        """Snap each individual's start for remaining cycle ``k`` to the
        nearest allowed step for that cycle's window.

        Args:
            starts_k: Integer start steps, shape ``(population_size,)``.
            cycle_index: Index into ``_allowed_steps`` rows.
            max_start: Maximum feasible start step (inclusive).

        Returns:
            Repaired start steps, same shape.
        """
        allowed_row = self._allowed_steps[cycle_index]
        allowed_indices = np.where(allowed_row)[0]

        if len(allowed_indices) == 0:
            return np.full_like(starts_k, max_start)

        repaired = starts_k.copy()
        is_disallowed = ~allowed_row[np.clip(starts_k, 0, len(allowed_row) - 1)]
        if not np.any(is_disallowed):
            return repaired

        dist = np.abs(starts_k[is_disallowed, np.newaxis] - allowed_indices[np.newaxis, :])
        repaired[is_disallowed] = allowed_indices[np.argmin(dist, axis=1)]
        return repaired

    # ------------------------------------------------------------------
    # Arbitration Phase
    # ------------------------------------------------------------------

    def build_device_request(
        self, state: HomeApplianceBatchState
    ) -> DeviceRequest | None:
        """Build an AC-sink energy request from the repaired schedule."""
        if self._step_interval_h is None:
            raise RuntimeError("Call setup_run() before build_device_request().")

        energy_wh = state.schedule * self._step_interval_h
        port_request = PortRequest(
            port_index=self._port_index,
            energy_wh=energy_wh,
            min_energy_wh=np.zeros_like(energy_wh),
            is_slack=False,
        )
        return DeviceRequest(
            device_index=self._device_index,
            port_requests=(port_request,),
        )

    def apply_device_grant(
        self,
        state: HomeApplianceBatchState,
        grant: DeviceGrant,
    ) -> None:
        """Store the arbitrated energy grant for cost evaluation."""
        if grant.port_grants:
            state.granted_energy_wh[:] = grant.port_grants[0].granted_wh
        else:
            state.granted_energy_wh[:] = 0.0

    # ------------------------------------------------------------------
    # Cost Evaluation
    # ------------------------------------------------------------------

    def compute_cost(self, state: HomeApplianceBatchState) -> np.ndarray:
        """Compute electricity cost across all remaining cycles.

        ``cost[i] = sum_t( granted_wh[i, t] × price_eur_per_wh[t] )``

        Returns:
            Shape ``(population_size, 1)`` [EUR].
        """
        if self._price_eur_per_wh is None:
            raise RuntimeError("Call setup_run() before compute_cost().")
        cost = state.granted_energy_wh @ self._price_eur_per_wh
        return cost[:, np.newaxis]

    # ------------------------------------------------------------------
    # Instruction Extraction
    # ------------------------------------------------------------------

    def extract_instructions(
        self,
        state: HomeApplianceBatchState,
        individual_index: int,
    ) -> list[OMBCInstruction]:
        """Extract per-step control instructions for one individual.

        Steps covered by a remaining cycle block => RUN.
        All other steps => OFF.

        Args:
            state: Batch state produced by apply_genome_batch.
            individual_index: Row index into state.start_steps.

        Returns:
            Ordered list of OMBCInstruction, one per horizon step.
        """
        duration = self._param.duration_h
        num_remaining = state.num_remaining_cycles
        starts = state.start_steps[individual_index].astype(int)

        active_steps: set[int] = set()
        for k in range(num_remaining):
            for d in range(duration):
                active_steps.add(int(starts[k]) + d)

        return [
            OMBCInstruction(
                id=None,  # Auto-generate
                resource_id=self.device_id,
                execution_time=step_time,
                abnormal_condition=False,
                operation_mode_id=(
                    str(ApplianceOperationMode.RUN)
                    if t in active_steps
                    else str(ApplianceOperationMode.OFF)
                ),
                operation_mode_factor=0.0  # Ignored but required
            )
            for t, step_time in enumerate(state.step_times)
        ]
