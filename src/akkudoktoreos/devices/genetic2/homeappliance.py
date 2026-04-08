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

Repair pipeline
---------------

1. Round each gene to the nearest integer step.
2. Clip to ``[0, max_start]``.
3. Window repair: snap to nearest allowed step.
4. Sort starts so ``start[k] <= start[k+1]`` (ordering, removes genome
   symmetry).
5. Gap enforcement: push ``start[k+1]`` forward if needed so the gap
   between consecutive cycles satisfies ``min_cycle_gap_h``.

Steps 4–5 iterate over the (small) cycle axis but remain fully
vectorised over the population axis.

Sign convention
---------------

Positive power/energy → consuming from bus (load).

Window scheduling
-----------------

The allowed start steps are determined from a ``TimeWindowSequence``
stored in the EOS config tree.  ``HomeApplianceParam`` holds a plain
``time_window_key: str | None`` — a ``/``-separated config path such as
``"devices/home_appliances/0/time_windows"`` — which ``setup_run``
resolves once via ``context.resolve_config_time_window_sequence(key)``.

This keeps ``HomeApplianceParam`` fully hashable (no mutable Pydantic
objects) and the device constructor uniform with all other devices::

    device = HomeApplianceDevice(param, device_index, port_index)

All cycles share the same window mask: the resolved array encodes *when
the appliance may run*, not how many times.  Cycle counting, gaps, and
ordering are handled entirely by the repair pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from akkudoktoreos.core.emplan import EnergyManagementInstruction, OMBCInstruction
from akkudoktoreos.devices.devicesabc import (
    ApplianceOperationMode,
    DeviceParam,
    EnergyDevice,
    EnergyPort,
    InstructionContext,
)
from akkudoktoreos.optimization.genetic2.genome import GenomeSlice
from akkudoktoreos.simulation.genetic2.arbitrator import (
    DeviceGrant,
    DeviceRequest,
    PortRequest,
)
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import DateTime

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
        time_window_key: ``/``-separated path into the EOS config tree
            pointing to the ``TimeWindowSequence`` that constrains when
            the appliance may start, e.g.
            ``"devices/home_appliances/0/time_windows"``.
            Resolved once during ``setup_run`` via
            ``context.resolve_config_time_window_sequence``.
            ``None`` means any step within the horizon is valid.
    """

    consumption_wh: float
    duration_h: int
    num_cycles: int = 1
    min_cycle_gap_h: int = 0
    price_forecast_key: str | None = None
    cycles_completed_measurement_key: str | None = None
    time_window_key: str | None = None

    def __post_init__(self) -> None:
        if self.consumption_wh <= 0:
            raise ValueError(f"{self.device_id}: consumption_wh must be > 0")
        if self.duration_h < 1:
            raise ValueError(f"{self.device_id}: duration_h must be >= 1")
        if self.num_cycles < 1:
            raise ValueError(f"{self.device_id}: num_cycles '{self.num_cycles}' must be >= 1")
        if self.min_cycle_gap_h < 0:
            raise ValueError(
                f"{self.device_id}: min_cycle_gap_h '{self.min_cycle_gap_h}' must be >= 0"
            )
        if not self.ports:
            raise ValueError(f"{self.device_id}: HomeApplianceParam requires at least one port")

    @property
    def effective_cycles_completed_key(self) -> str:
        """Measurement key to use, falling back to the device-scoped default."""
        return self.cycles_completed_measurement_key or f"{self.device_id}.cycles_completed"


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

    start_steps: np.ndarray  # (population_size, num_remaining_cycles)
    schedule: np.ndarray  # (population_size, horizon)  [W]
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
        Immutable device parameters.  ``param.time_window_key`` optionally
        points to a ``TimeWindowSequence`` in the EOS config tree that
        constrains when each cycle may start.
    device_index:
        Position of this device in the shared device list (used by the
        arbitrator to route grants).
    port_index:
        Index of the port within this device's port tuple to use for the
        energy request.
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
    # Topology
    # ------------------------------------------------------------------

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return self._param.ports

    @property
    def objective_names(self) -> list[str]:
        return ["energy_cost_amt"]

    # ------------------------------------------------------------------
    # Structure Phase
    # ------------------------------------------------------------------

    def setup_run(self, context: SimulationContext) -> None:
        """Setup for simulation run.

        Cache run-scoped quantities, read completed-cycle measurement,
        and build the feasibility mask from the config time-window sequence.

        The completed-cycle count is read from the measurement store via
        ``context.resolve_measurement``.  If no record exists yet
        (``None`` returned) the device defaults to 0 completed cycles.

        The window feasibility mask is resolved once via
        ``context.resolve_config_time_window_sequence`` when
        ``param.time_window_key`` is set, or marks all start steps as
        allowed when it is ``None``.

        Args:
            context: Immutable run context.
        """
        self._step_times = context.step_times
        self._horizon = context.horizon
        self._step_interval_h = context.step_interval.total_seconds() / 3600.0
        self._power_per_step_w = self._param.consumption_wh / float(self._param.duration_h)

        completed = self._resolve_completed_cycles(context)
        self._num_remaining_cycles = self._param.num_cycles - completed

        self._allowed_steps = self._build_allowed_steps(context)
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

    def _build_allowed_steps(self, context: SimulationContext) -> np.ndarray:
        """Build a ``(num_remaining_cycles, horizon)`` boolean feasibility array.

        ``allowed[k, t]`` is ``True`` when remaining cycle ``k`` may start at
        step ``t`` — i.e. step ``t`` falls inside a window belonging to that
        cycle's index and the full block ``[t, t+duration_steps)`` is
        feasible.

        When ``param.time_window_key`` is ``None`` all start positions up to
        ``horizon - duration_steps`` are allowed for every remaining cycle and
        the config is not queried.

        When windows are provided, ``context.resolve_config_cycle_time_windows``
        returns a ``(cycle_indices, raw_matrix)`` pair:

        * ``cycle_indices`` — sorted list of cycle numbers defined in the
          sequence (e.g. ``[0, 1, 2]``).
        * ``raw_matrix`` — shape ``(num_defined_cycles, horizon)`` with
          ``1.0`` when a step is inside that cycle's window, ``0.0`` outside.

        The block-feasibility check (whether a full ``duration_h``-step run
        fits) is applied per cycle row via a sliding-sum convolution, then
        the rows for *remaining* cycles (those not yet completed) are selected
        to produce the final ``(num_remaining, horizon)`` output.

        If ``num_remaining_cycles`` exceeds the number of cycles defined in the
        window sequence the extra rows are marked fully allowed (unconstrained),
        matching the behaviour of a ``None`` entry in the old list-based API.

        Args:
            context: Run context providing config resolution.

        Returns:
            Boolean array of shape ``(num_remaining_cycles, horizon)``.
        """
        horizon = context.horizon
        duration_steps = self._param.duration_h
        num_cycles = self._param.num_cycles
        num_remaining = self._num_remaining_cycles
        if num_remaining is None or num_cycles is None:
            raise RuntimeError("Call setup_run() before _build_allowed_steps().")
        completed = num_cycles - num_remaining

        if self._param.time_window_key is None or num_remaining == 0:
            # Unconstrained: any start step up to max_start is valid.
            allowed = np.zeros((num_remaining, horizon), dtype=bool)
            allowed[:, : horizon - duration_steps + 1] = True
            return allowed

        cycle_indices, raw_matrix = context.resolve_config_cycle_time_windows(
            self._param.time_window_key
        )

        # Map cycle index → row in raw_matrix for fast lookup.
        index_to_row: dict[int, int] = {c: i for i, c in enumerate(cycle_indices)}

        # Build sliding-sum feasibility for each defined cycle row.
        # feasible_matrix[i, t] = True iff the block [t, t+duration_steps) lies
        # entirely within windows for cycle cycle_indices[i].
        if duration_steps == 1:
            feasible_matrix = raw_matrix == 1.0
        else:
            kernel = np.ones(duration_steps)
            feasible_matrix = np.zeros_like(raw_matrix, dtype=bool)
            for i, row in enumerate(raw_matrix):
                block_sum = np.convolve(row, kernel, mode="full")[:horizon]
                feasible_matrix[i, : horizon - duration_steps + 1] = block_sum[
                    : horizon - duration_steps + 1
                ] == float(duration_steps)

        # Build output: one row per remaining cycle (completed cycles are skipped).
        # Cycles are ordered by index; the first `completed` are already done.
        all_cycle_indices_sorted = sorted(set(range(num_cycles)) | set(cycle_indices))
        remaining_cycle_indices = all_cycle_indices_sorted[completed:]

        allowed = np.zeros((num_remaining, horizon), dtype=bool)
        for k, cycle_idx in enumerate(remaining_cycle_indices[:num_remaining]):
            if cycle_idx in index_to_row:
                allowed[k] = feasible_matrix[index_to_row[cycle_idx]]
            else:
                # No window defined for this cycle → unconstrained
                allowed[k, : horizon - duration_steps + 1] = True

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
        """Snap each individual's start to the nearest allowed step.

        Sanps each individual's start for remaining cycle ``k`` to the
        nearest allowed step for that cycle's window.

        Args:
            starts_k: Integer start steps, shape ``(population_size,)``.
            cycle_index: Index into ``_allowed_steps`` rows.
            max_start: Maximum feasible start step (inclusive).

        Returns:
            Repaired start steps, same shape.
        """
        allowed_steps = self._allowed_steps
        if allowed_steps is None:
            raise RuntimeError("Call setup_run() before _repair_cycle_to_allowed().")
        allowed_row = allowed_steps[cycle_index]
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

    def build_device_request(self, state: HomeApplianceBatchState) -> DeviceRequest | None:
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
        instruction_context: InstructionContext | None = None,
    ) -> list[EnergyManagementInstruction]:
        """Extract S2 instructions for the best individual.

        Steps covered by a remaining cycle block → ``RUN``.
        All other steps → ``OFF``.

        Args:
            state: Device batch state produced by ``create_batch_state``
                and populated by the full simulation pipeline
                (``apply_genome_batch`` → arbitration → ``apply_device_grant``).
            individual_index: Row index into the batch. Pass ``0`` when the
                state was produced from a single-individual population (the
                normal case when called from
                ``GeneticOptimizer.extract_best_instructions``).
            instruction_context: Optional shared cross-device context
                populated by ``GeneticOptimizer.extract_best_instructions``.
                Provides the arbitrated grid exchange (``grid_granted_wh``)
                and the step interval so devices can select firmware-autonomous
                modes (e.g. ``SELF_CONSUMPTION``) when the optimizer already
                found a near-zero-export equilibrium at a given step.
                Devices that do not need cross-device context may ignore this
                argument; it defaults to ``None`` for backward compatibility.

        Returns:
            Ordered list of ``EnergyManagementInstruction``, one per step.
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
                resource_id=self.device_id,
                execution_time=step_time,
                operation_mode_id=(
                    ApplianceOperationMode.RUN.value
                    if t in active_steps
                    else ApplianceOperationMode.OFF.value
                ),
                operation_mode_factor=1.0,
            )
            for t, step_time in enumerate(state.step_times)
        ]
