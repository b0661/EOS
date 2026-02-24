"""Abstract base classes for simulation devices.

Every physical device in the simulation inherits from ``EnergyDevice`` and
implements the step-based protocol:

1. ``genome_requirements()`` — declare optimizer genome needs before any run.
2. ``apply_genome()``        — receive and interpret genome slice once per run.
3. ``request()``             — declare resource needs for a specific step (Phase 1 per step).
4. ``simulate_step()``       — simulate given granted resources (Phase 2 per step).
5. ``repair_genome()``       — optionally propose genome corrections after a run.

Devices own their own physics and internal state. The engine and optimizer
never access device internals directly.

Step-based design
-----------------
The simulation operates on uniform time steps of arbitrary duration
(e.g. 15 minutes, 1 hour). ``apply_genome()`` receives the datetime for
each step so devices can precompute time-dependent schedules. The genome
size is determined by ``num_steps`` and ``step_interval`` at setup time,
allowing the same device class to operate at different resolutions.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from akkudoktoreos.simulation.genetic2.flows import (
    EnergyFlows,
    ResourceGrant,
    ResourceRequest,
)

# ---------------------------------------------------------------------------
# GenomeSlice — device genome contract
# ---------------------------------------------------------------------------


@dataclass
class GenomeSlice:
    """Declares what portion of the optimizer genome a device requires.

    The ``GenomeAssembler`` collects one ``GenomeSlice`` from every registered
    device and assigns contiguous, non-overlapping index ranges in the flat
    genome array.  Bounds declared here are enforced by ``validate()`` both at
    dispatch time and when accepting ``GenomeRepairResult`` proposals.

    Attributes:
        device_id (str): ID of the device making this declaration.
        size (int): Number of genome slots required.  Must be > 0.
            Typically equal to the number of simulation steps, but a device
            may request fewer slots (e.g. a single start-time integer).
        dtype (type): Element type — ``int`` for discrete states, ``float``
            for continuous values.  Used by ``GenomeAssembler.random_genome()``
            to pick the correct random distribution.
        low (float): Inclusive lower bound for every genome slot.
        high (float): Inclusive upper bound for every genome slot.
        description (str): Human-readable explanation of the encoding, e.g.
            ``"0=idle, 1..N=charge factor index, N+1..2N=discharge factor index"``.
            Shown by ``GenomeAssembler.describe()`` for debugging.
    """

    device_id: str
    size: int
    dtype: type = int
    low: float = 0.0
    high: float = 1.0
    description: str = ""

    def validate(self, genome_slice: np.ndarray) -> None:
        """Validate that a genome slice conforms to this declaration.

        Checks both length and per-element bounds.  Called by
        ``GenomeAssembler.dispatch()`` before forwarding slices to devices,
        and by the engine before accepting ``GenomeRepairResult`` proposals.

        Args:
            genome_slice (np.ndarray): Slice to validate.

        Raises:
            ValueError: If ``len(genome_slice) != self.size``.
            ValueError: If any element is outside ``[self.low, self.high]``.
        """
        if len(genome_slice) != self.size:
            raise ValueError(
                f"Device '{self.device_id}': genome slice has {len(genome_slice)} "
                f"elements, expected {self.size}."
            )
        if np.any(genome_slice < self.low) or np.any(genome_slice > self.high):
            raise ValueError(
                f"Device '{self.device_id}': genome values must be in "
                f"[{self.low}, {self.high}], got "
                f"min={genome_slice.min()}, max={genome_slice.max()}."
            )


# ---------------------------------------------------------------------------
# GenomeRepairResult — post-simulation genome correction proposal
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GenomeRepairResult:
    """A device's proposal to correct its genome slice after a simulation run.

    Produced by ``EnergyDevice.repair_genome()`` *after* a complete simulation.
    The simulation that just finished used the *original* genome slice supplied
    by ``apply_genome()``.  The repaired slice returned here was **not** used
    during that run — it is a suggestion for the genetic engine to adopt in
    subsequent generations.

    The engine is the sole owner of the full genome and decides whether and
    how to integrate any proposal.  It **must** call
    ``GenomeSlice.validate()`` on ``repaired_slice`` before accepting it, to
    guard against a device returning out-of-bounds values.  It **must** copy
    the array (``repaired_slice.copy()``) before storing, because
    ``np.ndarray`` is mutable and the device may reuse its internal buffer.

    Attributes:
        repaired_slice (np.ndarray):
            Corrected genome slice.  Must satisfy all of the following:

            * ``len(repaired_slice) == genome_requirements().size`` —
              same length as the original declaration.
            * All values within ``[GenomeSlice.low, GenomeSlice.high]`` —
              within declared bounds.
            * Same dtype semantics as the original slice.

            The engine calls ``GenomeSlice.validate()`` on this array before
            accepting it; invalid slices are discarded with a warning rather
            than raising, to avoid a single misbehaving device aborting an
            entire optimization run.

        changed (bool):
            ``True`` if the device actually modified the slice relative to what
            was applied.  The engine **may** skip further processing when
            ``False``, but must not assume that ``changed=False`` means the
            original slice was feasible — the device may simply not implement
            repair.

    Note:
        Partial repairs (correcting only a subset of steps) are not currently
        expressible.  A device that needs to fix only steps 3–7 must still
        return the full slice with all other elements unchanged.  This is a
        known limitation; a future ``changed_indices`` field may address it.
    """

    repaired_slice: np.ndarray
    changed: bool = True


# ---------------------------------------------------------------------------
# EnergyDevice — abstract base for all simulation devices
# ---------------------------------------------------------------------------


class EnergyDevice(ABC):
    """Base class for all simulation devices.

    Each device is self-contained: it declares what genome slots it needs,
    interprets those slots internally, and exposes only standardised
    ``EnergyFlows`` to the engine.  The engine and optimizer never access
    device internals directly.

    Simulation lifecycle per run
    ----------------------------
    1. ``reset()`` — restore device to initial physical state.
    2. ``apply_genome(slice, step_times)`` — decode genome into schedule.
       Called once after ``reset()``, before any ``simulate_step()`` call.
    3. For each simulation step *i*:

       a. ``request(i)`` — declare resource needs.
       b. ``simulate_step(grant)`` — act on the arbitrated grant, update state.

    4. ``repair_genome()`` — optionally propose a corrected genome slice.

    reset() vs apply_genome() ordering
    -----------------------------------
    ``reset()`` restores *physical* state only (e.g. SoC, temperatures).
    It must **not** zero out decoded schedule arrays, because ``apply_genome()``
    is called immediately after ``reset()`` at the start of each run and will
    re-populate the schedules.  Clearing schedules in ``reset()`` is harmless
    when ``apply_genome()`` always follows — but if ``reset()`` is ever called
    mid-run (e.g. for a partial re-simulation), it would corrupt the schedule.
    Keep ``reset()`` strictly physical.

    Args:
        device_id (str): Unique identifier for this device instance.
            Must be unique within a ``DeviceRegistry``.
    """

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id

    # ------------------------------------------------------------------
    # Genome protocol
    # ------------------------------------------------------------------

    @abstractmethod
    def genome_requirements(
        self,
        num_steps: int,
        step_interval: timedelta,
    ) -> GenomeSlice | None:
        """Declare what portion of the optimizer genome this device needs.

        Called once during optimizer setup so the ``GenomeAssembler`` can
        assign non-overlapping genome slices to every device.  The genome
        size may depend on ``num_steps`` and ``step_interval`` — a device
        operating at 15-minute resolution over 24 hours needs 96 slots,
        while the same device at hourly resolution needs only 24.

        Args:
            num_steps (int): Total number of simulation steps in this run.
            step_interval (timedelta): Duration of one simulation step.
                Use this to calculate energy quantities correctly when the
                step duration differs from one hour (e.g. multiply power
                in W by ``step_interval.total_seconds() / 3600`` to get Wh).

        Returns:
            GenomeSlice: Shape and bounds of the required genome slice.
            None: If this device has no optimisable parameters (e.g. a
                fixed-forecast PV array whose output cannot be influenced
                by the optimizer).
        """

    @abstractmethod
    def apply_genome(
        self,
        genome_slice: np.ndarray,
        step_times: Sequence[datetime],
    ) -> None:
        """Decode the genome slice and precompute the per-step schedule.

        Called once per simulation run, after ``reset()`` and before the
        first ``simulate_step()`` call.  The device must store enough
        information internally to answer all subsequent ``request()`` calls.

        Length guarantees
        -----------------
        * ``len(genome_slice) == genome_requirements(num_steps, step_interval).size``
          when ``genome_requirements()`` returns a non-None slice.
          ``genome_slice`` will be empty (length 0) for devices that return
          ``None`` from ``genome_requirements()``.
        * ``len(step_times) == num_steps`` — always the *total* simulation
          length, regardless of genome size.  A device whose genome covers
          only an optimisation sub-horizon (e.g. the first 24 steps of a
          48-step prediction) will still receive all 48 datetimes.

        Args:
            genome_slice (np.ndarray): This device's genome slice.
                Empty (length 0) when ``genome_requirements()`` returns ``None``.
            step_times (Sequence[datetime]): Datetime of the *start* of each
                simulation step.  ``step_times[i]`` is the wall-clock time at
                step *i*.  Use these for time-of-day scheduling (e.g. only
                charge between 22:00 and 06:00).
        """

    def repair_genome(self) -> GenomeRepairResult | None:
        """Propose a corrected genome slice after a completed simulation run.

        Called once after the simulation finishes, before the fitness score
        is recorded.  The device may inspect its internal state (e.g. final
        SoC, curtailment count, constraint violations) and return a corrected
        slice for the genetic engine to consider for subsequent generations.

        Engine obligations when receiving a repair proposal
        ---------------------------------------------------
        1. Call ``genome_requirements().validate(result.repaired_slice)``
           before accepting.  Discard invalid proposals with a logged warning
           rather than raising — a misbehaving device must not abort the run.
        2. Copy the array: ``safe = result.repaired_slice.copy()``.
           ``np.ndarray`` is mutable; the device may reuse its internal buffer
           on the next call.
        3. Treat ``changed=False`` as an optimisation hint, not a correctness
           guarantee.  The device may return ``changed=False`` simply because
           it does not implement repair.

        Known limitation
        ----------------
        Partial repairs are not expressible.  A device fixing only steps 3–7
        must return the full slice with all other elements unchanged.  A future
        ``changed_indices`` field may address this.

        Returns:
            GenomeRepairResult: Proposed replacement slice (copied defensively
                by the engine) and a ``changed`` flag.
            None: Device has no genome, or no repair is necessary.
                Semantically equivalent to
                ``GenomeRepairResult(repaired_slice=original_slice, changed=False)``.
        """
        return None

    # ------------------------------------------------------------------
    # Step protocol
    # ------------------------------------------------------------------

    @abstractmethod
    def request(self, step_index: int) -> ResourceRequest:
        """Phase 1: Declare resource needs for a specific simulation step.

        Called before any device simulates that step, so the arbitrator sees
        all requests simultaneously before making allocation decisions.
        Implementations should base their request on current internal state
        (e.g. SoC) and the pre-computed schedule for ``step_index``.

        An idle device (no resource interaction this step) should return::

            ResourceRequest(device_id=self.device_id, step_index=step_index)

        Args:
            step_index (int): Zero-based index of the current simulation step.

        Returns:
            ResourceRequest: Declared resource needs for this step.
                Positive field values indicate consumption (device wants power).
                Negative field values indicate surplus being offered to the
                system (device has power to give).
        """

    @abstractmethod
    def simulate_step(self, grant: ResourceGrant) -> EnergyFlows:
        """Phase 2: Simulate one step given the arbitrated resource grant.

        Called after arbitration.  The device **must** respect grant limits —
        it may not consume more than granted, even if it requested more.
        When ``grant.curtailed`` is ``True``, the device should not run and
        must return zeroed ``EnergyFlows`` (with ``soc_pct`` still populated
        for storage devices so that SoC tracking remains consistent).

        Internal state (e.g. SoC) must be updated inside this method.  The
        updated state will be read by ``request()`` for the next step.

        Args:
            grant (ResourceGrant): Resources allocated by the arbitrator
                in response to this device's ``request()`` for the current step.

        Returns:
            EnergyFlows: Actual energy flows for this step.  Values may
                differ from the grant when internal constraints (e.g. SoC
                limits, minimum power thresholds, ramp rates) prevent full
                utilisation of the granted resources.
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> None:
        """Reset device to its initial physical state.

        Called at the start of each simulation run, *before* ``apply_genome()``.
        Must restore all mutable physical state (SoC, temperatures, runtime
        counters) to their initial values.

        **Must not zero out decoded schedule arrays.**  Schedules are
        populated by ``apply_genome()``, which is called immediately after
        ``reset()``.  Clearing schedules here would cause the device to run
        with a blank schedule for the entire simulation.  Keep ``reset()``
        strictly physical.
        """

    # ------------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------------

    def store_genome(self, genome_slice: np.ndarray) -> None:
        """Store the raw genome slice from the optimizer for later decoding.

        Called by ``GenomeAssembler.dispatch()`` after validation.
        The engine subsequently calls ``apply_genome(stored_slice, step_times)``
        with the value stored here, so that genome decoding and step-time
        scheduling happen together in a single call.

        The default implementation stores the slice in ``self._stored_genome``.
        Subclasses may override this to perform eager validation or copying,
        but should call ``super().store_genome(genome_slice)`` to preserve
        the default retrieval contract.

        Args:
            genome_slice (np.ndarray): Raw genome slice for this device,
                already validated against ``GenomeSlice`` bounds by the
                assembler.  The array is a view into the full genome; do not
                store the reference directly — copy if you need a durable value.
        """
        # Defensive copy: the slice is a view into the full genome array
        # which the optimizer may mutate before simulate() is called.
        self._stored_genome: np.ndarray = genome_slice.copy()

    def get_stored_genome(self) -> np.ndarray:
        """Return the genome slice stored by the most recent ``store_genome()`` call.

        Called by the engine inside ``_apply_genomes()`` to retrieve the raw
        slice before forwarding it to ``apply_genome()`` together with
        ``step_times``.

        Returns:
            np.ndarray: The stored genome slice, or an empty array if
                ``store_genome()`` has not yet been called for this device.
        """
        return getattr(self, "_stored_genome", np.empty(0))

    def to_dict(self) -> dict:
        """Serialise current device state to a plain dictionary.

        Override in subclasses to include device-specific fields such as
        current SoC, rated capacity, or efficiency parameters.  The base
        implementation returns only the device ID.

        Returns:
            dict: At minimum ``{"device_id": self.device_id}``.
        """
        return {"device_id": self.device_id}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device_id={self.device_id!r})"


# ---------------------------------------------------------------------------
# Specialised base classes
# ---------------------------------------------------------------------------


class StorageDevice(EnergyDevice, ABC):
    """Base class for devices that store energy (batteries, EVs, thermal tanks).

    Extends ``EnergyDevice`` with standardised SoC reporting so the optimizer
    and result layer can query storage state without knowing device internals.
    Subclasses must implement ``current_soc_percentage()`` and
    ``current_energy_content()`` in addition to the full ``EnergyDevice``
    protocol.
    """

    @abstractmethod
    def current_soc_percentage(self) -> float:
        """Return the current state of charge as a percentage.

        Returns:
            float: SoC in the range [0, 100].
        """

    @abstractmethod
    def current_energy_content(self) -> float:
        """Return the current usable energy available for discharge in Wh.

        Must account for minimum SoC limits and discharge efficiency so that
        the returned value represents energy that can actually be delivered
        to the AC bus, not just the raw energy stored in the medium.

        Returns:
            float: Deliverable energy in Wh (>= 0).
        """


class GenerationDevice(EnergyDevice, ABC):
    """Base class for energy generation devices (PV arrays, wind turbines).

    Generation devices typically return ``None`` from ``genome_requirements()``
    because their output is forecast-driven and not directly optimisable.
    Future extensions may add curtailment factors or orientation optimisation.
    """


class LoadDevice(EnergyDevice, ABC):
    """Base class for energy consumption devices with fixed or scheduled profiles.

    Fixed loads (e.g. a base household load curve) return ``None`` from
    ``genome_requirements()`` and implement ``apply_genome()`` as a no-op.
    Schedulable loads (e.g. a dishwasher or heat pump) may declare a genome
    slot encoding their preferred start step or operating mode per step.
    """
