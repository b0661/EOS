"""Immutable device parameters and abstract device base classes for energy simulation.

This module has two distinct responsibilities that are intentionally co-located
because they are tightly coupled by the device identity contract:

**Immutable parameter dataclasses** (``DeviceParam`` and its subclasses)
    Frozen, hashable value objects that fully describe the physical properties
    of an energy device. They carry no runtime state, no configuration logic,
    and no simulation behavior. Their immutability makes them safe to use as
    dictionary keys and cache keys inside the genetic algorithm.

**Abstract device base classes** (``EnergyDevice``, ``SingleStateEnergyDevice``)
    The simulation contract that concrete device implementations must fulfill.
    ``EnergyDevice`` defines the full vectorized batch lifecycle. ``SingleStateEnergyDevice``
    provides a reusable implementation of that lifecycle for devices with a single
    scalar internal state (e.g. batteries, thermal storage), leaving only the
    physics — ``initial_state``, ``state_transition_batch``, ``power_bounds`` —
    to concrete subclasses.

Design principles for parameter dataclasses
-------------------------------------------
- Fully immutable (``frozen=True`` dataclasses).
- Fully hashable — safe for GA caching.
- No mutable containers — tuples only, never lists or sets.
- No runtime state.
- No configuration logic.
- No Pydantic.
- No simulation behavior.
- Use ``float`` for physical quantities (avoids NumPy casting overhead).
- Use ``int`` for counts or indices.
- Use enums for symbolic modes.

Mutable configuration models (e.g. Pydantic models, ORM models) must live in a
separate module and convert to these parameter specs before passing them to the
simulation.

Simulation device lifecycle
----------------------------
Each concrete ``EnergyDevice`` follows this lifecycle within the engine::

    1. setup_run()           — called once per optimisation run
    2. genome_requirements() — called once to freeze genome structure
    3. create_batch_state()  ─┐
    4. apply_genome_batch()   │ repeated once per generation
    5. build_device_request() │ inside evaluate_population()
    6. apply_device_grant()   │
    7. compute_cost()        ─┘

See ``EnergySimulationEngine`` in ``simulation_engine.py`` for how the engine
orchestrates these steps across all devices.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from akkudoktoreos.core.emplan import EnergyManagementInstruction
from akkudoktoreos.optimization.genetic2.genome import GenomeSlice
from akkudoktoreos.simulation.genetic2.arbitrator import DeviceGrant, DeviceRequest
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import DateTime

# ============================================================
# Instruction Extraction Context
# ============================================================


@dataclass
class InstructionContext:
    """Shared cross-device context available during S2 instruction extraction.

    Passed by ``GeneticOptimizer.extract_best_instructions`` to every
    ``device.extract_instructions()`` call so devices can inspect the
    arbitrated outcome of *other* devices when deciding which operation
    mode to emit.

    The primary use-case is translating an explicit battery charge/discharge
    setpoint into a firmware-autonomous mode (e.g. ``SELF_CONSUMPTION``)
    when the arbitrated grid exchange at that step is near zero — meaning
    the optimizer already found a self-consumption equilibrium and the
    inverter firmware can maintain it without a precise setpoint.

    Attributes:
        grid_granted_wh:
            Net AC energy granted to the grid-connection device per step,
            shape ``(horizon,)`` [Wh].  Positive = import from grid;
            negative = export to grid.  ``None`` if no grid-connection
            device is registered (e.g. off-grid or island-mode setups).
        step_interval_sec:
            Duration of each simulation step [s].  Convenience copy so
            devices do not need to cache it separately.
    """

    grid_granted_wh: np.ndarray | None  # shape (horizon,), or None
    step_interval_sec: float


# ============================================================
# Operation Mode Enums
# ============================================================


class BatteryOperationMode(Enum):
    """Operating modes of a battery in a home energy management simulation.

    These modes express *intent* — which strategy the battery should follow.
    They require no direct awareness of electricity prices or carbon intensity;
    higher-level controllers or optimisers decide when to switch modes.

    Modes
    -----
    IDLE
        No charging or discharging.
    SELF_CONSUMPTION
        Charge from local surplus; discharge to meet local demand.
    NON_EXPORT
        Charge from on-site surplus to minimise or prevent export to the
        external grid. Discharging to the grid is not allowed.
    PEAK_SHAVING
        Discharge during local demand peaks to reduce grid draw.
    GRID_SUPPORT_EXPORT
        Discharge to support the upstream grid when commanded.
    GRID_SUPPORT_IMPORT
        Charge from the grid when instructed to absorb excess supply.
    FREQUENCY_REGULATION
        Perform fast bidirectional power adjustments based on grid
        frequency deviations.
    RAMP_RATE_CONTROL
        Smooth changes in local net load or generation.
    RESERVE_BACKUP
        Maintain a minimum state of charge for emergency use.
    OUTAGE_SUPPLY
        Discharge to power critical loads during a grid outage.
    FORCED_CHARGE
        Override all other logic and charge regardless of conditions.
    FORCED_DISCHARGE
        Override all other logic and discharge regardless of conditions.
    FAULT
        Battery is unavailable due to a fault or error state.
    """

    IDLE = "IDLE"
    SELF_CONSUMPTION = "SELF_CONSUMPTION"
    NON_EXPORT = "NON_EXPORT"
    PEAK_SHAVING = "PEAK_SHAVING"
    GRID_SUPPORT_EXPORT = "GRID_SUPPORT_EXPORT"
    GRID_SUPPORT_IMPORT = "GRID_SUPPORT_IMPORT"
    FREQUENCY_REGULATION = "FREQUENCY_REGULATION"
    RAMP_RATE_CONTROL = "RAMP_RATE_CONTROL"
    RESERVE_BACKUP = "RESERVE_BACKUP"
    OUTAGE_SUPPLY = "OUTAGE_SUPPLY"
    FORCED_CHARGE = "FORCED_CHARGE"
    FORCED_DISCHARGE = "FORCED_DISCHARGE"
    FAULT = "FAULT"


class ApplianceOperationMode(Enum):
    """Operating modes for a controllable appliance.

    Modes
    -----
    OFF
        Stop or prevent any active operation.
    RUN
        Start or continue normal operation.
    DEFER
        Postpone operation to a later time window based on scheduling
        or optimisation criteria.
    PAUSE
        Temporarily suspend an ongoing operation, keeping the option
        to resume later.
    RESUME
        Continue an operation that was previously paused or deferred.
    LIMIT_POWER
        Run under reduced power constraints, for example in response to
        load-management or demand-response signals.
    FORCED_RUN
        Start or maintain operation even if constraints or optimisation
        strategies would otherwise delay or limit it.
    FAULT
        Appliance is unavailable due to a fault or error state.
    """

    OFF = "OFF"
    RUN = "RUN"
    DEFER = "DEFER"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    LIMIT_POWER = "LIMIT_POWER"
    FORCED_RUN = "FORCED_RUN"
    FAULT = "FAULT"


# ============================================================
# Topology Enums
# ============================================================


class EnergyCarrier(Enum):
    """Energy carrier type used for bus topology validation."""

    AC = "ac"
    DC = "dc"
    HEAT = "heat"


class PortDirection(Enum):
    """Energy flow direction of a port relative to its owning device.

    The direction is defined from the perspective of the device, not the bus:

    SOURCE
        The device injects energy onto the bus (e.g. a PV array, a generator,
        or a battery discharging).
    SINK
        The device consumes energy from the bus (e.g. a load, a heat pump,
        or a battery charging).
    BIDIRECTIONAL
        The device can both inject and consume energy depending on operating
        state (e.g. a battery that charges and discharges, a grid connection
        that imports and exports). Counts as both a source and a sink for
        bus-level topology validation.
    """

    SOURCE = "source"
    SINK = "sink"
    BIDIRECTIONAL = "bidirectional"


# ============================================================
# Topology Layer
# ============================================================


@dataclass(frozen=True, slots=True)
class EnergyBusConstraint:
    """Optional structural constraints on bus connectivity.

    Attributes:
        max_sinks: Maximum number of consuming ports allowed on this bus,
            or ``None`` for no limit.
        max_sources: Maximum number of producing ports allowed on this bus,
            or ``None`` for no limit.
    """

    max_sinks: int | None = None
    max_sources: int | None = None


@dataclass(frozen=True, slots=True)
class EnergyBus:
    """Immutable energy bus descriptor.

    A bus is a connection point that devices attach to via ports. All ports
    on the same bus exchange energy of the same carrier type.

    Attributes:
        bus_id: Unique identifier for this bus.
        carrier: Energy carrier type flowing through this bus.
        constraint: Optional structural constraints on port counts.
    """

    bus_id: str
    carrier: EnergyCarrier
    constraint: EnergyBusConstraint | None = None


@dataclass(frozen=True, slots=True)
class EnergyPort:
    """Immutable port descriptor connecting a device to an energy bus.

    The carrier type is determined by the bus this port connects to — ports
    do not redeclare the carrier. The direction expresses energy flow from
    the perspective of the owning device.

    Attributes:
        port_id: Unique identifier for this port within the owning device.
            Must be unique per device; need not be globally unique.
        bus_id: Identifier of the bus this port connects to. Must reference
            a bus registered with the simulation engine.
        direction: Energy flow direction relative to the owning device.
            See ``PortDirection`` for semantics.
        max_power_w: Optional maximum power [W] allowed through this port,
            or ``None`` for no limit.
    """

    port_id: str
    bus_id: str
    direction: PortDirection
    max_power_w: float | None = None


# ============================================================
# Abstract Device Parameter Base
# ============================================================


@dataclass(frozen=True, slots=True)
class DeviceParam(ABC):
    """Abstract immutable base class for all device parameter objects.

    Subclasses are frozen dataclasses that describe the physical properties
    of one device instance. They are intentionally value objects: two
    ``DeviceParam`` instances with identical field values are considered equal
    and produce the same hash, making them safe as dictionary or cache keys.

    Note on ``ABC`` + frozen dataclass: Python allows combining these, but
    ``__init__`` and ``__hash__`` are generated by ``@dataclass``, not ``ABC``.
    Abstractness is enforced at instantiation time via ``ABCMeta`` as usual.

    Subclasses may optionally declare ``operation_modes`` as a field when the
    device supports multiple operating modes (see ``BatteryParam``,
    ``HeatPumpParam``). This is not required by the base class.

    Attributes:
        device_id: Unique identifier for this device instance.
        ports: Tuple of ports connecting this device to energy buses.
    """

    device_id: str
    ports: tuple[EnergyPort, ...]


# ============================================================
# Abstract Device Base Class
# ============================================================


class EnergyDevice(ABC):
    """Abstract base class for vectorized batch simulation devices.

    Devices are immutable structural elements of the simulation — they hold
    configuration but never mutable runtime data. All per-evaluation state
    lives in device-specific state objects created by ``create_batch_state``
    and passed explicitly through the lifecycle.

    Concrete subclasses must support vectorized evaluation over an entire
    population of candidate solutions simultaneously.

    Lifecycle
    ---------
    Called once per optimisation run::

        setup_run(context)
        genome_requirements()

    Called once per generation inside ``EnergySimulationEngine.evaluate_population``::

        state = create_batch_state(population_size, horizon)
        repaired = apply_genome_batch(state, genome_batch)
        request = build_device_request(state)
        apply_device_grant(state, grant)
        cost = compute_cost(state)

    Class attributes
    ----------------
    device_id : str
        Unique identifier for this device instance. Concrete subclasses must
        set this, typically via their ``__init__`` or as a constructor argument.
    """

    device_id: str

    # ==========================================================
    # Structure Phase
    # ==========================================================

    def setup_run(self, context: SimulationContext) -> None:
        """Configure the device for a new simulation run.

        Called once before ``genome_requirements`` at the start of each
        optimisation run. Implementations should store any derived quantities
        needed during batch evaluation (e.g. horizon length, step interval).

        The ``step_times`` are ``DateTime`` objects so that devices can use
        them directly as S2 ``execution_time`` values in
        ``extract_instructions`` without conversion. They are also passed
        into ``SingleStateBatchState.step_times`` so every lifecycle method
        (including ``compute_cost``) can access them for time-of-use logic.

        Args:
            context (SimulationContext):
                Holds the simulation context with:
                    step_times: Ordered tuple of ``DateTime`` timestamps defining
                        the simulation horizon. Length equals the genome horizon.
                    step_interval: Fixed time delta between consecutive steps [s].
        """
        raise NotImplementedError

    def genome_requirements(self) -> GenomeSlice | None:
        """Declare this device's genome slice requirements.

        Called once after ``setup_run`` to freeze the genome structure for
        the run. The engine assembles a global genome from all device slices.

        Returns:
            A ``GenomeSlice`` describing the shape and bounds of this device's
            genes, or ``None`` if this device is not genome-controlled (e.g.
            a fixed load or a purely reactive device).
        """
        raise NotImplementedError

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        """Tuple of energy ports connecting this device to buses.

        Declared statically — port topology does not change between runs.
        """
        raise NotImplementedError

    @property
    def objective_names(self) -> list[str]:
        """Names of the cost objectives this device contributes to.

        Column order must match the columns returned by ``compute_cost``.
        Multiple devices may declare the same objective name — the engine
        accumulates contributions into the same global fitness column.

        Returns:
            Ordered list of objective name strings,
            e.g. ``["energy_cost_eur", "peak_power_kw"]``.
        """
        raise NotImplementedError

    # ==========================================================
    # Batch Lifecycle
    # ==========================================================

    def create_batch_state(
        self,
        population_size: int,
        horizon: int,
    ) -> Any:
        """Allocate a fresh mutable batch state container for one generation.

        Called at the start of each ``evaluate_population`` call. The returned
        object holds all per-individual mutable data for this device and is
        passed explicitly to all subsequent lifecycle methods.

        Args:
            population_size: Number of individuals in the current population.
            horizon: Number of simulation time steps.

        Returns:
            A device-specific mutable batch state object. The type is
            defined by the concrete subclass.
        """
        raise NotImplementedError

    def apply_genome_batch(
        self,
        state: Any,
        genome_batch: np.ndarray,
    ) -> np.ndarray:
        """Decode and apply genome values to batch state.

        Implementations should store the genome into the state, run any
        forward simulation needed to evaluate feasibility, and repair
        infeasible values in-place. The repaired genome is returned so the
        engine can detect and surface changes to the optimiser.

        Args:
            state: Device batch state created by ``create_batch_state``.
            genome_batch: Genome slice of shape ``(population_size, horizon)``.

        Returns:
            The (possibly repaired) genome of shape ``(population_size, horizon)``.
            If no repair was performed, returning ``genome_batch`` unchanged
            (same object) allows the engine to skip the repair bookkeeping.
        """
        raise NotImplementedError

    # ==========================================================
    # Arbitration Phase
    # ==========================================================

    def build_device_request(self, state: Any) -> DeviceRequest | None:
        """Build this device's energy request for bus arbitration.

        Called after ``apply_genome_batch`` so the state reflects the
        simulated and repaired schedule. The request describes how much
        energy the device wants to draw from or inject into each of its ports.

        Args:
            state: Device batch state.

        Returns:
            A ``DeviceRequest`` describing this device's energy needs, or
            ``None`` if this device does not participate in arbitration
            (e.g. a device that only computes a cost term).
        """
        raise NotImplementedError

    def apply_device_grant(
        self,
        state: Any,
        grant: DeviceGrant,
    ) -> None:
        """Update device state with the arbitrated energy grant.

        Called after the bus arbitrator has resolved competing requests.
        Implementations should update the state to reflect the actually
        granted power, which may differ from the requested power.

        Args:
            state: Device batch state.
            grant: Arbitrated energy grant for this device.
        """
        raise NotImplementedError

    # ==========================================================
    # Cost Evaluation
    # ==========================================================

    def compute_cost(self, state: Any) -> np.ndarray:
        """Compute the local objective cost matrix after arbitration.

        Called after ``apply_device_grant``. The state at this point reflects
        the constrained, arbitrated simulation outcome.

        Args:
            state: Device batch state.

        Returns:
            Cost array of shape ``(population_size, num_local_objectives)``,
            where ``num_local_objectives == len(self.objective_names)``.
            Column order must match ``self.objective_names``.
        """
        raise NotImplementedError

    # ==========================================================
    # Instruction Extraction
    # ==========================================================

    def extract_instructions(
        self,
        state: Any,
        individual_index: int,
        instruction_context: InstructionContext | None = None,
    ) -> list[EnergyManagementInstruction]:
        """Extract S2 control instructions for one individual from the batch state.

        Called after the full simulation pipeline has completed for a
        population of size 1 containing the best individual. The state
        reflects the repaired, arbitrated result for that individual.

        Implementations slice row ``individual_index`` from the batch
        state arrays and convert the per-step values into a list of
        ``EnergyManagementInstruction`` objects — one per time step or
        per mode change, depending on the device's S2 control type.

        ``state.step_times`` provides the ordered ``DateTime`` timestamps
        to use as ``execution_time`` fields on each instruction. No
        conversion from float is needed.

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
            Ordered list of ``EnergyManagementInstruction`` objects covering
            the full simulation horizon for this individual. May be one
            instruction per step (e.g. PPBC ``PPBCScheduleInstruction``)
            or a compressed list of one instruction per mode change (e.g.
            OMBC ``OMBCInstruction``), at the device's discretion.

        Raises:
            NotImplementedError: Must be implemented by every concrete
                device subclass.
        """
        raise NotImplementedError


# ============================================================
# Single-State Convenience Base
# ============================================================


class SingleStateEnergyDevice(EnergyDevice):
    """Convenience base for devices with a single scalar internal state.

    Implements the full batch simulation loop, leaving only the physics to
    concrete subclasses. Suitable for devices whose state can be described
    by a single float per individual per time step — for example, a battery's
    state of charge, or a thermal store's temperature.

    Assumptions:
        - One gene per simulation time step.
        - A single scalar internal state that evolves forward in time.
        - Sequential simulation is unavoidable (each step depends on the
          previous state), but the population axis is fully vectorised.
        - Optional in-place genome repair at each step.

    Subclasses must implement
    -------------------------
    ``initial_state() -> float``
        Scalar initial state shared by all individuals at step 0.
    ``state_transition_batch(state, power, step_interval) -> np.ndarray``
        Vectorised one-step state update.
    ``power_bounds() -> tuple[float, float]``
        Static ``(min_power_w, max_power_w)`` used for genome bounds and
        default repair.
    ``ports``
        Static port declaration.
    ``objective_names``
        Objective names contributed to the global fitness matrix.
    ``build_device_request``
        Arbitration request built from the simulated schedule.
    ``apply_device_grant``
        Grant application to batch state.
    ``compute_cost``
        Local cost matrix after arbitration.
    """

    def __init__(self) -> None:
        super().__init__()
        self._step_times: tuple[DateTime, ...] | None = None
        self._num_steps: int | None = None
        self._step_interval: float | None = None

    # ------------------------------------------------------------------
    # Structure Phase
    # ------------------------------------------------------------------

    def setup_run(self, context: SimulationContext) -> None:
        """Store simulation context for batch simulation.

        Horizon length, step interval, and step times for batch simulation.

        Args:
            context (SimulationContext):
                Holds the simulation context with:
                    step_times: Ordered ``DateTime`` timestamps. Stored as
                        ``_step_times`` and forwarded into every `SingleStateBatchState`` so all
                        lifecycle methods can use them (time-of-use pricing, S2 instruction
                        construction, etc.).
                    step_interval: Fixed time delta between steps [s].
        """
        self._step_times = context.step_times
        self._num_steps = context.horizon
        self._step_interval = context.step_interval

    def genome_requirements(self) -> GenomeSlice | None:
        """Return one gene per simulation step, bounded by ``power_bounds``.

        Returns:
            A ``GenomeSlice`` of length ``num_steps`` with lower and upper
            bounds set to the values returned by ``power_bounds()``.
        """
        if self._num_steps is None:
            raise RuntimeError("Call setup_run() before genome_requirements().")
        lower, upper = self.power_bounds()

        return GenomeSlice(
            start=0,  # Re-indexed by the genome assembler
            size=self._num_steps,
            lower_bound=np.full(self._num_steps, lower),
            upper_bound=np.full(self._num_steps, upper),
        )

    # ports — remains abstract
    # objective_names — remains abstract

    # ------------------------------------------------------------------
    # Batch Lifecycle
    # ------------------------------------------------------------------

    def create_batch_state(
        self,
        population_size: int,
        horizon: int,
    ) -> SingleStateBatchState:
        """Allocate zeroed schedule and initial state arrays.

        Args:
            population_size: Number of individuals.
            horizon: Number of simulation time steps.

        Returns:
            ``SingleStateBatchState`` with a zeroed schedule, all
            individual states set to ``initial_state()``, and
            ``step_times`` populated from ``_step_times`` so that every
            lifecycle method (``compute_cost``, ``extract_instructions``)
            can access the simulation timestamps without any conversion.
        """
        if self._step_times is None:
            raise RuntimeError("Call setup_run() before create_batch_state().")
        return SingleStateBatchState(
            schedule=np.zeros((population_size, horizon)),
            state=np.full(population_size, self.initial_state()),
            population_size=population_size,
            horizon=horizon,
            step_times=self._step_times,
        )

    def apply_genome_batch(
        self,
        state: SingleStateBatchState,
        genome_batch: np.ndarray,
    ) -> np.ndarray:
        """Copy genome into state, run forward simulation with repair, return repaired genome.

        Args:
            state: Device batch state.
            genome_batch: Shape ``(population_size, horizon)``.

        Returns:
            Repaired schedule of shape ``(population_size, horizon)``.
            The schedule in ``state`` is updated in-place by ``_simulate_batch``;
            this method returns a reference to ``state.schedule``.
        """
        state.schedule[:] = genome_batch
        self._simulate_batch(state)
        return state.schedule

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _simulate_batch(self, state: SingleStateBatchState) -> None:
        """Run the forward simulation loop over all steps and all individuals.

        The time-step loop is sequential because each step's state depends on
        the previous one. The population axis is fully vectorised — each step
        operates on ``(population_size,)`` arrays.

        Repairs ``state.schedule`` in-place at each step by calling
        ``repair_batch`` before the state transition.

        Args:
            state: Device batch state. Modified in-place.
        """
        step_interval = self._step_interval
        if step_interval is None:
            raise RuntimeError("Call setup_run() before simulation.")

        state.state[:] = self.initial_state()

        horizon = state.horizon
        schedule = state.schedule
        internal_state = state.state

        for i in range(horizon):
            schedule[:, i] = self.repair_batch(i, schedule[:, i], internal_state)
            internal_state[:] = self.state_transition_batch(
                internal_state,
                schedule[:, i],
                step_interval,
            )

    # ------------------------------------------------------------------
    # Repair hook
    # ------------------------------------------------------------------

    def repair_batch(
        self,
        step: int,
        requested_power: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        """Repair infeasible power values for one time step across all individuals.

        Default implementation clamps to the static bounds returned by
        ``power_bounds()``. Override to implement state-dependent repair
        (e.g. clamping charge power based on current SoC to avoid overcharge).

        Args:
            step: Current time step index (0-based).
            requested_power: Requested power values, shape ``(population_size,)`` [W].
            current_state: Current internal state, shape ``(population_size,)``.
                Available so subclasses can apply state-dependent constraints
                without needing to access the full state object.

        Returns:
            Feasible power array of shape ``(populati   on_size,)`` [W].
        """
        lower, upper = self.power_bounds()
        return np.clip(requested_power, lower, upper)

    # ------------------------------------------------------------------
    # Arbitration Phase — remains abstract
    # ------------------------------------------------------------------

    def build_device_request(self, state: SingleStateBatchState) -> DeviceRequest | None:
        raise NotImplementedError

    def apply_device_grant(
        self,
        state: SingleStateBatchState,
        grant: DeviceGrant,
    ) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Cost Evaluation — remains abstract
    # ------------------------------------------------------------------

    def compute_cost(self, state: SingleStateBatchState) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Abstract physics — subclasses must implement
    # ------------------------------------------------------------------

    def initial_state(self) -> float:
        """Return the scalar initial state shared by all individuals.

        Called at the start of each ``_simulate_batch`` invocation.

        Returns:
            Initial state value (e.g. initial SoC in Wh, initial temperature
            in °C).
        """
        raise NotImplementedError

    def state_transition_batch(
        self,
        state: np.ndarray,
        power: np.ndarray,
        step_interval: float,
    ) -> np.ndarray:
        """Compute the next internal state for all individuals.

        Args:
            state: Current internal state, shape ``(population_size,)``.
            power: Applied (repaired) power, shape ``(population_size,)`` [W].
            step_interval: Duration of this time step [s].

        Returns:
            Updated state array of shape ``(population_size,)``.
        """
        raise NotImplementedError

    def power_bounds(self) -> tuple[float, float]:
        """Return the static power limits for this device.

        Used both to set genome bounds in ``genome_requirements`` and as the
        default repair constraint in ``repair_batch``.

        Returns:
            ``(min_power_w, max_power_w)`` where negative values indicate
            discharge (generation) and positive values indicate consumption,
            following the load convention.
        """
        raise NotImplementedError


# ============================================================
# Batch State
# ============================================================


@dataclass
class SingleStateBatchState:
    """Mutable batch state for ``SingleStateEnergyDevice``.

    Holds all per-individual mutable data for one generation's evaluation.
    Created fresh each generation by ``SingleStateEnergyDevice.create_batch_state``
    and never shared between devices.

    Attributes:
        schedule: Power schedule after genome decoding and repair,
            shape ``(population_size, horizon)`` [W].
        state: Internal device state at the current simulation step,
            shape ``(population_size,)``. Updated in-place during
            ``_simulate_batch``. After simulation completes, holds the
            final state at the end of the horizon.
        population_size: Number of individuals in this batch.
        horizon: Number of simulation time steps.
        step_times: Ordered ``DateTime`` timestamps for each simulation
            step. Length equals ``horizon``. Available to all lifecycle
            methods — ``compute_cost`` can use them for time-of-use
            pricing; ``extract_instructions`` uses them as S2
            ``execution_time`` values.
    """

    schedule: np.ndarray  # (population_size, horizon)
    state: np.ndarray  # (population_size,)
    population_size: int
    horizon: int
    step_times: tuple[DateTime, ...]  # length == horizon
