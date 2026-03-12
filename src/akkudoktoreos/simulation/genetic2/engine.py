"""Vectorized multi-objective evolutionary simulation engine.

Overview
--------
This module implements the core evaluation engine for population-based
optimization of energy systems. It is designed around three principles:

**Immutable structure, mutable state.**
    Devices are structural elements that never hold runtime data themselves.
    All per-evaluation data lives in state objects created fresh each generation,
    making the engine safe for repeated evaluation without re-instantiation.

**Vectorized batch evaluation.**
    An entire population of candidate solutions (genomes) is evaluated in a
    single call. Each device operates over a ``(population_size, horizon)``
    array rather than looping over individuals, keeping NumPy in control of
    the inner loops.

**Multi-objective cost accumulation.**
    Devices declare the objectives they contribute to by name. The engine
    builds a global objective index at ``setup_run()`` time and accumulates
    each device's local cost columns into the correct global columns.
    Multiple devices may share an objective name — their contributions are
    summed. The optimizer receives a raw ``(population_size, num_objectives)``
    fitness matrix and is responsible for scalarization, Pareto ranking, or
    any other aggregation strategy.

Engine Lifecycle
----------------
A typical usage sequence is::

    engine = EnergySimulationEngine(registry, buses, arbitrator)

    # Once per optimization run:
    engine.setup_run(context)
    genome_reqs = engine.genome_requirements()

    # Once per generation:
    result = engine.evaluate_population(genome_population)
    # result.fitness:          (population_size, num_objectives)
    # result.objective_names:  ["energy_cost_eur", "peak_power_kw", ...]
    # result.repaired_genomes: {device_id: repaired_array, ...}

    # Write repairs back into the population before selection:
    for device_id, repaired in result.repaired_genomes.items():
        genome_population[device_id] = repaired

Repair Contract
---------------
During genome application, devices may repair infeasible values in-place
(e.g. clamping a battery schedule to respect SoC limits). The engine detects
changes and surfaces only the modified genomes in ``EvaluationResult.repaired_genomes``.
The optimizer should write these back into the population before crossover and
selection so infeasible individuals do not persist across generations.

Objective Naming
----------------
Objective names are arbitrary strings declared by each device via the
``objective_names`` property. The engine assigns global column indices in
stable insertion order across all devices. Sharing a name across devices
(e.g. two devices both contributing to ``"energy_cost_eur"``) is intentional
and correct — contributions are summed into the same column.

Device index contract
---------------------
``DeviceRequest.device_index`` must be set by the device's own
``build_device_request()`` implementation and must match the index the device
expects to receive in ``apply_device_grant()``. The engine does not inject or
override device indices — devices own their own identity. The conventional
approach is for a device to store its index during ``setup_run()`` or at
construction time, or to derive it from the registry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from akkudoktoreos.devices.devicesabc import EnergyBus
from akkudoktoreos.optimization.genetic2.genome import GenomeSlice
from akkudoktoreos.simulation.genetic2.arbitrator import VectorizedBusArbitrator
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.simulation.genetic2.state import BatchSimulationState
from akkudoktoreos.simulation.genetic2.topology import TopologyValidator


class EngineState(Enum):
    """Lifecycle state of the simulation engine.

    Transitions::

        CREATED -> RUN_CONFIGURED  (via setup_run)
        RUN_CONFIGURED -> STRUCTURE_FROZEN  (via genome_requirements)
        STRUCTURE_FROZEN -> RUN_CONFIGURED  (via setup_run for a new run)
    """

    CREATED = auto()
    RUN_CONFIGURED = auto()
    STRUCTURE_FROZEN = auto()


@dataclass
class EvaluationResult:
    """Result of a single population evaluation.

    Attributes:
        fitness: Objective cost matrix of shape ``(population_size, num_objectives)``.
            Each row is one individual; each column is one named objective.
            Column order corresponds to ``objective_names``. Lower is better
            for all objectives.
        objective_names: Ordered list of global objective names matching the
            columns of ``fitness``.
        repaired_genomes: Mapping of ``device_id`` to repaired genome array of
            shape ``(population_size, horizon)``. Only contains entries for
            devices that actually modified the genome during simulation.
    """

    fitness: np.ndarray
    objective_names: list[str]
    repaired_genomes: dict[str, np.ndarray]


class EnergySimulationEngine:
    """Vectorized multi-objective evolutionary simulation engine.

    Evaluates a full population of candidate solutions in a single call by
    delegating to a registry of vectorized ``EnergyDevice`` instances. Devices
    interact through energy buses arbitrated by a ``VectorizedBusArbitrator``.

    The engine is stateful across its lifecycle but holds no per-individual
    mutable data itself — all runtime state is created fresh in each call to
    ``evaluate_population``.

    Args:
        registry: Registry of all participating devices.
        buses: All energy buses in the system.
        arbitrator: Bus arbitrator used to resolve competing energy requests.

    Raises:
        TopologyValidationError: If the device-bus topology is structurally
            invalid at construction time.
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        buses: Sequence[EnergyBus],
        arbitrator: VectorizedBusArbitrator,
    ) -> None:
        self._registry = registry
        self._buses = list(buses)
        self._arbitrator = arbitrator
        self._state = EngineState.CREATED
        self._context: SimulationContext | None = None
        self._genome_reqs: dict[str, GenomeSlice] | None = None
        self._objective_index: dict[str, int] = {}
        self._num_objectives: int = 0
        self._last_batch_state: BatchSimulationState | None = None

        TopologyValidator.validate(registry.all_devices(), buses)

    # ------------------------------------------------------------------
    # Setup Phase
    # ------------------------------------------------------------------

    def setup_run(self, context: SimulationContext) -> None:
        """Configure the engine for a new optimisation run.

        Calls ``setup_run`` on every registered device and builds the global
        objective index in stable insertion order. Multiple devices sharing an
        objective name contribute to the same fitness column.

        Can be called again from ``STRUCTURE_FROZEN`` to start a new run with
        different context without re-instantiating the engine.

        Args:
            context: Immutable simulation context configuration.

        Raises:
            RuntimeError: If called from ``RUN_CONFIGURED`` state.
        """
        if self._state == EngineState.RUN_CONFIGURED:
            raise RuntimeError(
                f"setup_run() called in invalid state {self._state}. "
                "Expected CREATED or STRUCTURE_FROZEN."
            )

        for device in self._registry.all_devices():
            device.setup_run(context)

        seen: dict[str, int] = {}
        for device in self._registry.all_devices():
            for name in device.objective_names:
                if name not in seen:
                    seen[name] = len(seen)

        self._objective_index = seen
        self._num_objectives = len(seen)
        self._context = context
        self._state = EngineState.RUN_CONFIGURED

    def genome_requirements(self) -> dict[str, GenomeSlice]:
        """Freeze the genome structure and return per-device slice definitions.

        Must be called after ``setup_run`` and before ``evaluate_population``.
        Transitions the engine to ``STRUCTURE_FROZEN``.

        Returns:
            Mapping of ``device_id`` to ``GenomeSlice`` for every genome-
            controlled device. Devices returning ``None`` are excluded.

        Raises:
            RuntimeError: If called before ``setup_run``.
        """
        if self._state != EngineState.RUN_CONFIGURED:
            raise RuntimeError(
                f"genome_requirements() called in invalid state {self._state}. "
                "Call setup_run() first."
            )

        cache: dict[str, GenomeSlice] = {}
        for device in self._registry.all_devices():
            req = device.genome_requirements()
            if req is not None:
                cache[device.device_id] = req

        self._genome_reqs = cache
        self._state = EngineState.STRUCTURE_FROZEN
        return cache

    @property
    def objective_names(self) -> list[str]:
        """Ordered global objective names matching ``EvaluationResult.fitness`` columns.

        Available after ``setup_run()``.

        Raises:
            RuntimeError: If called before ``setup_run``.
        """
        if self._state == EngineState.CREATED:
            raise RuntimeError("objective_names is not available before setup_run().")
        return list(self._objective_index.keys())

    # ------------------------------------------------------------------
    # Evaluation Phase
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        genome_population: dict[str, np.ndarray],
    ) -> EvaluationResult:
        """Evaluate an entire population across all registered objectives.

        Runs the full simulation pipeline — genome application, bus arbitration,
        grant application, and cost accumulation — for all individuals
        simultaneously using vectorized operations.

        Args:
            genome_population: Mapping of ``device_id`` to genome slice array
                of shape ``(population_size, horizon)``. Devices absent from
                this mapping are simulated without genome input.

        Returns:
            ``EvaluationResult`` with fitness, objective names, and any
            repaired genome slices.

        Raises:
            RuntimeError: If called before ``genome_requirements()``.
        """
        if self._state != EngineState.STRUCTURE_FROZEN:
            raise RuntimeError(
                f"evaluate_population() called in invalid state {self._state}. "
                "Call genome_requirements() first."
            )
        if self._context is None:
            raise RuntimeError("evaluate_population() called with None simulation context avail.")

        pop_size: int = next(iter(genome_population.values())).shape[0]
        horizon: int = len(self._context.step_times)

        total_cost = np.zeros((pop_size, self._num_objectives))
        repaired_genomes: dict[str, np.ndarray] = {}

        devices = list(self._registry.all_devices())

        # ---------------------------------------------------------
        # 0. Create batch state — fresh each generation
        # ---------------------------------------------------------
        device_states: dict[str, Any] = {
            device.device_id: device.create_batch_state(pop_size, horizon) for device in devices
        }
        batch_state = BatchSimulationState(
            device_states=device_states,
            population_size=pop_size,
        )

        # ---------------------------------------------------------
        # 1. Apply genomes, simulate, collect repairs
        # ---------------------------------------------------------
        for device in devices:
            genome_slice = genome_population.get(device.device_id)
            if genome_slice is None:
                continue

            device_state = batch_state.device_states[device.device_id]
            repaired = device.apply_genome_batch(device_state, genome_slice)

            # Identity check first — avoids O(n) array comparison in the
            # common no-repair case.
            if repaired is not genome_slice and not np.array_equal(repaired, genome_slice):
                repaired_genomes[device.device_id] = repaired

        # ---------------------------------------------------------
        # 2. Build device requests for bus arbitration
        # ---------------------------------------------------------
        device_requests = []
        for device in devices:
            device_state = batch_state.device_states[device.device_id]
            req = device.build_device_request(device_state)
            if req is not None:
                device_requests.append(req)

        # ---------------------------------------------------------
        # 3. Run bus arbitration
        # ---------------------------------------------------------
        device_grants = self._arbitrator.arbitrate(device_requests)
        grant_map = {g.device_index: g for g in device_grants}

        # ---------------------------------------------------------
        # 4. Apply grants to device states
        # ---------------------------------------------------------
        for device in devices:
            device_state = batch_state.device_states[device.device_id]
            req = device.build_device_request(device_state)
            if req is not None:
                grant = grant_map.get(req.device_index)
                if grant is not None:
                    device.apply_device_grant(device_state, grant)

        # ---------------------------------------------------------
        # 5. Accumulate local costs into global objective columns
        # ---------------------------------------------------------
        for device in devices:
            device_state = batch_state.device_states[device.device_id]
            local_cost = device.compute_cost(device_state)

            for local_idx, name in enumerate(device.objective_names):
                global_idx = self._objective_index[name]
                total_cost[:, global_idx] += local_cost[:, local_idx]

        self._last_batch_state = batch_state

        return EvaluationResult(
            fitness=total_cost,
            objective_names=self.objective_names,
            repaired_genomes=repaired_genomes,
        )

    @property
    def last_batch_state(self) -> BatchSimulationState:
        """The ``BatchSimulationState`` produced by the most recent ``evaluate_population`` call.

        Used by ``GeneticOptimizer.extract_best_instructions`` to retrieve
        device states after a single-individual re-evaluation of the best
        genome. The property is ``None`` before any call to
        ``evaluate_population``.

        Returns:
            The most recently produced ``BatchSimulationState``.

        Raises:
            RuntimeError: If ``evaluate_population`` has never been called.
        """
        if self._last_batch_state is None:
            raise RuntimeError(
                "last_batch_state is only available after evaluate_population() "
                "has been called at least once."
            )
        return self._last_batch_state
