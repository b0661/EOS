"""Genetic optimiser for the genetic2 framework.

This module provides two optimiser classes that operate on top of
``EnergySimulationEngine``:

``GeneticOptimizer``
    A standard real-valued genetic algorithm. The entire population is
    evaluated in a single vectorized ``evaluate_population()`` call per
    generation. Genome repair (Lamarckian learning) is applied each
    generation: when the engine reports that a device clamped infeasible
    genes, the corrected values are written back into the population
    before selection so the GA never wastes budget on unfixable genomes.

``RollingHorizonOptimizer``
    Wraps ``GeneticOptimizer`` to solve longer horizons by optimising one
    time window at a time and advancing the window by ``roll_steps`` each
    iteration. Only the first ``roll_steps`` of each optimised window are
    committed to the final schedule; the rest are re-optimised in the
    next window with updated context.

Design notes
------------
- **Vectorized evaluation.** ``evaluate_population()`` receives a
  ``dict[device_id, (population_size, horizon)]`` array and returns a
  ``(population_size, num_objectives)`` fitness matrix in one call. No
  Python loop over individuals exists inside the engine.

- **Genome representation.** The optimiser works exclusively with the
  *assembled* flat genome of shape ``(population_size, total_genome_size)``.
  The ``GenomeAssembler`` slices this into per-device sub-arrays before
  passing them to the engine, and reconstructs the full dict on every
  call. This keeps the GA operators (crossover, mutation) simple — they
  always work on flat arrays.

- **Repair / Lamarckian write-back.** After each ``evaluate_population()``
  call, ``EvaluationResult.repaired_genomes`` contains per-device arrays
  where the engine clamped infeasible values. The optimiser splices these
  corrections back into the flat population using the assembled slice
  indices so the repaired phenotype becomes the individual's genotype for
  the next generation.

- **Multi-objective fitness.** The engine returns a
  ``(population_size, num_objectives)`` matrix. The optimiser reduces
  this to a scalar via a caller-supplied ``scalarize`` function, defaulting
  to an equal-weight sum. This keeps multi-objective aggregation policy out
  of the engine and out of the core GA loop.

- **Bound enforcement.** Initial population is sampled uniformly within
  ``assembled.lower_bounds`` / ``assembled.upper_bounds``. Gaussian
  mutation perturbs genes and clips back to bounds. BLX-α crossover
  produces convex combinations of parents (already within bounds by
  convexity, but clipped for safety).

- **No threading.** NumPy releases the GIL during array operations, so
  ``ThreadPoolExecutor`` over NumPy calls adds synchronisation overhead
  without parallelism benefit. Vectorization over the population axis is
  the correct performance strategy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from akkudoktoreos.core.emplan import EnergyManagementInstruction
from akkudoktoreos.optimization.genetic2.genome import AssembledGenome
from akkudoktoreos.simulation.genetic2.engine import (
    EnergySimulationEngine,
    EvaluationResult,
)
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext

# ============================================================
# Types
# ============================================================

# Reduces a (population_size, num_objectives) matrix to (population_size,).
# Default: equal-weight sum across objectives.
ScalarizeFunction = Callable[[np.ndarray], np.ndarray]


def default_scalarize(fitness_matrix: np.ndarray) -> np.ndarray:
    """Equal-weight sum of all objectives.

    Args:
        fitness_matrix: Shape ``(population_size, num_objectives)``.

    Returns:
        Shape ``(population_size,)`` scalar fitness per individual.
    """
    return fitness_matrix.sum(axis=1)


# ============================================================
# Result dataclasses
# ============================================================


@dataclass
class GenerationStats:
    """Statistics captured at the end of one generation.

    Attributes:
        generation: Zero-based generation index.
        best_scalar_fitness: Lowest scalar fitness in this generation.
        mean_scalar_fitness: Mean scalar fitness across the population.
        num_repaired: Number of devices that had genomes repaired.
    """

    generation: int
    best_scalar_fitness: float
    mean_scalar_fitness: float
    num_repaired: int


@dataclass
class OptimizationResult:
    """Result of a full optimisation run.

    Attributes:
        best_genome: Flat genome of shape ``(total_genome_size,)`` for
            the best individual found across all generations.
        best_fitness_vector: Raw multi-objective fitness vector for the
            best individual, shape ``(num_objectives,)``.
        best_scalar_fitness: Scalar fitness of the best individual.
        objective_names: Ordered objective names matching
            ``best_fitness_vector`` columns.
        generations_run: Number of generations executed.
        history: Per-generation statistics, one entry per generation.
        assembled: The assembled genome used during optimisation.
            Use ``assembled.slices[device_id]`` to extract per-device
            schedules from ``best_genome``.
    """

    best_genome: np.ndarray
    best_fitness_vector: np.ndarray
    best_scalar_fitness: float
    objective_names: list[str]
    generations_run: int
    history: list[GenerationStats]
    assembled: AssembledGenome


# ============================================================
# GeneticOptimizer
# ============================================================


class GeneticOptimizer:
    """Real-valued genetic algorithm tightly integrated with ``EnergySimulationEngine``.

    Evaluates the entire population in one vectorized engine call per
    generation. Repairs are written back into the population each generation
    (Lamarckian learning).

    Args:
        engine: A fully constructed ``EnergySimulationEngine`` instance.
            The engine must be in ``CREATED`` or ``STRUCTURE_FROZEN`` state
            when ``optimize()`` is called — ``setup_run()`` and
            ``genome_requirements()`` are called internally.
        population_size: Number of individuals per generation.
        generations: Number of generations to run.
        crossover_rate: Probability that two parents undergo BLX-α
            crossover. Set to 0.0 to disable crossover.
        mutation_rate: Per-gene probability of Gaussian perturbation.
        mutation_sigma: Standard deviation of the Gaussian mutation noise,
            expressed as a fraction of each gene's bound range.
        tournament_size: Number of individuals drawn for each tournament
            selection. Higher values increase selection pressure.
        scalarize: Function mapping ``(population_size, num_objectives)``
            fitness matrix to ``(population_size,)`` scalar fitness.
            Defaults to equal-weight sum.
        random_seed: Optional seed for the NumPy RNG.
    """

    def __init__(
        self,
        engine: EnergySimulationEngine,
        population_size: int,
        generations: int,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.05,
        mutation_sigma: float = 0.1,
        tournament_size: int = 3,
        scalarize: ScalarizeFunction = default_scalarize,
        random_seed: int | None = None,
    ) -> None:
        self._engine = engine
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size
        self._scalarize = scalarize
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, context: SimulationContext) -> OptimizationResult:
        """Run the genetic optimisation for one horizon.

        Calls ``engine.setup_run()`` and ``engine.genome_requirements()``
        internally, then evolves the population for ``self.generations``
        generations. The engine is left in ``STRUCTURE_FROZEN`` state after
        the call so the caller can inspect ``engine.objective_names`` or
        chain another call.

        Args:
            context: Simulation context (step times and step interval).

        Returns:
            ``OptimizationResult`` containing the best genome, its fitness,
            generation history, and the assembled genome descriptor.

        Raises:
            ValueError: If the assembled genome has zero size (no device
                declared genome requirements).
        """
        # ----------------------------------------------------------
        # 1. Engine lifecycle: setup → freeze structure
        # ----------------------------------------------------------
        self._engine.setup_run(context)
        slices = self._engine.genome_requirements()  # transitions to STRUCTURE_FROZEN

        # ----------------------------------------------------------
        # 2. Assemble genome from engine slice definitions
        # ----------------------------------------------------------
        assembled = self._build_assembled_genome(slices)

        if assembled.total_size == 0:
            raise ValueError(
                "Assembled genome has zero size. "
                "At least one device must return a non-None GenomeSlice "
                "from genome_requirements()."
            )

        # ----------------------------------------------------------
        # 3. Initialise population uniformly within bounds
        # ----------------------------------------------------------
        population = self._init_population(assembled)

        best_genome: np.ndarray | None = None
        best_fitness_vector: np.ndarray | None = None
        best_scalar: float = np.inf
        history: list[GenerationStats] = []

        # ----------------------------------------------------------
        # 4. Evolution loop
        # ----------------------------------------------------------
        for gen in range(self.generations):
            # Build per-device genome dict from flat population
            genome_dict = self._split_population(population, assembled)

            # Vectorized evaluation — one engine call for the whole generation
            result: EvaluationResult = self._engine.evaluate_population(genome_dict)

            # Lamarckian repair write-back
            num_repaired = len(result.repaired_genomes)
            if result.repaired_genomes:
                self._apply_repairs(population, result.repaired_genomes, assembled)

            # Scalarise fitness for selection
            scalar_fitness = self._scalarize(result.fitness)  # (pop_size,)

            # Track global best
            gen_best_idx = int(np.argmin(scalar_fitness))
            gen_best_scalar = float(scalar_fitness[gen_best_idx])
            if gen_best_scalar < best_scalar:
                best_scalar = gen_best_scalar
                best_genome = population[gen_best_idx].copy()
                best_fitness_vector = result.fitness[gen_best_idx].copy()

            history.append(
                GenerationStats(
                    generation=gen,
                    best_scalar_fitness=gen_best_scalar,
                    mean_scalar_fitness=float(scalar_fitness.mean()),
                    num_repaired=num_repaired,
                )
            )

            # Breed next generation
            population = self._breed(population, scalar_fitness, assembled)

        return OptimizationResult(
            best_genome=best_genome,
            best_fitness_vector=best_fitness_vector,
            best_scalar_fitness=best_scalar,
            objective_names=self._engine.objective_names,
            generations_run=self.generations,
            history=history,
            assembled=assembled,
        )

    # ------------------------------------------------------------------
    # Instruction extraction
    # ------------------------------------------------------------------

    def extract_best_instructions(
        self,
        result: OptimizationResult,
        context: SimulationContext,
    ) -> dict[str, list[EnergyManagementInstruction]]:
        """Run the engine once for the best individual and extract S2 instructions.

        Re-evaluates the engine with a population of size 1 containing
        ``result.best_genome``, then calls ``device.extract_instructions(state, 0)``
        on every registered device. This avoids storing batch state across
        generations and keeps the GA loop clean.

        The engine must be in ``CREATED`` or ``STRUCTURE_FROZEN`` state
        before calling this method — i.e. call this after ``optimize()``,
        not during it. The engine is left in ``STRUCTURE_FROZEN`` state
        after the call.

        Args:
            result: The ``OptimizationResult`` returned by ``optimize()``.
                Provides ``best_genome`` and the ``assembled`` descriptor.
            context: The same ``SimulationContext`` used during
                optimisation. Needed to call ``setup_run`` so the engine
                reconstructs ``step_times`` in each device.

        Returns:
            ``dict[device_id, list[EnergyManagementInstruction]]`` — one
            entry per registered device (including non-genome devices that
            still have meaningful instructions, e.g. a fixed-schedule load).
            Devices for which ``extract_instructions`` raises
            ``NotImplementedError`` are silently omitted.
        """
        # Re-run the engine for a population of 1 using the best genome.
        self._engine.setup_run(context)
        self._engine.genome_requirements()  # transitions to STRUCTURE_FROZEN

        # Build a single-individual genome dict from the flat best genome.
        genome_dict: dict[str, np.ndarray] = {
            device_id: result.best_genome[slc.start : slc.end].reshape(1, -1)
            for device_id, slc in result.assembled.slices.items()
        }

        eval_result = self._engine.evaluate_population(genome_dict)
        # eval_result.fitness shape: (1, num_objectives) — not used here.

        # Retrieve the batch states the engine created for this single-individual run.
        # The engine exposes them via the registry after evaluate_population.
        batch_state = self._engine.last_batch_state  # see engine change below

        instructions: dict[str, list[EnergyManagementInstruction]] = {}
        for device in self._engine._registry.all_devices():
            device_state = batch_state.device_states.get(device.device_id)
            if device_state is None:
                continue
            try:
                instrs = device.extract_instructions(device_state, individual_index=0)
                instructions[device.device_id] = instrs
            except NotImplementedError:
                pass  # Device has not implemented extract_instructions yet — skip.

        return instructions

    # ------------------------------------------------------------------
    # Genome assembly helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_assembled_genome(slices: dict) -> AssembledGenome:
        """Reconstruct an ``AssembledGenome`` from the engine's slice dict.

        The engine returns ``dict[device_id, GenomeSlice]`` from
        ``genome_requirements()``. We rebuild the ``AssembledGenome``
        directly — no second pass through devices needed.

        Args:
            slices: Mapping of device_id to GenomeSlice as returned by
                ``engine.genome_requirements()``.

        Returns:
            ``AssembledGenome`` with correct total_size and bound vectors.
        """
        import numpy as np

        from akkudoktoreos.optimization.genetic2.genome import AssembledGenome

        if not slices:
            return AssembledGenome(
                total_size=0,
                slices={},
                lower_bounds=np.empty(0),
                upper_bounds=np.empty(0),
            )

        total_size = max(slc.end for slc in slices.values())
        lower_bounds = np.full(total_size, -np.inf)
        upper_bounds = np.full(total_size, np.inf)

        for slc in slices.values():
            if slc.lower_bound is not None:
                lower_bounds[slc.start : slc.end] = slc.lower_bound
            if slc.upper_bound is not None:
                upper_bounds[slc.start : slc.end] = slc.upper_bound

        return AssembledGenome(
            total_size=total_size,
            slices=slices,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def _init_population(self, assembled: AssembledGenome) -> np.ndarray:
        """Sample initial population uniformly within declared bounds.

        Genes without a finite lower or upper bound are initialised to 0.0
        and will be shaped by mutation. In practice all devices in this
        framework declare explicit bounds via ``power_bounds()``.

        Returns:
            Shape ``(population_size, total_genome_size)``.
        """
        lo = np.where(np.isfinite(assembled.lower_bounds), assembled.lower_bounds, 0.0)
        hi = np.where(np.isfinite(assembled.upper_bounds), assembled.upper_bounds, 0.0)
        return self._rng.uniform(lo, hi, size=(self.population_size, assembled.total_size))

    def _split_population(
        self,
        population: np.ndarray,
        assembled: AssembledGenome,
    ) -> dict[str, np.ndarray]:
        """Slice flat population into per-device genome dict.

        Args:
            population: Shape ``(population_size, total_genome_size)``.
            assembled: Genome assembly with per-device slice definitions.

        Returns:
            ``dict[device_id, (population_size, slice_size)]`` arrays.
        """
        return {
            device_id: population[:, slc.start : slc.end]
            for device_id, slc in assembled.slices.items()
        }

    def _apply_repairs(
        self,
        population: np.ndarray,
        repaired_genomes: dict[str, np.ndarray],
        assembled: AssembledGenome,
    ) -> None:
        """Write engine-repaired genome values back into the population in-place.

        Args:
            population: Shape ``(population_size, total_genome_size)``.
            repaired_genomes: ``EvaluationResult.repaired_genomes`` —
                only contains entries for devices that changed values.
            assembled: Genome assembly for slice index lookup.
        """
        for device_id, repaired in repaired_genomes.items():
            slc = assembled.slices.get(device_id)
            if slc is None:
                continue
            population[:, slc.start : slc.end] = repaired

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------

    def _breed(
        self,
        population: np.ndarray,
        scalar_fitness: np.ndarray,
        assembled: AssembledGenome,
    ) -> np.ndarray:
        """Produce the next generation via tournament selection, crossover, mutation.

        Args:
            population: Current population, shape ``(pop_size, genome_size)``.
            scalar_fitness: Scalar fitness per individual, shape ``(pop_size,)``.
            assembled: Used to enforce bounds during mutation.

        Returns:
            New population of the same shape.
        """
        pop_size, genome_size = population.shape
        next_pop = np.empty_like(population)
        i = 0

        while i < pop_size:
            p1 = self._tournament_select(population, scalar_fitness)
            p2 = self._tournament_select(population, scalar_fitness)
            c1, c2 = self._blx_crossover(p1, p2)
            self._mutate(c1, assembled)
            self._mutate(c2, assembled)
            next_pop[i] = c1
            i += 1
            if i < pop_size:
                next_pop[i] = c2
                i += 1

        return next_pop

    def _tournament_select(
        self,
        population: np.ndarray,
        scalar_fitness: np.ndarray,
    ) -> np.ndarray:
        """Tournament selection — lower fitness wins.

        Args:
            population: Shape ``(pop_size, genome_size)``.
            scalar_fitness: Shape ``(pop_size,)``.

        Returns:
            Copy of the winning individual's genome.
        """
        indices = self._rng.integers(0, len(population), size=self.tournament_size)
        winner = indices[int(np.argmin(scalar_fitness[indices]))]
        return population[winner].copy()

    def _blx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """BLX-α crossover (α = random blend coefficient).

        Produces two convex combinations of the parents. If crossover is
        not applied (probability ``1 - crossover_rate``), copies are
        returned unchanged.

        Args:
            parent1: Genome array, shape ``(genome_size,)``.
            parent2: Genome array, shape ``(genome_size,)``.

        Returns:
            Two child genome arrays of the same shape.
        """
        if self._rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        alpha = self._rng.random()
        child1 = alpha * parent1 + (1.0 - alpha) * parent2
        child2 = alpha * parent2 + (1.0 - alpha) * parent1
        return child1, child2

    def _mutate(self, genome: np.ndarray, assembled: AssembledGenome) -> None:
        """Gaussian mutation with per-gene adaptive scale and bound clipping.

        The noise standard deviation is ``mutation_sigma * bound_range``
        for each gene, so mutation is proportional to the gene's feasible
        range rather than using a fixed absolute sigma.

        Args:
            genome: Shape ``(genome_size,)``. Modified in-place.
            assembled: Used for bound range computation and clipping.
        """
        mask = self._rng.random(genome.shape) < self.mutation_rate
        if not np.any(mask):
            return

        lo = assembled.lower_bounds
        hi = assembled.upper_bounds
        ranges = np.where(
            np.isfinite(hi - lo),
            hi - lo,
            1.0,
        )
        noise = self._rng.normal(0.0, self.mutation_sigma * ranges[mask])
        genome[mask] += noise
        np.clip(genome, lo, hi, out=genome)


# ============================================================
# RollingHorizonOptimizer
# ============================================================


class RollingHorizonOptimizer:
    """Optimises a long horizon by solving overlapping windows sequentially.

    For each window, a ``GeneticOptimizer`` is run on the engine configured
    for just that window's time steps. Only the first ``roll_steps`` of the
    optimised window are committed to the final schedule; the remainder is
    re-optimised in the next window with updated initial conditions.

    This receding-horizon strategy is appropriate when the full horizon is
    too long to optimise at once, or when later steps have high forecast
    uncertainty and benefit from re-optimisation as the horizon advances.

    Args:
        engine: Simulation engine. Must be in ``CREATED`` or
            ``STRUCTURE_FROZEN`` state when ``optimize()`` is called.
        all_step_times: Complete time axis for the full horizon, as a
            tuple of floats (Unix timestamps or seconds offsets). Each
            window extracts a contiguous slice.
        step_interval: Duration of each time step in seconds.
        window_size: Number of steps per optimisation window.
        roll_steps: Number of steps to advance the window and commit to
            the output schedule per iteration. Must be ≤ ``window_size``.
        ga_kwargs: Keyword arguments forwarded to ``GeneticOptimizer``
            (``population_size``, ``generations``, ``mutation_rate``,
            ``crossover_rate``, ``scalarize``, ``random_seed``, etc.).
    """

    def __init__(
        self,
        engine: EnergySimulationEngine,
        all_step_times: tuple[float, ...],
        step_interval: float,
        window_size: int,
        roll_steps: int,
        **ga_kwargs: Any,
    ) -> None:
        if roll_steps > window_size:
            raise ValueError(f"roll_steps ({roll_steps}) must be ≤ window_size ({window_size}).")

        self._engine = engine
        self._all_step_times = all_step_times
        self._step_interval = step_interval
        self._window_size = window_size
        self._roll_steps = roll_steps
        self._ga_kwargs = ga_kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> dict[str, np.ndarray]:
        """Run rolling-horizon optimisation over the full time axis.

        For each window position, a fresh ``GeneticOptimizer`` is created
        and run. The committed portion of the best genome is written into
        the output schedule. Windows that extend past the end of the
        horizon are truncated automatically.

        Returns:
            ``dict[device_id, np.ndarray]`` mapping each genome-controlled
            device to its optimised schedule of shape ``(total_steps,)``.
            Steps are filled in order as windows commit their first
            ``roll_steps`` genes.
        """
        total_steps = len(self._all_step_times)
        schedule: dict[str, np.ndarray] = {}

        start = 0
        while start < total_steps:
            end = min(start + self._window_size, total_steps)
            window_times = self._all_step_times[start:end]
            commit_steps = min(self._roll_steps, end - start)

            context = SimulationContext(
                step_times=window_times,
                step_interval=self._step_interval,
            )

            ga = GeneticOptimizer(engine=self._engine, **self._ga_kwargs)
            result = ga.optimize(context)

            # Commit the first commit_steps of each device's schedule
            for device_id, slc in result.assembled.slices.items():
                if device_id not in schedule:
                    schedule[device_id] = np.zeros(total_steps)

                # Extract this device's genes from the best flat genome
                device_genes = result.best_genome[slc.start : slc.end]
                schedule[device_id][start : start + commit_steps] = device_genes[:commit_steps]

            start += commit_steps

        return schedule
