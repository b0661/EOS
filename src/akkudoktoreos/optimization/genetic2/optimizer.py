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

Best-individual extraction
--------------------------
After ``optimize()`` completes, call ``extract_best_result()`` to re-run
the engine for the single best individual and extract a comprehensive
``BestIndividualResult`` containing:

- S2 operation instructions per device (``instructions``).
- A ``solution_df`` pandas DataFrame with per-step energy and cost columns,
  suitable for directly populating ``OptimizationSolution.solution``.
- Scalar ``total_costs_amt`` and ``total_revenues_amt`` aggregated from
  the grid-connection device's per-step financials.

``extract_best_instructions()`` is kept as a thin backward-compatible
wrapper that returns only the instructions dict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from akkudoktoreos.core.emplan import EnergyManagementInstruction
from akkudoktoreos.devices.devicesabc import InstructionContext
from akkudoktoreos.optimization.genetic2.genome import AssembledGenome
from akkudoktoreos.simulation.genetic2.engine import (
    EnergySimulationEngine,
    EvaluationResult,
)
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext

# ============================================================
# Types
# ============================================================

ScalarizeFunction = Callable[[np.ndarray], np.ndarray]


def default_scalarize(fitness_matrix: np.ndarray) -> np.ndarray:
    """Equal-weight sum of all objectives."""
    return fitness_matrix.sum(axis=1)


# ============================================================
# Result dataclasses
# ============================================================


@dataclass
class GenerationStats:
    """Statistics captured at the end of one generation."""

    generation: int
    best_scalar_fitness: float
    mean_scalar_fitness: float
    num_repaired: int


@dataclass
class OptimizationResult:
    """Result of a full optimisation run."""

    best_genome: np.ndarray
    best_fitness_vector: np.ndarray
    best_scalar_fitness: float
    objective_names: list[str]
    generations_run: int
    history: list[GenerationStats]
    assembled: AssembledGenome


@dataclass
class BestIndividualResult:
    """Full information extracted from a single replay of the best individual.

    Produced by ``GeneticOptimizer.extract_best_result()``. Contains
    everything needed to populate ``OptimizationSolution`` without any
    further engine interaction.

    Attributes:
        instructions: S2 operation instructions per device, keyed by
            ``device_id``. Devices that do not implement
            ``extract_instructions`` are absent.
        solution_df: Per-step time-series DataFrame indexed by the
            simulation step timestamps. Columns follow the
            ``OptimizationSolution.solution`` field convention:

            - ``grid_energy_wh``             Grid import (+) / export (-) [Wh].
            - ``costs_amt``                  Per-step gross import cost [currency].
            - ``revenue_amt``                Per-step gross export revenue [currency].
            - ``load_energy_wh``             Sum of all consuming device energy [Wh].
            - ``losses_energy_wh``           Conversion losses [Wh] — always 0.0;
                                             the engine embeds losses in device
                                             physics but does not surface them as
                                             a discrete signal.
            - ``{id}_energy_wh``             Per-step granted energy for each device [Wh].
            - ``{id}_soc_factor``            SoC as fraction of capacity
                                             (inverter/battery devices only).
            - ``{id}_{mode}_op_mode``        1.0 when operation mode ``mode`` is active at
                                             that step, 0.0 otherwise. One column per
                                             distinct mode emitted by the device's S2
                                             instructions. Examples:
                                             ``inverter1_charge_op_mode``,
                                             ``inverter1_pv_utilise_op_mode``,
                                             ``dishwasher_run_op_mode``.
                                             ``PV_UTILISE`` columns are only emitted when
                                             at least one step has non-zero PV utilisation.
            - ``{id}_{mode}_op_factor``      Operation mode factor at that step (float in
                                             ``[0, 1]``). For inverter battery modes this is
                                             the charge/discharge magnitude; for home
                                             appliances it mirrors ``op_mode`` (1.0 when
                                             running, 0.0 otherwise). Zero when inactive.
        total_costs_amt: Aggregate gross import cost over the horizon [currency].
        total_revenues_amt: Aggregate gross export revenue over the horizon [currency].
    """

    instructions: dict[str, list[EnergyManagementInstruction]]
    solution_df: pd.DataFrame
    total_costs_amt: float
    total_revenues_amt: float


# ============================================================
# GeneticOptimizer
# ============================================================


class GeneticOptimizer:
    """Real-valued genetic algorithm tightly integrated with ``EnergySimulationEngine``."""

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
        """Run the genetic optimisation for one horizon."""
        self._engine.setup_run(context)
        slices = self._engine.genome_requirements()

        assembled = self._build_assembled_genome(slices)
        if assembled.total_size == 0:
            raise ValueError(
                "Assembled genome has zero size. "
                "At least one device must return a non-None GenomeSlice "
                "from genome_requirements()."
            )

        population = self._init_population(assembled)

        best_genome: np.ndarray | None = None
        best_fitness_vector: np.ndarray | None = None
        best_scalar: float = np.inf
        history: list[GenerationStats] = []

        for gen in range(self.generations):
            genome_dict = self._split_population(population, assembled)
            result: EvaluationResult = self._engine.evaluate_population(genome_dict)

            num_repaired = len(result.repaired_genomes)
            if result.repaired_genomes:
                self._apply_repairs(population, result.repaired_genomes, assembled)

            scalar_fitness = self._scalarize(result.fitness)

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

            population = self._breed(population, scalar_fitness, assembled)

        if best_genome is None or best_fitness_vector is None:
            raise RuntimeError(
                f"Optimization failed with best genome: {best_genome}, "
                f"best fitness vector: {best_fitness_vector}"
            )

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
    # Best-individual extraction
    # ------------------------------------------------------------------

    def extract_best_result(
        self,
        result: OptimizationResult,
        context: SimulationContext,
    ) -> BestIndividualResult:
        """Re-run the engine once for the best individual and extract full results.

        This is the canonical post-optimisation extraction call. It performs
        a single engine replay with ``population_size=1``, then reads all
        device batch states in one pass to produce:

        - S2 ``instructions`` per device.
        - A ``solution_df`` pandas DataFrame with per-step energy and cost
          columns, indexed by the step timestamps from ``context``.
        - Scalar ``total_costs_amt`` and ``total_revenues_amt``.

        Device states are read duck-typed by the attributes they expose:

        - **GridConnectionBatchState**: recognised by ``granted_wh`` (and
          the absence of ``bat_factors``).
        - **HybridInverterBatchState**: recognised by ``bat_factors`` and
          ``soc_wh``.
        - **HomeApplianceBatchState**: recognised by ``granted_energy_wh``
          and ``schedule``.

        The engine is left in ``STRUCTURE_FROZEN`` state after the call.

        Args:
            result: The ``OptimizationResult`` returned by ``optimize()``.
            context: The same ``SimulationContext`` used during optimisation.

        Returns:
            ``BestIndividualResult`` with instructions, solution DataFrame,
            and aggregated financials.
        """
        # ------------------------------------------------------------------
        # 1. Replay the best genome with population_size=1
        # ------------------------------------------------------------------
        self._engine.setup_run(context)
        self._engine.genome_requirements()

        genome_dict: dict[str, np.ndarray] = {
            device_id: result.best_genome[slc.start : slc.end].reshape(1, -1)
            for device_id, slc in result.assembled.slices.items()
        }
        self._engine.evaluate_population(genome_dict)
        batch_state = self._engine.last_batch_state

        # ------------------------------------------------------------------
        # 2. Resolve step_interval_sec — handles pendulum.Duration or float
        # ------------------------------------------------------------------
        step_interval_sec: float = (
            context.step_interval.total_seconds()
            if hasattr(context.step_interval, "total_seconds")
            else float(context.step_interval)
        )
        step_h = step_interval_sec / 3600.0
        horizon = len(context.step_times)
        step_index = list(context.step_times)

        # ------------------------------------------------------------------
        # 3. Locate the grid-connection device state for InstructionContext
        # ------------------------------------------------------------------
        grid_granted_wh: np.ndarray | None = None
        for device in self._engine._registry.all_devices():
            state = batch_state.device_states.get(device.device_id)
            if (
                state is not None
                and hasattr(state, "granted_wh")
                and not hasattr(state, "bat_factors")
            ):
                grid_granted_wh = state.granted_wh[0]  # shape (horizon,)
                break

        instruction_context = InstructionContext(
            grid_granted_wh=grid_granted_wh,
            step_interval_sec=step_interval_sec,
        )

        # ------------------------------------------------------------------
        # 4. Single pass: collect instructions + build solution columns
        # ------------------------------------------------------------------
        instructions: dict[str, list[EnergyManagementInstruction]] = {}
        columns: dict[str, np.ndarray] = {}
        load_energy_wh = np.zeros(horizon)

        for device in self._engine._registry.all_devices():
            device_id = device.device_id
            state = batch_state.device_states.get(device_id)
            if state is None:
                continue

            # ---- S2 instructions ----
            try:
                instrs = device.extract_instructions(
                    state,
                    individual_index=0,
                    instruction_context=instruction_context,
                )
                instructions[device_id] = instrs
            except NotImplementedError:
                pass

            # ---- Solution columns (duck-typed on batch state attributes) ----
            # state is typed as ``object`` in BatchSimulationState; cast both
            # state and device to Any so mypy allows attribute access on the
            # concrete types without requiring hard imports of device classes.
            s: Any = state
            d: Any = device

            # GridConnectionDevice: has granted_wh, no bat_factors, and the
            # param carries import_cost_per_kwh (distinguishes it from
            # FixedLoadDevice which shares the same state shape).
            if (
                hasattr(state, "granted_wh")
                and not hasattr(state, "bat_factors")
                and hasattr(d.param, "import_cost_per_kwh")
            ):
                granted: np.ndarray = s.granted_wh[0]  # (horizon,)
                columns["grid_energy_wh"] = granted

                # Per-step prices: read from the device's cached arrays (set
                # during setup_run) or fall back to the flat-rate param value.
                _import_cached: np.ndarray | None = getattr(d, "_import_price_per_kwh", None)
                _export_cached: np.ndarray | None = getattr(d, "_export_price_per_kwh", None)
                import_price: np.ndarray = (
                    _import_cached
                    if _import_cached is not None
                    else np.full(horizon, d.param.import_cost_per_kwh)
                )
                export_price: np.ndarray = (
                    _export_cached
                    if _export_cached is not None
                    else np.full(horizon, d.param.export_revenue_per_kwh)
                )
                import_wh = np.maximum(0.0, granted)
                export_wh = np.maximum(0.0, -granted)
                columns["costs_amt"] = (import_wh / 1000.0) * import_price
                columns["revenue_amt"] = (export_wh / 1000.0) * export_price
                load_energy_wh += import_wh

            # FixedLoadDevice: has granted_wh, no bat_factors, and the param
            # carries load_power_w_key (distinguishes it from GridConnection).
            elif (
                hasattr(state, "granted_wh")
                and not hasattr(state, "bat_factors")
                and hasattr(d.param, "load_power_w_key")
            ):
                energy_wh: np.ndarray = s.granted_wh[0]  # (horizon,)
                columns[f"{device_id}_energy_wh"] = energy_wh
                load_energy_wh += np.maximum(0.0, energy_wh)

            # HybridInverterDevice: has bat_factors + soc_wh
            elif hasattr(state, "bat_factors") and hasattr(state, "soc_wh"):
                ac_power_w: np.ndarray = s.ac_power_w[0]  # (horizon,) after apply_device_grant
                energy_wh = ac_power_w * step_h
                columns[f"{device_id}_energy_wh"] = energy_wh

                capacity_wh: float = d.param.battery_capacity_wh
                columns[f"{device_id}_soc_factor"] = (
                    s.soc_wh[0] / capacity_wh if capacity_wh > 0 else np.zeros(horizon)
                )

                load_energy_wh += np.maximum(0.0, energy_wh)

            # HomeApplianceDevice: has granted_energy_wh + schedule
            elif hasattr(state, "granted_energy_wh") and hasattr(state, "schedule"):
                energy_wh = s.granted_energy_wh[0]  # (horizon,)
                columns[f"{device_id}_energy_wh"] = energy_wh

                load_energy_wh += np.maximum(0.0, energy_wh)

            # ---- Instruction-driven per-mode columns -------------------------
            # Build {id}_{MODE}_op_mode (0.0/1.0) and {id}_{MODE}_op_factor
            # columns from the S2 instructions already collected above.
            # This is done for every device that produced instructions and has
            # an operation_mode_id attribute on its instructions (inverters and
            # appliances).  Grid instructions are empty lists so nothing is
            # emitted for the grid device.
            # The step → instruction mapping uses the instruction's
            # execution_time matched against step_index.  Multiple instructions
            # sharing the same execution_time (e.g. bat + PV_UTILISE from
            # HybridInverterDevice) each get their own mode column.
            if device_id in instructions:
                # Build a fast lookup: execution_time → step index
                time_to_step: dict[Any, int] = {t: i for i, t in enumerate(step_index)}

                # Accumulate per-mode arrays keyed by mode string
                mode_op: dict[str, np.ndarray] = {}
                mode_factor: dict[str, np.ndarray] = {}

                for instr in instructions[device_id]:
                    mode_id: str | None = getattr(instr, "operation_mode_id", None)
                    if mode_id is None:
                        continue
                    t = time_to_step.get(instr.execution_time)
                    if t is None:
                        continue
                    factor = float(getattr(instr, "operation_mode_factor", 1.0))

                    if mode_id not in mode_op:
                        mode_op[mode_id] = np.zeros(horizon)
                        mode_factor[mode_id] = np.zeros(horizon)
                    mode_op[mode_id][t] = 1.0
                    mode_factor[mode_id][t] = factor

                for mode_id, op_arr in mode_op.items():
                    # Skip PV_UTILISE entirely when no step actually used PV
                    # (pv_util was zero across the whole horizon).
                    if mode_id == "PV_UTILISE" and not np.any(op_arr):
                        continue
                    columns[f"{device_id}_{mode_id.lower()}_op_mode"] = op_arr
                    columns[f"{device_id}_{mode_id.lower()}_op_factor"] = mode_factor[mode_id]

        columns["load_energy_wh"] = load_energy_wh
        # Conversion losses are embedded in device physics but not surfaced
        # as a discrete signal by the engine.
        columns["losses_energy_wh"] = np.zeros(horizon)

        solution_df = pd.DataFrame(columns, index=step_index)

        # ------------------------------------------------------------------
        # 5. Aggregate financials
        # ------------------------------------------------------------------
        total_costs_amt = float(columns.get("costs_amt", np.zeros(horizon)).sum())
        total_revenues_amt = float(columns.get("revenue_amt", np.zeros(horizon)).sum())

        return BestIndividualResult(
            instructions=instructions,
            solution_df=solution_df,
            total_costs_amt=total_costs_amt,
            total_revenues_amt=total_revenues_amt,
        )

    def extract_best_instructions(
        self,
        result: OptimizationResult,
        context: SimulationContext,
    ) -> dict[str, list[EnergyManagementInstruction]]:
        """Thin backward-compatible wrapper around ``extract_best_result()``.

        Prefer calling ``extract_best_result()`` directly when you also need
        the solution DataFrame or aggregated financials.
        """
        return self.extract_best_result(result, context).instructions

    # ------------------------------------------------------------------
    # Genome assembly helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_assembled_genome(slices: dict) -> AssembledGenome:
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
        lo = np.where(np.isfinite(assembled.lower_bounds), assembled.lower_bounds, 0.0)
        hi = np.where(np.isfinite(assembled.upper_bounds), assembled.upper_bounds, 0.0)
        return self._rng.uniform(lo, hi, size=(self.population_size, assembled.total_size))

    def _split_population(
        self,
        population: np.ndarray,
        assembled: AssembledGenome,
    ) -> dict[str, np.ndarray]:
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
        indices = self._rng.integers(0, len(population), size=self.tournament_size)
        winner = indices[int(np.argmin(scalar_fitness[indices]))]
        return population[winner].copy()

    def _blx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        alpha = self._rng.random()
        child1 = alpha * parent1 + (1.0 - alpha) * parent2
        child2 = alpha * parent2 + (1.0 - alpha) * parent1
        return child1, child2

    def _mutate(self, genome: np.ndarray, assembled: AssembledGenome) -> None:
        mask = self._rng.random(genome.shape) < self.mutation_rate
        if not np.any(mask):
            return

        lo = assembled.lower_bounds
        hi = assembled.upper_bounds
        ranges = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
        noise = self._rng.normal(0.0, self.mutation_sigma * ranges[mask])
        genome[mask] += noise
        np.clip(genome, lo, hi, out=genome)


# ============================================================
# RollingHorizonOptimizer
# ============================================================


class RollingHorizonOptimizer:
    """Optimises a long horizon by solving overlapping windows sequentially."""

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
            raise ValueError(f"roll_steps ({roll_steps}) must be <= window_size ({window_size}).")

        self._engine = engine
        self._all_step_times = all_step_times
        self._step_interval = step_interval
        self._window_size = window_size
        self._roll_steps = roll_steps
        self._ga_kwargs = ga_kwargs

    def optimize(self) -> dict[str, np.ndarray]:
        """Run rolling-horizon optimisation over the full time axis."""
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

            for device_id, slc in result.assembled.slices.items():
                if device_id not in schedule:
                    schedule[device_id] = np.zeros(total_steps)
                device_genes = result.best_genome[slc.start : slc.end]
                schedule[device_id][start : start + commit_steps] = device_genes[:commit_steps]

            start += commit_steps

        return schedule
