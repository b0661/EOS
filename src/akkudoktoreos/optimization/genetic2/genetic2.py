"""Genetic2 optimisation entry point for EOS.

This module wires together the simulation engine, device registry, bus
topology, bus arbitrator, and genetic optimiser into a single callable
``Genetic2Optimization`` class that integrates with the EOS
``ConfigMixin`` / ``EnergyManagementSystemMixin`` protocol.

Supported concrete devices
--------------------------
``GridConnectionDevice``
    Slack AC-bus device.  Holds no genome — absorbs the bus residual
    after all other devices have been settled and converts it to an
    ``"energy_cost_amt"`` objective.

``HomeApplianceDevice``
    Shiftable fixed-duration load (dishwasher, washing machine, ...).
    One integer start-step gene per remaining run cycle.

``HybridInverterDevice``
    Hybrid inverter with continuous one-gene encoding (battery factor)
    in BATTERY, SOLAR, or HYBRID topology.

``EVChargerDevice``
    Electric vehicle charger with one-gene encoding (charge factor).

Return type
-----------
``optimize()`` returns a ``(OptimizationSolution, EnergyManagementPlan)``
tuple.

``OptimizationSolution`` is populated as follows:

- ``total_costs_amt``, ``total_revenues_amt`` -- aggregated from the
  grid-connection device's per-step financials (single replay run).
- ``total_losses_energy_wh`` -- always ``0.0``; conversion losses are
  embedded in device physics but not surfaced as a discrete signal.
- ``fitness_score`` -- the best scalar fitness from the GA as a singleton
  set, matching the convention of the legacy GENETIC algorithm.
- ``solution`` -- per-step DataFrame built from all device batch states
  during the replay run; indexed by step timestamps.
- ``prediction`` -- ``None``; GENETIC2 does not accumulate the forecast
  inputs (PV, prices, temperatures) into a result object.

Circular-import safety
----------------------
The simulation/optimizer imports are deferred to inside ``optimize()`` to
avoid circular-import failures during package initialization.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from loguru import logger

from akkudoktoreos.core.coreabc import ConfigMixin, EnergyManagementSystemMixin
from akkudoktoreos.core.emplan import EnergyManagementPlan
from akkudoktoreos.core.pydantic import PydanticDateTimeDataFrame
from akkudoktoreos.devices.devicesabc import EnergyBus, EnergyCarrier
from akkudoktoreos.devices.genetic2.evcharger import (
    EVChargerDevice,
    EVChargerParam,
)
from akkudoktoreos.devices.genetic2.fixedload import (
    FixedLoadDevice,
    FixedLoadParam,
)
from akkudoktoreos.devices.genetic2.gridconnection import (
    GridConnectionDevice,
    GridConnectionParam,
)
from akkudoktoreos.devices.genetic2.homeappliance import (
    HomeApplianceDevice,
    HomeApplianceParam,
)
from akkudoktoreos.devices.genetic2.hybridinverter import (
    HybridInverterDevice,
    HybridInverterParam,
)
from akkudoktoreos.optimization.optimization import OptimizationSolution
from akkudoktoreos.utils.datetimeutil import to_datetime

if TYPE_CHECKING:
    from akkudoktoreos.optimization.genetic2.optimizer import (
        BestIndividualResult,
        OptimizationResult,
    )
    from akkudoktoreos.simulation.genetic2.simulation import SimulationContext


# ---------------------------------------------------------------------------
# Device factory
# ---------------------------------------------------------------------------


def _build_devices(
    device_params: list[Union[EVChargerParam, FixedLoadParam, GridConnectionParam, HomeApplianceParam, HybridInverterParam]],
    buses: list[EnergyBus],
) -> tuple[list, list[EnergyBus]]:
    """Instantiate concrete EnergyDevice objects from immutable param objects."""
    import warnings

    devices: list[
        Union[EVChargerDevice, FixedLoadDevice, GridConnectionDevice, HomeApplianceDevice, HybridInverterDevice]
    ] = []
    device_index = 0
    port_index = 0
    dev: Union[EVChargerDevice, FixedLoadDevice, GridConnectionDevice, HomeApplianceDevice, HybridInverterDevice]

    logger.debug("_build_devices: {} params, {} buses", len(device_params), len(buses))
    for param in device_params:
        logger.debug("  param type={} device_id={}", type(param).__name__, param.device_id)


        if isinstance(param, EVChargerParam):
            dev = EVChargerDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
            devices.append(dev)
            logger.debug(
                "  → EVChargerDevice device_index={} port_index={} "
                "capacity_wh={} ev_connected_measurement_key={}",
                device_index,
                port_index,
                param.ev_battery_capacity_wh,
                param.ev_connected_measurement_key,
            )
            port_index += len(dev.ports)
            device_index += 1

        elif isinstance(param, FixedLoadParam):
            dev = FixedLoadDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
            devices.append(dev)
            logger.debug(
                "  → FixedLoadDevice device_index={} port_index={}", device_index, port_index
            )
            port_index += len(dev.ports)
            device_index += 1

        elif isinstance(param, GridConnectionParam):
            dev = GridConnectionDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
            devices.append(dev)
            logger.debug(
                "  → GridConnectionDevice device_index={} port_index={}", device_index, port_index
            )
            port_index += len(dev.ports)
            device_index += 1

        elif isinstance(param, HomeApplianceParam):
            dev = HomeApplianceDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
            devices.append(dev)
            logger.debug(
                "  → HomeApplianceDevice device_index={} port_index={}", device_index, port_index
            )
            port_index += len(dev.ports)
            device_index += 1

        elif isinstance(param, HybridInverterParam):
            dev = HybridInverterDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
            devices.append(dev)
            logger.debug(
                "  → HybridInverterDevice device_index={} port_index={} type={} "
                "capacity_wh={} pv_key={} lcos={}",
                device_index,
                port_index,
                param.inverter_type,
                param.battery_capacity_wh,
                param.pv_power_w_key,
                param.battery_lcos_amt_kwh,
            )
            port_index += len(dev.ports)
            device_index += 1

        else:
            warnings.warn(
                f"Unknown DeviceParam type '{type(param).__name__}' for device "
                f"'{param.device_id}'. Skipping.",
                stacklevel=3,
            )
            logger.warning("  → SKIPPED unknown param type {}", type(param).__name__)

    logger.debug("_build_devices: built {} devices", len(devices))
    for d in devices:
        logger.debug("  device_id={} type={}", d.device_id, type(d).__name__)
    return devices, buses


# ---------------------------------------------------------------------------
# Topology builder
# ---------------------------------------------------------------------------


def _build_topology(
    devices: list,
    buses: list[EnergyBus],
    horizon: int,
) -> tuple:
    """Build the bus topology and arbitrator from the constructed devices."""
    from akkudoktoreos.simulation.genetic2.arbitrator import (
        BusTopology,
        VectorizedBusArbitrator,
    )

    bus_id_to_index: dict[str, int] = {b.bus_id: i for i, b in enumerate(buses)}
    port_bus_assignments: list[int] = []

    for dev in devices:
        for port in dev.ports:
            bus_idx = bus_id_to_index.get(port.bus_id)
            if bus_idx is None:
                raise ValueError(
                    f"Device '{dev.device_id}', port '{port.port_id}' references "
                    f"unknown bus '{port.bus_id}'. "
                    f"Known buses: {list(bus_id_to_index.keys())}"
                )
            port_bus_assignments.append(bus_idx)

    topo = BusTopology(
        port_to_bus=np.array(port_bus_assignments, dtype=int),
        num_buses=len(buses),
    )
    arb = VectorizedBusArbitrator(topo, horizon=horizon)
    return topo, arb


# ---------------------------------------------------------------------------
# Result converters
# ---------------------------------------------------------------------------


def _collect_predictions(context: SimulationContext) -> PydanticDateTimeDataFrame:
    """Collect all the predictions that were used in the simulation."""
    predictions = context.resolved_predictions()
    logger.debug("Resolved predictions keys: {}", list(predictions.keys()))

    return PydanticDateTimeDataFrame.from_dataframe(predictions)


def _best_to_solution(
    best: BestIndividualResult,
    result: OptimizationResult,
    context: SimulationContext,
) -> OptimizationSolution:
    """Build an ``OptimizationSolution`` from the best-individual replay result.

    All fields are derived from the single replay run stored in ``best``:

    ``total_costs_amt``
        Aggregate gross import cost from grid financials.
    ``total_revenues_amt``
        Aggregate gross export revenue from grid financials.
    ``total_losses_energy_wh``
        Always ``0.0`` -- losses are implicit in device physics, not tracked.
    ``fitness_score``
        ``{best_scalar_fitness}`` -- singleton set matching legacy convention.
    ``solution``
        Per-step DataFrame wrapped in ``PydanticDateTimeDataFrame``.
    ``prediction``
        ``None`` -- GENETIC2 does not accumulate forecast inputs.
    """
    optimization_log = None
    if result.progress_df is not None:
        import pandas as pd

        df = result.progress_df.reset_index()  # generation becomes a regular column
        # PydanticDateTimeDataFrame needs a datetime index — use epoch + generation seconds
        df.index = pd.to_datetime(df["generation"], unit="s", utc=True)
        optimization_log = PydanticDateTimeDataFrame.from_dataframe(df)

    return OptimizationSolution(
        id=str(uuid.uuid4()),
        generated_at=to_datetime(),
        valid_from=context.step_times[0] if context.step_times else None,
        valid_until=context.step_times[-1] if context.step_times else None,
        total_costs_amt=best.total_costs_amt,
        total_revenues_amt=best.total_revenues_amt,
        total_losses_energy_wh=0.0,
        fitness_score={result.best_scalar_fitness},
        solution=PydanticDateTimeDataFrame.from_dataframe(best.solution_df),
        prediction=_collect_predictions(context),
        optimization_log=optimization_log,
        run_summary=result.run_summary if result.run_summary else None,
    )


def _best_to_plan(best: BestIndividualResult) -> EnergyManagementPlan:
    """Build an ``EnergyManagementPlan`` from the best-individual instructions."""
    all_instructions = [
        instr for device_instructions in best.instructions.values() for instr in device_instructions
    ]
    return EnergyManagementPlan(
        id=str(uuid.uuid4()),
        generated_at=to_datetime(),
        instructions=all_instructions,
    )


# ---------------------------------------------------------------------------
# Main optimisation class
# ---------------------------------------------------------------------------


class Genetic2Optimization(ConfigMixin, EnergyManagementSystemMixin):
    """EOS energy management optimisation using the genetic2 vectorised GA."""

    def optimize(self) -> tuple[OptimizationSolution, EnergyManagementPlan]:
        """Run the genetic optimisation using the current EOS configuration.

        Returns:
            ``(OptimizationSolution, EnergyManagementPlan)``
        """
        # Deferred imports to prevent circular imports during package init.
        from pendulum import (
            Duration,  # Do not use utils, we do not need and want pydantic here
        )

        from akkudoktoreos.optimization.genetic2.optimizer import (
            GeneticOptimizer,
            default_scalarize,
        )
        from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine
        from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
        from akkudoktoreos.simulation.genetic2.simulation import SimulationContext

        # ------------------------------------------------------------------
        # 1. Resolve optimisation parameters from config
        # ------------------------------------------------------------------
        step_interval_sec: Optional[int] = self.config.optimization.interval
        if step_interval_sec is None or step_interval_sec <= 0:
            raise ValueError(f"Optimization step interval invalid: {step_interval_sec} seconds")

        step_interval: Duration = Duration(seconds=int(step_interval_sec))

        horizon: int = self.config.optimization.horizon
        if horizon <= 0:
            raise ValueError(
                f"Computed horizon is zero "
                f"(horizon={horizon}, step_interval_sec={step_interval_sec})"
            )

        genetic_cfg = self.config.optimization.genetic
        pop_size: int = genetic_cfg.individuals
        generations: int = genetic_cfg.generations
        crossover_rate: float = genetic_cfg.crossover_rate
        mutation_rate: float = genetic_cfg.mutation_rate
        mutation_sigma: float = genetic_cfg.mutation_sigma
        tournament_size: int = genetic_cfg.tournament_size
        elitism_count: int = genetic_cfg.elitism_count
        stagnation_window: int = genetic_cfg.stagnation_window
        stagnation_boost: float = genetic_cfg.stagnation_boost
        random_seed: Optional[int] = genetic_cfg.seed
        log_progress_interval: int = genetic_cfg.log_progress_interval

        # ------------------------------------------------------------------
        # 2. Build the time axis
        # ------------------------------------------------------------------
        start_datetime = self.ems.start_datetime
        step_times = tuple(
            start_datetime.add(seconds=int(i * step_interval_sec)) for i in range(horizon)
        )

        context = SimulationContext(
            step_times=step_times,
            step_interval=step_interval,
        )

        # ------------------------------------------------------------------
        # 3. Build buses from config
        # ------------------------------------------------------------------
        buses: list[EnergyBus] = self.config.buses.to_genetic2_param()

        ac_buses = [b for b in buses if b.carrier == EnergyCarrier.AC]
        if not ac_buses:
            default_ac = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
            buses = [default_ac] + buses

        # ------------------------------------------------------------------
        # 4. Instantiate concrete devices from param objects
        # ------------------------------------------------------------------
        device_params = self.config.devices.to_genetic2_params()
        devices, buses = _build_devices(device_params, buses)

        if not devices:
            raise ValueError(
                "No controllable devices could be instantiated from the current "
                "configuration. Check that at least one GridConnectionDevice, "
                "HomeApplianceDevice, or HybridInverterDevice is configured."
            )

        # ------------------------------------------------------------------
        # 5. Build bus topology and arbitrator
        # ------------------------------------------------------------------
        _topo, arb = _build_topology(devices, buses, horizon)

        # ------------------------------------------------------------------
        # 6. Build device registry and simulation engine
        # ------------------------------------------------------------------
        registry = DeviceRegistry()
        for dev in devices:
            registry.register(dev)

        engine = EnergySimulationEngine(registry, buses, arb)

        # ------------------------------------------------------------------
        # 7. Run the genetic optimiser
        # ------------------------------------------------------------------
        optimizer = GeneticOptimizer(
            engine=engine,
            population_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            mutation_sigma=mutation_sigma,
            tournament_size=tournament_size,
            scalarize=default_scalarize,
            random_seed=random_seed,
            log_progress_interval=log_progress_interval,
            elitism_count=elitism_count,
            stagnation_window=stagnation_window,
            stagnation_boost=stagnation_boost,
        )

        result = optimizer.optimize(context)

        # ------------------------------------------------------------------
        # 8. Single replay run: extract instructions + solution data in one pass
        # ------------------------------------------------------------------
        best = optimizer.extract_best_result(result, context)

        # ------------------------------------------------------------------
        # 9. Convert to EOS-native types and return
        # ------------------------------------------------------------------
        solution = _best_to_solution(best, result, context)
        plan = _best_to_plan(best)

        return solution, plan
