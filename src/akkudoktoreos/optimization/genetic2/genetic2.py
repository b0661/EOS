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
    ``"energy_cost_eur"`` objective.

``HomeApplianceDevice``
    Shiftable fixed-duration load (dishwasher, washing machine, ...).
    One integer start-step gene per remaining run cycle.

``HybridInverterDevice``
    Hybrid inverter with continuous two-gene encoding (battery factor +
    PV utilisation factor) in BATTERY, SOLAR, or HYBRID topology.

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
from pendulum import Duration
from pendulum import now as pendulum_now

from akkudoktoreos.core.coreabc import ConfigMixin, EnergyManagementSystemMixin
from akkudoktoreos.core.emplan import EnergyManagementPlan
from akkudoktoreos.core.pydantic import PydanticDateTimeDataFrame
from akkudoktoreos.devices.devicesabc import EnergyBus, EnergyCarrier
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
    device_params: list[Union[GridConnectionParam, HomeApplianceParam, HybridInverterParam]],
    buses: list[EnergyBus],
) -> tuple[list, list[EnergyBus]]:
    """Instantiate concrete EnergyDevice objects from immutable param objects."""
    import warnings

    devices: list[Union[GridConnectionDevice, HomeApplianceDevice, HybridInverterDevice]] = []
    device_index = 0
    port_index = 0
    dev: Union[GridConnectionDevice, HomeApplianceDevice, HybridInverterDevice]

    for param in device_params:
        if isinstance(param, GridConnectionParam):
            dev = GridConnectionDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
            devices.append(dev)
            port_index += len(dev.ports)
            device_index += 1

        elif isinstance(param, HomeApplianceParam):
            dev = HomeApplianceDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
            devices.append(dev)
            port_index += len(dev.ports)
            device_index += 1

        elif isinstance(param, HybridInverterParam):
            dev = HybridInverterDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
            devices.append(dev)
            port_index += len(dev.ports)
            device_index += 1

        else:
            warnings.warn(
                f"Unknown DeviceParam type '{type(param).__name__}' for device "
                f"'{param.device_id}'. Skipping.",
                stacklevel=3,
            )
            port_index += len(param.ports)

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
    return OptimizationSolution(
        id=str(uuid.uuid4()),
        generated_at=pendulum_now(),
        valid_from=context.step_times[0] if context.step_times else None,
        valid_until=context.step_times[-1] if context.step_times else None,
        total_costs_amt=best.total_costs_amt,
        total_revenues_amt=best.total_revenues_amt,
        total_losses_energy_wh=0.0,
        fitness_score={result.best_scalar_fitness},
        solution=PydanticDateTimeDataFrame.from_dataframe(best.solution_df),
        prediction=PydanticDateTimeDataFrame(data={}, dtypes={}, datetime_columns=[]),
    )


def _best_to_plan(best: BestIndividualResult) -> EnergyManagementPlan:
    """Build an ``EnergyManagementPlan`` from the best-individual instructions."""
    all_instructions = [
        instr for device_instructions in best.instructions.values() for instr in device_instructions
    ]
    return EnergyManagementPlan(
        id=str(uuid.uuid4()),
        generated_at=pendulum_now(),
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
        from akkudoktoreos.optimization.genetic2.optimizer import GeneticOptimizer
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
        random_seed: Optional[int] = genetic_cfg.seed

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
            random_seed=random_seed,
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
