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
    Shiftable fixed-duration load (dishwasher, washing machine, …).
    One integer start-step gene per remaining run cycle.

``HybridInverterDevice``
    Hybrid inverter with continuous two-gene encoding (battery factor +
    PV utilisation factor) in BATTERY, SOLAR, or HYBRID topology.

Return type
-----------
``optimize()`` returns a ``(Genetic2Solution, EnergyManagementPlan)``
tuple.

``Genetic2Solution`` is a lightweight dataclass defined in this module.

Circular-import safety
----------------------
``genetic2.py`` is imported at module level by ``ems.py``, which itself
is transitively imported during config loading.  All heavy imports
(optimizer, engine, simulation, device classes) are deferred to inside
``optimize()`` to avoid circular-import failures during package
initialization.  Only the public API surface (``Genetic2Solution``,
``ConfigMixin``, ``EnergyManagementSystemMixin``, ``EnergyManagementPlan``,
``EnergyBus``, ``EnergyCarrier``) is imported at module level.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from pendulum import Duration
from pendulum import now as pendulum_now

from akkudoktoreos.core.coreabc import ConfigMixin, EnergyManagementSystemMixin
from akkudoktoreos.core.emplan import EnergyManagementPlan
from akkudoktoreos.devices.devicesabc import EnergyBus, EnergyCarrier

if TYPE_CHECKING:
    # Only used in type annotations on the private helpers — not needed at runtime.
    from akkudoktoreos.optimization.genetic2.optimizer import (
        GeneticOptimizer,
        OptimizationResult,
    )
    from akkudoktoreos.simulation.genetic2.arbitrator import (
        BusTopology,
        VectorizedBusArbitrator,
    )
    from akkudoktoreos.simulation.genetic2.simulation import SimulationContext


# ---------------------------------------------------------------------------
# Lightweight result type
# ---------------------------------------------------------------------------


@dataclass
class Genetic2Solution:
    """Optimisation result returned by ``Genetic2Optimization.optimize()``.

    A lightweight dataclass that avoids the heavyweight EOS
    ``OptimizationSolution`` Pydantic model (which requires fields such as
    ``id``, ``generated_at``, and full time-series DataFrames that are not
    produced by the GA).  Callers that need the canonical EOS type can
    convert from these fields.

    Attributes:
        cost: Scalar fitness of the best individual (lower is better).
        objective_values: Mapping of objective name to its value for the
            best individual.
        schedule: Per-device gene lists.  Keys are ``device_id`` strings;
            values are flat ``list[float]`` of length equal to the
            device's genome size (``2 * horizon`` for inverter devices;
            number of remaining cycles for appliance devices).
        generations_run: Number of GA generations actually executed.
    """

    cost: float
    objective_values: dict[str, float]
    schedule: dict[str, list[float]]
    generations_run: int


# ---------------------------------------------------------------------------
# Device factory
# ---------------------------------------------------------------------------


def _build_devices(
    device_params: list,
    buses: list[EnergyBus],
) -> tuple[list, list[EnergyBus]]:
    """Instantiate concrete EnergyDevice objects from immutable param objects.

    Assigns sequential ``device_index`` and ``port_index`` values so that
    the ``BusTopology`` port-to-bus array can be built without a second
    pass.

    Unsupported param types emit a ``UserWarning`` and are skipped; the
    port counter still advances so subsequent devices receive the correct
    ``port_index``.

    Args:
        device_params: Flat list of param instances.
        buses: All energy buses in the system.

    Returns:
        ``(devices, buses)`` where ``devices`` is the ordered list of
        concrete ``EnergyDevice`` objects and ``buses`` is returned
        unchanged.
    """
    import warnings

    # Deferred imports — avoid circular dependency at module load time.
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

    devices = []
    device_index = 0
    port_index = 0

    for param in device_params:
        if isinstance(param, GridConnectionParam):
            dev = GridConnectionDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
        elif isinstance(param, HomeApplianceParam):
            dev = HomeApplianceDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
        elif isinstance(param, HybridInverterParam):
            dev = HybridInverterDevice(
                param=param,
                device_index=device_index,
                port_index=port_index,
            )
        else:
            warnings.warn(
                f"Unknown DeviceParam type '{type(param).__name__}' for device "
                f"'{param.device_id}'. Skipping.",
                stacklevel=3,
            )
            port_index += len(param.ports)
            continue

        devices.append(dev)
        port_index += len(dev.ports)
        device_index += 1

    return devices, buses


# ---------------------------------------------------------------------------
# Topology builder
# ---------------------------------------------------------------------------


def _build_topology(
    devices: list,
    buses: list[EnergyBus],
    horizon: int,
) -> tuple[BusTopology, VectorizedBusArbitrator]:
    """Build the bus topology and arbitrator from the constructed devices.

    Args:
        devices: Concrete ``EnergyDevice`` instances (port indices already
            assigned during construction).
        buses: All energy buses — order determines the bus index.
        horizon: Simulation horizon length (number of time steps).

    Returns:
        ``(BusTopology, VectorizedBusArbitrator)``

    Raises:
        ValueError: If a device port references a bus_id not present in
            ``buses``.
    """
    # Deferred imports — avoid circular dependency at module load time.
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


def _result_to_solution(
    result: OptimizationResult,
    context: SimulationContext,
) -> Genetic2Solution:
    """Convert a genetic2 ``OptimizationResult`` to a ``Genetic2Solution``."""
    schedule: dict[str, list[float]] = {}
    for device_id, slc in result.assembled.slices.items():
        genes = result.best_genome[slc.start : slc.end]
        schedule[device_id] = genes.tolist()

    return Genetic2Solution(
        cost=result.best_scalar_fitness,
        objective_values={
            name: float(result.best_fitness_vector[i])
            for i, name in enumerate(result.objective_names)
        },
        schedule=schedule,
        generations_run=result.generations_run,
    )


def _result_to_plan(
    optimizer: GeneticOptimizer,
    result: OptimizationResult,
    context: SimulationContext,
) -> EnergyManagementPlan:
    """Extract S2 instructions from the best individual and build an EMS plan."""
    instructions_by_device = optimizer.extract_best_instructions(result, context)
    all_instructions = [
        instr
        for device_instructions in instructions_by_device.values()
        for instr in device_instructions
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
    """EOS energy management optimisation using the genetic2 vectorised GA.

    The ``optimize()`` method is the single public entry point.
    """

    def optimize(self) -> tuple[Genetic2Solution, EnergyManagementPlan]:
        """Run the genetic optimisation using the current EOS configuration.

        Returns:
            ``(Genetic2Solution, EnergyManagementPlan)``

        Raises:
            ValueError: If the configured step interval or horizon is
                invalid, or no controllable devices can be instantiated.
        """
        # All simulation/optimizer imports are deferred here to prevent
        # circular imports during package initialization (ems.py imports
        # this module at module level, and the simulation stack transitively
        # reaches back through coreabc/cache to ems.py).
        from akkudoktoreos.optimization.genetic2.optimizer import GeneticOptimizer
        from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine
        from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
        from akkudoktoreos.simulation.genetic2.simulation import SimulationContext

        # ------------------------------------------------------------------
        # 1. Resolve optimisation parameters from config
        # ------------------------------------------------------------------
        step_interval_sec: Optional[int] = self.config.optimization.interval
        if step_interval_sec is None or step_interval_sec <= 0:
            raise ValueError(
                f"Optimization step interval invalid: {step_interval_sec} seconds"
            )

        step_interval: Duration = Duration(seconds=int(step_interval_sec))

        # Note: config.optimization.horizon is a @computed_field whose
        # implementation currently has a missing return statement (always
        # returns None).  We compute the step count from horizon_hours
        # directly to avoid that dependency.
        horizon_hours: Optional[int] = self.config.optimization.horizon_hours
        if horizon_hours is None or horizon_hours <= 0:
            raise ValueError(
                f"Optimization horizon_hours invalid: {horizon_hours}"
            )
        horizon: int = int(horizon_hours * 3600 / step_interval_sec)
        if horizon <= 0:
            raise ValueError(
                f"Computed horizon is zero "
                f"(horizon_hours={horizon_hours}, "
                f"step_interval_sec={step_interval_sec})"
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
            start_datetime.add(seconds=int(i * step_interval_sec))
            for i in range(horizon)
        )

        context = SimulationContext(
            step_times=step_times,
            step_interval=step_interval,
        )

        # ------------------------------------------------------------------
        # 3. Build buses from config
        # ------------------------------------------------------------------
        buses: list[EnergyBus] = self.config.buses.to_genetic2_param()

        # Ensure there is at least one AC bus.
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
        # 8. Convert result and return
        # ------------------------------------------------------------------
        solution = _result_to_solution(result, context)
        plan = _result_to_plan(optimizer, result, context)

        return solution, plan
