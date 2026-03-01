from akkudoktoreos.core.coreabc import ConfigMixin
from akkudoktoreos.core.emplan import EnergyManagementPlan
from akkudoktoreos.optimization.genetic2.optimizer import (
    GeneticOptimizer,
)
from akkudoktoreos.optimization.optimization import OptimizationSolution
from akkudoktoreos.simulation.genetic2.arbitrator import (
    BusTopology,
    VectorizedBusArbitrator,
)
from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry


class Genetic2Optimization(ConfigMixin):
    def optimize(self) -> tuple[OptimizationSolution, EnergyManagementPlan]:
        """Optimize using the configured values."""
        # Device registry for simulation
        registry = DeviceRegistry()
        for dev in devices:
            registry.register(dev)

        # Ports list of all devices
        total_ports = sum(len(d.ports) for d in devices)

        # Bus topology
        topo = BusTopology(
            port_to_bus=np.zeros(total_ports, dtype=int),
            num_buses=1,
        )

        # Arbitrator for simulation
        arb = VectorizedBusArbitrator(topo, horizon=horizon)

        # Simulation engine
        engine = EnergySimulationEngine(registry, [AC_BUS], arb)

        # Optimizer
        optimizer = GeneticOptimizer(
            engine=engine,
            population_size=pop,
            generations=gens,
            random_seed=seed,
            **kwargs,
        )
