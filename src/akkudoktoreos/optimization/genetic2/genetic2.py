





class Genetic2(ConfigMixin):

    def __init__(self):
        pass

    def optimize(self) -> solution, plan:
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

