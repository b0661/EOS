"""Energy simulation engine.

The engine coordinates the per-hour simulation loop across all registered
devices. It is intentionally stateless between ``simulate()`` calls —
all mutable state lives inside the devices themselves.

The engine never accesses device internals. It only calls the three public
device methods: ``request()``, ``simulate_hour()``, and (via registry)
``reset()``.
"""

from akkudoktoreos.devices.genetic2.base import EnergyDevice
from akkudoktoreos.simulation.genetic2.arbitrator import (
    ArbitratorBase,
    PriorityArbitrator,
)
from akkudoktoreos.simulation.genetic2.flows import ResourceGrant
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.result import (
    DeviceHourlyState,
    HourlyResult,
    SimulationResult,
)
from akkudoktoreos.simulation.genetic2.timeseries import SimulationInput


class EnergySimulationEngine:
    """Stateless energy flow simulation engine supporting multiple devices of each type.

    The engine runs a two-phase protocol each hour:

    **Phase 1 — Request:** every device declares what resources it wants.
    **Phase 2 — Arbitrate + Simulate:** the arbitrator allocates resources,
    then each device simulates given its grant.

    Adding a new device type requires no engine changes — implement the device
    interface, register it, and the engine picks it up automatically.

    Args:
        registry (DeviceRegistry): All devices participating in simulation.
        arbitrator (ArbitratorBase | None): Resource allocation strategy.
            Defaults to ``PriorityArbitrator`` if None.

    Example:
        >>> engine = EnergySimulationEngine(registry)
        >>> assembler.dispatch(genome, registry)  # apply genome to devices first
        >>> result = engine.simulate(inputs)
        >>> print(result.net_balance_eur)
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        arbitrator: ArbitratorBase | None = None,
    ) -> None:
        self._registry = registry
        self._arbitrator = arbitrator or PriorityArbitrator()

    def simulate(self, inputs: SimulationInput) -> SimulationResult:
        """Run a full simulation over the input time horizon.

        Resets all devices first, then simulates each hour in sequence.
        Device state (e.g. SoC) carries forward between hours within a run.

        Args:
            inputs (SimulationInput): Time-series data and schedules for this run.

        Returns:
            SimulationResult: Per-hour detail and aggregate metrics.
        """
        self._registry.reset_all()
        result = SimulationResult()

        for hour in range(inputs.start_hour, inputs.end_hour):
            result.hours.append(self._simulate_hour(hour, inputs))

        return result

    def _simulate_hour(self, hour: int, inputs: SimulationInput) -> HourlyResult:
        """Simulate a single hour across all devices.

        Implements the request → arbitrate → simulate protocol.

        Args:
            hour (int): Current simulation hour.
            inputs (SimulationInput): Full simulation inputs (price arrays etc.)

        Returns:
            HourlyResult: Aggregated energy flows and device states for this hour.
        """
        # Phase 1: collect requests from all devices
        requests = [device.request(hour) for device in self._registry.all_of_type(EnergyDevice)]

        # Phase 2: arbitrate — decide who gets what
        grants = self._arbitrator.arbitrate(requests)

        # Phase 3: simulate — each device acts on its grant
        device_states: dict[str, DeviceHourlyState] = {}
        for device in self._registry.all_of_type(EnergyDevice):
            grant = grants.get(device.device_id, ResourceGrant.idle(device.device_id))
            flows = device.simulate_hour(hour, grant)
            device_states[device.device_id] = DeviceHourlyState(
                device_id=device.device_id,
                flows=flows,
                curtailed=grant.curtailed,
            )

        # Phase 4: aggregate flows and compute financials
        return self._aggregate(hour, device_states, inputs)

    def _aggregate(
        self,
        hour: int,
        device_states: dict[str, DeviceHourlyState],
        inputs: SimulationInput,
    ) -> HourlyResult:
        """Aggregate per-device flows into system-level totals for one hour.

        The grid acts as the balancing element: any net surplus becomes
        feed-in, any net deficit becomes grid import.

        AC power sign convention (from EnergyFlows):
            Positive = device provides AC to system (reduces grid import).
            Negative = device consumes AC from system (increases grid import).

        Args:
            hour (int): Current simulation hour.
            device_states (dict[str, DeviceHourlyState]): All device states this hour.
            inputs (SimulationInput): Simulation inputs for price/tariff lookup.

        Returns:
            HourlyResult: Aggregated result for this hour.
        """
        # Base load from external input (not contributed by a device)
        total_load_wh = inputs.load_wh[hour]
        total_losses_wh = 0.0
        net_ac_wh = 0.0  # Positive = surplus (feed-in candidate), negative = deficit

        for state in device_states.values():
            flows = state.flows
            # Device load contributions (fixed loads, EV charging drawn from grid)
            total_load_wh += flows.load_wh
            total_losses_wh += flows.losses_wh
            # Net AC balance: sources add, sinks subtract
            net_ac_wh += flows.ac_power_wh

        # The base household load is a sink — subtract it from net
        net_ac_wh -= inputs.load_wh[hour]

        # Grid balances the remainder
        feedin_wh = max(net_ac_wh, 0.0)
        grid_import_wh = max(-net_ac_wh, 0.0)

        # Self-consumption: generation that stayed in the system (not fed in)
        total_generation = sum(
            s.flows.ac_power_wh for s in device_states.values() if s.flows.is_ac_source
        )
        self_consumption_wh = max(total_generation - feedin_wh, 0.0)

        price = inputs.electricity_price[hour]
        tariff = inputs.feed_in_tariff[hour]

        return HourlyResult(
            hour=hour,
            total_load_wh=total_load_wh,
            feedin_wh=feedin_wh,
            grid_import_wh=grid_import_wh,
            total_losses_wh=total_losses_wh,
            self_consumption_wh=self_consumption_wh,
            cost_eur=grid_import_wh * price,
            revenue_eur=feedin_wh * tariff,
            devices=device_states,
        )
