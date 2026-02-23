"""Complete usage example for the Akkudoktor-EOS simulation engine.

Shows how to wire together the registry, genome assembler, arbitrator,
and engine for a two-battery + PV + EV scenario.

Run with:
    python example_usage.py
"""

import numpy as np

from akkudoktoreos.devices.genetic2.base import EnergyDevice, GenomeSlice, StorageDevice
from akkudoktoreos.optimization.genetic2.genome import GenomeAssembler
from akkudoktoreos.simulation.genetic2.arbitrator import PriorityArbitrator
from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine

# ── imports (adjust paths to match your project layout) ──────────────────────
from akkudoktoreos.simulation.genetic2.flows import (
    EnergyFlows,
    Priority,
    ResourceGrant,
    ResourceRequest,
)
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.timeseries import SimulationInput

# ── minimal concrete device implementations for the example ──────────────────

class SimpleBattery(StorageDevice):
    """Minimal battery implementation for demonstration."""

    def __init__(
        self,
        device_id: str,
        capacity_wh: float,
        max_power_wh: float,
        initial_soc_pct: float,
        prediction_hours: int,
    ) -> None:
        super().__init__(device_id)
        self.capacity_wh = capacity_wh
        self.max_power_wh = max_power_wh
        self.initial_soc_pct = initial_soc_pct
        self.prediction_hours = prediction_hours
        self._soc_wh = (initial_soc_pct / 100) * capacity_wh
        self._charge_schedule = np.zeros(prediction_hours)
        self._discharge_schedule = np.zeros(prediction_hours)

    # ── GenomeSlice: one slot per hour, 0=idle, 1=discharge, 2=charge ────────
    def genome_requirements(self) -> GenomeSlice:
        return GenomeSlice(
            device_id=self.device_id,
            size=self.prediction_hours,
            dtype=int,
            low=0.0,
            high=2.0,
            description="0=idle, 1=discharge, 2=charge",
        )

    def apply_genome(self, genome_slice: np.ndarray) -> None:
        self._charge_schedule = (genome_slice == 2).astype(float)
        self._discharge_schedule = (genome_slice == 1).astype(float)

    def request(self, hour: int) -> ResourceRequest:
        if self._discharge_schedule[hour]:
            deliverable = min(self._soc_wh, self.max_power_wh)
            return ResourceRequest(
                device_id=self.device_id,
                hour=hour,
                priority=Priority.NORMAL,
                ac_power_wh=-deliverable,  # offering
            )
        if self._charge_schedule[hour]:
            headroom = self.capacity_wh - self._soc_wh
            wanted = min(headroom, self.max_power_wh)
            return ResourceRequest(
                device_id=self.device_id,
                hour=hour,
                priority=Priority.LOW,
                ac_power_wh=wanted,
                min_ac_power_wh=0.0,
            )
        return ResourceRequest(device_id=self.device_id, hour=hour)

    def simulate_hour(self, hour: int, grant: ResourceGrant) -> EnergyFlows:
        flows = EnergyFlows(soc_pct=self.current_soc_percentage())
        if grant.curtailed:
            return flows
        if grant.ac_power_wh < 0:
            # Discharging
            delivered = min(-grant.ac_power_wh, self._soc_wh)
            self._soc_wh -= delivered
            flows.ac_power_wh = delivered
        elif grant.ac_power_wh > 0:
            # Charging
            stored = min(grant.ac_power_wh, self.capacity_wh - self._soc_wh)
            self._soc_wh += stored
            flows.ac_power_wh = -stored
        return flows

    def reset(self) -> None:
        self._soc_wh = (self.initial_soc_pct / 100) * self.capacity_wh
        self._charge_schedule = np.zeros(self.prediction_hours)
        self._discharge_schedule = np.zeros(self.prediction_hours)

    def current_soc_percentage(self) -> float:
        return (self._soc_wh / self.capacity_wh) * 100

    def current_energy_content(self) -> float:
        return self._soc_wh


class SimplePV(EnergyDevice):
    """PV array — no genome, output is forecast-driven."""

    def __init__(self, device_id: str, forecast_wh: np.ndarray) -> None:
        super().__init__(device_id)
        self._forecast = forecast_wh

    def genome_requirements(self) -> None:
        return None

    def apply_genome(self, genome_slice: np.ndarray) -> None:
        pass

    def request(self, hour: int) -> ResourceRequest:
        generation = float(self._forecast[hour])
        return ResourceRequest(
            device_id=self.device_id,
            hour=hour,
            priority=Priority.CRITICAL,
            ac_power_wh=-generation,  # always offering
        )

    def simulate_hour(self, hour: int, grant: ResourceGrant) -> EnergyFlows:
        return EnergyFlows(
            ac_power_wh=float(self._forecast[hour]),
            generation_wh=float(self._forecast[hour]),
        )

    def reset(self) -> None:
        pass


# ── wire everything together ─────────────────────────────────────────────────

def main() -> None:
    hours = 24
    rng = np.random.default_rng(42)

    # Synthetic data
    pv_forecast = np.clip(np.sin(np.linspace(0, np.pi, hours)) * 3000, 0, None)
    load = np.full(hours, 500.0)
    price = np.full(hours, 0.30)   # EUR/Wh
    tariff = np.full(hours, 0.08)  # EUR/Wh feed-in

    # Build registry
    registry = DeviceRegistry()
    registry.register(SimplePV("pv_main", pv_forecast))
    registry.register(SimpleBattery("battery_main", capacity_wh=10000, max_power_wh=2500,
                                     initial_soc_pct=20, prediction_hours=hours))
    registry.register(SimpleBattery("battery_garage", capacity_wh=5000, max_power_wh=1500,
                                     initial_soc_pct=50, prediction_hours=hours))

    # Build genome assembler — queries all devices automatically
    assembler = GenomeAssembler(registry)
    print(assembler.describe())
    print()

    # Generate a random genome and dispatch to devices
    genome = assembler.random_genome(rng)
    assembler.dispatch(genome, registry)

    # Build simulation inputs
    inputs = SimulationInput(
        start_hour=0,
        end_hour=hours,
        load_wh=load,
        electricity_price=price,
        feed_in_tariff=tariff,
    )

    # Run simulation
    engine = EnergySimulationEngine(registry, arbitrator=PriorityArbitrator())
    result = engine.simulate(inputs)

    # Report
    print(f"Net balance:    {result.net_balance_eur:.2f} EUR")
    print(f"Total cost:     {result.total_cost_eur:.2f} EUR")
    print(f"Total revenue:  {result.total_revenue_eur:.2f} EUR")
    print(f"Total losses:   {result.total_losses_wh:.1f} Wh")
    print(f"Total feed-in:  {result.total_feedin_wh:.1f} Wh")
    print()

    # Per-device SoC trajectories
    for device_id in ["battery_main", "battery_garage"]:
        soc = result.soc_per_hour(device_id)
        print(f"{device_id} SoC (first 6h): {[f'{s:.1f}%' if s is not None else 'N/A' for s in soc[:6]]}")

    # Backwards-compatible dict export
    legacy = result.to_dict()
    print(f"\nLegacy keys: {list(legacy.keys())}")


if __name__ == "__main__":
    main()
