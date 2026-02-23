"""Typed simulation result containers."""

from dataclasses import dataclass, field

from akkudoktoreos.simulation.genetic2.flows import EnergyFlows


@dataclass
class DeviceHourlyState:
    """State snapshot and energy flows for one device during one hour.

    Attributes:
        device_id (str): Device this state belongs to.
        flows (EnergyFlows): Actual energy flows reported by the device.
        curtailed (bool): True if the device was curtailed by the arbitrator.
    """

    device_id: str
    flows: EnergyFlows
    curtailed: bool = False

    @property
    def soc_pct(self) -> float | None:
        """State of charge at start of hour, or None for non-storage devices."""
        return self.flows.soc_pct


@dataclass
class HourlyResult:
    """Aggregated energy flows and financials for one simulation hour.

    Attributes:
        hour (int): Simulation hour index.
        total_load_wh (float): Total system load including device contributions.
        feedin_wh (float): Energy fed into the grid in Wh.
        grid_import_wh (float): Energy drawn from the grid in Wh.
        total_losses_wh (float): Total conversion losses across all devices.
        self_consumption_wh (float): Energy consumed directly from generation.
        cost_eur (float): Cost of grid import this hour in EUR.
        revenue_eur (float): Revenue from grid feed-in this hour in EUR.
        devices (dict[str, DeviceHourlyState]): Per-device state this hour,
            keyed by device_id.
    """

    hour: int
    total_load_wh: float
    feedin_wh: float
    grid_import_wh: float
    total_losses_wh: float
    self_consumption_wh: float
    cost_eur: float
    revenue_eur: float
    devices: dict[str, DeviceHourlyState] = field(default_factory=dict)

    def device(self, device_id: str) -> DeviceHourlyState | None:
        """Retrieve state for a specific device this hour.

        Args:
            device_id (str): Device to look up.

        Returns:
            DeviceHourlyState or None if device had no activity this hour.
        """
        return self.devices.get(device_id)


@dataclass
class SimulationResult:
    """Full simulation result spanning all simulated hours.

    Provides both per-hour detail via ``hours`` and aggregate properties
    for use as optimizer fitness signals.

    Attributes:
        hours (list[HourlyResult]): Ordered list of per-hour results.
            Index 0 corresponds to ``start_hour`` from SimulationInput.
    """

    hours: list[HourlyResult] = field(default_factory=list)

    # --- Aggregate properties ---

    @property
    def total_cost_eur(self) -> float:
        """Total grid import cost over all hours in EUR."""
        return sum(h.cost_eur for h in self.hours)

    @property
    def total_revenue_eur(self) -> float:
        """Total feed-in revenue over all hours in EUR."""
        return sum(h.revenue_eur for h in self.hours)

    @property
    def total_losses_wh(self) -> float:
        """Total conversion losses over all hours in Wh."""
        return sum(h.total_losses_wh for h in self.hours)

    @property
    def net_balance_eur(self) -> float:
        """Net financial balance (cost minus revenue) in EUR.

        Used as the primary optimizer fitness signal — lower is better.
        """
        return self.total_cost_eur - self.total_revenue_eur

    @property
    def total_feedin_wh(self) -> float:
        """Total energy fed into the grid over all hours in Wh."""
        return sum(h.feedin_wh for h in self.hours)

    @property
    def total_grid_import_wh(self) -> float:
        """Total energy drawn from the grid over all hours in Wh."""
        return sum(h.grid_import_wh for h in self.hours)

    # --- Per-device queries ---

    def soc_per_hour(self, device_id: str) -> list[float | None]:
        """Return the SoC trajectory for a specific device.

        Args:
            device_id (str): Device to query.

        Returns:
            list[float | None]: SoC percentage at the start of each simulated
                hour. None for hours where the device had no recorded state,
                or for non-storage devices.
        """
        return [
            h.devices[device_id].soc_pct if device_id in h.devices else None for h in self.hours
        ]

    def flows_per_hour(self, device_id: str) -> list[EnergyFlows | None]:
        """Return the energy flow trajectory for a specific device.

        Args:
            device_id (str): Device to query.

        Returns:
            list[EnergyFlows | None]: Flows for each simulated hour,
                None for hours where the device had no recorded state.
        """
        return [h.devices[device_id].flows if device_id in h.devices else None for h in self.hours]

    def all_device_ids(self) -> set[str]:
        """Return all device IDs that appear in any hourly result.

        Returns:
            set[str]: Set of device IDs with recorded state.
        """
        return {device_id for h in self.hours for device_id in h.devices}

    # --- Backwards-compatible export ---

    def to_dict(self) -> dict:
        """Export to the legacy EOS output dictionary format.

        Provides backwards compatibility with existing API consumers and
        visualization code. New code should use the typed properties directly.

        Returns:
            dict: Output matching the legacy ``simulate()`` return format.
        """
        device_ids = self.all_device_ids()
        return {
            "Last_Wh_pro_Stunde": [h.total_load_wh for h in self.hours],
            "Netzeinspeisung_Wh_pro_Stunde": [h.feedin_wh for h in self.hours],
            "Netzbezug_Wh_pro_Stunde": [h.grid_import_wh for h in self.hours],
            "Kosten_Euro_pro_Stunde": [h.cost_eur for h in self.hours],
            "Einnahmen_Euro_pro_Stunde": [h.revenue_eur for h in self.hours],
            "Verluste_Pro_Stunde": [h.total_losses_wh for h in self.hours],
            "Gesamtbilanz_Euro": self.net_balance_eur,
            "Gesamtkosten_Euro": self.total_cost_eur,
            "Gesamteinnahmen_Euro": self.total_revenue_eur,
            "Gesamt_Verluste": self.total_losses_wh,
            # Per-device SoC, keyed by device_id
            "soc_per_hour": {device_id: self.soc_per_hour(device_id) for device_id in device_ids},
        }
