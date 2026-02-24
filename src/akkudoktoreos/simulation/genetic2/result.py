"""Typed simulation result containers.

Result types mirror the step-based simulation design introduced in
``SimulationInput`` and ``EnergyDevice``.  The primary change from the
hourly design is that every result object is indexed by *step* rather
than by calendar hour, and carries a ``step_time`` timestamp so
consumers never need to reconstruct wall-clock times from an index.

Repair proposals returned by ``EnergyDevice.repair_genome()`` after a
simulation run are also collected here, in ``SimulationResult.repairs``.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from akkudoktoreos.simulation.genetic2.flows import EnergyFlows


@dataclass
class DeviceStepState:
    """State snapshot and energy flows for one device during one simulation step.

    Attributes:
        device_id (str): Device this state belongs to.
        flows (EnergyFlows): Actual energy flows reported by the device for
            this step.
        curtailed (bool): True if the arbitrator curtailed this device for at
            least one resource type this step.
    """

    device_id: str
    flows: EnergyFlows
    curtailed: bool = False

    @property
    def soc_pct(self) -> float | None:
        """State of charge at the *start* of this step, or None for non-storage devices."""
        return self.flows.soc_pct


@dataclass
class StepResult:
    """Aggregated energy flows and financials for one simulation step.

    Attributes:
        step_index (int): Zero-based position of this step in the run.
        step_time (datetime): Wall-clock datetime at the start of this step.
        total_load_wh (float): Total system load for this step in Wh,
            including the base load and all device load contributions.
        feedin_wh (float): Energy fed into the grid in Wh.
        grid_import_wh (float): Energy drawn from the grid in Wh.
        total_losses_wh (float): Total conversion losses across all devices in Wh.
        self_consumption_wh (float): Generation consumed directly within the
            system (not fed to grid) in Wh.
        cost_eur (float): Cost of grid import for this step in EUR.
        revenue_eur (float): Revenue from grid feed-in for this step in EUR.
        devices (dict[str, DeviceStepState]): Per-device state for this step,
            keyed by device_id.
    """

    step_index: int
    step_time: datetime
    total_load_wh: float
    feedin_wh: float
    grid_import_wh: float
    total_losses_wh: float
    self_consumption_wh: float
    cost_eur: float
    revenue_eur: float
    devices: dict[str, DeviceStepState] = field(default_factory=dict)

    def device(self, device_id: str) -> DeviceStepState | None:
        """Retrieve state for a specific device this step.

        Args:
            device_id (str): Device to look up.

        Returns:
            DeviceStepState or None if device had no recorded state this step.
        """
        return self.devices.get(device_id)


@dataclass
class GenomeRepairRecord:
    """Record of a genome repair proposal accepted from a device after a run.

    Stored in ``SimulationResult.repairs`` so the genetic engine can apply
    repairs to the population after evaluating fitness.

    Attributes:
        device_id (str): Device that proposed the repair.
        repaired_slice (np.ndarray): Validated, defensively copied slice
            ready for the engine to write back into the full genome.
    """

    device_id: str
    repaired_slice: np.ndarray


@dataclass
class SimulationResult:
    """Full simulation result spanning all simulated steps.

    Provides per-step detail via ``steps``, aggregate financial and energy
    metrics as properties (for use as optimizer fitness signals), and any
    genome repair proposals collected from devices after the run.

    Attributes:
        steps (list[StepResult]): Ordered list of per-step results.
            ``steps[i]`` corresponds to ``SimulationInput.step_times[i]``.
        repairs (list[GenomeRepairRecord]): Validated genome repair proposals
            from devices, ready for the genetic engine to apply.  Empty if no
            device proposed a repair this run.
    """

    steps: list[StepResult] = field(default_factory=list)
    repairs: list[GenomeRepairRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Aggregate financial properties
    # ------------------------------------------------------------------

    @property
    def total_cost_eur(self) -> float:
        """Total grid import cost over all steps in EUR."""
        return sum(s.cost_eur for s in self.steps)

    @property
    def total_revenue_eur(self) -> float:
        """Total feed-in revenue over all steps in EUR."""
        return sum(s.revenue_eur for s in self.steps)

    @property
    def net_balance_eur(self) -> float:
        """Net financial balance (cost minus revenue) in EUR.

        The primary optimizer fitness signal — lower is better.
        A negative value means the system generated more revenue than it spent.
        """
        return self.total_cost_eur - self.total_revenue_eur

    # ------------------------------------------------------------------
    # Aggregate energy properties
    # ------------------------------------------------------------------

    @property
    def total_losses_wh(self) -> float:
        """Total conversion losses over all steps in Wh."""
        return sum(s.total_losses_wh for s in self.steps)

    @property
    def total_feedin_wh(self) -> float:
        """Total energy fed into the grid over all steps in Wh."""
        return sum(s.feedin_wh for s in self.steps)

    @property
    def total_grid_import_wh(self) -> float:
        """Total energy drawn from the grid over all steps in Wh."""
        return sum(s.grid_import_wh for s in self.steps)

    # ------------------------------------------------------------------
    # Per-device queries
    # ------------------------------------------------------------------

    def soc_per_step(self, device_id: str) -> list[float | None]:
        """Return the SoC trajectory for a specific storage device.

        Args:
            device_id (str): Device to query.

        Returns:
            list[float | None]: SoC percentage at the start of each step.
                None for steps where the device had no recorded state, or
                for non-storage devices.
        """
        return [
            s.devices[device_id].soc_pct if device_id in s.devices else None for s in self.steps
        ]

    def flows_per_step(self, device_id: str) -> list[EnergyFlows | None]:
        """Return the energy flow trajectory for a specific device.

        Args:
            device_id (str): Device to query.

        Returns:
            list[EnergyFlows | None]: Flows for each step, or None for steps
                where the device had no recorded state.
        """
        return [s.devices[device_id].flows if device_id in s.devices else None for s in self.steps]

    def all_device_ids(self) -> set[str]:
        """Return all device IDs that appear in any step result.

        Returns:
            set[str]: Device IDs with at least one recorded state.
        """
        return {device_id for s in self.steps for device_id in s.devices}

    def step_times(self) -> list[datetime]:
        """Return the wall-clock start time of every simulated step.

        Returns:
            list[datetime]: Ordered list of step start times.
        """
        return [s.step_time for s in self.steps]

    # ------------------------------------------------------------------
    # Backwards-compatible export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Export to the legacy EOS output dictionary format.

        Provides backwards compatibility with existing API consumers and
        visualisation code.  New code should use the typed properties and
        per-device query methods directly.

        Note:
            The legacy format uses per-*hour* keys (``_pro_Stunde``).  When
            running at sub-hourly resolution each list element represents one
            step, not one hour.  Callers that depend on hourly semantics must
            resample the output themselves.

        Returns:
            dict: Output matching the legacy ``simulate()`` return format.
        """
        device_ids = self.all_device_ids()
        return {
            "Last_Wh_pro_Stunde": [s.total_load_wh for s in self.steps],
            "Netzeinspeisung_Wh_pro_Stunde": [s.feedin_wh for s in self.steps],
            "Netzbezug_Wh_pro_Stunde": [s.grid_import_wh for s in self.steps],
            "Kosten_Euro_pro_Stunde": [s.cost_eur for s in self.steps],
            "Einnahmen_Euro_pro_Stunde": [s.revenue_eur for s in self.steps],
            "Verluste_Pro_Stunde": [s.total_losses_wh for s in self.steps],
            "Gesamtbilanz_Euro": self.net_balance_eur,
            "Gesamtkosten_Euro": self.total_cost_eur,
            "Gesamteinnahmen_Euro": self.total_revenue_eur,
            "Gesamt_Verluste": self.total_losses_wh,
            "step_times": [s.step_time.isoformat() for s in self.steps],
            "soc_per_step": {device_id: self.soc_per_step(device_id) for device_id in device_ids},
        }
