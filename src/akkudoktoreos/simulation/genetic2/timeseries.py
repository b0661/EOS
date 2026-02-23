"""Typed input and result containers for the simulation engine."""

# simulation/timeseries.py
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeviceSchedule:
    """Per-device hourly schedule for one simulation run.

    In the new design, schedules are decoded from the genome by each
    device itself via ``apply_genome()``. This class is used for
    devices that receive external schedules (e.g. fixed price-signal
    based charging) rather than genome-derived ones.

    Attributes:
        device_id (str): Device this schedule applies to.
        charge_factors (np.ndarray): Per-hour charge factors in [0, 1].
            0 means no charging this hour.
        discharge_factors (np.ndarray): Per-hour discharge factors in [0, 1].
            0 means no discharging this hour.
    """

    device_id: str
    charge_factors: np.ndarray
    discharge_factors: np.ndarray

    def __post_init__(self) -> None:
        if len(self.charge_factors) != len(self.discharge_factors):
            raise ValueError(
                f"Device '{self.device_id}': charge_factors length "
                f"({len(self.charge_factors)}) must match discharge_factors "
                f"length ({len(self.discharge_factors)})."
            )
        if np.any(self.charge_factors < 0) or np.any(self.charge_factors > 1):
            raise ValueError(f"Device '{self.device_id}': charge_factors must be in [0, 1].")
        if np.any(self.discharge_factors < 0) or np.any(self.discharge_factors > 1):
            raise ValueError(f"Device '{self.device_id}': discharge_factors must be in [0, 1].")


@dataclass
class SimulationInput:
    """All time-series inputs required for one simulation run.

    Arrays must all have the same length (``end_hour``). Indices correspond
    directly to hour numbers, so ``load_wh[5]`` is the load in hour 5.

    Attributes:
        start_hour (int): First hour to simulate (inclusive).
        end_hour (int): Last hour to simulate (exclusive).
        load_wh (np.ndarray): Base household load per hour in Wh.
            Does not include device loads (EV, appliances) — those are
            contributed by the devices themselves.
        electricity_price (np.ndarray): Grid electricity price in EUR/Wh per hour.
        feed_in_tariff (np.ndarray): Feed-in compensation in EUR/Wh per hour.
        external_schedules (dict[str, DeviceSchedule]): Optional external
            schedules for devices that don't use genome-derived scheduling.
    """

    start_hour: int
    end_hour: int
    load_wh: np.ndarray
    electricity_price: np.ndarray
    feed_in_tariff: np.ndarray
    external_schedules: dict[str, DeviceSchedule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.end_hour
        for name, arr in [
            ("load_wh", self.load_wh),
            ("electricity_price", self.electricity_price),
            ("feed_in_tariff", self.feed_in_tariff),
        ]:
            if len(arr) < n:
                raise ValueError(
                    f"SimulationInput: '{name}' has {len(arr)} elements, "
                    f"need at least {n} (end_hour)."
                )

    def get_schedule(self, device_id: str) -> DeviceSchedule | None:
        """Retrieve an external schedule for a device, or None if not set."""
        return self.external_schedules.get(device_id)

    def add_schedule(self, schedule: DeviceSchedule) -> None:
        """Register an external schedule for a device."""
        self.external_schedules[schedule.device_id] = schedule

    @property
    def total_hours(self) -> int:
        """Number of hours in this simulation run."""
        return self.end_hour - self.start_hour
