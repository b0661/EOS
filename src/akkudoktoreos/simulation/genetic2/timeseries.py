"""Typed input containers for the simulation engine.

``SimulationInput`` is the single object passed to
``EnergySimulationEngine.simulate()``.  It carries all time-series data
(loads, prices, tariffs) indexed by *step*, together with the step
schedule itself (``step_times`` and ``step_interval``).

The step-based design replaces the old ``start_hour`` / ``end_hour``
integer pair.  The number of steps and their wall-clock times are now
explicit, which lets the engine and devices operate at any uniform
resolution (15 min, 30 min, 1 h, …).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np


@dataclass
class DeviceSchedule:
    """Per-device per-step schedule for devices that do not use genome-derived scheduling.

    In the primary design, schedules are decoded from the genome by each
    device via ``apply_genome()``.  ``DeviceSchedule`` is used for devices
    that receive *external* schedules (e.g. a rule-based fixed-price
    charging strategy) rather than optimizer-derived ones.

    Both arrays must have the same length as ``SimulationInput.num_steps``.
    Factors are in [0, 1]: 0 means inactive, 1 means full power.

    Attributes:
        device_id (str): Device this schedule applies to.
        charge_factors (np.ndarray): Per-step charge factors in [0, 1].
        discharge_factors (np.ndarray): Per-step discharge factors in [0, 1].
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
    """All inputs required for one simulation run.

    The simulation operates on ``num_steps`` uniform time steps of duration
    ``step_interval``.  Every array attribute must have exactly ``num_steps``
    elements; index *i* corresponds to step *i* and wall-clock time
    ``step_times[i]``.

    The engine passes ``step_times`` and ``step_interval`` to
    ``EnergyDevice.apply_genome()`` so devices can precompute
    time-dependent schedules and scale energy quantities correctly.

    Attributes:
        step_times (list[datetime]): Wall-clock datetime at the *start* of
            each simulation step.  ``len(step_times)`` defines ``num_steps``.
            Steps must be uniformly spaced by ``step_interval``.
        step_interval (timedelta): Duration of one simulation step.
            Devices use this to convert power (W) to energy (Wh):
            ``energy_wh = power_w * step_interval.total_seconds() / 3600``.
        load_wh (np.ndarray): Base household load for each step in Wh.
            Does not include device loads (EV, appliances) — those are
            reported by the devices themselves via ``EnergyFlows.load_wh``.
        electricity_price (np.ndarray): Grid electricity price in EUR/Wh
            for each step.
        feed_in_tariff (np.ndarray): Feed-in compensation in EUR/Wh
            for each step.
        external_schedules (dict[str, DeviceSchedule]): Optional external
            per-device schedules for devices that bypass genome-derived
            scheduling.
    """

    step_times: list[datetime]
    step_interval: timedelta
    load_wh: np.ndarray
    electricity_price: np.ndarray
    feed_in_tariff: np.ndarray
    external_schedules: dict[str, DeviceSchedule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.num_steps
        if n == 0:
            raise ValueError("SimulationInput: step_times must not be empty.")
        for name, arr in [
            ("load_wh", self.load_wh),
            ("electricity_price", self.electricity_price),
            ("feed_in_tariff", self.feed_in_tariff),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"SimulationInput: '{name}' has {len(arr)} elements, "
                    f"expected {n} (= len(step_times))."
                )
        if self.step_interval.total_seconds() <= 0:
            raise ValueError(
                f"SimulationInput: step_interval must be positive, got {self.step_interval}."
            )

    @property
    def num_steps(self) -> int:
        """Total number of simulation steps."""
        return len(self.step_times)

    @property
    def total_duration(self) -> timedelta:
        """Total wall-clock duration of the simulation."""
        return self.step_interval * self.num_steps

    def get_schedule(self, device_id: str) -> DeviceSchedule | None:
        """Retrieve an external schedule for a device, or None if not set.

        Args:
            device_id (str): Device to look up.

        Returns:
            DeviceSchedule or None.
        """
        return self.external_schedules.get(device_id)

    def add_schedule(self, schedule: DeviceSchedule) -> None:
        """Register or replace an external schedule for a device.

        Args:
            schedule (DeviceSchedule): Schedule to register.
        """
        self.external_schedules[schedule.device_id] = schedule
