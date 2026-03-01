"""Vectorized battery device implementation."""

from dataclasses import dataclass

import numpy as np

from akkudoktoreos.device.devicesabc import EnergyDevice
from akkudoktoreos.optimization.genetic2.genome import GenomeSlice


@dataclass
class BatteryBatchState:
    """Batch runtime state for BatteryDevice.

    Attributes:
        power: Power schedule (population_size, horizon).
        soc: State of charge (population_size, horizon).
        repair: Optional repair array if constraints violated.
    """

    power: np.ndarray
    soc: np.ndarray
    repair: np.ndarray | None = None


class BatteryDevice(EnergyDevice):
    """Vectorized battery model supporting batch evaluation."""

    def __init__(
        self,
        device_id: str,
        capacity_kwh: float,
        max_power_kw: float,
    ) -> None:
        """Initialize battery device.

        Args:
            device_id: Unique device identifier.
            capacity_kwh: Total storage capacity.
            max_power_kw: Maximum charge/discharge power.
        """
        self.device_id = device_id
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw

    def setup_run(self, context):
        self._horizon = context.horizon
        self._dt = context.step_interval

    def genome_requirements(self):
        return GenomeSlice(
            start=0,
            length=self._horizon,
            lower_bound=-self.max_power_kw,
            upper_bound=self.max_power_kw,
        )

    def create_batch_state(self, population_size, horizon):
        return BatteryBatchState(
            power=np.zeros((population_size, horizon)),
            soc=np.zeros((population_size, horizon)),
        )

    def apply_genome_batch(self, state, genome_batch):
        state.power[:] = genome_batch

    def compute_batch(self, state):
        dt = self._dt
        capacity = self.capacity_kwh

        state.soc = np.cumsum(state.power * dt, axis=1)
        state.soc /= capacity

        violations = (state.soc < 0) | (state.soc > 1)

        if np.any(violations):
            repaired = np.clip(state.soc, 0, 1)
            state.repair = repaired.copy()
            state.soc[:] = repaired

        cost = np.sum(np.abs(state.power), axis=1) * 0.01
        return cost
