"""Shared fixtures for the genetic2 simulation test suite.

Fixtures are organised by scope:
- session-scoped: expensive objects reused across all tests
- function-scoped (default): fresh state per test to prevent bleed-through

Concrete device implementations (SimpleBattery, SimplePV) live here so every
test module can import them without duplicating code.
"""

import os
import sys

import numpy as np
import pytest

from akkudoktoreos.devices.genetic2.base import EnergyDevice, GenomeSlice, StorageDevice
from akkudoktoreos.optimization.genetic2.genome import GenomeAssembler
from akkudoktoreos.simulation.genetic2.arbitrator import PriorityArbitrator
from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine
from akkudoktoreos.simulation.genetic2.flows import (
    EnergyFlows,
    Priority,
    ResourceGrant,
    ResourceRequest,
)
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.timeseries import SimulationInput

# Make the project root importable regardless of working directory
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))



# ---------------------------------------------------------------------------
# Minimal concrete device implementations used across all tests
# ---------------------------------------------------------------------------

class SimpleBattery(StorageDevice):
    """Minimal battery for testing. No efficiency losses for predictable results."""

    def __init__(
        self,
        device_id: str,
        capacity_wh: float = 10_000.0,
        max_power_wh: float = 2_500.0,
        initial_soc_pct: float = 50.0,
        prediction_hours: int = 24,
    ) -> None:
        super().__init__(device_id)
        self.capacity_wh = capacity_wh
        self.max_power_wh = max_power_wh
        self.initial_soc_pct = initial_soc_pct
        self.prediction_hours = prediction_hours
        self._soc_wh = (initial_soc_pct / 100) * capacity_wh
        self._charge_schedule = np.zeros(prediction_hours)
        self._discharge_schedule = np.zeros(prediction_hours)
        # Track what apply_genome received for assertion purposes
        self.last_genome_slice: np.ndarray | None = None

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
        self.last_genome_slice = genome_slice.copy()
        self._charge_schedule = (genome_slice == 2).astype(float)
        self._discharge_schedule = (genome_slice == 1).astype(float)

    def request(self, hour: int) -> ResourceRequest:
        if self._discharge_schedule[hour]:
            deliverable = min(self._soc_wh, self.max_power_wh)
            return ResourceRequest(
                device_id=self.device_id,
                hour=hour,
                priority=Priority.NORMAL,
                ac_power_wh=-deliverable,
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
            delivered = min(-grant.ac_power_wh, self._soc_wh)
            self._soc_wh -= delivered
            flows.ac_power_wh = delivered
        elif grant.ac_power_wh > 0:
            stored = min(grant.ac_power_wh, self.capacity_wh - self._soc_wh)
            self._soc_wh += stored
            flows.ac_power_wh = -stored
        return flows

    def reset(self) -> None:
        # Only reset physical state (SoC). Schedules survive reset because
        # apply_genome() is called once before simulate() and the engine calls
        # reset_all() inside simulate() — schedules must persist across that reset.
        self._soc_wh = (self.initial_soc_pct / 100) * self.capacity_wh

    def current_soc_percentage(self) -> float:
        return (self._soc_wh / self.capacity_wh) * 100

    def current_energy_content(self) -> float:
        return self._soc_wh


class SimplePV(EnergyDevice):
    """Fixed-forecast PV array. No genome — output is entirely forecast-driven."""

    def __init__(self, device_id: str, forecast_wh: np.ndarray) -> None:
        super().__init__(device_id)
        self._forecast = forecast_wh

    def genome_requirements(self) -> None:
        return None

    def apply_genome(self, genome_slice: np.ndarray) -> None:
        pass  # No genome

    def request(self, hour: int) -> ResourceRequest:
        return ResourceRequest(
            device_id=self.device_id,
            hour=hour,
            priority=Priority.CRITICAL,
            ac_power_wh=-float(self._forecast[hour]),
        )

    def simulate_hour(self, hour: int, grant: ResourceGrant) -> EnergyFlows:
        generation = float(self._forecast[hour])
        return EnergyFlows(
            ac_power_wh=generation,
            generation_wh=generation,
        )

    def reset(self) -> None:
        pass


class FixedLoad(EnergyDevice):
    """Fixed load device with no genome — always requests the same power each hour."""

    def __init__(self, device_id: str, load_wh: float, prediction_hours: int = 24) -> None:
        super().__init__(device_id)
        self._load_wh = load_wh
        self.prediction_hours = prediction_hours

    def genome_requirements(self) -> None:
        return None

    def apply_genome(self, genome_slice: np.ndarray) -> None:
        pass

    def request(self, hour: int) -> ResourceRequest:
        return ResourceRequest(
            device_id=self.device_id,
            hour=hour,
            priority=Priority.HIGH,
            ac_power_wh=self._load_wh,
            min_ac_power_wh=self._load_wh,
        )

    def simulate_hour(self, hour: int, grant: ResourceGrant) -> EnergyFlows:
        return EnergyFlows(
            ac_power_wh=-grant.ac_power_wh,
            load_wh=grant.ac_power_wh,
        )

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HOURS = 24


@pytest.fixture
def hours() -> int:
    return HOURS


@pytest.fixture
def flat_pv_forecast() -> np.ndarray:
    """Constant 1000 Wh generation every hour."""
    return np.full(HOURS, 1000.0)


@pytest.fixture
def zero_pv_forecast() -> np.ndarray:
    """No PV generation — forces grid import."""
    return np.zeros(HOURS)


@pytest.fixture
def price_array() -> np.ndarray:
    return np.full(HOURS, 0.30)


@pytest.fixture
def tariff_array() -> np.ndarray:
    return np.full(HOURS, 0.08)


@pytest.fixture
def base_load_array() -> np.ndarray:
    """500 Wh base load every hour."""
    return np.full(HOURS, 500.0)


@pytest.fixture
def zero_load_array() -> np.ndarray:
    return np.zeros(HOURS)


@pytest.fixture
def battery(hours) -> SimpleBattery:
    return SimpleBattery("battery_main", prediction_hours=hours)


@pytest.fixture
def battery_full(hours) -> SimpleBattery:
    return SimpleBattery("battery_full", initial_soc_pct=100.0, prediction_hours=hours)


@pytest.fixture
def battery_empty(hours) -> SimpleBattery:
    return SimpleBattery("battery_empty", initial_soc_pct=0.0, prediction_hours=hours)


@pytest.fixture
def pv(flat_pv_forecast) -> SimplePV:
    return SimplePV("pv_main", flat_pv_forecast)


@pytest.fixture
def registry(battery, pv) -> DeviceRegistry:
    """Basic registry with one battery and one PV array."""
    reg = DeviceRegistry()
    reg.register(pv)
    reg.register(battery)
    return reg


@pytest.fixture
def simulation_input(hours, base_load_array, price_array, tariff_array) -> SimulationInput:
    return SimulationInput(
        start_hour=0,
        end_hour=hours,
        load_wh=base_load_array,
        electricity_price=price_array,
        feed_in_tariff=tariff_array,
    )


@pytest.fixture
def assembler(registry) -> GenomeAssembler:
    return GenomeAssembler(registry)


@pytest.fixture
def engine(registry) -> EnergySimulationEngine:
    return EnergySimulationEngine(registry, arbitrator=PriorityArbitrator())


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for reproducible random genomes in tests."""
    return np.random.default_rng(42)
