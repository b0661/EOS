"""Shared fixtures for the genetic2 simulation test suite.

Fixtures are organised by scope:
- session-scoped: expensive objects reused across all tests
- function-scoped (default): fresh state per test to prevent bleed-through

Concrete device implementations (SimpleBattery, SimplePV, FixedLoad) live
here so every test module can import them without duplicating code.

Step-based design
-----------------
All devices implement the updated EnergyDevice protocol:

  genome_requirements(num_steps, step_interval) -> GenomeSlice | None
  apply_genome(genome_slice, step_times) -> None   [ABSTRACT — must implement]
  request(step_index) -> ResourceRequest
  simulate_step(grant) -> EnergyFlows
  reset() -> None   [physical state only, not schedules]

dispatch() in GenomeAssembler now calls device.store_genome() instead of
apply_genome(). The engine calls apply_genome() at the start of simulate().
Devices read their stored raw values from get_stored_genome() inside
apply_genome() and decode them alongside step_times.
"""

import os
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta

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

#sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))



# ---------------------------------------------------------------------------
# Concrete device implementations
# ---------------------------------------------------------------------------

class SimpleBattery(StorageDevice):
    """Minimal battery for testing. No efficiency losses for predictable results.

    genome encoding per step: 0=idle, 1=discharge, 2=charge.

    apply_genome() reads the raw slice from get_stored_genome() (written by
    GenomeAssembler.dispatch via store_genome()) rather than from the
    genome_slice parameter, which the engine passes as empty.
    """

    def __init__(
        self,
        device_id: str,
        capacity_wh: float = 10_000.0,
        max_power_wh: float = 2_500.0,
        initial_soc_pct: float = 50.0,
        num_steps: int = 24,
    ) -> None:
        super().__init__(device_id)
        self.capacity_wh = capacity_wh
        self.max_power_wh = max_power_wh
        self.initial_soc_pct = initial_soc_pct
        self.num_steps = num_steps
        self._soc_wh = (initial_soc_pct / 100) * capacity_wh
        self._charge_schedule = np.zeros(num_steps)
        self._discharge_schedule = np.zeros(num_steps)
        # Recorded by apply_genome() so tests can assert on step_times forwarding
        self.last_step_times: list[datetime] | None = None

    def genome_requirements(
        self, num_steps: int, step_interval: timedelta
    ) -> GenomeSlice:
        return GenomeSlice(
            device_id=self.device_id,
            size=num_steps,
            dtype=int,
            low=0.0,
            high=2.0,
            description="0=idle, 1=discharge, 2=charge",
        )

    def apply_genome(
        self, genome_slice: np.ndarray, step_times: Sequence[datetime]
    ) -> None:
        self.last_step_times = list(step_times)
        # Raw slice was written by GenomeAssembler.dispatch() via store_genome().
        # The engine passes an empty array as genome_slice; real values come
        # from get_stored_genome().
        raw = self.get_stored_genome()
        if len(raw) == 0:
            self._charge_schedule = np.zeros(self.num_steps)
            self._discharge_schedule = np.zeros(self.num_steps)
            return
        self._charge_schedule = (raw == 2).astype(float)
        self._discharge_schedule = (raw == 1).astype(float)

    def request(self, step_index: int) -> ResourceRequest:
        if self._discharge_schedule[step_index]:
            deliverable = min(self._soc_wh, self.max_power_wh)
            return ResourceRequest(
                device_id=self.device_id,
                hour=step_index,
                priority=Priority.NORMAL,
                ac_power_wh=-deliverable,
            )
        if self._charge_schedule[step_index]:
            headroom = self.capacity_wh - self._soc_wh
            wanted = min(headroom, self.max_power_wh)
            return ResourceRequest(
                device_id=self.device_id,
                hour=step_index,
                priority=Priority.LOW,
                ac_power_wh=wanted,
                min_ac_power_wh=0.0,
            )
        return ResourceRequest(device_id=self.device_id, hour=step_index)

    def simulate_step(self, grant: ResourceGrant) -> EnergyFlows:
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
        # Restore physical state only. Schedules are populated by apply_genome()
        # which is called immediately after reset() at the start of each run.
        self._soc_wh = (self.initial_soc_pct / 100) * self.capacity_wh

    def current_soc_percentage(self) -> float:
        return (self._soc_wh / self.capacity_wh) * 100

    def current_energy_content(self) -> float:
        return self._soc_wh


class SimplePV(EnergyDevice):
    """Fixed-forecast PV array. No genome — output is entirely forecast-driven.

    Caches the current step_index in request() so simulate_step() can read
    the correct forecast element without needing step_index again.
    """

    def __init__(self, device_id: str, forecast_wh: np.ndarray) -> None:
        super().__init__(device_id)
        self._forecast = forecast_wh
        self._current_step = 0

    def genome_requirements(
        self, num_steps: int, step_interval: timedelta
    ) -> None:
        return None

    def apply_genome(
        self, genome_slice: np.ndarray, step_times: Sequence[datetime]
    ) -> None:
        self._current_step = 0

    def request(self, step_index: int) -> ResourceRequest:
        self._current_step = step_index
        return ResourceRequest(
            device_id=self.device_id,
            hour=step_index,
            priority=Priority.CRITICAL,
            ac_power_wh=-float(self._forecast[step_index]),
        )

    def simulate_step(self, grant: ResourceGrant) -> EnergyFlows:
        generation = float(self._forecast[self._current_step])
        return EnergyFlows(
            ac_power_wh=generation,
            generation_wh=generation,
        )

    def reset(self) -> None:
        self._current_step = 0


class FixedLoad(EnergyDevice):
    """Fixed load device with no genome — requests the same power every step."""

    def __init__(self, device_id: str, load_wh: float) -> None:
        super().__init__(device_id)
        self._load_wh = load_wh

    def genome_requirements(
        self, num_steps: int, step_interval: timedelta
    ) -> None:
        return None

    def apply_genome(
        self, genome_slice: np.ndarray, step_times: Sequence[datetime]
    ) -> None:
        pass

    def request(self, step_index: int) -> ResourceRequest:
        return ResourceRequest(
            device_id=self.device_id,
            hour=step_index,
            priority=Priority.HIGH,
            ac_power_wh=self._load_wh,
            min_ac_power_wh=self._load_wh,
        )

    def simulate_step(self, grant: ResourceGrant) -> EnergyFlows:
        return EnergyFlows(
            ac_power_wh=-grant.ac_power_wh,
            load_wh=grant.ac_power_wh,
        )

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Test constants and helpers (also imported by test modules)
# ---------------------------------------------------------------------------

STEPS = 24
STEP_INTERVAL = timedelta(hours=1)
START_TIME = datetime(2024, 1, 1, 0, 0, 0)


def make_step_times(
    n: int = STEPS,
    interval: timedelta = STEP_INTERVAL,
    start: datetime = START_TIME,
) -> list[datetime]:
    """Build a list of n step datetimes at *interval* spacing starting at *start*."""
    return [start + i * interval for i in range(n)]


def make_input(
    n: int = STEPS,
    load: float = 500.0,
    price: float = 0.30,
    tariff: float = 0.08,
    interval: timedelta = STEP_INTERVAL,
    start: datetime = START_TIME,
) -> SimulationInput:
    """Factory for SimulationInput — single source of truth for constructor args."""
    times = make_step_times(n, interval, start)
    return SimulationInput(
        step_times=times,
        step_interval=interval,
        load_wh=np.full(n, load),
        electricity_price=np.full(n, price),
        feed_in_tariff=np.full(n, tariff),
    )


def make_assembler(
    registry: DeviceRegistry,
    n: int = STEPS,
    interval: timedelta = STEP_INTERVAL,
) -> GenomeAssembler:
    """Factory for GenomeAssembler — passes num_steps and step_interval."""
    return GenomeAssembler(registry, num_steps=n, step_interval=interval)


# ---------------------------------------------------------------------------
# pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def num_steps() -> int:
    return STEPS


@pytest.fixture
def step_interval() -> timedelta:
    return STEP_INTERVAL


@pytest.fixture
def step_times(num_steps, step_interval) -> list[datetime]:
    return make_step_times(num_steps, step_interval)


@pytest.fixture
def flat_pv_forecast(num_steps) -> np.ndarray:
    return np.full(num_steps, 1000.0)


@pytest.fixture
def zero_pv_forecast(num_steps) -> np.ndarray:
    return np.zeros(num_steps)


@pytest.fixture
def price_array(num_steps) -> np.ndarray:
    return np.full(num_steps, 0.30)


@pytest.fixture
def tariff_array(num_steps) -> np.ndarray:
    return np.full(num_steps, 0.08)


@pytest.fixture
def base_load_array(num_steps) -> np.ndarray:
    return np.full(num_steps, 500.0)


@pytest.fixture
def zero_load_array(num_steps) -> np.ndarray:
    return np.zeros(num_steps)


@pytest.fixture
def battery(num_steps) -> SimpleBattery:
    return SimpleBattery("battery_main", num_steps=num_steps)


@pytest.fixture
def battery_full(num_steps) -> SimpleBattery:
    return SimpleBattery("battery_full", initial_soc_pct=100.0, num_steps=num_steps)


@pytest.fixture
def battery_empty(num_steps) -> SimpleBattery:
    return SimpleBattery("battery_empty", initial_soc_pct=0.0, num_steps=num_steps)


@pytest.fixture
def pv(flat_pv_forecast) -> SimplePV:
    return SimplePV("pv_main", flat_pv_forecast)


@pytest.fixture
def registry(battery, pv) -> DeviceRegistry:
    reg = DeviceRegistry()
    reg.register(pv)
    reg.register(battery)
    return reg


@pytest.fixture
def simulation_input(
    step_times, step_interval, base_load_array, price_array, tariff_array
) -> SimulationInput:
    return SimulationInput(
        step_times=step_times,
        step_interval=step_interval,
        load_wh=base_load_array,
        electricity_price=price_array,
        feed_in_tariff=tariff_array,
    )


@pytest.fixture
def assembler(registry, num_steps, step_interval) -> GenomeAssembler:
    return GenomeAssembler(registry, num_steps=num_steps, step_interval=step_interval)


@pytest.fixture
def engine(registry) -> EnergySimulationEngine:
    return EnergySimulationEngine(registry, arbitrator=PriorityArbitrator())


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
