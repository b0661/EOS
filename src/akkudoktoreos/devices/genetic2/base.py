"""Abstract base classes for simulation devices.

Every physical device in the simulation inherits from EnergyDevice and
implements the three-phase protocol:

1. ``genome_requirements()`` — declare optimizer genome needs before any run.
2. ``apply_genome()``        — receive and interpret genome slice once per run.
3. ``request()``             — declare resource needs for a specific hour (Phase 1 per hour).
4. ``simulate_hour()``       — simulate given granted resources (Phase 2 per hour).

Devices own their own physics and internal state. The engine and optimizer
never access device internals directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from akkudoktoreos.simulation.genetic2.flows import (
    EnergyFlows,
    ResourceGrant,
    ResourceRequest,
)


@dataclass
class GenomeSlice:
    """Declares what portion of the optimizer genome a device requires.

    Attributes:
        device_id (str): ID of the device making this declaration.
        size (int): Number of genome slots required (one per prediction hour
            is the most common case).
        dtype (type): Element type — int for discrete states, float for
            continuous values.
        low (float): Inclusive lower bound for each genome slot.
        high (float): Inclusive upper bound for each genome slot.
        description (str): Human-readable explanation of what each slot
            encodes, e.g. "0=idle, 1..N=charge factor index".
    """

    device_id: str
    size: int
    dtype: type = int
    low: float = 0.0
    high: float = 1.0
    description: str = ""

    def validate(self, genome_slice: np.ndarray) -> None:
        """Validate that a genome slice matches this declaration.

        Args:
            genome_slice (np.ndarray): Slice to validate.

        Raises:
            ValueError: If size or bounds are violated.
        """
        if len(genome_slice) != self.size:
            raise ValueError(
                f"Device '{self.device_id}': genome slice has {len(genome_slice)} elements, "
                f"expected {self.size}."
            )
        if np.any(genome_slice < self.low) or np.any(genome_slice > self.high):
            raise ValueError(
                f"Device '{self.device_id}': genome values must be in "
                f"[{self.low}, {self.high}], got min={genome_slice.min()}, "
                f"max={genome_slice.max()}."
            )


class EnergyDevice(ABC):
    """Base class for all simulation devices.

    Each device is self-contained: it declares what genome slots it needs,
    interprets those slots internally, and exposes only standardized
    EnergyFlows to the engine. The engine and optimizer never access
    device internals directly.

    The simulation lifecycle per run:

    1. ``apply_genome(slice)`` called once — device stores its schedule.
    2. For each hour:
        a. ``request(hour)`` called — device declares what it wants.
        b. ``simulate_hour(hour, grant)`` called — device acts on its grant.
    3. ``reset()`` called before the next run.

    Args:
        device_id (str): Unique identifier for this device instance.
            Must be unique within a DeviceRegistry.
    """

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id

    @abstractmethod
    def genome_requirements(self) -> GenomeSlice | None:
        """Declare what portion of the optimizer genome this device needs.

        Called once during optimizer setup to build the full genome structure.
        The GenomeAssembler uses all device declarations to assign non-overlapping
        genome slices.

        Returns:
            GenomeSlice describing required genome shape and bounds, or None
            if this device has no optimizable parameters (e.g. fixed load profile).
        """

    @abstractmethod
    def apply_genome(self, genome_slice: np.ndarray) -> None:
        """Receive and interpret this device's genome slice.

        Called once per simulation run, before any ``simulate_hour()`` call.
        The device should decode the genome into its internal schedule
        (e.g. per-hour charge factors, on/off states).

        Args:
            genome_slice (np.ndarray): Array of length ``genome_requirements().size``.
                Will be empty if ``genome_requirements()`` returns None.
        """

    @abstractmethod
    def request(self, hour: int) -> ResourceRequest:
        """Phase 1: Declare resource needs for a specific hour.

        Called before any device simulates that hour, so all requests are
        visible to the arbitrator simultaneously. Devices should base requests
        on their current internal state (e.g. SoC) and their decoded schedule.

        Args:
            hour (int): Simulation hour index.

        Returns:
            ResourceRequest describing what this device wants to produce
            or consume. Return an idle ResourceRequest if the device has
            no interaction this hour.
        """

    @abstractmethod
    def simulate_hour(self, hour: int, grant: ResourceGrant) -> EnergyFlows:
        """Phase 2: Simulate one hour given the granted resources.

        Called after arbitration. The device must respect grant limits —
        it cannot consume more than granted, even if it requested more.
        If ``grant.curtailed`` is True, the device should not run.

        Internal state (e.g. SoC) must be updated here.

        Args:
            hour (int): Simulation hour index.
            grant (ResourceGrant): Resources allocated by the arbitrator.

        Returns:
            EnergyFlows describing actual energy flows this hour.
            These may differ from the grant if internal constraints
            (e.g. SoC limits) prevent full utilization.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset device to its initial state.

        Called before each simulation run. Must restore SoC, schedules,
        and any other mutable state to initial values.
        """

    def to_dict(self) -> dict:
        """Serialize current device state to a dictionary.

        Override in subclasses to include device-specific fields.

        Returns:
            dict with at minimum ``device_id``.
        """
        return {"device_id": self.device_id}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device_id={self.device_id!r})"


class StorageDevice(EnergyDevice, ABC):
    """Base class for devices that store energy (Battery, EV, ThermalStorage).

    Adds shared SoC tracking and energy content reporting on top of
    the base EnergyDevice protocol.
    """

    @abstractmethod
    def current_soc_percentage(self) -> float:
        """Return current state of charge as a percentage (0–100).

        Returns:
            float: Current SoC in percent.
        """

    @abstractmethod
    def current_energy_content(self) -> float:
        """Return current usable energy available for discharge in Wh.

        Accounts for minimum SoC and discharge efficiency.

        Returns:
            float: Usable energy in Wh (>= 0).
        """


class GenerationDevice(EnergyDevice, ABC):
    """Base class for devices that generate energy (PV arrays, wind turbines).

    Generation devices typically have no genome (output is forecast-driven)
    but may have optimizable parameters in future (e.g. curtailment factor).
    """


class LoadDevice(EnergyDevice, ABC):
    """Base class for devices with a fixed or scheduled load profile.

    Fixed loads have no genome — their consumption is determined by
    an external forecast or profile. Schedulable loads (e.g. dishwasher)
    may have a genome that encodes their start time.
    """
