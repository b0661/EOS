"""Standardized energy flow and resource types for the simulation engine.

All power values use Wh (watt-hours) per simulation hour.
Sign convention for EnergyFlows: positive = providing to system, negative = consuming.
Sign convention for ResourceRequest: positive = wants to consume, negative = offering to provide.
"""

from dataclasses import dataclass
from enum import IntEnum


class Priority(IntEnum):
    """Arbitration priority for resource allocation.

    Lower value means higher priority. Consumers with higher priority
    are served first when supply is insufficient.
    """

    CRITICAL = 0  # Life-safety or contractual must-run loads
    HIGH = 1  # Must-charge windows, heating demand
    NORMAL = 2  # Standard battery cycling
    LOW = 3  # Opportunistic charging, deferrable loads


@dataclass
class EnergyFlows:
    """Standardized per-hour energy flows reported by a device after simulation.

    Sign convention: positive = providing power to system, negative = consuming.

    Attributes:
        ac_power_wh (float): AC electrical flow. Positive = source (e.g. battery
            discharging), negative = sink (e.g. EV charging).
        dc_power_wh (float): DC electrical flow. Positive = source (e.g. PV
            generation), negative = sink.
        heat_provided_wh (float): Heat delivered into the system (always >= 0).
        heat_consumed_wh (float): Heat drawn from the system (always >= 0).
        losses_wh (float): Conversion losses this hour (always >= 0).
        soc_pct (float | None): State of charge percentage at start of hour.
            None for devices without storage.
        generation_wh (float): Raw generation before conversion (e.g. PV DC output).
        load_wh (float): Fixed load contribution from this device this hour.
    """

    ac_power_wh: float = 0.0
    dc_power_wh: float = 0.0
    heat_provided_wh: float = 0.0
    heat_consumed_wh: float = 0.0
    losses_wh: float = 0.0
    soc_pct: float | None = None
    generation_wh: float = 0.0
    load_wh: float = 0.0

    @property
    def is_ac_source(self) -> bool:
        """True if device is providing AC power to the system."""
        return self.ac_power_wh > 0

    @property
    def is_ac_sink(self) -> bool:
        """True if device is consuming AC power from the system."""
        return self.ac_power_wh < 0

    @property
    def is_heat_source(self) -> bool:
        """True if device is providing heat to the system."""
        return self.heat_provided_wh > 0


@dataclass
class ResourceRequest:
    """A device's declared resource needs for one simulation hour.

    Devices submit requests before any device simulates, allowing the
    arbitrator to see the full picture before making allocation decisions.

    Sign convention: positive = wants to consume, negative = offering to provide.

    Attributes:
        device_id (str): Unique identifier of the requesting device.
        hour (int): Simulation hour this request covers.
        priority (Priority): Arbitration priority for this request.
        ac_power_wh (float): Desired AC power. Negative means offering surplus.
        dc_power_wh (float): Desired DC power. Negative means offering surplus.
        heat_wh (float): Desired heat. Negative means offering heat.
        cold_wh (float): Desired cooling (reserved for future use).
        min_ac_power_wh (float): Minimum AC below which device prefers not to run.
        min_dc_power_wh (float): Minimum DC below which device prefers not to run.
        min_heat_wh (float): Minimum heat below which device prefers not to run.
    """

    device_id: str
    hour: int
    priority: Priority = Priority.NORMAL
    ac_power_wh: float = 0.0
    dc_power_wh: float = 0.0
    heat_wh: float = 0.0
    cold_wh: float = 0.0
    min_ac_power_wh: float = 0.0
    min_dc_power_wh: float = 0.0
    min_heat_wh: float = 0.0

    @property
    def is_producer(self) -> bool:
        """True if this device is offering resources rather than requesting them."""
        return self.ac_power_wh < 0 or self.dc_power_wh < 0 or self.heat_wh < 0

    @property
    def is_consumer(self) -> bool:
        """True if this device is requesting resources from the system."""
        return self.ac_power_wh > 0 or self.dc_power_wh > 0 or self.heat_wh > 0

    @property
    def is_idle(self) -> bool:
        """True if device has no resource interaction this hour."""
        return not self.is_producer and not self.is_consumer


@dataclass
class ResourceGrant:
    """The engine's allocation decision for one device for one hour.

    Produced by the arbitrator after reviewing all requests. Devices
    must respect grant limits — they may not consume more than granted.

    Attributes:
        device_id (str): Device this grant applies to.
        ac_power_wh (float): Allocated AC power in Wh.
        dc_power_wh (float): Allocated DC power in Wh.
        heat_wh (float): Allocated heat in Wh.
        cold_wh (float): Allocated cooling in Wh.
        curtailed (bool): True if device received less than its minimum
            request and should not run this hour.
    """

    device_id: str
    ac_power_wh: float = 0.0
    dc_power_wh: float = 0.0
    heat_wh: float = 0.0
    cold_wh: float = 0.0
    curtailed: bool = False

    @classmethod
    def idle(cls, device_id: str) -> "ResourceGrant":
        """Create a zero-allocation grant for an idle device."""
        return cls(device_id=device_id)

    @classmethod
    def curtailed_grant(cls, device_id: str) -> "ResourceGrant":
        """Create a curtailed grant — device should not run this hour."""
        return cls(device_id=device_id, curtailed=True)
