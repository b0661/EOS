"""Shared topology configuration and base classes for all device settings.

This module contains the building blocks that every device settings class
depends on:

- ``PortConfig`` / ``BusConstraintConfig`` / ``BusConfig`` / ``BusesConfig``:
  topology declarations (ports and buses).
- ``PortsMixin``: validated ``ports`` field mixin.
- ``DevicesBaseSettings``: common ``device_id`` field for all devices.

Nothing in this module is optimizer-specific.
"""

import json
from typing import Any, Optional, TextIO, cast

from pydantic import Field, model_validator

from akkudoktoreos.config.configabc import SettingsBaseModel
from akkudoktoreos.core.cache import CacheFileStore
from akkudoktoreos.core.coreabc import ConfigMixin, SingletonMixin
from akkudoktoreos.core.emplan import ResourceStatus
from akkudoktoreos.core.pydantic import ConfigDict, PydanticBaseModel
from akkudoktoreos.devices.devicesabc import (
    EnergyBus,
    EnergyBusConstraint,
    EnergyCarrier,
    EnergyPort,
    PortDirection,
)
from akkudoktoreos.utils.datetimeutil import DateTime, to_datetime


# ============================================================
# Topology config
# ============================================================


class PortConfig(PydanticBaseModel):
    """Configuration for a single device port.

    Each port connects a device to one energy bus. The carrier is
    determined by the bus — ports do not redeclare it.

    Attributes:
        port_id: Unique identifier for this port within the owning device.
            Must be unique per device; need not be globally unique.
        bus_id: References a ``BusConfig.bus_id`` that must exist in the
            same ``BusesConfig``.
        direction: Energy flow direction from the device's perspective.
            ``SOURCE`` = device injects energy onto the bus.
            ``SINK`` = device consumes energy from the bus.
            ``BIDIRECTIONAL`` = device can do both (battery, grid).
        max_power_w: Optional maximum power [W] through this port.
            ``None`` means no port-level limit (bus or device limits still
            apply).
    """

    port_id: str = Field(
        ...,
        json_schema_extra={
            "description": "Unique port identifier within this device.",
            "examples": ["p_dc", "p_ac"],
        },
    )
    bus_id: str = Field(
        ...,
        json_schema_extra={
            "description": "ID of the bus this port connects to.",
            "examples": ["bus_dc", "bus_ac"],
        },
    )
    direction: PortDirection = Field(
        ...,
        json_schema_extra={
            "description": (
                "Energy flow direction from the device's perspective. "
                "'source' = injects energy, 'sink' = consumes energy, "
                "'bidirectional' = both."
            ),
            "examples": ["source", "sink", "bidirectional"],
        },
    )
    max_power_w: Optional[float] = Field(
        default=None,
        gt=0,
        json_schema_extra={
            "description": "Maximum power through this port [W]. null means no limit.",
            "examples": [5000, None],
        },
    )

    def to_genetic2_param(self) -> EnergyPort:
        """Return an immutable ``EnergyPort`` domain object."""
        return EnergyPort(
            port_id=self.port_id,
            bus_id=self.bus_id,
            direction=self.direction,
            max_power_w=self.max_power_w,
        )


class BusConstraintConfig(PydanticBaseModel):
    """Optional structural constraints on port counts for a bus.

    Attributes:
        max_sinks: Maximum number of sink ports allowed on this bus,
            or ``None`` for no limit.
        max_sources: Maximum number of source ports allowed on this bus,
            or ``None`` for no limit.
    """

    max_sinks: Optional[int] = Field(
        default=None,
        ge=1,
        json_schema_extra={
            "description": "Maximum number of sink ports on this bus.",
            "examples": [1, None],
        },
    )
    max_sources: Optional[int] = Field(
        default=None,
        ge=1,
        json_schema_extra={
            "description": "Maximum number of source ports on this bus.",
            "examples": [1, None],
        },
    )

    def to_genetic2_param(self) -> EnergyBusConstraint:
        """Return an immutable ``EnergyBusConstraint`` domain object."""
        return EnergyBusConstraint(
            max_sinks=self.max_sinks,
            max_sources=self.max_sources,
        )


class BusConfig(PydanticBaseModel):
    """Configuration for a single energy bus.

    A bus is a connection point that devices attach to via ports. All ports
    on the same bus exchange energy of the same carrier type.

    Attributes:
        bus_id: Unique identifier for this bus. Referenced by
            ``PortConfig.bus_id`` on all devices that connect to it.
        carrier: Energy carrier type flowing through this bus.
            ``"ac"`` = AC electricity, ``"dc"`` = DC electricity,
            ``"heat"`` = thermal energy.
        constraint: Optional structural limits on the number of source or
            sink ports. Used by topology validation at engine construction.
    """

    bus_id: str = Field(
        ...,
        json_schema_extra={
            "description": "Unique bus identifier. Referenced by device port configs.",
            "examples": ["bus_ac", "bus_dc", "bus_heat"],
        },
    )
    carrier: EnergyCarrier = Field(
        ...,
        json_schema_extra={
            "description": "Energy carrier type: 'ac', 'dc', or 'heat'.",
            "examples": ["ac", "dc", "heat"],
        },
    )
    constraint: Optional[BusConstraintConfig] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Optional structural constraints on the number of source or sink "
                "ports. Validated at engine construction time."
            ),
            "examples": [None, {"max_sinks": 1}],
        },
    )

    def to_genetic2_param(self) -> EnergyBus:
        """Return an immutable ``EnergyBus`` domain object."""
        return EnergyBus(
            bus_id=self.bus_id,
            carrier=self.carrier,
            constraint=self.constraint.to_genetic2_param() if self.constraint else None,
        )


# ============================================================
# Shared port validation mixin
# ============================================================


class PortsMixin(PydanticBaseModel):
    """Mixin that adds a validated ``ports`` field to any device config.

    Validates that all ``port_id`` values within the device are unique.
    """

    ports: list[PortConfig] = Field(
        ...,
        min_length=1,
        json_schema_extra={
            "description": (
                "Ports connecting this device to energy buses. "
                "At least one port is required. "
                "Each port_id must be unique within this device."
            ),
            "examples": [[{"port_id": "p_dc", "bus_id": "bus_dc", "direction": "bidirectional"}]],
        },
    )

    @model_validator(mode="after")
    def _validate_unique_port_ids(self) -> "PortsMixin":
        seen: set[str] = set()
        for port in self.ports:
            if port.port_id in seen:
                raise ValueError(
                    f"Duplicate port_id '{port.port_id}' in device "
                    f"'{getattr(self, 'device_id', '<unknown>')}'"
                )
            seen.add(port.port_id)
        return self

    def _domain_ports(self) -> tuple[EnergyPort, ...]:
        """Return ports as an immutable tuple of domain objects."""
        return tuple(p.to_genetic2_param() for p in self.ports)


# ============================================================
# Base settings
# ============================================================


class DevicesBaseSettings(SettingsBaseModel):
    """Base class for all device settings — provides ``device_id``."""

    device_id: str = Field(
        default="<unknown>",
        json_schema_extra={
            "description": "ID of device",
            "examples": ["battery1", "ev1", "inverter1", "dishwasher"],
        },
    )


# ============================================================
# Resource registry
# ============================================================


class ResourceKey(PydanticBaseModel):
    """Key identifying a resource and optionally an actuator."""

    resource_id: str
    actuator_id: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    def __hash__(self) -> int:
        return hash(self.resource_id + (self.actuator_id or ""))

    def as_tuple(self) -> tuple[str, Optional[str]]:
        """Return the key as a tuple for internal dictionary indexing."""
        return (self.resource_id, self.actuator_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ResourceKey):
            return NotImplemented
        return self.resource_id == other.resource_id and self.actuator_id == other.actuator_id


class ResourceRegistry(SingletonMixin, ConfigMixin, PydanticBaseModel):
    """Registry for collecting and retrieving device status reports.

    Maintains the latest and optionally historical status reports for
    each resource.
    """

    keep_history: bool = False
    history_size: int = 100

    latest: dict[ResourceKey, ResourceStatus] = Field(
        default_factory=dict,
        json_schema_extra={
            "description": "Latest resource status per resource key.",
            "example": [],
        },
    )
    history: dict[ResourceKey, list[tuple[DateTime, ResourceStatus]]] = Field(
        default_factory=dict,
        json_schema_extra={
            "description": "History of resource stati per resource key.",
            "example": [],
        },
    )

    @model_validator(mode="after")
    def _enforce_history_limits(self) -> "ResourceRegistry":
        if self.keep_history:
            for key, records in self.history.items():
                if len(records) > self.history_size:
                    self.history[key] = records[-self.history_size :]
        return self

    def update_status(self, key: ResourceKey, status: ResourceStatus) -> None:
        """Update the latest status and optionally store in history."""
        self.latest[key] = status
        if self.keep_history:
            timestamp = getattr(status, "transition_timestamp", None) or to_datetime()
            self.history.setdefault(key, []).append((timestamp, status))
            if len(self.history[key]) > self.history_size:
                self.history[key] = self.history[key][-self.history_size :]

    def status_latest(self, key: ResourceKey) -> Optional[ResourceStatus]:
        """Retrieve the most recent status for a resource."""
        return self.latest.get(key)

    def status_history(self, key: ResourceKey) -> list[tuple[DateTime, ResourceStatus]]:
        """Retrieve historical status reports for a resource."""
        if not self.keep_history:
            raise RuntimeError("History tracking is disabled.")
        return self.history.get(key, [])

    def status_exists(self, key: ResourceKey) -> bool:
        """Check if a status report exists for the given resource."""
        return key in self.latest

    def save(self) -> None:
        """Save the registry to file."""
        cache_file = cast(
            TextIO,
            CacheFileStore().create(key="resource_registry", mode="w+", suffix=".json"),
        )
        cache_file.seek(0)
        cache_file.write(self.model_dump_json(indent=4))
        cache_file.truncate()

    def load(self) -> None:
        """Load registry state from file and update the current instance."""
        cache_file = CacheFileStore().get(key="resource_registry")
        if cache_file:
            try:
                cache_file.seek(0)
                data = json.load(cache_file)
                loaded = self.__class__.model_validate(data)
                self.keep_history = loaded.keep_history
                self.history_size = loaded.history_size
                self.latest = loaded.latest
                self.history = loaded.history
            except Exception as e:
                from loguru import logger
                logger.error("Can not load resource registry: {}", e)
