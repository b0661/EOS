"""Device configuration — umbrella module.

Concepts
--------
1. **Pydantic Config Models (mutable, user-facing)**
   - Read, validate, and document user-provided configuration.
   - Each device type has a corresponding CommonSettings class defined
     in ``devices/settings/<device>.py``.
   - Multiple devices of the same type are represented as a list in
     ``DevicesCommonSettings`` below.

2. **Optimizer-specific domain conversion**
   - Each CommonSettings class exposes ``to_genetic2_param()`` to produce
     the immutable frozen dataclass required by the GENETIC2 optimizer.
   - Other optimizers add their own ``to_<optimizer>_param()`` methods to
     the same CommonSettings classes when needed.
   - ``DevicesCommonSettings.to_genetic2_params()`` aggregates all devices
     into a flat ``list[DeviceParam]`` for GENETIC2.

3. **Topology Config**
   - ``BusConfig`` / ``BusesConfig`` — declare energy buses explicitly.
   - Each device config carries a ``ports`` field wiring it to buses.

4. **Backward compatibility**
   - All names previously importable from this module remain importable
     here. The definitions have moved to ``devices/settings/`` but this
     file re-exports everything so existing import sites are unaffected.

Usage (GENETIC2)
----------------
1. Load ``BusesConfig`` + ``DevicesCommonSettings`` from YAML/JSON.
2. Pydantic validates all fields and constraints.
3. ``buses_config.to_genetic2_param()``  →  ``list[EnergyBus]``
4. ``devices_config.to_genetic2_params()``  →  ``list[DeviceParam]``
5. Instantiate concrete ``EnergyDevice`` objects and pass them to
   ``EnergySimulationEngine``.
"""

from typing import Optional

from loguru import logger
from pydantic import Field, computed_field, model_validator

from akkudoktoreos.config.configabc import SettingsBaseModel
from akkudoktoreos.devices.devicesabc import DeviceParam, EnergyBus
from akkudoktoreos.devices.settings.devicebasesettings import (
    BusConfig,
    BusConstraintConfig,
    DevicesBaseSettings,
    PortConfig,
    PortsMixin,
    ResourceKey,
    ResourceRegistry,
)
from akkudoktoreos.devices.settings.batterysettings import BATTERY_DEFAULT_CHARGE_RATES, BatteriesCommonSettings
from akkudoktoreos.devices.settings.fixedloadsettings import FixedLoadSettings
from akkudoktoreos.devices.settings.gridconnectionsettings import GridConnectionSettings
from akkudoktoreos.devices.settings.heatpumpsettings import HeatPumpCommonSettings
from akkudoktoreos.devices.settings.homeappliancesettings import HomeApplianceCommonSettings
from akkudoktoreos.devices.settings.invertersettings import InverterCommonSettings
from akkudoktoreos.devices.settings.pvsettings import PVCommonSettings


# ============================================================
# Top-level device collection
# ============================================================


class BusesCommonSettings(SettingsBaseModel):
    """Configuration for all energy buses in the system.

    Validates that all ``bus_id`` values are unique. Each device's
    ``ports`` field must reference a ``bus_id`` that exists here —
    this cross-reference is validated by the engine at construction time,
    not here (to keep config loading independent of engine instantiation).

    Call ``to_genetic2_param()`` to obtain the ``list[EnergyBus]`` required by
    ``EnergySimulationEngine``.
    """

    buses: list[BusConfig] = Field(
        default_factory=list,
        json_schema_extra={
            "description": "List of energy buses in the system. Used by GENETIC2 only.",
            "examples": [
                [
                    {"bus_id": "bus_dc", "carrier": "dc"},
                    {"bus_id": "bus_ac", "carrier": "ac"},
                ]
            ],
        },
    )

    @model_validator(mode="after")
    def _validate_unique_bus_ids(self) -> "BusesCommonSettings":
        seen: set[str] = set()
        for bus in self.buses:
            if bus.bus_id in seen:
                raise ValueError(f"Duplicate bus_id: '{bus.bus_id}'")
            seen.add(bus.bus_id)
        return self

    def to_genetic2_param(self) -> list[EnergyBus]:
        """Return an immutable list of ``EnergyBus`` domain objects."""
        return [bus.to_genetic2_param() for bus in self.buses]


class DevicesCommonSettings(SettingsBaseModel):
    """Configuration for all controllable devices in the simulation.

    Each device list is independent and individually validated. Devices
    reference buses by ``bus_id`` in their ``ports`` field; the
    corresponding buses must be declared in a separate ``BusesConfig``
    that is passed alongside this config when building the engine.

    Call ``to_genetic2_params()`` to obtain a flat ``list[DeviceParam]``
    for all devices that have a complete GENETIC2 domain class. Device
    types whose domain class is not yet implemented
    (``GridConnectionSettings``, ``FixedLoadSettings``) are skipped with
    a warning.
    """

    # ---- Batteries ----
    batteries: Optional[list[BatteriesCommonSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Stationary battery storage devices.",
            "examples": [[{"device_id": "bat0", "capacity_wh": 8000, "ports": [...]}]],
        },
    )
    max_batteries: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Maximum number of batteries allowed.", "examples": [1]},
    )

    # ---- Electric vehicles ----
    electric_vehicles: Optional[list[BatteriesCommonSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Electric vehicle battery packs (uses same settings as batteries).",
            "examples": [[{"device_id": "ev0", "capacity_wh": 60000, "ports": [...]}]],
        },
    )
    max_electric_vehicles: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Maximum number of EVs allowed.", "examples": [1]},
    )

    # ---- Inverters ----
    inverters: Optional[list[InverterCommonSettings]] = Field(
        default=None,
        json_schema_extra={"description": "Inverter devices.", "examples": [[]]},
    )
    max_inverters: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Maximum number of inverters allowed.", "examples": [1]},
    )

    # ---- PV arrays ----
    pvs: Optional[list[PVCommonSettings]] = Field(
        default=None,
        json_schema_extra={"description": "Photovoltaic array devices.", "examples": [[]]},
    )
    max_pvs: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Maximum number of PV arrays allowed.", "examples": [2]},
    )

    # ---- Grid connections ----
    grid_connections: Optional[list[GridConnectionSettings]] = Field(
        default=None,
        json_schema_extra={"description": "Grid connection points.", "examples": [[]]},
    )
    max_grid_connections: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "Maximum number of grid connections allowed.",
            "examples": [1],
        },
    )

    # ---- Heat pumps ----
    heat_pumps: Optional[list[HeatPumpCommonSettings]] = Field(
        default=None,
        json_schema_extra={"description": "Heat pump devices.", "examples": [[]]},
    )
    max_heat_pumps: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "Maximum number of heat pumps allowed.",
            "examples": [1],
        },
    )

    # ---- Fixed loads ----
    fixed_loads: Optional[list[FixedLoadSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Non-controllable fixed household loads.",
            "examples": [[{"device_id": "base_load", "peak_power_w": 500, "ports": [...]}]],
        },
    )
    max_fixed_loads: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "Maximum number of fixed loads allowed.",
            "examples": [3],
        },
    )

    # ---- Controllable home appliances ----
    home_appliances: Optional[list[HomeApplianceCommonSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Shiftable home appliance devices.",
            "examples": [[{"device_id": "dishwasher", "consumption_wh": 1500, "ports": [...]}]],
        },
    )
    max_home_appliances: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "Maximum number of home appliances allowed.",
            "examples": [3],
        },
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """All measurement keys across all configured devices."""
        keys: list[str] = []
        for device_list in [
            self.batteries or [],
            self.electric_vehicles or [],
            self.inverters or [],
            self.pvs or [],
            self.grid_connections or [],
            self.heat_pumps or [],
            self.fixed_loads or [],
            self.home_appliances or [],
        ]:
            for device in device_list:
                keys.extend(device.measurement_keys)
        return keys

    def to_genetic2_params(self) -> list[DeviceParam]:
        """Return a flat list of GENETIC2 domain objects for all devices.

        Device types whose domain class is not yet implemented
        (``GridConnectionSettings``, ``FixedLoadSettings``) are skipped
        with a warning rather than raising, so a partial system can still
        be configured and optimised.

        Returns:
            Ordered list of ``DeviceParam`` subclass instances, in the
            order: batteries, electric_vehicles, inverters, pvs,
            heat_pumps, home_appliances. Grid connections and fixed loads
            are skipped until their domain classes exist.
        """
        params: list[DeviceParam] = []

        def _add(device_list: Optional[list]) -> None:
            for device in device_list or []:
                try:
                    params.append(device.to_genetic2_param())
                except NotImplementedError:
                    logger.warning(
                        "Skipping device '{}' ({}): to_genetic2_param() not yet implemented.",
                        device.device_id,
                        type(device).__name__,
                    )

        _add(self.batteries)
        _add(self.electric_vehicles)
        _add(self.inverters)
        _add(self.pvs)
        _add(self.heat_pumps)
        _add(self.home_appliances)
        # Grid connections and fixed loads skipped until domain classes exist:
        _add(self.grid_connections)
        _add(self.fixed_loads)

        return params
