"""Device configuration — umbrella module.

Concepts
--------

1. **Pydantic Config Models (mutable, user-facing)**
    - Read, validate, and document user-provided configuration.
    - Each device type has a corresponding CommonSettings class defined
        in ``devices/settings/<device>.py``.
    - Multiple devices of the same type are represented as a
        ``dict[str, <Settings>]`` in ``DevicesCommonSettings``, keyed by
        ``device_id``.  The dict key must match the ``device_id`` field
        inside the value.

2. **Optimizer-specific domain conversion**
    - Each CommonSettings class exposes ``to_genetic2_param()`` to produce
        the immutable frozen dataclass required by the GENETIC2 optimizer.
    - Other optimizers add their own ``to_<optimizer>_param()`` methods to
        the same CommonSettings classes when needed.
    - ``DevicesCommonSettings.to_genetic2_params()`` aggregates all devices
        into a flat ``list[DeviceParam]`` for GENETIC2.

3. **Topology Config**
    - ``BusConfig`` / ``BusesCommonSettings`` — declare energy buses explicitly.
    - Each device config carries a ``ports`` field wiring it to buses.

4. **Backward compatibility**
    - All names previously importable from this module remain importable
        here. The definitions have moved to ``devices/settings/`` but this
        file re-exports everything so existing import sites are unaffected.

Usage (GENETIC2)
----------------

1. Load ``BusesCommonSettings`` + ``DevicesCommonSettings`` from YAML/JSON.
2. Pydantic validates all fields and constraints.
3. ``buses_config.to_domain()``  →  ``list[EnergyBus]``
4. ``devices_config.to_genetic2_params()``  →  ``list[DeviceParam]``
5. Instantiate concrete ``EnergyDevice`` objects and pass them to
   ``EnergySimulationEngine``.
"""

from typing import Optional

from loguru import logger
from pydantic import Field, computed_field, model_validator

from akkudoktoreos.config.configabc import ConfigScope, SettingsBaseModel
from akkudoktoreos.devices.devicesabc import DeviceParam, EnergyBus
from akkudoktoreos.devices.settings.batterysettings import (
    BatteriesCommonSettings,
)
from akkudoktoreos.devices.settings.devicebasesettings import (
    BusConfig,
)
from akkudoktoreos.devices.settings.fixedloadsettings import FixedLoadSettings
from akkudoktoreos.devices.settings.gridconnectionsettings import GridConnectionSettings
from akkudoktoreos.devices.settings.heatpumpsettings import HeatPumpCommonSettings
from akkudoktoreos.devices.settings.homeappliancesettings import (
    HomeApplianceCommonSettings,
)
from akkudoktoreos.devices.settings.invertersettings import InverterCommonSettings

# ============================================================
# Top-level energy bus collection
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
            "description": "List of energy buses in the system.",
            "examples": [
                [
                    {"bus_id": "bus_dc", "carrier": "dc"},
                    {"bus_id": "bus_ac", "carrier": "ac"},
                ]
            ],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # ------------------------------------------------------------------
    # GENETIC2 domain conversion
    # ------------------------------------------------------------------

    # TBD

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


# ============================================================
# Top-level device collection
# ============================================================


class DevicesCommonSettings(SettingsBaseModel):
    """Configuration for all controllable devices in the simulation.

    Every device collection is a ``dict[str, <Settings>]`` keyed by
    ``device_id``.  This makes config paths stable regardless of
    declaration order and lets each device settings class build its own
    config path from ``self.device_id`` without needing an external index.

    Devices reference buses by ``bus_id`` in their ``ports`` field; the
    corresponding buses must be declared in a separate ``BusesCommonSettings``
    that is passed alongside this config when building the engine.

    Call ``to_genetic2_params()`` to obtain a flat ``list[DeviceParam]``
    for all devices that have a complete GENETIC2 domain class. Device
    types whose domain class is not yet implemented
    (``GridConnectionSettings``, ``FixedLoadSettings``) are skipped with
    a warning.
    """

    # ---- Batteries ----
    batteries: Optional[dict[str, BatteriesCommonSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Stationary battery storage devices, keyed by device_id.",
            "examples": [{"bat0": {"device_id": "bat0", "capacity_wh": 8000, "ports": []}}],
            "x-scope": [str(ConfigScope.GENETIC)],
        },
    )
    max_batteries: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Maximum number of batteries allowed.", "examples": [1]},
    )

    # ---- Electric vehicles ----
    electric_vehicles: Optional[dict[str, BatteriesCommonSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Electric vehicle battery packs, keyed by device_id.",
            "examples": [{"ev0": {"device_id": "ev0", "capacity_wh": 60000, "ports": []}}],
            "x-scope": [str(ConfigScope.GENETIC)],
        },
    )
    max_electric_vehicles: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Maximum number of EVs allowed.", "examples": [1]},
    )

    # ---- Inverters ----
    inverters: Optional[dict[str, InverterCommonSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Inverter devices, keyed by device_id.",
            "examples": [{}],
            "x-scope": [str(ConfigScope.GENETIC), str(ConfigScope.GENETIC2)],
        },
    )
    max_inverters: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={"description": "Maximum number of inverters allowed.", "examples": [1]},
    )

    # ---- Grid connections ----
    grid_connections: Optional[dict[str, GridConnectionSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Grid connection points, keyed by device_id.",
            "examples": [{}],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
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
    heat_pumps: Optional[dict[str, HeatPumpCommonSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Heat pump devices, keyed by device_id.",
            "examples": [{}],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
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
    fixed_loads: Optional[dict[str, FixedLoadSettings]] = Field(
        default=None,
        json_schema_extra={
            "description": "Non-controllable fixed household loads, keyed by device_id.",
            "examples": [
                {
                    "base_load": {
                        "device_id": "base_load",
                        "peak_power_w": 500,
                        "ports": [{"bus_id": "bus_ac", "port_id": "p_ac", "direction": "sink"}],
                    }
                }
            ],
            "x-scope": [str(ConfigScope.GENETIC2)],
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
    home_appliances: dict[str, HomeApplianceCommonSettings] = Field(
        default_factory=dict,
        json_schema_extra={
            "description": "Shiftable home appliance devices, keyed by device_id.",
            "examples": [
                {
                    "dishwasher": {
                        "device_id": "dishwasher",
                        "consumption_wh": 1500,
                        "duration_h": 2.0,  # required field
                        "ports": [{"bus_id": "bus_ac", "port_id": "p_ac", "direction": "sink"}],
                    }
                }
            ],
            "x-scope": [str(ConfigScope.GENETIC), str(ConfigScope.GENETIC2)],
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
        for device_dict in [
            self.batteries,
            self.electric_vehicles,
            self.inverters,
            self.grid_connections,
            self.heat_pumps,
            self.fixed_loads,
            self.home_appliances,
        ]:
            for device in (device_dict or {}).values():
                keys.extend(device.measurement_keys)
        return keys

    # ------------------------------------------------------------------
    # GENETIC2 domain conversion
    # ------------------------------------------------------------------

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

        def _add(device_dict: Optional[dict]) -> None:
            for device in (device_dict or {}).values():
                try:
                    params.append(device.to_genetic2_param())
                except NotImplementedError:
                    logger.warning(
                        "Skipping device '{}' ({}): to_genetic2_param() not yet implemented.",
                        device.device_id,
                        type(device).__name__,
                    )

        # Batteries and EVs not used in GENETIC2 (only GENETIC)
        # Grid connections and fixed loads skipped until domain classes exist:
        _add(self.fixed_loads)
        _add(self.heat_pumps)
        _add(self.home_appliances)
        _add(self.inverters)
        _add(self.grid_connections)

        return params
