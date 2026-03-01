"""General configuration settings for simulated devices.

Concepts
--------
1. **Pydantic Config Models (mutable, user-facing)**
   - Read, validate, and document user-provided configuration.
   - Each device type has a corresponding Config model.
   - Multiple devices of the same type are represented as a list in
     DevicesCommonSettings.

2. **Immutable Domain Specs (frozen, GA-usable)**
   - Created via ``to_domain()`` methods on Config models.
   - Immutable and hashable for caching and GA memoization.
   - Carry ``device_id`` for mapping within the simulation engine.

3. **Topology Config**
   - ``BusConfig`` / ``BusesConfig`` — declare energy buses explicitly.
   - Each device config carries a ``ports`` field that wires it to buses.
   - Buses and ports are always declared explicitly by the user; no
     topology is inferred from device type.

4. **Top-level DevicesCommonSettings**
   - Aggregates all device configs.
   - ``to_domain()`` returns a flat list of ``DeviceParam`` objects.
   - ``BusesConfig.to_domain()`` returns the ``list[EnergyBus]`` list
     separately — the two are combined by the engine builder.

Usage
-----
1. Create or load ``BusesConfig`` + ``DevicesCommonSettings``
   (e.g., from YAML/JSON).
2. Pydantic validates all fields and constraints.
3. Call ``buses_config.to_domain()`` → ``list[EnergyBus]``.
4. Call ``devices_config.to_domain()`` → ``list[DeviceParam]``.
5. Use the results to instantiate concrete ``EnergyDevice`` objects and
   pass them to ``EnergySimulationEngine``.

Missing domain classes
----------------------
``GridConnectionParam`` and ``FixedLoadParam`` are not yet defined in
``devicesabc``. The ``to_domain()`` methods for those device types are
marked with ``# TODO`` and return ``None`` until those classes exist.
"""

import json
import re
from typing import Any, Optional, TextIO, cast

import numpy as np
from loguru import logger
from pydantic import Field, computed_field, field_validator, model_validator

from akkudoktoreos.config.configabc import SettingsBaseModel, TimeWindowSequence
from akkudoktoreos.core.cache import CacheFileStore
from akkudoktoreos.core.coreabc import ConfigMixin, SingletonMixin
from akkudoktoreos.core.emplan import ResourceStatus
from akkudoktoreos.core.pydantic import ConfigDict, PydanticBaseModel
from akkudoktoreos.devices.devicesabc import DevicesBaseSettings
from akkudoktoreos.utils.datetimeutil import DateTime, to_datetime
from akkudoktoreos.devices.devicesabc import (
    ApplianceOperationMode,
    BatteryOperationMode,
    BatteryParam,
    DeviceParam,
    EnergyBus,
    EnergyBusConstraint,
    EnergyCarrier,
    EnergyPort,
    HeatPumpParam,
    InverterParam,
    PortDirection,
    PVParam,
)

# Default charge rates for battery
BATTERY_DEFAULT_CHARGE_RATES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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

    def to_domain(self) -> EnergyPort:
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

    def to_domain(self) -> EnergyBusConstraint:
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

    def to_domain(self) -> EnergyBus:
        """Return an immutable ``EnergyBus`` domain object."""
        return EnergyBus(
            bus_id=self.bus_id,
            carrier=self.carrier,
            constraint=self.constraint.to_domain() if self.constraint else None,
        )


class BusesConfig(PydanticBaseModel):
    """Configuration for all energy buses in the system.

    Validates that all ``bus_id`` values are unique. Each device's
    ``ports`` field must reference a ``bus_id`` that exists here —
    this cross-reference is validated by the engine at construction time,
    not here (to keep config loading independent of engine instantiation).

    Call ``to_domain()`` to obtain the ``list[EnergyBus]`` required by
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
        },
    )

    @model_validator(mode="after")
    def _validate_unique_bus_ids(self) -> "BusesConfig":
        seen: set[str] = set()
        for bus in self.buses:
            if bus.bus_id in seen:
                raise ValueError(f"Duplicate bus_id: '{bus.bus_id}'")
            seen.add(bus.bus_id)
        return self

    def to_domain(self) -> list[EnergyBus]:
        """Return an immutable list of ``EnergyBus`` domain objects."""
        return [bus.to_domain() for bus in self.buses]


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
        return tuple(p.to_domain() for p in self.ports)


# ============================================================
# Base settings
# ============================================================


class DevicesBaseSettings(SettingsBaseModel):
    """Base devices setting."""

    device_id: str = Field(
        default="<unknown>",
        json_schema_extra={
            "description": "ID of device",
            "examples": ["battery1", "ev1", "inverter1", "dishwasher"],
        },
    )


# ============================================================
# Battery / EV
# ============================================================


class BatteriesCommonSettings(PortsMixin, DevicesBaseSettings):
    """Battery and electric vehicle device settings.

    Used for both stationary batteries and EV battery packs. The port
    configuration declares how the battery connects to the DC bus (or
    directly to the AC bus for AC-coupled systems).

    Port wiring guidance
    --------------------
    Standard DC-coupled battery::

        ports:
          - port_id: p_dc
            bus_id: bus_dc
            direction: bidirectional  # charges and discharges
            max_power_w: 5000         # optional port-level limit
    """

    capacity_wh: int = Field(
        default=8000,
        gt=0,
        json_schema_extra={"description": "Capacity [Wh].", "examples": [8000]},
    )
    charging_efficiency: float = Field(
        default=0.88,
        gt=0,
        le=1,
        json_schema_extra={
            "description": "Charging efficiency [0.01 ... 1.00].",
            "examples": [0.88],
        },
    )
    discharging_efficiency: float = Field(
        default=0.88,
        gt=0,
        le=1,
        json_schema_extra={
            "description": "Discharge efficiency [0.01 ... 1.00].",
            "examples": [0.88],
        },
    )
    levelized_cost_of_storage_kwh: float = Field(
        default=0.0,
        json_schema_extra={
            "description": (
                "Levelized cost of storage (LCOS), the average lifetime cost "
                "of delivering one kWh [€/kWh]."
            ),
            "examples": [0.12],
        },
    )
    max_charge_power_w: Optional[float] = Field(
        default=5000,
        gt=0,
        json_schema_extra={
            "description": "Maximum charging power [W].",
            "examples": [5000],
        },
    )
    min_charge_power_w: Optional[float] = Field(
        default=50,
        gt=0,
        json_schema_extra={
            "description": "Minimum charging power [W].",
            "examples": [50],
        },
    )
    charge_rates: Optional[list[float]] = Field(
        default=BATTERY_DEFAULT_CHARGE_RATES,
        json_schema_extra={
            "description": (
                "Charge rates as factor of maximum charging power [0.00 ... 1.00]. "
                "None triggers fallback to default charge-rates."
            ),
            "examples": [[0.0, 0.25, 0.5, 0.75, 1.0], None],
        },
    )
    min_soc_percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        json_schema_extra={
            "description": (
                "Minimum state of charge (SOC) as percentage of capacity [%]. "
                "This is the target SoC for charging."
            ),
            "examples": [10],
        },
    )
    max_soc_percentage: int = Field(
        default=100,
        ge=0,
        le=100,
        json_schema_extra={
            "description": "Maximum state of charge (SOC) as percentage of capacity [%].",
            "examples": [100],
        },
    )
    operation_modes: list[BatteryOperationMode] = Field(
        default_factory=lambda: [BatteryOperationMode.SELF_CONSUMPTION],
        json_schema_extra={
            "description": "Supported operating modes for this battery.",
            "examples": [["SELF_CONSUMPTION", "PEAK_SHAVING"]],
        },
    )

    @field_validator("charge_rates", mode="before")
    @classmethod
    def validate_and_sort_charge_rates(cls, v: Any) -> list[float]:
        if v is None:
            return BATTERY_DEFAULT_CHARGE_RATES.copy()
        if isinstance(v, str):
            numbers = re.split(r"[,\s]+", v.strip("[]"))
            arr = np.array([float(x) for x in numbers if x])
        else:
            arr = np.array(v, dtype=float)
        if arr.size == 0:
            raise ValueError("charge_rates must contain at least one value.")
        if (arr < 0.0).any() or (arr > 1.0).any():
            raise ValueError("charge_rates must be within [0.0, 1.0].")
        arr = np.unique(arr)
        return arr.tolist()

    @model_validator(mode="after")
    def _validate_soc_range(self) -> "BatteriesCommonSettings":
        if self.min_soc_percentage >= self.max_soc_percentage:
            raise ValueError("min_soc_percentage must be < max_soc_percentage")
        if (
            self.min_charge_power_w is not None
            and self.max_charge_power_w is not None
            and self.min_charge_power_w > self.max_charge_power_w
        ):
            raise ValueError("min_charge_power_w must be <= max_charge_power_w")
        return self

    def to_domain(self) -> BatteryParam:
        """Return an immutable ``BatteryParam`` domain object."""
        return BatteryParam(
            device_id=self.device_id,
            ports=self._domain_ports(),
            operation_modes=tuple(self.operation_modes),
            capacity_wh=float(self.capacity_wh),
            charging_efficiency=self.charging_efficiency,
            discharging_efficiency=self.discharging_efficiency,
            levelized_cost_of_storage_kwh=self.levelized_cost_of_storage_kwh,
            max_charge_power_w=float(self.max_charge_power_w or 5000),
            min_charge_power_w=float(self.min_charge_power_w or 0),
            charge_rates=(tuple(self.charge_rates) if self.charge_rates is not None else None),
            min_soc_factor=self.min_soc_percentage / 100.0,
            max_soc_factor=self.max_soc_percentage / 100.0,
        )

    # ---- Measurement keys (unchanged) ----

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_key_soc_factor(self) -> str:
        """Measurement key for SoC as factor of total capacity [0.0 ... 1.0]."""
        return f"{self.device_id}-soc-factor"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_key_power_l1_w(self) -> str:
        """Measurement key for L1 power [W]."""
        return f"{self.device_id}-power-l1-w"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_key_power_l2_w(self) -> str:
        """Measurement key for L2 power [W]."""
        return f"{self.device_id}-power-l2-w"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_key_power_l3_w(self) -> str:
        """Measurement key for L3 power [W]."""
        return f"{self.device_id}-power-l3-w"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_key_power_3_phase_sym_w(self) -> str:
        """Measurement key for symmetric 3-phase power [W]."""
        return f"{self.device_id}-power-3-phase-sym-w"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """All measurement keys for this battery."""
        return [
            self.measurement_key_soc_factor,
            self.measurement_key_power_l1_w,
            self.measurement_key_power_l2_w,
            self.measurement_key_power_l3_w,
            self.measurement_key_power_3_phase_sym_w,
        ]


# ============================================================
# Inverter
# ============================================================


class InverterCommonSettings(PortsMixin, DevicesBaseSettings):
    """Inverter device settings.

    An inverter bridges a DC bus (PV / battery) and an AC bus (grid /
    household). It must therefore have at least one DC port and one AC
    port.

    Port wiring guidance
    --------------------
    Standard string inverter (PV-only, no battery)::

        ports:
          - port_id: p_dc
            bus_id: bus_dc
            direction: sink       # consumes DC power from PV
          - port_id: p_ac
            bus_id: bus_ac
            direction: source     # injects AC power onto the AC bus

    Hybrid inverter with battery::

        ports:
          - port_id: p_dc
            bus_id: bus_dc
            direction: bidirectional  # DC bus shared with battery
          - port_id: p_ac
            bus_id: bus_ac
            direction: bidirectional  # can import AC for battery charging

    Notes:
    -----
    ``battery_id`` is retained for backward compatibility with existing
    configurations but is not used by ``to_domain()`` or the simulation
    engine. Port wiring replaces that coupling.
    """

    max_power_w: Optional[float] = Field(
        default=None,
        gt=0,
        json_schema_extra={
            "description": "Maximum AC output power [W].",
            "examples": [10000],
        },
    )
    ac_to_dc_efficiency: float = Field(
        default=1.0,
        ge=0,
        le=1,
        json_schema_extra={
            "description": (
                "Efficiency of AC→DC conversion for grid-to-battery charging (0–1). "
                "Set to 0 to disable AC charging. Default 1.0."
            ),
            "examples": [0.95, 1.0, 0.0],
        },
    )
    dc_to_ac_efficiency: float = Field(
        default=1.0,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Efficiency of DC→AC conversion for battery discharging (0–1). Default 1.0."
            ),
            "examples": [0.95, 1.0],
        },
    )
    max_ac_charge_power_w: Optional[float] = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": (
                "Maximum AC charging power [W]. "
                "null means no additional limit. 0 disables AC charging."
            ),
            "examples": [None, 0, 5000],
        },
    )
    # Retained for backward compatibility — not used by to_domain().
    battery_id: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Deprecated. Use port wiring to connect inverter to battery DC bus. "
                "Retained for backward compatibility only."
            ),
            "examples": [None],
        },
    )

    def to_domain(self) -> InverterParam:
        """Return an immutable ``InverterParam`` domain object.

        Note: ``ac_to_dc_efficiency``, ``dc_to_ac_efficiency``, and
        ``max_ac_charge_power_w`` are not yet fields on ``InverterParam``.
        They are preserved in the config for future extension.
        """
        return InverterParam(
            device_id=self.device_id,
            ports=self._domain_ports(),
            max_power_w=self.max_power_w or float("inf"),
            efficiency=self.dc_to_ac_efficiency,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this inverter (currently none)."""
        return []


# ============================================================
# PV array (new)
# ============================================================


class PVCommonSettings(PortsMixin, DevicesBaseSettings):
    """Photovoltaic array device settings.

    A PV array is a pure source — it injects power onto the DC bus (for
    string inverters) or directly onto the AC bus (for micro-inverters).

    Port wiring guidance
    --------------------
    DC-coupled PV (most common)::

        ports:
          - port_id: p_dc
            bus_id: bus_dc
            direction: source    # injects DC power

    AC-coupled PV (micro-inverter)::

        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: source    # injects AC power directly
    """

    peak_power_w: float = Field(
        ...,
        gt=0,
        json_schema_extra={
            "description": "Peak power output under standard test conditions [W].",
            "examples": [10000, 5000],
        },
    )
    tilt_deg: float = Field(
        default=30.0,
        ge=0,
        le=90,
        json_schema_extra={
            "description": "Array tilt angle from horizontal [degrees].",
            "examples": [30.0],
        },
    )
    azimuth_deg: float = Field(
        default=180.0,
        ge=0,
        lt=360,
        json_schema_extra={
            "description": (
                "Array orientation [degrees]: 0°=North, 90°=East, 180°=South, 270°=West."
            ),
            "examples": [180.0],
        },
    )

    def to_domain(self) -> PVParam:
        """Return an immutable ``PVParam`` domain object."""
        return PVParam(
            device_id=self.device_id,
            ports=self._domain_ports(),
            peak_power_w=self.peak_power_w,
            tilt_deg=self.tilt_deg,
            azimuth_deg=self.azimuth_deg,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this PV array (currently none)."""
        return []


# ============================================================
# Grid connection (new)
# ============================================================


class GridConnectionSettings(PortsMixin, DevicesBaseSettings):
    """Grid connection device settings.

    Represents the household's connection point to the public AC grid.
    The grid is bidirectional — it can import power (positive, consuming
    from the grid) or export power (negative, injecting into the grid).

    Port wiring guidance
    --------------------
    Standard grid connection::

        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: bidirectional
            max_power_w: 11000     # grid connection limit [W]

    Notes:
    -----
    ``GridConnectionParam`` does not yet exist in ``devicesabc``.
    ``to_domain()`` raises ``NotImplementedError`` until it is added.
    The config fields are fully specified here so the configuration
    layer is stable independent of the simulation domain layer.
    """

    max_import_power_w: float = Field(
        default=11000,
        gt=0,
        json_schema_extra={
            "description": "Maximum power import from the grid [W].",
            "examples": [11000, 25000],
        },
    )
    max_export_power_w: float = Field(
        default=11000,
        gt=0,
        json_schema_extra={
            "description": "Maximum power export to the grid [W].",
            "examples": [11000, 0],
        },
    )
    import_cost_per_kwh: float = Field(
        default=0.30,
        ge=0,
        json_schema_extra={
            "description": "Cost of importing 1 kWh from the grid [currency/kWh].",
            "examples": [0.30],
        },
    )
    export_revenue_per_kwh: float = Field(
        default=0.08,
        ge=0,
        json_schema_extra={
            "description": "Revenue from exporting 1 kWh to the grid [currency/kWh].",
            "examples": [0.08],
        },
    )

    def to_domain(self) -> DeviceParam:
        """Return an immutable domain object.

        Raises:
            NotImplementedError: ``GridConnectionParam`` is not yet defined
                in ``devicesabc``. Add it there first.
        """
        # TODO: return GridConnectionParam(...) once defined in devicesabc.
        raise NotImplementedError(
            "GridConnectionParam is not yet defined in devicesabc. "
            "Add it before calling to_domain() on GridConnectionSettings."
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this grid connection (currently none)."""
        return []


# ============================================================
# Heat pump (new)
# ============================================================


class HeatPumpCommonSettings(PortsMixin, DevicesBaseSettings):
    """Heat pump device settings.

    A heat pump consumes AC electrical power and produces thermal output.
    It therefore needs an AC electrical port and a heat bus port.

    Port wiring guidance
    --------------------
    Standard heat pump::

        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: sink       # consumes AC power
          - port_id: p_heat
            bus_id: bus_heat
            direction: source     # delivers thermal output
    """

    thermal_power_w: float = Field(
        ...,
        gt=0,
        json_schema_extra={
            "description": "Rated thermal output power [W].",
            "examples": [8000],
        },
    )
    cop: float = Field(
        default=3.0,
        gt=0,
        json_schema_extra={
            "description": (
                "Coefficient of performance: thermal output / electrical input. "
                "Electrical input = thermal_power_w / cop."
            ),
            "examples": [3.0, 4.5],
        },
    )
    operation_modes: list[ApplianceOperationMode] = Field(
        default_factory=lambda: [ApplianceOperationMode.RUN, ApplianceOperationMode.OFF],
        json_schema_extra={
            "description": "Supported operating modes.",
            "examples": [["RUN", "OFF", "DEFER"]],
        },
    )

    def to_domain(self) -> HeatPumpParam:
        """Return an immutable ``HeatPumpParam`` domain object."""
        return HeatPumpParam(
            device_id=self.device_id,
            ports=self._domain_ports(),
            operation_modes=tuple(self.operation_modes),
            thermal_power_w=self.thermal_power_w,
            cop=self.cop,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this heat pump (currently none)."""
        return []


# ============================================================
# Fixed load (new)
# ============================================================


class FixedLoadSettings(PortsMixin, DevicesBaseSettings):
    """Fixed (non-controllable) household load settings.

    Represents a device whose consumption is driven entirely by an
    external forecast — it has no genome and the optimizer cannot
    shift or reduce its load. Typical examples: refrigerator, always-on
    equipment, base load.

    Port wiring guidance
    --------------------
    Fixed AC load::

        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: sink

    Notes:
    -----
    ``FixedLoadParam`` does not yet exist in ``devicesabc``.
    ``to_domain()`` raises ``NotImplementedError`` until it is added.
    """

    peak_power_w: float = Field(
        ...,
        gt=0,
        json_schema_extra={
            "description": (
                "Peak power consumption of this load [W]. "
                "Actual consumption per step is provided by a forecast."
            ),
            "examples": [500, 1200],
        },
    )

    def to_domain(self) -> DeviceParam:
        """Return an immutable domain object.

        Raises:
            NotImplementedError: ``FixedLoadParam`` is not yet defined in
                ``devicesabc``. Add it there first.
        """
        # TODO: return FixedLoadParam(...) once defined in devicesabc.
        raise NotImplementedError(
            "FixedLoadParam is not yet defined in devicesabc. "
            "Add it before calling to_domain() on FixedLoadSettings."
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this load (currently none)."""
        return []


# ============================================================
# Home appliance (controllable)
# ============================================================


class HomeApplianceCommonSettings(PortsMixin, DevicesBaseSettings):
    """Controllable home appliance device settings.

    Represents a shiftable load whose start time can be deferred by the
    optimizer within allowed time windows (e.g. dishwasher, washing machine,
    EV charging session).

    Port wiring guidance
    --------------------
    Standard AC appliance::

        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: sink
    """

    consumption_wh: int = Field(
        ...,
        gt=0,
        json_schema_extra={
            "description": "Energy consumption per run cycle [Wh].",
            "examples": [2000],
        },
    )
    duration_h: int = Field(
        ...,
        gt=0,
        le=24,
        json_schema_extra={
            "description": "Run duration per cycle [h] (1–24).",
            "examples": [1],
        },
    )
    time_windows: Optional[TimeWindowSequence] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Allowed scheduling time windows. Defaults to the global optimisation time window."
            ),
            "examples": [{"windows": [{"start_time": "10:00", "duration": "2 hours"}]}],
        },
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this appliance (currently none)."""
        return []


# ============================================================
# Top-level device collection
# ============================================================


class DevicesCommonSettings(SettingsBaseModel):
    """Configuration for all controllable devices in the simulation.

    Each device list is independent and individually validated. Devices
    reference buses by ``bus_id`` in their ``ports`` field; the
    corresponding buses must be declared in a separate ``BusesConfig``
    that is passed alongside this config when building the engine.

    Call ``to_domain()`` to obtain a flat ``list[DeviceParam]`` covering
    all device types that have a complete domain class. Device types
    whose domain class is not yet implemented (``GridConnectionSettings``,
    ``FixedLoadSettings``) are skipped with a warning.
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

    def to_domain(self) -> list[DeviceParam]:
        """Return a flat list of immutable domain objects for all devices.

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

        def _add(device_list: Optional[list], skip_types: tuple = ()) -> None:
            for device in device_list or []:
                try:
                    params.append(device.to_domain())
                except NotImplementedError:
                    logger.warning(
                        "Skipping device '{}' ({}): to_domain() not yet implemented.",
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


# ============================================================
# Resource registry (unchanged)
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
                logger.error("Can not load resource registry: {}", e)
