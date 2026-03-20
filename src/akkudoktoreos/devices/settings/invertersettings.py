"""Inverter device settings."""

from typing import TYPE_CHECKING, Optional

from pydantic import Field, computed_field, model_validator

from akkudoktoreos.config.configabc import ConfigScope
from akkudoktoreos.devices.settings.devicebasesettings import (
    DevicesBaseSettings,
    PortsMixin,
)

if TYPE_CHECKING:
    from akkudoktoreos.devices.genetic2.hybridinverter import HybridInverterParam
    from akkudoktoreos.devices.genetic.inverter import InverterParameters


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

    GENETIC2 inverter type inference
    ---------------------------------

    The ``inverter_type`` passed to ``HybridInverterParam`` is inferred
    automatically from the fields that are set:

    - ``HYBRID``: ``pv_power_w_key`` is set **and** ``battery_capacity_wh``
      is set (and > 0).
    - ``SOLAR``: ``pv_power_w_key`` is set but ``battery_capacity_wh`` is
      ``None`` or 0.
    - ``BATTERY``: ``pv_power_w_key`` is ``None`` and
      ``battery_capacity_wh`` is set (and > 0).

    Notes:
    ------
    ``battery_id`` is retained for backward compatibility with existing
    configurations but is not used by ``to_genetic2_param()``. Port wiring
    replaces that coupling.
    """

    # ------------------------------------------------------------------
    # Shared fields (GENETIC + GENETIC2)
    # ------------------------------------------------------------------

    max_power_w: Optional[float] = Field(
        default=None,
        gt=0,
        json_schema_extra={
            "description": "Maximum AC output power [W].",
            "examples": [10000],
            "x-scope": [str(ConfigScope.GENETIC), str(ConfigScope.GENETIC2)],
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
            "x-scope": [str(ConfigScope.GENETIC), str(ConfigScope.GENETIC2)],
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
            "x-scope": [str(ConfigScope.GENETIC), str(ConfigScope.GENETIC2)],
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
            "x-scope": [str(ConfigScope.GENETIC)],
            "x-ui": {
                "widget": "number",
                "unit": "W",
            },
        },
    )
    battery_id: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": ("Device ID of the battery."),
            "examples": [None],
            "x-scope": [str(ConfigScope.GENETIC)],
            "x-ui": {
                "widget": "id",
                "unit": "",
            },
        },
    )

    # ------------------------------------------------------------------
    # GENETIC2-only fields
    # ------------------------------------------------------------------

    # Auxiliary power consumption
    off_state_power_consumption_w: float = Field(
        default=0.0,
        ge=0,
        json_schema_extra={
            "description": (
                "Standby power consumed when the inverter is fully idle "
                "(battery=0 and PV=0) [W]. Default 0.0."
            ),
            "examples": [5.0, 0.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
            "x-ui": {
                "widget": "number",
                "unit": "W",
            },
        },
    )
    on_state_power_consumption_w: float = Field(
        default=0.0,
        ge=0,
        json_schema_extra={
            "description": (
                "Auxiliary power consumed whenever the inverter is active "
                "(non-zero AC power) [W]. Default 0.0."
            ),
            "examples": [10.0, 0.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
            "x-ui": {
                "widget": "number",
                "unit": "W",
            },
        },
    )

    # PV parameters (used for SOLAR and HYBRID inverter types)
    pv_to_ac_efficiency: float = Field(
        default=1.0,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Efficiency of PV DC→AC conversion (0–1). "
                "Required when pv_power_w_key is set (SOLAR or HYBRID). Default 1.0."
            ),
            "examples": [0.97, 1.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    pv_to_battery_efficiency: float = Field(
        default=1.0,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Efficiency of PV DC→battery charging path (0–1). "
                "Used for HYBRID inverters only. Default 1.0."
            ),
            "examples": [0.98, 1.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    pv_max_power_w: Optional[float] = Field(
        default=None,
        gt=0,
        json_schema_extra={
            "description": (
                "Maximum DC PV power fed into the inverter [W]. "
                "Required when pv_power_w_key is set (SOLAR or HYBRID). "
                "Values from pv_power_w_key are clipped to this limit."
            ),
            "examples": [8000.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
            "x-ui": {
                "widget": "number",
                "unit": "W",
            },
        },
    )
    pv_min_power_w: float = Field(
        default=0.0,
        ge=0,
        json_schema_extra={
            "description": (
                "Minimum DC PV power threshold [W]. Steps with available PV "
                "below this value are treated as zero. Default 0.0."
            ),
            "examples": [50.0, 0.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
            "x-ui": {
                "widget": "number",
                "unit": "W",
            },
        },
    )
    pv_power_w_key: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "SimulationContext prediction key resolving to a per-step PV "
                "power forecast array [W] of shape (horizon,). "
                "Set for SOLAR and HYBRID inverter types; leave None for BATTERY."
            ),
            "examples": ["pv_forecast_w", None],
            "x-scope": [str(ConfigScope.GENETIC2)],
            "x-ui": {
                "widget": "prediction_key",
                "unit": "",
            },
        },
    )

    # Battery parameters (used for BATTERY and HYBRID inverter types)
    battery_capacity_wh: Optional[float] = Field(
        default=None,
        gt=0,
        json_schema_extra={
            "description": (
                "Usable battery capacity [Wh]. Required for BATTERY and HYBRID inverter types."
            ),
            "examples": [10000.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
            "x-ui": {
                "widget": "number",
                "unit": "Wh",
            },
        },
    )
    battery_charge_rates: Optional[list[float]] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Optional list of discrete charge rate fractions (each in (0, 1]). "
                "When set, the battery is constrained to these specific fractions "
                "of battery_max_charge_rate. null means continuous charging. "
                "All values must be in (0, 1]."
            ),
            "examples": [None, [0.25, 0.5, 1.0]],
            "x-scope": [str(ConfigScope.GENETIC2)],
            "x-ui": {
                "widget": "list_number",
                "unit": "",
            },
        },
    )
    battery_min_charge_rate: float = Field(
        default=0.0,
        ge=0,
        le=1,
        json_schema_extra={
            "description": (
                "Minimum non-zero charge rate as a fraction of battery_max_charge_rate. "
                "Charge commands below this threshold are rounded to zero. Default 0.0."
            ),
            "examples": [0.1, 0.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    battery_max_charge_rate: float = Field(
        default=1.0,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Maximum charge rate as a fraction of the 1C rate "
                "(1C = battery_capacity_wh W). bat_factor=+1 maps to this rate. "
                "Default 1.0 (full 1C charge)."
            ),
            "examples": [0.5, 1.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    battery_min_discharge_rate: float = Field(
        default=0.0,
        ge=0,
        le=1,
        json_schema_extra={
            "description": (
                "Minimum non-zero discharge rate as a fraction of battery_max_discharge_rate. "
                "Discharge commands below this threshold are rounded to zero. Default 0.0."
            ),
            "examples": [0.1, 0.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    battery_max_discharge_rate: float = Field(
        default=1.0,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Maximum discharge rate as a fraction of the 1C rate. "
                "bat_factor=−1 maps to this rate. Default 1.0 (full 1C discharge)."
            ),
            "examples": [0.5, 1.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    battery_min_soc_factor: float = Field(
        default=0.0,
        ge=0,
        lt=1,
        json_schema_extra={
            "description": (
                "Minimum allowed state of charge as a fraction of battery_capacity_wh. "
                "Must be < battery_max_soc_factor. Default 0.0."
            ),
            "examples": [0.1, 0.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    battery_max_soc_factor: float = Field(
        default=1.0,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Maximum allowed state of charge as a fraction of battery_capacity_wh. "
                "Must be > battery_min_soc_factor. Default 1.0."
            ),
            "examples": [0.9, 1.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    battery_initial_soc_factor_key: str = Field(
        default="",
        json_schema_extra={
            "description": (
                "SimulationContext measurement key resolving to the initial battery "
                "SoC as a fraction of battery_capacity_wh, in [min_soc_factor, max_soc_factor]. "
                "An empty string means the device uses battery_min_soc_factor as the "
                "initial SoC (fully depleted to the minimum)."
            ),
            "examples": ["battery1.soc_factor", ""],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_soc_factors(self) -> "InverterCommonSettings":
        if self.battery_min_soc_factor >= self.battery_max_soc_factor:
            raise ValueError(
                "battery_min_soc_factor must be strictly less than battery_max_soc_factor"
            )
        return self

    # ------------------------------------------------------------------
    # GENETIC domain conversion
    # ------------------------------------------------------------------

    def to_genetic_param(self) -> "InverterParameters":
        """Return InverterParameters for the GENETIC optimizer."""
        from akkudoktoreos.devices.genetic.inverter import InverterParameters

        return InverterParameters(
            device_id=self.device_id,
            max_power_wh=self.max_power_w,
            battery_id=self.battery_id,
            ac_to_dc_efficiency=self.ac_to_dc_efficiency,
            dc_to_ac_efficiency=self.dc_to_ac_efficiency,
            max_ac_charge_power_w=self.max_ac_charge_power_w,
        )

    # ------------------------------------------------------------------
    # GENETIC2 domain conversion
    # ------------------------------------------------------------------

    def to_genetic2_param(self) -> "HybridInverterParam":
        """Return an immutable ``HybridInverterParam`` for the GENETIC2 optimizer.

        The inverter type is inferred from the configured fields:

        - ``HYBRID``  — ``pv_power_w_key`` is set **and** ``battery_capacity_wh`` > 0.
        - ``SOLAR``   — ``pv_power_w_key`` is set, no battery capacity.
        - ``BATTERY`` — ``battery_capacity_wh`` > 0, no PV key.

        The single AC port is the first port in ``self.ports`` — the
        ``HybridInverterParam`` inherits ``device_id`` and ``ports`` from
        ``DeviceParam``; ``ports`` is populated via ``self._domain_ports()``.

        ``battery_charge_rates`` is converted from ``list[float] | None`` to
        ``tuple[float, ...] | None`` as required by the frozen dataclass.
        """
        from akkudoktoreos.devices.genetic2.hybridinverter import (
            HybridInverterParam,
            InverterType,
        )

        has_pv = self.pv_power_w_key is not None
        battery_capacity_wh = (
            self.battery_capacity_wh
            if self.battery_capacity_wh is not None and self.battery_capacity_wh > 0
            else 0.0
        )

        if has_pv and battery_capacity_wh > 0.0:
            inverter_type = InverterType.HYBRID
        elif has_pv:
            inverter_type = InverterType.SOLAR
        else:
            inverter_type = InverterType.BATTERY

        return HybridInverterParam(
            device_id=self.device_id,
            ports=self.ports_to_genetic2_param(),
            inverter_type=inverter_type,
            # Auxiliary consumption
            off_state_power_consumption_w=self.off_state_power_consumption_w,
            on_state_power_consumption_w=self.on_state_power_consumption_w,
            # PV (used by SOLAR and HYBRID; HybridInverterParam validation
            # only checks these for the relevant inverter types)
            pv_to_ac_efficiency=self.pv_to_ac_efficiency,
            pv_to_battery_efficiency=self.pv_to_battery_efficiency,
            pv_max_power_w=self.pv_max_power_w if self.pv_max_power_w is not None else float("inf"),
            pv_min_power_w=self.pv_min_power_w,
            pv_power_w_key=self.pv_power_w_key,
            # AC <-> Battery efficiency (used by BATTERY and HYBRID)
            ac_to_battery_efficiency=self.ac_to_dc_efficiency,
            battery_to_ac_efficiency=self.dc_to_ac_efficiency,
            # Battery sizing
            battery_capacity_wh=battery_capacity_wh,
            battery_charge_rates=(
                tuple(self.battery_charge_rates) if self.battery_charge_rates is not None else None
            ),
            # Rate bounds
            battery_min_charge_rate=self.battery_min_charge_rate,
            battery_max_charge_rate=self.battery_max_charge_rate,
            battery_min_discharge_rate=self.battery_min_discharge_rate,
            battery_max_discharge_rate=self.battery_max_discharge_rate,
            # SoC constraints
            battery_min_soc_factor=self.battery_min_soc_factor,
            battery_max_soc_factor=self.battery_max_soc_factor,
            battery_initial_soc_factor_key=self.battery_initial_soc_factor_key,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this inverter.

        Returns the ``battery_initial_soc_factor_key`` if non-empty, so
        the EMS measurement store knows to watch for this key.
        """
        if self.battery_initial_soc_factor_key:
            return [self.battery_initial_soc_factor_key]
        return []
