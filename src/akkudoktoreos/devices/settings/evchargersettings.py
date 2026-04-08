"""EV charger device settings."""

from typing import TYPE_CHECKING, Optional

from pydantic import Field, computed_field, model_validator

from akkudoktoreos.config.configabc import ConfigScope
from akkudoktoreos.devices.settings.devicebasesettings import (
    DevicesBaseSettings,
    PortsMixin,
)

if TYPE_CHECKING:
    from akkudoktoreos.devices.genetic2.evcharger import EVChargerParam


class EVChargerCommonSettings(PortsMixin, DevicesBaseSettings):
    """EV charger (EVSE / wall-box) device settings.

    Models a wall-box connected to an electric vehicle, including:

    - Charger power limits and deadband.
    - A multi-stage efficiency chain from the AC wall socket to the EV battery.
    - EV battery sizing, SoC constraints, and departure target.
    - Connection-state resolution (measurement or time-window).
    - Optional battery wear cost (LCOS).

    Port wiring guidance
    --------------------
    Standard AC charger (sink only)::

        ports:
          - port_id: p_ac
            bus_id:  bus_ac
            direction: sink

    Loss model overview
    -------------------
    Power flows from the AC bus through the following efficiency stages:

    ::

        AC_wall ──[charger_efficiency]──► DC_cable
                ──[hold_time_efficiency × control_efficiency]──► effective DC
                ──[ev_charger_efficiency_{low|high}]──► EV onboard output
                ──[ev_battery_efficiency]──► stored in EV battery

    Additionally:

    - ``standby_power_w``      — draw when EV is connected but not charging.
    - ``deep_standby_power_w`` — draw when EV is absent.

    The boundary between low-current and high-current EV onboard charger
    efficiency is set by ``high_current_threshold_w``.

    Connection-state resolution
    ---------------------------
    Priority (highest → lowest):

    1. **Measurement** (``ev_connected_measurement_key``) — ground truth for
       real-time dispatch.  ``1.0`` = connected, ``0.0`` = absent.
    2. **Time window** (``connection_time_window_key``) — expected presence
       schedule for forward-planning / forecast runs.  Points to a
       ``CycleTimeWindowSequence`` in the config tree.
    3. **Always-connected fallback** — used when neither source is available.
    """

    # ------------------------------------------------------------------
    # Power limits
    # ------------------------------------------------------------------

    min_charge_power_w: float = Field(
        default=1380.0,
        ge=0,
        json_schema_extra={
            "description": (
                "Minimum non-zero charge power at the AC wall socket [W]. "
                "Charge commands below this threshold are rounded to zero. "
                "Typical: 1 380 W (6 A × 230 V single-phase)."
            ),
            "examples": [1380.0, 0.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    max_charge_power_w: float = Field(
        default=11000.0,
        gt=0,
        json_schema_extra={
            "description": (
                "Maximum charge power at the AC wall socket [W]. "
                "charge_factor=1 maps to this value. "
                "Typical: 11 000 W (16 A × 3-phase)."
            ),
            "examples": [11000.0, 7400.0, 3700.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    high_current_threshold_w: float = Field(
        default=3680.0,
        ge=0,
        json_schema_extra={
            "description": (
                "AC power threshold [W] above which the high-current EV onboard "
                "charger efficiency applies. Below this value the low-current "
                "efficiency is used. "
                "Typical: 3 680 W (single-phase 16 A)."
            ),
            "examples": [3680.0, 4000.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # ------------------------------------------------------------------
    # Loss model
    # ------------------------------------------------------------------

    deep_standby_power_w: float = Field(
        default=1.0,
        ge=0,
        json_schema_extra={
            "description": (
                "AC draw of the charger hardware when the EV is absent [W]. "
                "Typical: 1–3 W."
            ),
            "examples": [1.0, 2.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    standby_power_w: float = Field(
        default=5.0,
        ge=0,
        json_schema_extra={
            "description": (
                "AC draw when the EV is connected but charging is paused [W]. "
                "Typical: 3–10 W."
            ),
            "examples": [5.0, 8.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    charger_efficiency: float = Field(
        default=0.95,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Efficiency of the charger's AC→DC conversion stage (0, 1]. "
                "Typical: 0.95."
            ),
            "examples": [0.95, 0.97],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    ev_charger_efficiency_low: float = Field(
        default=0.77,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Efficiency of the EV's onboard charger at low AC current (0, 1]. "
                "Applies when AC wall power < high_current_threshold_w. "
                "Typical single-phase 6 A: 0.77."
            ),
            "examples": [0.77, 0.80],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    ev_charger_efficiency_high: float = Field(
        default=0.90,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Efficiency of the EV's onboard charger at high AC current (0, 1]. "
                "Applies when AC wall power >= high_current_threshold_w. "
                "Typical three-phase 16 A: 0.90."
            ),
            "examples": [0.90, 0.92],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    ev_battery_efficiency: float = Field(
        default=0.96,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Round-trip efficiency of the EV's high-voltage battery when "
                "charging (DC in → stored kWh) (0, 1].  Typical: 0.96."
            ),
            "examples": [0.96, 0.98],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    hold_time_efficiency: float = Field(
        default=0.98,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Scheduling efficiency factor (0, 1].  Accounts for energy wasted "
                "because the charger cannot transition instantaneously between power "
                "levels within a time step.  Typical: 0.98."
            ),
            "examples": [0.98, 1.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    control_efficiency: float = Field(
        default=0.97,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Power-tracking efficiency factor (0, 1].  Accounts for the "
                "charger's inability to follow a rapidly varying power source "
                "(e.g. solar) exactly within a time step.  Typical: 0.97."
            ),
            "examples": [0.97, 1.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # ------------------------------------------------------------------
    # EV battery
    # ------------------------------------------------------------------

    ev_battery_capacity_wh: float = Field(
        default=60000.0,
        gt=0,
        json_schema_extra={
            "description": "Usable EV battery capacity [Wh]. Typical: 40 000–100 000 Wh.",
            "examples": [60000.0, 40000.0, 82000.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    ev_min_soc_factor: float = Field(
        default=0.1,
        ge=0,
        lt=1,
        json_schema_extra={
            "description": (
                "Minimum allowed EV SoC as a fraction of ev_battery_capacity_wh "
                "[0, ev_max_soc_factor).  Must be < ev_max_soc_factor."
            ),
            "examples": [0.1, 0.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    ev_max_soc_factor: float = Field(
        default=0.9,
        gt=0,
        le=1,
        json_schema_extra={
            "description": (
                "Maximum allowed EV SoC as a fraction of ev_battery_capacity_wh "
                "(ev_min_soc_factor, 1].  Must be > ev_min_soc_factor."
            ),
            "examples": [0.9, 1.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    ev_target_soc_factor: float = Field(
        default=0.8,
        ge=0,
        le=1,
        json_schema_extra={
            "description": (
                "Desired EV SoC at end of the optimisation horizon as a fraction "
                "of ev_battery_capacity_wh.  The SoC shortfall below this target "
                "is penalised in compute_cost at the mean import price. "
                "Set equal to ev_max_soc_factor to always fully charge."
            ),
            "examples": [0.8, 1.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    ev_initial_soc_factor_key: str = Field(
        default="",
        json_schema_extra={
            "description": (
                "SimulationContext measurement key resolving to the EV's current "
                "SoC as a fraction of ev_battery_capacity_wh. "
                "An empty string causes the device to default to ev_min_soc_factor."
            ),
            "examples": ["ev1_soc_factor", ""],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    ev_connected_measurement_key: str = Field(
        default="",
        json_schema_extra={
            "description": (
                "SimulationContext measurement key resolving to the EV plug state: "
                "1.0 = connected, 0.0 = absent. "
                "Takes precedence over connection_time_window_key. "
                "An empty string disables measurement-based detection."
            ),
            "examples": ["ev1_connected", ""],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    connection_time_window_key: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Config path (/-separated) to a CycleTimeWindowSequence that "
                "describes when the EV is expected to be plugged in. "
                "Used for forward-planning / forecast runs. "
                "null disables time-window-based connection detection."
            ),
            "examples": [
                "devices/ev_chargers/ev_charger1/connection_time_windows",
                None,
            ],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # ------------------------------------------------------------------
    # Wear cost
    # ------------------------------------------------------------------

    ev_lcos_amt_kwh: float = Field(
        default=0.0,
        ge=0,
        json_schema_extra={
            "description": (
                "Levelized cost of EV battery storage [currency / kWh cycled]. "
                "Penalises unnecessary cycling to prevent the GA from scheduling "
                "charge/discharge cycles with no net cost benefit. "
                "Typical EV Li-ion: 0.02–0.05 currency/kWh. "
                "Defaults to 0.0 (no wear cost)."
            ),
            "examples": [0.0, 0.03],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_soc_factors(self) -> "EVChargerCommonSettings":
        if self.ev_min_soc_factor >= self.ev_max_soc_factor:
            raise ValueError(
                "ev_min_soc_factor must be strictly less than ev_max_soc_factor"
            )
        if not (self.ev_min_soc_factor <= self.ev_target_soc_factor <= self.ev_max_soc_factor):
            raise ValueError(
                "ev_target_soc_factor must be in [ev_min_soc_factor, ev_max_soc_factor]"
            )
        if self.min_charge_power_w > self.max_charge_power_w:
            raise ValueError("min_charge_power_w must be <= max_charge_power_w")
        return self

    # ------------------------------------------------------------------
    # GENETIC2 domain conversion
    # ------------------------------------------------------------------

    def to_genetic2_param(self) -> "EVChargerParam":
        """Return an immutable ``EVChargerParam`` for the GENETIC2 optimizer."""
        from akkudoktoreos.devices.genetic2.evcharger import EVChargerParam

        return EVChargerParam(
            device_id=self.device_id,
            ports=self.ports_to_genetic2_param(),
            min_charge_power_w=self.min_charge_power_w,
            max_charge_power_w=self.max_charge_power_w,
            high_current_threshold_w=self.high_current_threshold_w,
            deep_standby_power_w=self.deep_standby_power_w,
            standby_power_w=self.standby_power_w,
            charger_efficiency=self.charger_efficiency,
            ev_charger_efficiency_low=self.ev_charger_efficiency_low,
            ev_charger_efficiency_high=self.ev_charger_efficiency_high,
            ev_battery_efficiency=self.ev_battery_efficiency,
            hold_time_efficiency=self.hold_time_efficiency,
            control_efficiency=self.control_efficiency,
            ev_battery_capacity_wh=self.ev_battery_capacity_wh,
            ev_min_soc_factor=self.ev_min_soc_factor,
            ev_max_soc_factor=self.ev_max_soc_factor,
            ev_target_soc_factor=self.ev_target_soc_factor,
            ev_initial_soc_factor_key=self.ev_initial_soc_factor_key,
            ev_connected_measurement_key=self.ev_connected_measurement_key,
            connection_time_window_key=self.connection_time_window_key,
            import_price_amt_kwh_key="elecprice_marketprice_amt_kwh",
            export_price_amt_kwh_key="feed_in_tariff_amt_kwh",
            ev_lcos_amt_kwh=self.ev_lcos_amt_kwh,
        )

    # ------------------------------------------------------------------
    # Measurement keys
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys this EV charger reads from the measurement store."""
        keys: list[str] = []
        if self.ev_initial_soc_factor_key:
            keys.append(self.ev_initial_soc_factor_key)
        if self.ev_connected_measurement_key:
            keys.append(self.ev_connected_measurement_key)
        return keys
