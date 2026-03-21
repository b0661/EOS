"""Battery and electric vehicle device settings.

Note: Used for the GENETIC algorithm only.
"""

import re
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from pydantic import Field, computed_field, field_validator, model_validator

from akkudoktoreos.devices.devicesabc import BatteryOperationMode
from akkudoktoreos.devices.settings.devicebasesettings import DevicesBaseSettings

if TYPE_CHECKING:
    from akkudoktoreos.devices.genetic.battery import (
        ElectricVehicleParameters,
        SolarPanelBatteryParameters,
    )


# Default charge rates for battery
BATTERY_DEFAULT_CHARGE_RATES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class BatteriesCommonSettings(DevicesBaseSettings):
    """Battery and electric vehicle device settings.

    Used for both stationary batteries and EV battery packs.

    Note: Used for the GENETIC algorithm only.
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
    levelized_cost_of_storage_amt_kwh: float = Field(
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
    operation_modes: list[str] = Field(
        default_factory=lambda: [str(BatteryOperationMode.SELF_CONSUMPTION)],
        json_schema_extra={
            "description": "Supported operating modes for this battery.",
            "examples": [["SELF_CONSUMPTION", "PEAK_SHAVING"]],
        },
    )

    # ------------------------------------------------------------------
    # GENETIC domain conversion
    # ------------------------------------------------------------------

    def to_genetic_pv_bat_param(self) -> "SolarPanelBatteryParameters":
        """Return SolarPanelBatteryParameters for the GENETIC optimizer."""
        from akkudoktoreos.devices.genetic.battery import SolarPanelBatteryParameters

        return SolarPanelBatteryParameters(
            device_id=self.device_id,
            capacity_wh=self.capacity_wh,
            charging_efficiency=self.charging_efficiency,
            discharging_efficiency=self.discharging_efficiency,
            max_charge_power_w=self.max_charge_power_w,
            min_soc_percentage=self.min_soc_percentage,
            max_soc_percentage=self.max_soc_percentage,
        )

    def to_genetic_ev_bat_param(self) -> "ElectricVehicleParameters":
        """Return ElectricVehicleParameters for the GENETIC optimizer."""
        from akkudoktoreos.devices.genetic.battery import ElectricVehicleParameters

        return ElectricVehicleParameters(
            device_id=self.device_id,
            capacity_wh=self.capacity_wh,
            charging_efficiency=self.charging_efficiency,
            discharging_efficiency=self.discharging_efficiency,
            charge_rates=self.charge_rates,
            max_charge_power_w=self.max_charge_power_w,
            min_soc_percentage=self.min_soc_percentage,
            max_soc_percentage=self.max_soc_percentage,
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
