from dataclasses import dataclass

from akkudoktoreos.devices.devicesabc import DeviceParam


@dataclass(frozen=True, slots=True)
class HeatPumpParam(DeviceParam):
    """Immutable heat pump parameters.

    Attributes:
        device_id: Unique identifier for this heat pump.
        ports: Ports connecting the heat pump to energy buses.
        operation_modes: Supported operating modes.
        thermal_power_w: Rated thermal output power [W].
        cop: Coefficient of performance (thermal output / electrical input).
    """

    thermal_power_w: float
    cop: float

    def __post_init__(self) -> None:
        if self.thermal_power_w <= 0:
            raise ValueError("thermal_power_w must be > 0")
        if self.cop <= 0:
            raise ValueError("cop must be > 0")
