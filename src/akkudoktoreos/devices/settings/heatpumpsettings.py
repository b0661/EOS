"""Heat pump device settings."""

from typing import TYPE_CHECKING

from pydantic import Field, computed_field

from akkudoktoreos.config.configabc import ConfigScope
from akkudoktoreos.devices.settings.devicebasesettings import (
    DevicesBaseSettings,
    PortsMixin,
)

if TYPE_CHECKING:
    from akkudoktoreos.devices.genetic2.heatpump import HeatPumpParam


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
            "x-scope": [str(ConfigScope.GENETIC2)],
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
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # ------------------------------------------------------------------
    # GENETIC2 domain conversion
    # ------------------------------------------------------------------

    def to_genetic2_param(self) -> "HeatPumpParam":
        """Return an immutable ``HeatPumpParam`` for the GENETIC2 optimizer."""
        from akkudoktoreos.devices.genetic2.heatpump import HeatPumpParam

        return HeatPumpParam(
            device_id=self.device_id,
            ports=self.ports_to_genetic2_param(),
            thermal_power_w=self.thermal_power_w,
            cop=self.cop,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this heat pump (currently none)."""
        return []
