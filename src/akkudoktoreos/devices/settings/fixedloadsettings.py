"""Fixed (non-controllable) household load settings."""

from typing import TYPE_CHECKING

from pydantic import Field, computed_field

from akkudoktoreos.config.configabc import ConfigScope
from akkudoktoreos.devices.settings.devicebasesettings import (
    DevicesBaseSettings,
    PortsMixin,
)

if TYPE_CHECKING:
    from akkudoktoreos.devices.genetic2.fixedload import FixedLoadParam


class FixedLoadCommonSettings(PortsMixin, DevicesBaseSettings):
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
    """

    load_power_w_key: str = Field(
        default="loadforecast_power_w",
        json_schema_extra={
            "description": (
                "SimulationContext prediction key resolving to a per-step "
                "load forecast array [W] of shape (horizon,). "
            ),
            "examples": ["loadforecast_power_w", ""],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    def to_genetic2_param(self) -> "FixedLoadParam":
        """Return an immutable domain object for the GENETIC2 optimizer."""
        from akkudoktoreos.devices.genetic2.fixedload import FixedLoadParam

        return FixedLoadParam(
            device_id=self.device_id,
            ports=self.ports_to_genetic2_param(),
            load_power_w_key=self.load_power_w_key,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this load (currently none)."""
        return []
