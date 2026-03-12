"""Fixed (non-controllable) household load settings."""

from pydantic import Field, computed_field

from akkudoktoreos.devices.devicesabc import DeviceParam
from akkudoktoreos.devices.settings.devicebasesettings import (
    DevicesBaseSettings,
    PortsMixin,
)


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
    ``to_genetic2_param()`` raises ``NotImplementedError`` until it is
    added.
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

    def to_genetic2_param(self) -> DeviceParam:
        """Return an immutable domain object for the GENETIC2 optimizer.

        Raises:
            NotImplementedError: ``FixedLoadParam`` is not yet defined in
                ``devicesabc``. Add it there first.
        """
        # TODO: return FixedLoadParam(...) once defined in devicesabc.
        raise NotImplementedError(
            "FixedLoadParam is not yet defined in devicesabc. "
            "Add it before calling to_genetic2_param() on FixedLoadSettings."
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this load (currently none)."""
        return []
