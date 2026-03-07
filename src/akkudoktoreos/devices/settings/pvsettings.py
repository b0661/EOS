"""Photovoltaic array device settings."""

from pydantic import Field, computed_field

from akkudoktoreos.devices.devicesabc import PVParam

from akkudoktoreos.devices.settings.devicebasesettings import DevicesBaseSettings, PortsMixin


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

    def to_genetic2_param(self) -> PVParam:
        """Return an immutable ``PVParam`` for the GENETIC2 optimizer."""
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
