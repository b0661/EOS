"""Inverter device settings."""

from typing import Optional

from pydantic import Field, computed_field

from akkudoktoreos.devices.devicesabc import InverterParam

from akkudoktoreos.devices.settings.devicebasesettings import DevicesBaseSettings, PortsMixin


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

    Notes
    -----
    ``battery_id`` is retained for backward compatibility with existing
    configurations but is not used by ``to_genetic2_param()`` or the
    simulation engine. Port wiring replaces that coupling.
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
    # Retained for backward compatibility — not used by to_genetic2_param().
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

    def to_genetic2_param(self) -> InverterParam:
        """Return an immutable ``InverterParam`` for the GENETIC2 optimizer.

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
