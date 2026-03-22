"""Grid connection device settings."""

from typing import TYPE_CHECKING

from pydantic import Field, computed_field

from akkudoktoreos.devices.settings.devicebasesettings import (
    DevicesBaseSettings,
    PortsMixin,
)

if TYPE_CHECKING:
    from akkudoktoreos.devices.genetic2.gridconnection import GridConnectionParam


class GridConnectionCommonSettings(PortsMixin, DevicesBaseSettings):
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
    include_peak_power_objective: bool = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Include peak import power [kW] as a second fitness objective. "
                "When True, the optimiser minimises both energy cost and peak grid import. "
                "Note: peak_import_kw is in kW while energy_cost is in EUR — they are summed "
                "with equal weight by default, so a 5 kW peak dominates a 0.50 EUR cost. "
                "Only enable when peak demand charges apply to your tariff. Default False."
            ),
            "examples": [False, True],
        },
    )

    def to_genetic2_param(self) -> "GridConnectionParam":
        """Return an immutable parameter object for the GENETIC2 optimizer."""
        from akkudoktoreos.devices.genetic2.gridconnection import GridConnectionParam

        return GridConnectionParam(
            device_id=self.device_id,
            ports=self.ports_to_genetic2_param(),
            max_import_power_w=self.max_import_power_w,
            max_export_power_w=self.max_export_power_w,
            import_cost_per_kwh=self.import_cost_per_kwh,
            export_revenue_per_kwh=self.export_revenue_per_kwh,
            import_price_amt_kwh_key="elecprice_marketprice_amt_kwh",
            export_price_amt_kwh_key="feed_in_tariff_amt_kwh",
            include_peak_power_objective=self.include_peak_power_objective,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys for this grid connection (currently none)."""
        return []
