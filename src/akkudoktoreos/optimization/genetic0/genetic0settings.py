"""Settings for the GENETIC0 optimization algorithm.

Kept in an extra module to avoid cyclic dependencies on package import.
"""

from typing import Optional, Union

from pydantic import Field

from akkudoktoreos.config.configabc import SettingsBaseModel


class Genetic0CommonSettings(SettingsBaseModel):
    """GENETIC0 Optimization Algorithm Configuration."""

    individuals: Optional[int] = Field(
        default=300,
        ge=10,
        json_schema_extra={
            "description": "Number of individuals (solutions) in the population [>= 10]. Defaults to 300.",
            "examples": [300],
        },
    )

    generations: Optional[int] = Field(
        default=400,
        ge=10,
        json_schema_extra={
            "description": "Number of generations to evolve [>= 10]. Defaults to 400.",
            "examples": [400],
        },
    )

    seed: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "Random seed for reproducibility. None = random.",
            "examples": [None, 42],
        },
    )

    # --- Penalties (existing) -------------------------------------------------

    penalties: dict[str, Union[float, int, str]] = Field(
        default_factory=lambda: {
            "ev_soc_miss": 10,
            "ac_charge_break_even": 1.0,
        },
        json_schema_extra={
            "description": "Penalty parameters used in fitness evaluation.",
            "examples": [{"ev_soc_miss": 10}],
        },
    )
