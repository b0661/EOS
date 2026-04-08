from typing import Dict, Optional, Union

from pydantic import Field, computed_field, model_validator

from akkudoktoreos.config.configabc import ConfigScope, SettingsBaseModel
from akkudoktoreos.core.coreabc import get_ems
from akkudoktoreos.core.pydantic import (
    PydanticBaseModel,
    PydanticDateTimeDataFrame,
)
from akkudoktoreos.utils.datetimeutil import DateTime


class GeneticCommonSettings(SettingsBaseModel):
    """General Genetic Optimization Algorithm Configuration."""

    individuals: Optional[int] = Field(
        default=300,
        ge=10,
        json_schema_extra={
            "description": "Number of individuals (solutions) in the population [>= 10]. Defaults to 300.",
            "examples": [300],
            "x-scope": [str(ConfigScope.GENETIC), str(ConfigScope.GENETIC2)],
        },
    )

    generations: Optional[int] = Field(
        default=400,
        ge=10,
        json_schema_extra={
            "description": "Number of generations to evolve [>= 10]. Defaults to 400.",
            "examples": [400],
            "x-scope": [str(ConfigScope.GENETIC), str(ConfigScope.GENETIC2)],
        },
    )

    seed: Optional[int] = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "Random seed for reproducibility. None = random.",
            "examples": [None, 42],
            "x-scope": [str(ConfigScope.GENETIC), str(ConfigScope.GENETIC2)],
        },
    )

    # --- Penalties (existing) -------------------------------------------------

    penalties: Dict[str, Union[float, int, str]] = Field(
        default_factory=lambda: {
            "ev_soc_miss": 10,
            "ac_charge_break_even": 1.0,
        },
        json_schema_extra={
            "description": "Penalty parameters used in fitness evaluation.",
            "examples": [{"ev_soc_miss": 10}],
            "x-scope": [str(ConfigScope.GENETIC)],
        },
    )

    # --- Core GA behavior -----------------------------------------------------

    crossover_rate: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "description": "Probability of applying crossover between two parents [0–1]. Higher values increase exploitation. Defaults to 0.9.",
            "examples": [0.8, 0.9],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    mutation_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "description": "Probability of mutating each gene [0–1]. Controls exploration. Defaults to 0.05.",
            "examples": [0.01, 0.05, 0.1],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    mutation_sigma: float = Field(
        default=0.03,
        ge=0.0,
        json_schema_extra={
            "description": "Standard deviation of mutation noise. Controls mutation strength. Defaults to 0.03.",
            "examples": [0.05, 0.1, 0.2],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    tournament_size: int = Field(
        default=3,
        ge=2,
        json_schema_extra={
            "description": "Number of individuals competing in tournament selection. Higher values increase selection pressure. Defaults to 3.",
            "examples": [2, 3, 5],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    elitism_count: int = Field(
        default=2,
        ge=0,
        json_schema_extra={
            "description": "Number of top individuals copied unchanged to the next generation. Prevents loss of best solutions. Defaults to 2.",
            "examples": [0, 1, 2, 5],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # --- Adaptive mutation / stagnation ---------------------------------------

    stagnation_window: int = Field(
        default=20,
        ge=1,
        json_schema_extra={
            "description": "Number of generations without improvement before triggering mutation boost. Defaults to 20.",
            "examples": [10, 20, 50],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    stagnation_boost: float = Field(
        default=3.0,
        ge=1.0,
        json_schema_extra={
            "description": "Multiplier applied to mutation_rate and mutation_sigma when stagnation is detected. Defaults to 3.0.",
            "examples": [2.0, 3.0, 5.0],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    # --- Optimization process logging ----------------------------------------

    log_progress_interval: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": (
                "Log optimization progress every N generations. "
                "0 = disabled (default, no overhead). "
                "Positive integer = log every N generations plus the final generation. "
                "Output includes generation index, best scalar fitness, mean scalar fitness, "
                "and number of repaired genomes."
            ),
            "examples": [0, 10, 50],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )


class OptimizationCommonSettings(SettingsBaseModel):
    """General Optimization Configuration."""

    horizon_hours: int = Field(
        default=24,
        ge=0,
        json_schema_extra={
            "description": "The general time window within which the energy optimization goal shall be achieved [h]. Defaults to 24 hours.",
            "examples": [24],
        },
    )

    interval: int = Field(
        default=3600,
        ge=15 * 60,
        le=60 * 60,
        json_schema_extra={
            "description": "The optimization interval [sec]. Defaults to 3600 seconds (1 hour)",
            "examples": [60 * 60, 15 * 60],
        },
    )

    algorithm: str = Field(
        default="GENETIC",
        json_schema_extra={
            "description": "The optimization algorithm. Defaults to GENETIC",
            "examples": ["GENETIC"],
        },
    )

    genetic: GeneticCommonSettings = Field(
        default_factory=GeneticCommonSettings,
        json_schema_extra={
            "description": "Genetic optimization algorithm configuration.",
            "examples": [{"individuals": 400, "seed": None, "penalties": {"ev_soc_miss": 10}}],
        },
    )

    # Computed fields
    @computed_field  # type: ignore[prop-decorator]
    @property
    def keys(self) -> list[str]:
        """The keys of the solution."""
        try:
            ems_eos = get_ems()
        except:
            # ems might not be initialized
            return []

        key_list = []
        optimization_solution = ems_eos.optimization_solution()
        if optimization_solution:
            # Prepare mapping
            df = optimization_solution.solution.to_dataframe()
            key_list = df.columns.tolist()
        return sorted(set(key_list))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def horizon(self) -> int:
        """Number of optimization steps."""
        if self.interval is None or self.interval == 0 or self.horizon_hours is None:
            return 0
        num_steps = int(float(self.horizon_hours * 3600) / self.interval)
        return num_steps

    # Validators
    @model_validator(mode="after")
    def _enforce_algorithm_configuration(self) -> "OptimizationCommonSettings":
        """Ensure algorithm default configuration is set."""
        if self.algorithm is not None:
            if self.algorithm.lower() == "genetic" and self.genetic is None:
                self.genetic = GeneticCommonSettings()
        return self


class OptimizationSolution(PydanticBaseModel):
    """General Optimization Solution."""

    id: str = Field(
        ..., json_schema_extra={"description": "Unique ID for the optimization solution."}
    )

    generated_at: DateTime = Field(
        ..., json_schema_extra={"description": "Timestamp when the solution was generated."}
    )

    comment: Optional[str] = Field(
        default=None,
        json_schema_extra={"description": "Optional comment or annotation for the solution."},
    )

    valid_from: Optional[DateTime] = Field(
        default=None, json_schema_extra={"description": "Start time of the optimization solution."}
    )

    valid_until: Optional[DateTime] = Field(
        default=None, json_schema_extra={"description": "End time of the optimization solution."}
    )

    total_losses_energy_wh: float = Field(
        json_schema_extra={"description": "The total losses in watt-hours over the entire period."}
    )

    total_revenues_amt: float = Field(
        json_schema_extra={"description": "The total revenues [money amount]."}
    )

    total_costs_amt: float = Field(
        json_schema_extra={"description": "The total costs [money amount]."}
    )

    fitness_score: set[float] = Field(
        json_schema_extra={"description": "The fitness score as a set of fitness values."}
    )

    prediction: PydanticDateTimeDataFrame = Field(
        json_schema_extra={
            "description": (
                "Datetime data frame with time series prediction data per optimization interval:"
                "- pv_energy_wh: PV energy prediction (positive) in wh"
                "- elec_price_amt_kwh: Electricity price prediction in money per kwh"
                "- feed_in_tariff_amt_kwh: Feed in tariff prediction in money per kwh"
                "- weather_temp_air_celcius: Temperature in °C"
                "- loadforecast_energy_wh: Load mean energy prediction in wh"
                "- loadakkudoktor_std_energy_wh: Load energy standard deviation prediction in wh"
                "- loadakkudoktor_mean_energy_wh: Load mean energy prediction in wh"
            )
        }
    )

    solution: PydanticDateTimeDataFrame = Field(
        json_schema_extra={
            "description": (
                "Datetime data frame with time series solution data per optimization interval:"
                "- load_energy_wh: Load of all energy consumers in wh"
                "- grid_energy_wh: Grid energy feed in (negative) or consumption (positive) in wh"
                "- costs_amt: Costs in money amount"
                "- revenue_amt: Revenue in money amount"
                "- losses_energy_wh: Energy losses in wh"
                "- <device-id>_soc_factor: State of charge of a battery/ electric vehicle device as factor of total capacity."
                "- <device-id>_energy_wh: Energy consumption (positive) of a device in wh."
                "- <device-id>_<mode>_op_mode: Operation mode <mode> active (1.0) or inactive (0.0)."
                "- <device-id>_<mode>_op_factor: Operation mode factor for <mode> (0.0 when inactive)."
            )
        }
    )

    optimization_log: Optional[PydanticDateTimeDataFrame] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Per-generation optimization progress log. Only populated when "
                "genetic.log_progress_interval > 0. Rows are indexed by generation number. "
                "Columns include: best_scalar_fitness, mean_scalar_fitness, num_repaired, "
                "obj_<name> per objective, bat_factor_min/mean/max, pv_util_min/mean/max."
            )
        },
    )

    run_summary: Optional[dict] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "JSON-serialisable run-level metadata. Only populated when "
                "genetic.log_progress_interval > 0. Includes device list, "
                "PV/price forecast summary, convergence generation, and elapsed time."
            )
        },
    )
