"""Abstract and base classes for optimization.

Implements the interface for the Optimization working as the Central Energy Controller (CEC)
for device simulations. It leans on the S2 energy management standard (EN 50491-12-2:2022).
"""

from typing import Union

from akkudoktor.simulation.simulationabc import (
    ID,
    ControlType,
    DDBCAverageDemandRateForecast,
    DDBCSystemDescription,
    FRBCFillLevelTargetProfile,
    FRBCLeakageBehaviour,
    FRBCSystemDescription,
    FRBCUsageForecast,
    OMBCSystemDescription,
    PEBCEnergyConstraints,
    PEBCPowerConstraints,
    PowerForecast,
    PowerMeasurement,
    PPBCPowerProfileDefinition,
    ReceptionStatus,
    ReceptionStatusValues,
    SimulationDetails,
)
from pydantic import Field

from akkudoktoreos.core.coreabc import ConfigMixin
from akkudoktoreos.core.logging import get_logger
from akkudoktoreos.core.pydantic import PydanticBaseModel

logger = get_logger(__name__)


class OptimizationBase(ConfigMixin, PydanticBaseModel):
    """Base class for an Optimization.

    Enables access to EOS configuration data (attribute `config`).

    An optimization run is organised in three phases:

    - Configuration (C): All simulations need to be identified and configured.
    - Scheduling (S):  Collecting relevant data for energy management scheduling. This data includes
        simulation descriptions, power and price profiles.
    - Optimisation (O):  Optimization of energy production, consumption and storage for all energy
        management participants.

    The Optimization works as the Central Energy Controller (CEC) for device simulations.
    It defines the abstract interface for handling simulation updates, control instructions,
    system descriptions, and constraints from device simulations, in line with
    the S2 standard.
    """

    power_forecast: PowerForecast = Field(
        ...,
        description=(
            "Power available to the device simulations. "
            "Updated by the simulations during optimisation by providing power measurements."
        ),
    )

    def process_simulation_update(
        self,
        simulation_id: ID,
        update: Union[
            SimulationDetails,
            ControlType,
            PEBCPowerConstraints,
            PEBCEnergyConstraints,
            PPBCPowerProfileDefinition,
            OMBCSystemDescription,
            FRBCSystemDescription,
            FRBCLeakageBehaviour,
            FRBCUsageForecast,
            FRBCFillLevelTargetProfile,
            DDBCSystemDescription,
            DDBCAverageDemandRateForecast,
        ],
    ) -> ReceptionStatus:
        """Process a simulation update from a device simulation.

        Args:
            simulation_id (ID): The ID of the device simulation.
            update (Union[...]): One of the allowed update types from the simulation.

        Returns:
            ReceptionStatus: Status of the update processing.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def process_power_measurement(
        self, simulation_id: ID, power_measurement: PowerMeasurement
    ) -> ReceptionStatus:
        """Process a power measurement from a device simulation.

        Args:
            simulation_id (ID): The ID of the device simulation.
            power_measurement (PowerMeasurement): Power measurement.

        Returns:
            ReceptionStatus: Status of the forecast processing.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def process_power_forecast(
        self, simulation_id: ID, power_forecast: PowerForecast
    ) -> ReceptionStatus:
        """Process a power forecast from a device simulation.

        Args:
            simulation_id (ID): The ID of the device simulation.
            power_forecast (PowerForecast): Forecast of future power usage or availability.

        Returns:
            ReceptionStatus: Status of the forecast processing.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_power_forecast(self, simulation_id: ID) -> ReceptionStatus:
        """Revoke a previously submitted power forecast.

        Args:
            simulation_id (ID): The ID of the device simulation.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_pebc_power_constraints(self, simulation_id: ID, id: ID) -> ReceptionStatus:
        """Revoke previously submitted PEBC power constraints.

        Args:
            simulation_id (ID): The ID of the device simulation.
            id (ID): The ID of the constraints to revoke.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_pebc_energy_constraints(self, simulation_id: ID, id: ID) -> ReceptionStatus:
        """Revoke previously submitted PEBC energy constraints.

        Args:
            simulation_id (ID): The ID of the device simulation.
            id (ID): The ID of the constraints to revoke.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_ppbc_power_profile_definition(self, simulation_id: ID, id: ID) -> ReceptionStatus:
        """Revoke a PPBC power profile definition.

        Args:
            simulation_id (ID): The ID of the device simulation.
            id (ID): The ID of the profile definition to revoke.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_ombc_system_description(self, simulation_id: ID, id: ID) -> ReceptionStatus:
        """Revoke a previously submitted OMBC system description.

        Args:
            simulation_id (ID): The ID of the device simulation.
            id (ID): The ID of the system description to revoke.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_frbc_system_description(self, simulation_id: ID, id: ID) -> ReceptionStatus:
        """Revoke a previously submitted FRBC system description.

        Args:
            simulation_id (ID): The ID of the device simulation.
            id (ID): The ID of the system description to revoke.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_frbc_leakage_behaviour(self, simulation_id: ID) -> ReceptionStatus:
        """Revoke FRBC leakage behaviour description.

        Args:
            simulation_id (ID): The ID of the device simulation.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_frbc_usage_forecast(self, simulation_id: ID) -> ReceptionStatus:
        """Revoke a previously submitted FRBC usage forecast.

        Args:
            simulation_id (ID): The ID of the device simulation.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_frbc_fill_level_target_profile(self, simulation_id: ID) -> ReceptionStatus:
        """Revoke a previously submitted FRBC fill level target profile.

        Args:
            simulation_id (ID): The ID of the device simulation.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_ddbc_system_description(self, simulation_id: ID) -> ReceptionStatus:
        """Revoke a previously submitted DDBC system description.

        Args:
            simulation_id (ID): The ID of the device simulation.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )

    def revoke_ddbc_average_demand_rate_forecast(self, simulation_id: ID) -> ReceptionStatus:
        """Revoke a previously submitted DDBC average demand rate forecast.

        Args:
            simulation_id (ID): The ID of the device simulation.

        Returns:
            ReceptionStatus: Status of the revocation.
        """
        return ReceptionStatus(
            status=ReceptionStatusValues.REJECTED, diagnostic_label="abstract method"
        )
