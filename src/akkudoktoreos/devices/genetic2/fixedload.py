"""Fixed (non-controllable) household load device for the genetic2 simulation framework.

A ``FixedLoadDevice`` models a load whose power consumption is driven
entirely by an external forecast resolved from the ``SimulationContext``
during ``setup_run``. The optimizer has no control over it — there is no
genome, no repair, and no instruction output.

Typical uses: refrigerator, base load, always-on equipment, or any load
whose schedule cannot be shifted.

Role in the simulation pipeline
--------------------------------

1. **Arbitration** — ``build_device_request`` submits the forecast power
   as a sink request each step. The arbitrator grants as much as the AC bus
   can supply; any shortfall is covered by the grid slack device.

2. **Cost evaluation** — ``compute_cost`` returns an empty
   ``(population_size, 0)`` array. The fixed load has no private cost
   objective; its energy draw is implicitly reflected in the grid
   connection's import cost.

No genome
---------

``genome_requirements`` returns ``None``. No genome slice is allocated and
``apply_genome_batch`` is never called by the engine.

Sign convention
---------------

Consistent with the rest of the framework:

    positive energy_wh  → consuming from the AC bus (load)
    negative energy_wh  → injecting into the AC bus (not applicable here)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from akkudoktoreos.core.emplan import EnergyManagementInstruction
from akkudoktoreos.devices.devicesabc import (
    DeviceParam,
    EnergyDevice,
    EnergyPort,
    InstructionContext,
)
from akkudoktoreos.simulation.genetic2.arbitrator import (
    DeviceGrant,
    DeviceRequest,
    PortRequest,
)
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import DateTime

# ============================================================
# Immutable parameter dataclass
# ============================================================


@dataclass(frozen=True, slots=True)
class FixedLoadParam(DeviceParam):
    """Immutable parameters for a fixed (non-controllable) household load.

    Frozen, slotted, and hashable — safe as a dictionary or cache key
    inside the genetic algorithm. Carries no mutable state or simulation
    logic.

    Attributes:
        device_id:
            Unique identifier for this device instance.
        ports:
            Ports connecting this device to energy buses. Typically a
            single AC sink port.
        load_power_w_key:
            ``SimulationContext`` prediction key resolving to a per-step
            load forecast array [W] of shape ``(horizon,)``. Positive
            values represent consumption (load convention). Must be set
            explicitly — no default is provided here; use
            ``FixedLoadCommonSettings`` which defaults to
            ``"loadforecast_power_w"``.
    """

    load_power_w_key: str

    def __post_init__(self) -> None:
        if not self.ports:
            raise ValueError("FixedLoadParam requires at least one port.")
        if not self.load_power_w_key:
            raise ValueError("load_power_w_key must be a non-empty string.")


# ============================================================
# Mutable batch state
# ============================================================


@dataclass
class FixedLoadBatchState:
    """Mutable batch state for ``FixedLoadDevice``.

    Created fresh each generation by ``create_batch_state``.
    Never shared between devices or between generations.

    Attributes:
        granted_wh:
            Energy granted by the arbitrator per individual per step [Wh],
            shape ``(population_size, horizon)``. Positive = consumed from
            the bus. Populated by ``apply_device_grant``; all zeros until
            then.
        population_size:
            Number of individuals in this batch.
        horizon:
            Number of simulation time steps.
        step_times:
            Ordered ``DateTime`` timestamps, length == horizon.
    """

    granted_wh: np.ndarray          # (population_size, horizon)  float64
    population_size: int
    horizon: int
    step_times: tuple[DateTime, ...]


# ============================================================
# Device implementation
# ============================================================


class FixedLoadDevice(EnergyDevice):
    """Non-controllable fixed household load device.

    Reads a load forecast from the ``SimulationContext`` during
    ``setup_run`` and submits it unchanged as a sink request to the bus
    arbitrator every generation. The optimizer cannot shift or reduce
    this load.

    Parameters
    ----------
    param:
        Immutable device parameters.
    device_index:
        Position of this device in the shared device list, used by the
        arbitrator to route grants back to this device.
    port_index:
        Index of this device's AC port in the arbitrator's port-to-bus
        topology array.
    """

    def __init__(
        self,
        param: FixedLoadParam,
        device_index: int,
        port_index: int,
    ) -> None:
        super().__init__()
        self.param = param
        self.device_id: str = param.device_id
        self._device_index = device_index
        self._port_index = port_index

        # Populated during setup_run
        self._step_times: tuple[DateTime, ...] | None = None
        self._step_interval_sec: float | None = None
        # Forecast power per step [W], shape (horizon,). Resolved once per run.
        self._load_power_w: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return self.param.ports

    @property
    def objective_names(self) -> list[str]:
        """No private cost objectives — returns an empty list."""
        return []

    # ------------------------------------------------------------------
    # Structure Phase
    # ------------------------------------------------------------------

    def setup_run(self, context: SimulationContext) -> None:
        """Resolve the load forecast from the context for this run.

        Args:
            context: Run-scoped simulation context providing the horizon,
                step interval, timestamps, and prediction resolution.

        Raises:
            ValueError: If the resolved forecast array does not have shape
                ``(horizon,)``.
        """
        horizon = context.horizon
        self._step_times = context.step_times
        self._step_interval_sec = context.step_interval.total_seconds()

        load = context.resolve_prediction(self.param.load_power_w_key)
        if load.shape != (horizon,):
            raise ValueError(
                f"{self.device_id}: load forecast must have shape ({horizon},), "
                f"got {load.shape}."
            )
        # Clamp to non-negative: a fixed load only consumes, never injects.
        self._load_power_w = np.maximum(0.0, load)

    def genome_requirements(self) -> None:
        """The fixed load has no genome — return ``None``.

        The engine excludes devices that return ``None`` from genome
        assembly, so no genome slice is allocated and
        ``apply_genome_batch`` is never called.
        """
        return None

    # ------------------------------------------------------------------
    # Batch Lifecycle
    # ------------------------------------------------------------------

    def create_batch_state(
        self,
        population_size: int,
        horizon: int,
    ) -> FixedLoadBatchState:
        """Allocate a fresh batch state for one generation.

        Args:
            population_size: Number of individuals in the current population.
            horizon: Number of simulation time steps.

        Returns:
            ``FixedLoadBatchState`` with ``granted_wh`` initialised to
            zeros and ``step_times`` forwarded from the last
            ``setup_run`` call.
        """
        if self._step_times is None:
            raise RuntimeError("Call setup_run() before create_batch_state().")

        return FixedLoadBatchState(
            granted_wh=np.zeros((population_size, horizon), dtype=np.float64),
            population_size=population_size,
            horizon=horizon,
            step_times=self._step_times,
        )

    def apply_genome_batch(
        self,
        state: FixedLoadBatchState,
        genome_batch: np.ndarray,
    ) -> np.ndarray:
        """No-op — the fixed load has no genome.

        Defined to satisfy the ``EnergyDevice`` ABC and to guard against
        unexpected calls in tests.
        """
        return genome_batch

    # ------------------------------------------------------------------
    # Arbitration
    # ------------------------------------------------------------------

    def build_device_request(
        self,
        state: FixedLoadBatchState,
    ) -> DeviceRequest:
        """Submit the forecast load as a sink request to the bus arbitrator.

        The same forecast applies to every individual in the population —
        the fixed load is identical across all candidates. The request
        energy is broadcast over the population axis.

        Args:
            state: Batch state (``granted_wh`` not yet populated).

        Returns:
            ``DeviceRequest`` with the forecast energy per step for the
            full population batch.
        """
        if self._step_interval_sec is None or self._load_power_w is None:
            raise RuntimeError("Call setup_run() before build_device_request().")

        step_h = self._step_interval_sec / 3600.0
        # Convert power [W] → energy [Wh] and broadcast over population axis.
        # Shape: (population_size, horizon)
        energy_wh = np.broadcast_to(
            (self._load_power_w * step_h)[np.newaxis, :],
            (state.population_size, state.horizon),
        ).copy()

        return DeviceRequest(
            device_index=self._device_index,
            port_requests=(
                PortRequest(
                    port_index=self._port_index,
                    energy_wh=energy_wh,
                    # min_energy_wh = 0: accept partial supply gracefully
                    # (grid slack will cover any shortfall).
                    min_energy_wh=np.zeros_like(energy_wh),
                    is_slack=False,
                ),
            ),
        )

    def apply_device_grant(
        self,
        state: FixedLoadBatchState,
        grant: DeviceGrant,
    ) -> None:
        """Record the arbitrated energy grant.

        Args:
            state: Mutable batch state.
            grant: Arbitrated energy grant from the bus arbitrator.
        """
        state.granted_wh[:] = grant.port_grants[0].granted_wh

    # ------------------------------------------------------------------
    # Cost Evaluation
    # ------------------------------------------------------------------

    def compute_cost(self, state: FixedLoadBatchState) -> np.ndarray:
        """Return an empty cost matrix — fixed loads have no private objective.

        The energy draw of this device is implicitly reflected in the grid
        connection's import cost. Returning shape ``(population_size, 0)``
        tells the engine that this device contributes no fitness columns.

        Args:
            state: Batch state after ``apply_device_grant``.

        Returns:
            Zero-column array of shape ``(population_size, 0)``.
        """
        return np.zeros((state.population_size, 0))

    # ------------------------------------------------------------------
    # Instruction Extraction
    # ------------------------------------------------------------------

    def extract_instructions(
        self,
        state: FixedLoadBatchState,
        individual_index: int,
        instruction_context: InstructionContext | None = None,
    ) -> list[EnergyManagementInstruction]:
        """Return an empty instruction list.

        A fixed load is not a controllable S2 resource — there are no
        instructions to send. Its behaviour is fully determined by the
        forecast resolved during ``setup_run``.

        Returns:
            Empty list.
        """
        return []
