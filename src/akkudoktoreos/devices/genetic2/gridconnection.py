"""Grid connection device for the genetic2 simulation framework.

The grid connection is the **slack device** on the AC bus. After all
controllable devices (inverters, batteries, heat pumps, …) have submitted
their energy requests and the bus arbitrator has settled them, the grid
connection absorbs whatever import or export residual remains. It is the
only device in the system that translates the arbitrated AC bus outcome
into an electricity cost objective.

Role in the simulation pipeline
--------------------------------
1. **Arbitration** — ``build_device_request`` offers the full grid
   connection window (up to ``max_import_power_w`` on the consume side,
   up to ``max_export_power_w`` on the inject side) so the arbitrator can
   always clear any bus imbalance through this port. The granted energy
   reflects actual import / export after all other devices have been
   settled first.

2. **Cost evaluation** — ``compute_cost`` converts the granted import/
   export energy into an ``"energy_cost_eur"`` fitness column using
   per-step electricity prices. An optional second column
   ``"peak_import_kw"`` supports peak-shaving objectives.

No genome
---------
The grid is not an optimizer-controlled resource — there are no decisions
for the GA to make about it. ``genome_requirements`` returns ``None``, so
no genome slice is allocated and ``apply_genome_batch`` is never called by
the engine.

Electricity pricing
-------------------
Two pricing modes are supported and may be combined per direction:

**Flat rate** (always active)
    ``import_cost_per_kwh`` and ``export_revenue_per_kwh`` in
    ``GridConnectionParam`` apply uniformly across every step.

**Time-of-use** (optional, context-resolved)
    If ``import_price_key`` or ``export_price_key`` is set, the
    corresponding time-series is resolved from the ``SimulationContext``
    during ``setup_run`` and used *instead of* the flat rate for that
    direction. This supports day-ahead spot prices, dynamic feed-in
    tariffs, and any step-varying tariff.

Sign convention
---------------
Consistent with the rest of the framework:

    positive energy_wh  → consuming from the AC bus (grid import)
    negative energy_wh  → injecting into the AC bus (grid export)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from akkudoktoreos.core.emplan import EnergyManagementInstruction
from akkudoktoreos.devices.devicesabc import (
    EnergyDevice,
    EnergyPort,
    InstructionContext,
    PortDirection,
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
class GridConnectionParam:
    """Immutable parameters for a grid connection device.

    Frozen, slotted, and hashable — safe as a dictionary or cache key
    inside the genetic algorithm. Carries no mutable state or simulation
    logic.

    Attributes:
    ----------
    device_id :
        Unique identifier for this device instance.
    port_id :
        Identifier of the single bidirectional AC port. Must be unique
        within the device and must match the port index registered with
        the bus arbitrator's topology map.
    bus_id :
        ID of the AC bus this grid connection is attached to.
    max_import_power_w :
        Maximum continuous power the grid can supply to the local AC bus
        [W]. Must be > 0. Caps the import side of the arbitration request.
        Typical residential values: 11 000 – 25 000 W.
    max_export_power_w :
        Maximum continuous power the local AC bus can push back to the
        grid [W]. Must be ≥ 0. Set to 0 to disable export (non-export
        tariff or inverter export limit). Caps the export side of the
        arbitration request.
    import_cost_per_kwh :
        Flat-rate electricity import cost [currency / kWh]. Applied
        uniformly to every imported kWh unless overridden by a resolved
        ``import_price_key`` time-series. Must be ≥ 0.
    export_revenue_per_kwh :
        Flat-rate feed-in revenue [currency / kWh]. Subtracted from total
        cost for every exported kWh unless overridden by a resolved
        ``export_price_key`` time-series. Must be ≥ 0.
    import_price_key :
        Optional key in the ``SimulationContext`` time-series store
        resolving to a per-step import price array [currency / kWh] of
        shape ``(horizon,)``. When set, replaces ``import_cost_per_kwh``
        step-by-step. Useful for day-ahead market prices.
    export_price_key :
        Optional key in the ``SimulationContext`` time-series store
        resolving to a per-step export revenue array [currency / kWh] of
        shape ``(horizon,)``. When set, replaces
        ``export_revenue_per_kwh`` step-by-step.
    include_peak_power_objective :
        When ``True``, ``compute_cost`` emits a second fitness column
        ``"peak_import_kw"`` equal to the maximum import power [kW]
        observed across the horizon for each individual. Enables
        peak-shaving alongside cost minimisation without changing the
        energy cost objective.
    """

    device_id: str
    port_id: str
    bus_id: str
    max_import_power_w: float
    max_export_power_w: float
    import_cost_per_kwh: float
    export_revenue_per_kwh: float
    import_price_key: str | None = None
    export_price_key: str | None = None
    include_peak_power_objective: bool = False

    def __post_init__(self) -> None:
        if self.max_import_power_w <= 0:
            raise ValueError("max_import_power_w must be > 0")
        if self.max_export_power_w < 0:
            raise ValueError("max_export_power_w must be >= 0")
        if self.import_cost_per_kwh < 0:
            raise ValueError("import_cost_per_kwh must be >= 0")
        if self.export_revenue_per_kwh < 0:
            raise ValueError("export_revenue_per_kwh must be >= 0")


# ============================================================
# Mutable batch state
# ============================================================


@dataclass
class GridConnectionBatchState:
    """Mutable batch state for ``GridConnectionDevice``.

    Created fresh each generation by ``create_batch_state``.
    Never shared between devices or between generations.

    Attributes:
    ----------
    granted_wh :
        Net AC energy granted by the arbitrator per individual per step
        [Wh], shape ``(population_size, horizon)``. Positive = imported
        from grid; negative = exported to grid. Populated by
        ``apply_device_grant``; all zeros until then.
    population_size :
        Number of individuals in this batch.
    horizon :
        Number of simulation time steps.
    step_times :
        Ordered ``DateTime`` timestamps, length == horizon.
    """

    granted_wh: np.ndarray  # (population_size, horizon)  float64
    population_size: int
    horizon: int
    step_times: tuple[DateTime, ...]


# ============================================================
# Device implementation
# ============================================================


class GridConnectionDevice(EnergyDevice):
    """Slack-bus grid connection device.

    Represents the household's point of connection to the public AC grid.

    Parameters
    ----------
    param :
        Immutable device parameters. Validated at construction.
    device_index :
        Index used by the bus arbitrator to identify this device's port
        request and match it to its grant. Must be unique across all
        devices registered with the same engine.
    port_index :
        Index of this device's AC port in the arbitrator's port-to-bus
        topology array.
    """

    def __init__(
        self,
        param: GridConnectionParam,
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
        self._num_steps: int | None = None
        self._step_interval_sec: float | None = None

        # Resolved per-step price arrays, shape (horizon,).
        # None when the corresponding price key is not configured;
        # the flat-rate scalar from param is used in that case.
        self._import_price_per_kwh: np.ndarray | None = None
        self._export_price_per_kwh: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        """Single bidirectional AC port connecting to the public grid."""
        return (
            EnergyPort(
                port_id=self.param.port_id,
                bus_id=self.param.bus_id,
                direction=PortDirection.BIDIRECTIONAL,
            ),
        )

    @property
    def objective_names(self) -> list[str]:
        """Fitness column names contributed to the global objective matrix.

        Always includes ``"energy_cost_eur"``.
        Includes ``"peak_import_kw"`` when
        ``param.include_peak_power_objective`` is ``True``.
        """
        names = ["energy_cost_eur"]
        if self.param.include_peak_power_objective:
            names.append("peak_import_kw")
        return names

    # ------------------------------------------------------------------
    # Structure Phase
    # ------------------------------------------------------------------

    def setup_run(self, context: SimulationContext) -> None:
        """Store run configuration and resolve optional time-of-use prices.

        Args:
            context: Run-scoped simulation context providing the horizon,
                step interval, timestamps, and a key-value resolution API
                for time-series forecasts and scalar measurements.

        Raises:
            ValueError: If a configured price key resolves to an array
                whose shape does not match ``(horizon,)``.
        """
        horizon = context.horizon
        self._step_times = context.step_times
        self._num_steps = horizon
        self._step_interval_sec = context.step_interval.total_seconds()

        if self.param.import_price_key is not None:
            prices = context.resolve(self.param.import_price_key)
            if prices.shape != (horizon,):
                raise ValueError(
                    f"{self.device_id}: import price series must have shape "
                    f"({horizon},), got {prices.shape}."
                )
            self._import_price_per_kwh = prices
        else:
            self._import_price_per_kwh = None

        if self.param.export_price_key is not None:
            prices = context.resolve(self.param.export_price_key)
            if prices.shape != (horizon,):
                raise ValueError(
                    f"{self.device_id}: export price series must have shape "
                    f"({horizon},), got {prices.shape}."
                )
            self._export_price_per_kwh = prices
        else:
            self._export_price_per_kwh = None

    def genome_requirements(self) -> None:
        """The grid connection has no genome — return ``None``.

        The engine excludes devices that return ``None`` from the genome
        assembly step, so no genome slice is allocated and
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
    ) -> GridConnectionBatchState:
        """Allocate a fresh batch state for one generation.

        Args:
            population_size: Number of individuals in the current population.
            horizon: Number of simulation time steps.

        Returns:
            ``GridConnectionBatchState`` with ``granted_wh`` initialised
            to zeros and ``step_times`` forwarded from the last
            ``setup_run`` call.
        """
        assert self._step_times is not None, "Call setup_run() before create_batch_state()."
        return GridConnectionBatchState(
            granted_wh=np.zeros((population_size, horizon), dtype=np.float64),
            population_size=population_size,
            horizon=horizon,
            step_times=self._step_times,
        )

    def apply_genome_batch(
        self,
        state: GridConnectionBatchState,
        genome_batch: np.ndarray,
    ) -> np.ndarray:
        """No-op — the grid has no genome to apply.

        The engine will not call this method because ``genome_requirements``
        returns ``None``. Defined here to satisfy the ``EnergyDevice`` ABC
        and to guard against unexpected calls in tests.
        """
        return genome_batch

    # ------------------------------------------------------------------
    # Arbitration
    # ------------------------------------------------------------------

    def build_device_request(
        self,
        state: GridConnectionBatchState,
    ) -> DeviceRequest:
        """Offer the full grid connection window to the bus arbitrator.

        Requests ``max_import_power_w * step_h`` Wh of potential import
        (positive) and sets ``min_energy_wh`` to the negated maximum
        export energy so the arbitrator knows it may grant up to
        ``max_export_power_w * step_h`` Wh of export (negative) through
        this port. The arbitrator grants only what is needed to balance
        the bus after all other devices have been settled.

        The request is identical for every individual and every step —
        the grid connection capacity does not depend on the genome.

        Args:
            state: Batch state (``granted_wh`` not yet populated).

        Returns:
            ``DeviceRequest`` covering the full grid connection window
            for the entire population batch.
        """
        assert self._step_interval_sec is not None, (
            "Call setup_run() before build_device_request()."
        )
        p = self.param
        step_h = self._step_interval_sec / 3600.0
        pop = state.population_size
        horizon = state.horizon

        max_import_wh = p.max_import_power_w * step_h
        max_export_wh = p.max_export_power_w * step_h

        return DeviceRequest(
            device_index=self._device_index,
            port_requests=(
                PortRequest(
                    port_index=self._port_index,
                    energy_wh=np.full((pop, horizon), max_import_wh),
                    min_energy_wh=np.full((pop, horizon), -max_export_wh),
                    is_slack=True,
                ),
            ),
        )

    def apply_device_grant(
        self,
        state: GridConnectionBatchState,
        grant: DeviceGrant,
    ) -> None:
        """Record the arbitrated grid energy for cost computation.

        Args:
            state: Mutable batch state.
            grant: Arbitrated energy grant from the bus arbitrator.
                ``grant.port_grants[0].granted_wh`` has shape
                ``(population_size, horizon)``.
        """
        state.granted_wh[:] = grant.port_grants[0].granted_wh

    # ------------------------------------------------------------------
    # Cost Evaluation
    # ------------------------------------------------------------------

    def compute_cost(
        self,
        state: GridConnectionBatchState,
    ) -> np.ndarray:
        """Compute electricity cost (and optionally peak power) objectives.

        Energy cost column (``"energy_cost_eur"``)
        -------------------------------------------
        For each individual ``i`` and step ``t``:

        - Import step (``granted_wh[i, t] > 0``):
          ``cost += granted_wh[i, t] / 1000 * import_price[t]``
        - Export step (``granted_wh[i, t] < 0``):
          ``cost -= abs(granted_wh[i, t]) / 1000 * export_price[t]``

        ``import_price[t]`` is the time-of-use array resolved from the
        context during ``setup_run`` if ``import_price_key`` was set,
        otherwise a uniform array filled with ``param.import_cost_per_kwh``.
        Likewise for ``export_price[t]``.

        Net cost can be negative when export revenue exceeds import cost —
        the optimizer treats lower as better, so this correctly drives the
        GA toward maximising self-consumption and valuable export.

        Peak import power column (``"peak_import_kw"``, optional)
        -----------------------------------------------------------
        Emitted only when ``param.include_peak_power_objective`` is
        ``True``. Equals the maximum per-step import power [kW] across
        the horizon for each individual:

            ``peak_kw[i] = max(max(granted_wh[i, :] / step_h, 0)) / 1000``

        Export steps do not contribute (clamped to zero).

        Args:
            state: Batch state after ``apply_device_grant``.

        Returns:
            Cost array of shape ``(population_size, num_objectives)``
            where ``num_objectives`` is 1 normally or 2 when
            ``include_peak_power_objective`` is ``True``.
        """
        assert self._step_interval_sec is not None, "Call setup_run() before compute_cost()."
        p = self.param
        granted = state.granted_wh  # (pop, horizon)
        step_h = self._step_interval_sec / 3600.0
        horizon = state.horizon

        # Build per-step price vectors, shape (horizon,).
        import_price = (
            self._import_price_per_kwh
            if self._import_price_per_kwh is not None
            else np.full(horizon, p.import_cost_per_kwh)
        )
        export_price = (
            self._export_price_per_kwh
            if self._export_price_per_kwh is not None
            else np.full(horizon, p.export_revenue_per_kwh)
        )

        # Separate import (>0) and export (>0 in magnitude) energy.
        import_wh = np.maximum(0.0, granted)  # (pop, horizon)
        export_wh = np.maximum(0.0, -granted)  # (pop, horizon)

        # Broadcast (horizon,) price over population axis → (pop, horizon),
        # then sum over horizon to get per-individual net cost.
        energy_cost = (
            (import_wh / 1000.0) * import_price - (export_wh / 1000.0) * export_price
        ).sum(axis=1)  # (pop,)

        if not p.include_peak_power_objective:
            return energy_cost[:, np.newaxis]  # (pop, 1)

        # Peak import power [kW]: max import-only power across the horizon.
        peak_import_kw = np.maximum(0.0, granted).max(axis=1) / step_h / 1000.0  # (pop,)

        return np.column_stack([energy_cost, peak_import_kw])  # (pop, 2)

    # ------------------------------------------------------------------
    # Instruction Extraction
    # ------------------------------------------------------------------

    def extract_instructions(
        self,
        state: GridConnectionBatchState,
        individual_index: int,
        instruction_context: InstructionContext | None = None,
    ) -> list[EnergyManagementInstruction]:
        """Return an empty instruction list.

        The grid connection is not a controllable S2 resource — there are
        no instructions to send to the grid operator. All control happens
        on the local devices (inverter, battery, …), and the grid simply
        responds.

        Returns:
            Empty list.
        """
        return []
