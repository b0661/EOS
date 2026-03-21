"""Hybrid inverter energy device — continuous two-gene genome encoding.

Models the same three inverter topologies as ``hybridinverter.py``:

- **Battery inverter** (``BATTERY``) — manages a battery pack on the AC
  bus. No PV input.
- **Solar inverter** (``SOLAR``) — converts DC PV power to AC. No battery.
- **Hybrid inverter** (``HYBRID``) — manages both a battery pack and PV
  input behind a single AC port.

This module replaces the discrete mode + fractional-factor single-gene
encoding with a **continuous two-gene** encoding that expresses battery
power and PV utilisation directly as physical fractions, avoiding the
need for an explicit mode enumeration.

Genome structure
----------------
Two genes are interleaved per time step as
``[bat_factor₀, pv_util₀, bat_factor₁, pv_util₁, …]``, giving a
genome of size ``2 × horizon`` for all inverter types.

``bat_factor[t] ∈ [−1, +1]``
    Signed fraction of the rated battery power:

    * ``bat_factor > 0`` — charge the battery at
      ``bat_factor × battery_max_charge_rate × battery_capacity_wh`` W
      (battery side).
    * ``bat_factor < 0`` — discharge the battery at
      ``|bat_factor| × battery_max_discharge_rate × battery_capacity_wh`` W
      (battery side).
    * ``bat_factor = 0`` — battery is idle.

    The factor is thus a *fraction of the configured maximum rate*, not
    of the raw 1C rate, so ``bat_factor = ±1`` always means the maximum
    configured charge or discharge power.

    SOLAR inverters have no battery; ``bat_factor`` is always repaired to
    ``0.0`` and contributes nothing to the physics.

``pv_util[t] ∈ [0, 1]``
    Fraction of the available clipped DC PV power that is utilised:

    * ``pv_util = 0`` — all PV is curtailed.
    * ``pv_util = 1`` — all available PV power is used.

    Of the utilised DC power, **battery charging takes priority**: as much
    as needed to meet the battery charge target is routed to the battery
    first (subject to ``pv_to_battery_efficiency``); any surplus is
    converted to AC via ``pv_to_ac_efficiency``.

    BATTERY inverters have no PV; ``pv_util`` is always repaired to
    ``0.0`` and contributes nothing to the physics.

Repair and Lamarckian write-back
---------------------------------
After decoding, both genes are repaired in place:

* ``bat_factor`` magnitude is clipped to the configured min/max rate for
  the active direction; values below the minimum deadband are zeroed;
  SoC headroom/floor caps further reduce the magnitude; discrete rate
  snapping is applied when ``battery_charge_rates`` is set.
* ``pv_util`` is clipped to ``[0, 1]`` (bounds are already enforced by
  the genome, but clipping guards against floating-point overshoot).

The repaired values are written back to ``genome_batch`` so that the GA
sees feasible individuals on the next generation (Lamarckian repair).

Power and energy sign convention
---------------------------------
Identical to ``hybridinverter.py``:

* Positive AC power/energy → consuming from the bus (load, charging).
* Negative AC power/energy → injecting into the bus (PV, discharging).

AC power calculation
---------------------
Given repaired ``bat_factor`` and ``pv_util`` for one time step:

1. **Battery charge demand** (AC side, gross, before PV offset)::

       charge_power_bat_w  = bat_factor × bat_max_charge_rate × capacity   [bat side, W]
       charge_power_ac_gross_w = charge_power_bat_w / ac_to_bat_eff

2. **PV contribution**::

       pv_dc_avail_w   = clip(pv_forecast[t], pv_min, pv_max)
       pv_dc_used_w    = pv_util × pv_dc_avail_w

       # Priority: battery first
       pv_for_bat_dc_w = pv_dc_used_w × pv_to_bat_eff          # DC→bat side
       pv_to_bat_w     = min(pv_for_bat_dc_w, charge_power_bat_w)   # capped by demand
       pv_surplus_dc_w = pv_dc_used_w − pv_to_bat_w / pv_to_bat_eff # remainder DC
       pv_surplus_ac_w = pv_surplus_dc_w × pv_to_ac_eff

3. **Net AC power**::

       # Charging scenario:
       ac_offset_from_pv_w = pv_to_bat_w / ac_to_bat_eff        # equiv. AC saved
       net_charge_ac_w     = charge_power_ac_gross_w − ac_offset_from_pv_w − pv_surplus_ac_w

       # Discharging scenario:
       discharge_bat_ac_w = |bat_factor| × bat_max_dis_rate × capacity × bat_to_ac_eff
       net_discharge_ac_w = discharge_bat_ac_w + pv_surplus_ac_w   # both inject

       # Idle battery, PV present:
       net_idle_ac_w = −pv_surplus_ac_w   # injection only

   On-state power consumption is added for non-zero AC power; off-state
   consumption is used when the device is fully idle (bat=0 and pv=0).

SoC advance
-----------
::

    stored_wh  = (pv_to_bat_w + ac_used_for_bat_w × ac_to_bat_eff) × step_h
    drawn_wh   = |bat_factor| × bat_max_dis_rate × capacity × step_h   [bat side]

where ``ac_used_for_bat_w = max(0, charge_power_bat_w − pv_to_bat_w)``.

PV forecast and initial SoC handling are identical to ``hybridinverter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from akkudoktoreos.core.emplan import EnergyManagementInstruction, OMBCInstruction
from akkudoktoreos.devices.devicesabc import (
    DeviceParam,
    EnergyDevice,
    EnergyPort,
    InstructionContext,
)
from akkudoktoreos.optimization.genetic2.genome import GenomeSlice
from akkudoktoreos.simulation.genetic2.arbitrator import (
    DeviceGrant,
    DeviceRequest,
    PortRequest,
)
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import DateTime

# ============================================================
# Constants
# ============================================================

# Net grid power threshold below which a step is considered a self-consumption
# equilibrium.  When the arbitrated grid exchange is within this band around
# zero, ``extract_instructions`` emits ``SELF_CONSUMPTION`` instead of an
# explicit ``CHARGE``/``DISCHARGE`` setpoint, delegating real-time balancing
# to the inverter firmware.  50 W is a practical deadband for residential
# inverters; adjust per hardware specification if needed.
_SELF_CONSUMPTION_THRESHOLD_W: float = 50.0

# ============================================================
# Enumerations
# ============================================================


class InverterType(IntEnum):
    """Physical topology of the inverter."""

    BATTERY = 0  # Battery only, no PV
    SOLAR = 1  # PV only, no battery
    HYBRID = 2  # PV + battery behind one AC port


# ============================================================
# Immutable parameter dataclass
# ============================================================


@dataclass(frozen=True, slots=True)
class HybridInverterParam(DeviceParam):
    """Immutable parameters for a continuous two-gene hybrid inverter device."""

    # device_id and ports are inherited from DeviceParam.
    inverter_type: InverterType

    # Auxiliary power consumption
    off_state_power_consumption_w: float
    on_state_power_consumption_w: float

    # PV parameters (SOLAR, HYBRID)
    pv_to_ac_efficiency: float
    pv_to_battery_efficiency: float
    pv_max_power_w: float
    pv_min_power_w: float
    pv_power_w_key: str | None

    # AC <-> Battery conversion (BATTERY, HYBRID)
    ac_to_battery_efficiency: float
    battery_to_ac_efficiency: float

    # Battery sizing and discrete rate control
    battery_capacity_wh: float
    battery_charge_rates: tuple[float, ...] | None  # discrete fractions of max_charge_rate

    # Continuous rate bounds (fractions of the 1C rate, i.e. capacity in Wh == 1C in W)
    battery_min_charge_rate: float  # minimum non-zero charge fraction
    battery_max_charge_rate: float  # maximum charge fraction (bat_factor=+1 maps here)
    battery_min_discharge_rate: float  # minimum non-zero discharge fraction
    battery_max_discharge_rate: float  # maximum discharge fraction (bat_factor=-1 maps here)

    # SoC constraints
    battery_min_soc_factor: float
    battery_max_soc_factor: float
    battery_initial_soc_factor_key: str

    # Battery wear cost — penalises unnecessary cycling so the GA does not
    # charge from the grid when there is no price-spread benefit.
    # Typical value: 0.05 EUR/kWh cycled (half-cycle cost of a residential
    # Li-ion battery at ~4000 cycles over 10 kWh capacity).
    # Set to 0.0 to disable.
    levelized_cost_of_storage_amt_kwh: float = 0.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:  # noqa: C901
        if self.off_state_power_consumption_w < 0:
            raise ValueError("off_state_power_consumption_w must be >= 0")
        if self.on_state_power_consumption_w < 0:
            raise ValueError("on_state_power_consumption_w must be >= 0")

        if self.inverter_type in (InverterType.SOLAR, InverterType.HYBRID):
            if not (0 < self.pv_to_ac_efficiency <= 1):
                raise ValueError("pv_to_ac_efficiency must be in (0, 1]")
            if not (0 < self.pv_to_battery_efficiency <= 1):
                raise ValueError("pv_to_battery_efficiency must be in (0, 1]")
            if self.pv_max_power_w <= 0:
                raise ValueError("pv_max_power_w must be > 0")
            if not (0 <= self.pv_min_power_w <= self.pv_max_power_w):
                raise ValueError("pv_min_power_w must be in [0, pv_max_power_w]")

        if self.inverter_type in (InverterType.BATTERY, InverterType.HYBRID):
            if not (0 < self.ac_to_battery_efficiency <= 1):
                raise ValueError("ac_to_battery_efficiency must be in (0, 1]")
            if not (0 < self.battery_to_ac_efficiency <= 1):
                raise ValueError("battery_to_ac_efficiency must be in (0, 1]")
            if self.battery_capacity_wh <= 0:
                raise ValueError("battery_capacity_wh must be > 0")
            if not (0 <= self.battery_min_charge_rate <= self.battery_max_charge_rate <= 1):
                raise ValueError(
                    "battery charge rates must satisfy 0 <= min_charge_rate <= max_charge_rate <= 1"
                )
            if not (0 <= self.battery_min_discharge_rate <= self.battery_max_discharge_rate <= 1):
                raise ValueError(
                    "battery discharge rates must satisfy "
                    "0 <= min_discharge_rate <= max_discharge_rate <= 1"
                )
            if not (0 <= self.battery_min_soc_factor < self.battery_max_soc_factor <= 1):
                raise ValueError(
                    "SoC factors must satisfy 0 <= min_soc_factor < max_soc_factor <= 1"
                )
            if self.battery_charge_rates is not None:
                if len(self.battery_charge_rates) == 0:
                    raise ValueError("battery_charge_rates must not be empty if set")
                if any(r <= 0 or r > 1 for r in self.battery_charge_rates):
                    raise ValueError(
                        "battery_charge_rates values must be in (0, 1] — "
                        "use bat_factor=0 to stop charging, not rate 0.0"
                    )

    @property
    def battery_min_soc_wh(self) -> float:
        return self.battery_min_soc_factor * self.battery_capacity_wh

    @property
    def battery_max_soc_wh(self) -> float:
        return self.battery_max_soc_factor * self.battery_capacity_wh


# ============================================================
# Mutable batch state
# ============================================================


@dataclass
class HybridInverterBatchState:
    """Mutable batch state for ``HybridInverterDevice``.

    All arrays have shape ``(population_size, horizon)`` unless noted.

    Attributes:
        bat_factors:   Repaired battery power factors in ``[−1, +1]``.
                       Positive = charging, negative = discharging.
        pv_util_factors: Repaired PV utilisation factors in ``[0, 1]``.
        soc_wh:        Battery state of charge after each step [Wh].
        ac_power_w:    Net AC port power after each step [W].
                       Positive = consuming from bus, negative = injecting.
        population_size: Number of individuals.
        horizon:       Number of time steps.
        step_times:    Tuple of ``DateTime`` objects, one per step.
    """

    bat_factors: np.ndarray  # (population_size, horizon)  float64
    pv_util_factors: np.ndarray  # (population_size, horizon)  float64
    soc_wh: np.ndarray  # (population_size, horizon)  float64
    ac_power_w: np.ndarray  # (population_size, horizon)  float64
    population_size: int
    horizon: int
    step_times: tuple[DateTime, ...]


# ============================================================
# Device implementation
# ============================================================


class HybridInverterDevice(EnergyDevice):
    """Vectorised hybrid inverter with continuous two-gene genome encoding.

    Genome layout
    -------------
    Genes are interleaved per step::

        genome = [bat_factor₀, pv_util₀, bat_factor₁, pv_util₁, …]

    ``bat_factor ∈ [−1, +1]``: signed battery power fraction.
    ``pv_util    ∈ [0,  1]``: PV utilisation fraction.

    For BATTERY inverters, ``pv_util`` genes are always zeroed by repair.
    For SOLAR inverters,   ``bat_factor`` genes are always zeroed by repair.
    Both genes are always present so the genome layout is uniform across
    all inverter types.
    """

    def __init__(
        self,
        param: HybridInverterParam,
        device_index: int,
        port_index: int,
    ) -> None:
        super().__init__()
        self.param = param
        self.device_id: str = param.device_id
        self._device_index = device_index
        self._port_index = port_index

        # Populated by setup_run
        self._step_times: tuple[DateTime, ...] | None = None
        self._num_steps: int | None = None
        self._step_interval_sec: float | None = None
        self._pv_power_w: np.ndarray | None = None  # shape (horizon,), SOLAR/HYBRID
        self._battery_initial_soc_wh: float | None = None  # BATTERY/HYBRID

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return self.param.ports

    @property
    def objective_names(self) -> list[str]:
        names = []
        if self.param.levelized_cost_of_storage_amt_kwh > 0.0:
            names.append("energy_cost_amt")
        return names

    # ------------------------------------------------------------------
    # Structure Phase
    # ------------------------------------------------------------------

    def setup_run(self, context: SimulationContext) -> None:
        """Resolve run-scoped forecasts and initial conditions from context."""
        horizon = context.horizon
        self._step_times = context.step_times
        self._num_steps = horizon
        self._step_interval_sec = context.step_interval.total_seconds()

        if self.param.inverter_type in (InverterType.SOLAR, InverterType.HYBRID):
            if self.param.pv_power_w_key is None:
                raise ValueError(
                    f"{self.device_id}: pv_power_w_key must be set for SOLAR or HYBRID inverter."
                )
            pv = context.resolve_prediction(self.param.pv_power_w_key)
            if pv.shape != (horizon,):
                raise ValueError(
                    f"{self.device_id}: PV forecast must have shape ({horizon},), got {pv.shape}."
                )
            self._pv_power_w = pv

        if self.param.inverter_type in (InverterType.BATTERY, InverterType.HYBRID):
            soc_factor = None
            if self.param.battery_initial_soc_factor_key:  # skip if empty string
                soc_factor = context.resolve_measurement(self.param.battery_initial_soc_factor_key)
            if soc_factor is None:
                soc_factor = self.param.battery_min_soc_factor  # use min_soc
            if not (0.0 <= soc_factor <= 1.0):
                raise ValueError(
                    f"{self.device_id}: initial SoC factor {soc_factor} outside allowed bounds "
                    "[0.0, 1.0]."
                )
            self._battery_initial_soc_wh = soc_factor * self.param.battery_capacity_wh

    def genome_requirements(self) -> GenomeSlice:
        """Return genome slice descriptor for the two-gene-per-step encoding.

        Layout: ``[bat_factor₀, pv_util₀, bat_factor₁, pv_util₁, …]``

        * ``bat_factor`` bounds: ``[−1.0, +1.0]``
        * ``pv_util``    bounds: ``[ 0.0,  1.0]``
        """
        if self._num_steps is None:
            raise RuntimeError("Call setup_run() before genome_requirements().")
        n = self._num_steps
        lower = np.empty(2 * n)
        upper = np.empty(2 * n)
        lower[0::2] = -1.0  # bat_factor lower bound
        upper[0::2] = +1.0  # bat_factor upper bound
        lower[1::2] = 0.0  # pv_util lower bound
        upper[1::2] = 1.0  # pv_util upper bound
        return GenomeSlice(start=0, size=2 * n, lower_bound=lower, upper_bound=upper)

    # ------------------------------------------------------------------
    # Batch Lifecycle
    # ------------------------------------------------------------------

    def create_batch_state(self, population_size: int, horizon: int) -> HybridInverterBatchState:
        """Allocate batch state; soc_wh pre-filled with resolved initial SoC."""
        if self._step_times is None:
            raise RuntimeError("Call setup_run() before create_batch_state().")
        initial_soc = (
            self._battery_initial_soc_wh if self._battery_initial_soc_wh is not None else 0.0
        )
        return HybridInverterBatchState(
            bat_factors=np.zeros((population_size, horizon), dtype=np.float64),
            pv_util_factors=np.zeros((population_size, horizon), dtype=np.float64),
            soc_wh=np.full((population_size, horizon), initial_soc, dtype=np.float64),
            ac_power_w=np.zeros((population_size, horizon), dtype=np.float64),
            population_size=population_size,
            horizon=horizon,
            step_times=self._step_times,
        )

    def apply_genome_batch(
        self,
        state: HybridInverterBatchState,
        genome_batch: np.ndarray,
    ) -> np.ndarray:
        """Decode, repair, and simulate the genome for the full population.

        Args:
            state: Pre-allocated batch state (mutated in place).
            genome_batch: Float array of shape ``(population_size, 2 * horizon)``
                with interleaved ``[bat_factor, pv_util]`` genes per step.
                Mutated in place with repaired values (Lamarckian repair).

        Returns:
            The mutated ``genome_batch``.
        """
        if self._step_interval_sec is None:
            raise RuntimeError("Call setup_run() before apply_genome_batch().")
        pv_power_w = self._pv_power_w
        pv_min_power_w = self.param.pv_min_power_w
        pv_max_power_w = self.param.pv_max_power_w

        soc = np.full(
            state.population_size,
            self._battery_initial_soc_wh if self._battery_initial_soc_wh is not None else 0.0,
        )

        for t in range(state.horizon):
            raw_bat = genome_batch[:, 2 * t]  # shape (pop,)
            raw_pv = genome_batch[:, 2 * t + 1]  # shape (pop,)

            if pv_power_w is not None:
                pv_dc_avail_w = float(np.clip(pv_power_w[t], pv_min_power_w, pv_max_power_w))
            else:
                pv_dc_avail_w = 0.0

            bat_t, pv_t = self._repair_genes(raw_bat, raw_pv, soc, pv_dc_avail_w)

            state.bat_factors[:, t] = bat_t
            state.pv_util_factors[:, t] = pv_t

            state.ac_power_w[:, t] = self._compute_ac_power(bat_t, pv_t, pv_dc_avail_w, soc)

            soc = self._advance_soc(soc, bat_t, pv_t, pv_dc_avail_w)
            state.soc_wh[:, t] = soc

            # Lamarckian write-back
            genome_batch[:, 2 * t] = bat_t
            genome_batch[:, 2 * t + 1] = pv_t

        return genome_batch

    # ------------------------------------------------------------------
    # Arbitration
    # ------------------------------------------------------------------

    def build_device_request(self, state: HybridInverterBatchState) -> DeviceRequest:
        if self._step_interval_sec is None:
            raise RuntimeError("Call setup_run() before build_device_request().")
        step_h = self._step_interval_sec / 3600.0
        return DeviceRequest(
            device_index=self._device_index,
            port_requests=(
                PortRequest(
                    port_index=self._port_index,
                    energy_wh=state.ac_power_w * step_h,
                    min_energy_wh=self._compute_min_energy(state, step_h),
                ),
            ),
        )

    def apply_device_grant(self, state: HybridInverterBatchState, grant: DeviceGrant) -> None:
        if self._step_interval_sec is None:
            raise RuntimeError("Call setup_run() before apply_device_grant().")
        step_h = self._step_interval_sec / 3600.0
        state.ac_power_w[:] = grant.port_grants[0].granted_wh / step_h

    def compute_cost(self, state: HybridInverterBatchState) -> np.ndarray:
        """Compute levelized cost of storage (LCOS) for battery cycling.

        Penalises every Wh cycled through the battery (charge + discharge)
        by ``levelized_cost_of_storage_amt_kwh``.  This gives the GA a
        signal to avoid unnecessary cycling when there is no price-spread
        benefit — preventing the pathological case of charging to maximum
        from the grid and staying there.

        Returns shape ``(population_size, 1)`` when LCOS > 0, else
        ``(population_size, 0)``.
        """
        p = self.param
        if p.levelized_cost_of_storage_amt_kwh == 0.0:
            return np.zeros((state.population_size, 0))

        if self._step_interval_sec is None:
            raise RuntimeError("Call setup_run() before compute_cost().")

        step_h = self._step_interval_sec / 3600.0
        # bat_factors > 0 → charging; < 0 → discharging.
        # Total cycled energy per individual = sum of |bat_factor| × max_rate × capacity × step_h
        # (bat_factors are already repaired so they reflect what actually happened).
        bat_f = state.bat_factors  # (population_size, horizon)
        cycled_wh = (
            np.abs(bat_f) * p.battery_max_charge_rate * p.battery_capacity_wh * step_h
        ).sum(axis=1)  # (population_size,)

        lcos = cycled_wh / 1000.0 * p.levelized_cost_of_storage_amt_kwh  # (population_size,)
        return lcos[:, np.newaxis]  # (population_size, 1)

    # ------------------------------------------------------------------
    # S2 instruction extraction
    # ------------------------------------------------------------------

    def extract_instructions(
        self,
        state: HybridInverterBatchState,
        individual_index: int,
        instruction_context: InstructionContext | None = None,
    ) -> list[EnergyManagementInstruction]:
        """Emit two ``OMBCInstruction`` objects per time step.

        The first carries the battery command:

        * ``operation_mode_id = "SELF_CONSUMPTION"`` when the arbitrated
            grid exchange at this step is within ``_SELF_CONSUMPTION_THRESHOLD_W``
            of zero *and* the battery is active (discharging or charging).  In
            this mode the inverter firmware autonomously balances local supply
            and demand without an explicit power setpoint.  Requires an
            ``InstructionContext`` with a non-``None`` ``grid_granted_wh``
            vector; falls back to explicit ``CHARGE``/``DISCHARGE``/``IDLE``
            if context is absent.  ``operation_mode_factor`` carries the
            optimised ``abs(bat_factor)`` magnitude so that the GA's decision
            is visible in the solution DataFrame even though the firmware
            ignores the explicit setpoint.
        * ``operation_mode_id = "CHARGE"``   when ``bat_factor > 0``
        * ``operation_mode_id = "DISCHARGE"`` when ``bat_factor < 0``
        * ``operation_mode_id = "IDLE"``      when ``bat_factor = 0``
        * ``operation_mode_factor = abs(bat_factor)``  always non-negative;
          the mode name carries the direction.

        The second carries the PV command:

        * ``operation_mode_id = "PV_UTILISE"``
        * ``operation_mode_factor = pv_util_factor ∈ [0, 1]``
        """
        # Derive the per-step power threshold for SELF_CONSUMPTION detection.
        # A net grid exchange below this value means the optimizer already
        # found a near-zero-export equilibrium — the inverter firmware can
        # maintain that without a precise setpoint.
        grid_wh: np.ndarray | None = None
        threshold_wh: float = 0.0
        if instruction_context is not None and instruction_context.grid_granted_wh is not None:
            grid_wh = instruction_context.grid_granted_wh
            step_h = instruction_context.step_interval_sec / 3600.0
            threshold_wh = _SELF_CONSUMPTION_THRESHOLD_W * step_h

        instructions: list[EnergyManagementInstruction] = []
        for t, (bat_f, pv_f, dt) in enumerate(
            zip(
                state.bat_factors[individual_index],
                state.pv_util_factors[individual_index],
                state.step_times,
            )
        ):
            # --- Determine battery operation mode ---
            battery_active = bat_f != 0.0
            if grid_wh is not None and battery_active and abs(grid_wh[t]) <= threshold_wh:
                # The arbitrated AC bus is at (or very near) zero net exchange.
                # The optimizer found a self-consumption equilibrium; delegate
                # real-time balancing to the inverter firmware.
                # The factor carries the optimised bat_factor magnitude so that
                # downstream consumers (logging, dashboards) can see what the GA
                # decided, even though the firmware ignores the explicit setpoint.
                bat_mode = "SELF_CONSUMPTION"
                bat_factor_out = 1.0 # abs(float(bat_f))
            elif bat_f > 0.0:
                bat_mode = "CHARGE"
                bat_factor_out = abs(float(bat_f))
            elif bat_f < 0.0:
                bat_mode = "DISCHARGE"
                bat_factor_out = abs(float(bat_f))
            else:
                bat_mode = "IDLE"
                bat_factor_out = 0.0

            instructions.append(
                OMBCInstruction(
                    resource_id=self.device_id,
                    execution_time=dt,
                    operation_mode_id=bat_mode,
                    operation_mode_factor=bat_factor_out,
                )
            )
            instructions.append(
                OMBCInstruction(
                    resource_id=self.device_id,
                    execution_time=dt,
                    operation_mode_id="PV_UTILISE",
                    operation_mode_factor=float(pv_f),
                )
            )
        return instructions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _repair_genes(
        self,
        raw_bat: np.ndarray,
        raw_pv: np.ndarray,
        soc_wh: np.ndarray,
        pv_dc_avail_w: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Repair bat_factor and pv_util genes for one time step.

        Args:
            raw_bat:       Raw battery factor genes, shape ``(pop,)``, in ``[−1, +1]``.
            raw_pv:        Raw PV utilisation genes, shape ``(pop,)``, in ``[0, 1]``.
            soc_wh:        Current SoC per individual, shape ``(pop,)`` [Wh].
            pv_dc_avail_w: Clipped DC PV power available this step [W].

        Returns:
            ``(bat_factor, pv_util)``: repaired arrays, both shape ``(pop,)``.

        Repair steps for ``bat_factor``:

        1. SOLAR type: zero all (no battery).
        2. Positive values (charge):
           a. Scale by ``battery_max_charge_rate`` to get an absolute rate fraction.
           b. Clip magnitude to ``[battery_min_charge_rate, battery_max_charge_rate]``.
           c. Snap to nearest discrete rate if ``battery_charge_rates`` is set.
           d. Cap by SoC headroom; zero if below minimum deadband.
        3. Negative values (discharge):
           a. Scale by ``battery_max_discharge_rate`` symmetrically.
           b. Clip magnitude to ``[battery_min_discharge_rate, battery_max_discharge_rate]``.
           c. Cap by available SoC; zero if below minimum deadband.
        4. After repair, rescale back to the ``[−1, +1]`` genome domain for
           write-back (so the gene reflects what was actually done).

        Repair steps for ``pv_util``:

        1. BATTERY type: zero all (no PV).
        2. Clip to ``[0, 1]``.
        """
        if self._step_interval_sec is None:
            raise RuntimeError("Step interval is None.")
        p = self.param
        step_h = self._step_interval_sec / 3600.0

        # --- PV utilisation -------------------------------------------------
        if p.inverter_type == InverterType.BATTERY or pv_dc_avail_w == 0.0:
            pv = np.zeros_like(raw_pv)
        else:
            pv = np.clip(raw_pv, 0.0, 1.0)

        # --- Battery factor -------------------------------------------------
        if p.inverter_type == InverterType.SOLAR:
            bat = np.zeros_like(raw_bat)
            return bat, pv

        bat = raw_bat.copy()

        # ---- Charging (bat > 0) ----
        charge_mask = bat > 0.0
        if charge_mask.any():
            # Map gene [0,1] → absolute rate [0, max_charge_rate]
            abs_rate = bat[charge_mask] * p.battery_max_charge_rate
            abs_rate = np.clip(abs_rate, p.battery_min_charge_rate, p.battery_max_charge_rate)

            if p.battery_charge_rates is not None:
                rates = np.array(p.battery_charge_rates)
                # Snap to nearest discrete rate
                idx = np.argmin(np.abs(abs_rate[:, np.newaxis] - rates[np.newaxis, :]), axis=1)
                abs_rate = rates[idx]

            # SoC headroom cap: how much can we still store this step?
            # charge_power_bat = abs_rate × capacity  [bat-side W]
            # stored_wh = charge_power_bat × step_h × ac_to_bat_eff  (worst case, no PV assist)
            # max abs_rate from headroom = headroom / (capacity × step_h × ac_eff)
            headroom_wh = p.battery_max_soc_wh - soc_wh[charge_mask]
            max_abs_rate = np.clip(
                headroom_wh / (p.battery_capacity_wh * step_h * p.ac_to_battery_efficiency),
                0.0,
                p.battery_max_charge_rate,
            )
            abs_rate = np.minimum(abs_rate, max_abs_rate)
            abs_rate[abs_rate < p.battery_min_charge_rate] = 0.0

            # Convert back to genome domain: gene = abs_rate / max_charge_rate
            bat[charge_mask] = (
                abs_rate / p.battery_max_charge_rate if p.battery_max_charge_rate > 0 else 0.0
            )

        # ---- Discharging (bat < 0) ----
        discharge_mask = bat < 0.0
        if discharge_mask.any():
            # Map gene [-1, 0) → absolute rate [0, max_discharge_rate]
            abs_rate = (-bat[discharge_mask]) * p.battery_max_discharge_rate
            abs_rate = np.clip(abs_rate, p.battery_min_discharge_rate, p.battery_max_discharge_rate)

            # SoC floor cap
            available_wh = soc_wh[discharge_mask] - p.battery_min_soc_wh
            max_abs_rate = np.clip(
                available_wh / (p.battery_capacity_wh * step_h),
                0.0,
                p.battery_max_discharge_rate,
            )
            abs_rate = np.minimum(abs_rate, max_abs_rate)
            abs_rate[abs_rate < p.battery_min_discharge_rate] = 0.0

            bat[discharge_mask] = (
                -(abs_rate / p.battery_max_discharge_rate)
                if p.battery_max_discharge_rate > 0
                else 0.0
            )

        return bat, pv

    def _bat_power_w(self, bat_factors: np.ndarray) -> np.ndarray:
        """Convert repaired bat_factor array to battery-side power [W].

        Positive result = charge power; negative result = discharge power
        (battery side, before any efficiency correction).

        Args:
            bat_factors: Shape ``(pop,)`` or ``(pop, horizon)``.

        Returns:
            Array of the same shape.  Units: W (battery side).
        """
        p = self.param
        result = np.where(
            bat_factors > 0.0,
            bat_factors * p.battery_max_charge_rate * p.battery_capacity_wh,
            np.where(
                bat_factors < 0.0,
                bat_factors * p.battery_max_discharge_rate * p.battery_capacity_wh,
                0.0,
            ),
        )
        return result

    def _compute_ac_power(
        self,
        bat_factors: np.ndarray,
        pv_util: np.ndarray,
        pv_dc_avail_w: float,
        soc_wh: np.ndarray,
    ) -> np.ndarray:
        """Compute net AC port power for one time step across all individuals.

        Args:
            bat_factors:   Repaired battery factors, shape ``(pop,)``.
            pv_util:       Repaired PV utilisation factors, shape ``(pop,)``.
            pv_dc_avail_w: Clipped available DC PV power [W] (scalar).
            soc_wh:        Current SoC, shape ``(pop,)`` [Wh]. (Unused here;
                           included for API symmetry with future extensions.)

        Returns:
            Net AC power, shape ``(pop,)`` [W].  Positive = consuming from
            bus; negative = injecting into bus.

        Physics summary (per individual)::

            pv_dc_used = pv_util × pv_dc_avail_w

            # --- Charging ---
            charge_bat_w = bat_factor × max_charge_rate × capacity     [bat-side]
            pv_for_bat   = min(pv_dc_used × pv_to_bat_eff, charge_bat_w)
            pv_surplus_dc = pv_dc_used − pv_for_bat / pv_to_bat_eff
            pv_surplus_ac = pv_surplus_dc × pv_to_ac_eff
            ac_for_bat    = max(0, charge_bat_w − pv_for_bat) / ac_to_bat_eff
            net_ac        = ac_for_bat − pv_surplus_ac + on_state

            # --- Discharging ---
            dis_bat_ac   = |bat_factor| × max_dis_rate × capacity × bat_to_ac_eff
            pv_surplus_ac = pv_dc_used × pv_to_ac_eff   (all PV → AC when discharging)
            net_ac       = −(dis_bat_ac + pv_surplus_ac) + on_state

            # --- Idle battery ---
            net_ac       = −pv_surplus_ac + on_state   (or off_state if pv=0 too)
        """
        p = self.param
        pop = len(bat_factors)
        ac = np.zeros(pop)

        pv_dc_used = pv_util * pv_dc_avail_w  # shape (pop,)

        charge_mask = bat_factors > 0.0
        discharge_mask = bat_factors < 0.0
        idle_mask = ~charge_mask & ~discharge_mask

        # ----------------------------------------------------------------
        # Charging individuals
        # ----------------------------------------------------------------
        if charge_mask.any():
            bf = bat_factors[charge_mask]
            pv_used = pv_dc_used[charge_mask]

            charge_bat_w = bf * p.battery_max_charge_rate * p.battery_capacity_wh

            if p.inverter_type in (InverterType.SOLAR, InverterType.HYBRID):
                pv_for_bat_avail = pv_used * p.pv_to_battery_efficiency  # bat-side W
                pv_for_bat = np.minimum(pv_for_bat_avail, charge_bat_w)
                # Remaining DC PV after battery takes its share
                pv_surplus_dc = pv_used - pv_for_bat / p.pv_to_battery_efficiency
                pv_surplus_ac = pv_surplus_dc * p.pv_to_ac_efficiency
                # AC grid must cover what PV doesn't supply to the battery
                ac_for_bat_w = (
                    np.maximum(0.0, charge_bat_w - pv_for_bat) / p.ac_to_battery_efficiency
                )
                ac[charge_mask] = ac_for_bat_w - pv_surplus_ac + p.on_state_power_consumption_w
            else:
                # BATTERY: no PV
                ac_for_bat_w = charge_bat_w / p.ac_to_battery_efficiency
                ac[charge_mask] = ac_for_bat_w + p.on_state_power_consumption_w

        # ----------------------------------------------------------------
        # Discharging individuals
        # ----------------------------------------------------------------
        if discharge_mask.any():
            bf = bat_factors[discharge_mask]
            pv_used = pv_dc_used[discharge_mask]

            dis_bat_ac_w = (
                (-bf)
                * p.battery_max_discharge_rate
                * p.battery_capacity_wh
                * p.battery_to_ac_efficiency
            )

            if p.inverter_type in (InverterType.SOLAR, InverterType.HYBRID):
                # When discharging, battery and PV both inject into AC
                pv_ac_w = pv_used * p.pv_to_ac_efficiency
                ac[discharge_mask] = -(dis_bat_ac_w + pv_ac_w) + p.on_state_power_consumption_w
            else:
                ac[discharge_mask] = -dis_bat_ac_w + p.on_state_power_consumption_w

        # ----------------------------------------------------------------
        # Idle battery individuals
        # ----------------------------------------------------------------
        if idle_mask.any():
            pv_used = pv_dc_used[idle_mask]
            if p.inverter_type in (InverterType.SOLAR, InverterType.HYBRID):
                pv_ac_w = pv_used * p.pv_to_ac_efficiency
                active_pv = pv_ac_w > 0.0
                ac[idle_mask] = np.where(
                    active_pv,
                    -pv_ac_w + p.on_state_power_consumption_w,
                    p.off_state_power_consumption_w,
                )
            else:
                # BATTERY, idle: standby draw
                ac[idle_mask] = p.off_state_power_consumption_w

        return ac

    def _advance_soc(
        self,
        soc_wh: np.ndarray,
        bat_factors: np.ndarray,
        pv_util: np.ndarray,
        pv_dc_avail_w: float,
    ) -> np.ndarray:
        """Advance SoC by one time step.

        Args:
            soc_wh:        Current SoC per individual, shape ``(pop,)`` [Wh].
            bat_factors:   Repaired battery factors, shape ``(pop,)``.
            pv_util:       Repaired PV utilisation factors, shape ``(pop,)``.
            pv_dc_avail_w: Clipped available DC PV power this step [W].

        Returns:
            New SoC array, shape ``(pop,)`` [Wh], clamped to ``[min_soc_wh, max_soc_wh]``.

        For SOLAR type, ``battery_min_soc_wh == battery_max_soc_wh == 0`` so
        the clamp is a no-op and SoC stays at 0.
        """
        if self._step_interval_sec is None:
            raise RuntimeError("Step interval is None.")
        p = self.param
        step_h = self._step_interval_sec / 3600.0
        new_soc = soc_wh.copy()

        pv_dc_used = pv_util * pv_dc_avail_w  # shape (pop,)

        # ---- Charging ----
        charge_mask = bat_factors > 0.0
        if charge_mask.any():
            bf = bat_factors[charge_mask]
            pv_used = pv_dc_used[charge_mask]
            charge_bat_w = bf * p.battery_max_charge_rate * p.battery_capacity_wh

            if p.inverter_type in (InverterType.SOLAR, InverterType.HYBRID):
                pv_for_bat_avail = pv_used * p.pv_to_battery_efficiency
                pv_for_bat = np.minimum(pv_for_bat_avail, charge_bat_w)
                ac_used_w = np.maximum(0.0, charge_bat_w - pv_for_bat)
                # PV contribution enters battery at full DC efficiency;
                # AC contribution is subject to ac_to_battery_efficiency.
                stored_wh = (pv_for_bat + ac_used_w * p.ac_to_battery_efficiency) * step_h
            else:
                stored_wh = charge_bat_w * p.ac_to_battery_efficiency * step_h

            new_soc[charge_mask] += stored_wh

        # ---- Discharging ----
        discharge_mask = bat_factors < 0.0
        if discharge_mask.any():
            bf = bat_factors[discharge_mask]
            dis_bat_w = (-bf) * p.battery_max_discharge_rate * p.battery_capacity_wh
            new_soc[discharge_mask] -= dis_bat_w * step_h

        return np.clip(new_soc, p.battery_min_soc_wh, p.battery_max_soc_wh)

    def _compute_min_energy(
        self,
        state: HybridInverterBatchState,
        step_h: float,
    ) -> np.ndarray:
        """Compute minimum acceptable AC energy per individual per step.

        Used by the arbitrator to determine the tightest AC commitment.

        * For charging steps: minimum is the AC energy needed when charging
          at the minimum charge rate (``battery_min_charge_rate``) with no
          PV assist.
        * For discharging steps: minimum injection is the AC energy produced
          when discharging at the minimum discharge rate plus any PV that
          is already being utilised.
        * For idle/PV-only steps: minimum is ``0`` (no battery constraint).

        Note: The minimum is expressed as a *request lower bound*, not a
        hard guarantee — the arbitrator may still grant less.
        """
        p = self.param
        pv_power_w = self._pv_power_w  # None for BATTERY

        min_e = np.zeros_like(state.ac_power_w)

        # Minimum charge AC energy (no PV assist in the conservative case)
        min_charge_ac_w = (
            p.battery_min_charge_rate
            * p.battery_max_charge_rate
            * p.battery_capacity_wh
            / p.ac_to_battery_efficiency
        )

        # Minimum discharge AC injection
        min_dis_bat_ac_w = (
            p.battery_min_discharge_rate
            * p.battery_max_discharge_rate
            * p.battery_capacity_wh
            * p.battery_to_ac_efficiency
        )

        charge_mask = state.bat_factors > 0.0
        discharge_mask = state.bat_factors < 0.0

        if charge_mask.any():
            if p.inverter_type == InverterType.HYBRID and pv_power_w is not None:
                pv_clipped = np.clip(pv_power_w, p.pv_min_power_w, p.pv_max_power_w)
                # Use the per-step PV utilisation to determine how much PV assists
                # the minimum charge.  We look at each step that is charging.
                pop_idx, step_idx = np.where(charge_mask)
                pv_util_at_step = state.pv_util_factors[pop_idx, step_idx]
                pv_dc_used = pv_util_at_step * pv_clipped[step_idx]
                pv_for_bat_avail = pv_dc_used * p.pv_to_battery_efficiency
                min_charge_bat_w = (
                    p.battery_min_charge_rate * p.battery_max_charge_rate * p.battery_capacity_wh
                )
                pv_for_bat = np.minimum(pv_for_bat_avail, min_charge_bat_w)
                pv_surplus_dc = pv_dc_used - pv_for_bat / p.pv_to_battery_efficiency
                pv_surplus_ac = pv_surplus_dc * p.pv_to_ac_efficiency
                ac_for_bat = (
                    np.maximum(0.0, min_charge_bat_w - pv_for_bat) / p.ac_to_battery_efficiency
                )
                net_min_ac_w = np.maximum(0.0, ac_for_bat - pv_surplus_ac)
                min_e[pop_idx, step_idx] = net_min_ac_w * step_h
            else:
                min_e[charge_mask] = min_charge_ac_w * step_h

        if discharge_mask.any():
            if p.inverter_type == InverterType.HYBRID and pv_power_w is not None:
                pv_clipped = np.clip(pv_power_w, p.pv_min_power_w, p.pv_max_power_w)
                pop_idx, step_idx = np.where(discharge_mask)
                pv_util_at_step = state.pv_util_factors[pop_idx, step_idx]
                pv_dc_used = pv_util_at_step * pv_clipped[step_idx]
                pv_inject_ac_w = pv_dc_used * p.pv_to_ac_efficiency
                total_min_inject = min_dis_bat_ac_w + pv_inject_ac_w
                min_e[pop_idx, step_idx] = -total_min_inject * step_h
            else:
                min_e[discharge_mask] = -min_dis_bat_ac_w * step_h

        # Idle battery + PV generating: the inverter must inject the PV surplus.
        # Without this, min_energy_wh stays 0 and the arbitrator may grant 0,
        # causing apply_device_grant to overwrite ac_power_w with 0 and
        # silently drop all PV generation on idle-battery steps.
        idle_mask = state.bat_factors == 0.0
        if idle_mask.any() and p.inverter_type in (
            InverterType.SOLAR, InverterType.HYBRID
        ) and pv_power_w is not None:
            pv_clipped = np.clip(pv_power_w, p.pv_min_power_w, p.pv_max_power_w)
            pop_idx, step_idx = np.where(idle_mask)
            pv_util_at_step = state.pv_util_factors[pop_idx, step_idx]
            pv_ac_w = pv_util_at_step * pv_clipped[step_idx] * p.pv_to_ac_efficiency
            # Only set minimum where PV is actually generating (negative = injection).
            pv_injection = -pv_ac_w * step_h  # negative Wh = injection
            min_e[pop_idx, step_idx] = np.minimum(min_e[pop_idx, step_idx], pv_injection)

        return min_e
