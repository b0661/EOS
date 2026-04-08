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

Repair and Lamarckian write-back
---------------------------------
After decoding, both genes are repaired in place:

* ``bat_factor`` magnitude is clipped to the configured min/max rate for
  the active direction; values below the minimum deadband are zeroed;
  SoC headroom/floor caps further reduce the magnitude; discrete rate
  snapping is applied when ``battery_charge_rates`` is set.

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

import numpy as np

from dataclasses import dataclass
from enum import IntEnum
from loguru import logger

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
    battery_lcos_amt_kwh: float = 0.0

    # Shadow price rewarding kWh discharged from the battery.
    # Adds a negative cost (benefit) of this amount per kWh of AC energy
    # delivered by the battery, on top of the grid import reduction that
    # GridConnectionDevice already captures. Useful when the GA struggles
    # to discover discharge because the load-matching rate is small relative
    # to mutation noise.
    # Suggested starting value: import_price - export_price - lcos
    # e.g. 0.30 - 0.08 - 0.05 = 0.17 EUR/kWh.
    # Set to 0.0 to disable (default).
    battery_discharge_reward_amt_kwh: float = 0.0

    # Keys for resolving per-step price forecasts used by SoC value
    # accounting in compute_cost.  Should match the keys used by
    # GridConnectionDevice.
    import_price_amt_kwh_key: str | None = None
    export_price_amt_kwh_key: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:  # noqa: C901
        if self.off_state_power_consumption_w < 0:
            raise ValueError(f"{self.device_id}: off_state_power_consumption_w must be >= 0")
        if self.on_state_power_consumption_w < 0:
            raise ValueError(f"{self.device_id}: on_state_power_consumption_w must be >= 0")

        if self.inverter_type in (InverterType.SOLAR, InverterType.HYBRID):
            if not (0 < self.pv_to_ac_efficiency <= 1):
                raise ValueError(f"{self.device_id}: pv_to_ac_efficiency must be in (0, 1]")
            if not (0 < self.pv_to_battery_efficiency <= 1):
                raise ValueError(f"{self.device_id}: pv_to_battery_efficiency must be in (0, 1]")
            if self.pv_max_power_w <= 0:
                raise ValueError(f"{self.device_id}: pv_max_power_w must be > 0")
            if not (0 <= self.pv_min_power_w <= self.pv_max_power_w):
                raise ValueError(f"{self.device_id}: pv_min_power_w must be in [0, pv_max_power_w]")

        if self.inverter_type in (InverterType.BATTERY, InverterType.HYBRID):
            if not (0 < self.ac_to_battery_efficiency <= 1):
                raise ValueError(f"{self.device_id}: ac_to_battery_efficiency must be in (0, 1]")
            if not (0 < self.battery_to_ac_efficiency <= 1):
                raise ValueError(f"{self.device_id}: battery_to_ac_efficiency must be in (0, 1]")
            if self.battery_capacity_wh <= 0:
                raise ValueError(f"{self.device_id}: battery_capacity_wh must be > 0")
            if not (0 <= self.battery_min_charge_rate <= self.battery_max_charge_rate <= 1):
                raise ValueError(
                    f"{self.device_id}: battery charge rates must satisfy "
                    "0 <= min_charge_rate <= max_charge_rate <= 1"
                )
            if not (0 <= self.battery_min_discharge_rate <= self.battery_max_discharge_rate <= 1):
                raise ValueError(
                    f"{self.device_id}: battery discharge rates must satisfy "
                    "0 <= min_discharge_rate <= max_discharge_rate <= 1"
                )
            if not (0 <= self.battery_min_soc_factor < self.battery_max_soc_factor <= 1):
                raise ValueError(
                    f"{self.device_id}: SoC factors must satisfy "
                    "0 <= min_soc_factor < max_soc_factor <= 1"
                )
            if self.battery_charge_rates is not None:
                if len(self.battery_charge_rates) == 0:
                    raise ValueError(
                        f"{self.device_id}: battery_charge_rates must not be empty if set"
                    )
                if any(r <= 0 or r > 1 for r in self.battery_charge_rates):
                    raise ValueError(
                        f"{self.device_id}: battery_charge_rates values must be in (0, 1] — "
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

        genome = [bat_factor, …]

    ``bat_factor ∈ [−1, +1]``: signed battery power fraction.

    For SOLAR inverters,   ``bat_factor`` genes are always zeroed by repair.
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
        self._import_price_per_kwh: np.ndarray | None = None  # shape (horizon,)
        self._export_price_per_kwh: np.ndarray | None = (
            None  # shape (horizon,), for SoC value accounting
        )

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return self.param.ports

    @property
    def objective_names(self) -> list[str]:
        if self.param.inverter_type == InverterType.SOLAR:
            return []
        # BATTERY/HYBRID always contributes energy_cost_amt via the SoC value
        # accounting (initial vs terminal SoC valued at export price) and
        # optionally LCOS. SOLAR has no battery.
        names = ["energy_cost_amt"]
        if self.param.battery_discharge_reward_amt_kwh != 0.0:
            names.append("battery_discharge_reward_amt")
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
            if not np.isfinite(self.param.pv_max_power_w):
                logger.warning(
                    "{}: pv_max_power_w is not set (null). PV power will not be clipped. "
                    "Set devices.inverters.{}.pv_max_power_w to the rated DC peak power [W] "
                    "of your PV array for accurate simulation.",
                    self.device_id,
                    self.device_id,
                )
            logger_pv_stats = f"min={pv.min():.0f} mean={pv.mean():.0f} max={pv.max():.0f} W"
            logger.info("{}: PV forecast resolved: {}", self.device_id, logger_pv_stats)

        if self.param.inverter_type in (InverterType.BATTERY, InverterType.HYBRID):
            soc_factor = context.resolve_measurement(self.param.battery_initial_soc_factor_key)
            if soc_factor is None:
                # No measurement available — default to midpoint of usable range.
                # The initial SoC is assigned a monetary value (export price) in
                # compute_cost, so the GA cannot exploit it as free energy.
                # Set battery_initial_soc_factor_key to the actual SoC measurement
                # key for accurate real-world planning.
                soc_factor = (
                    self.param.battery_min_soc_factor + self.param.battery_max_soc_factor
                ) / 2.0
            if not (0.0 <= soc_factor <= 1.0):
                raise ValueError(
                    f"{self.device_id}: initial SoC factor {soc_factor} outside allowed bounds "
                    "[0.0, 1.0]."
                )
            self._battery_initial_soc_wh = soc_factor * self.param.battery_capacity_wh

            # Resolve import and export prices for SoC value accounting in compute_cost.
            if self.param.import_price_amt_kwh_key:
                try:
                    self._import_price_per_kwh = context.resolve_prediction(
                        self.param.import_price_amt_kwh_key
                    )
                except Exception:
                    self._import_price_per_kwh = None
            else:
                self._import_price_per_kwh = None

            # Export price for SoC value accounting
            if self.param.export_price_amt_kwh_key:
                try:
                    self._export_price_per_kwh = context.resolve_prediction(
                        self.param.export_price_amt_kwh_key
                    )
                except Exception:
                    self._export_price_per_kwh = None
            else:
                self._export_price_per_kwh = None

    def genome_requirements(self) -> GenomeSlice:
        """Return genome slice descriptor for the one-gene-per-step encoding.

        Only ``bat_factor`` is a genome gene. ``pv_util`` is always 1.0 —
        PV is always fully utilised. Curtailment (pv_util < 1) is almost
        never optimal in a home energy system: available PV either offsets
        import cost or earns export revenue, both of which are better than
        curtailing. Making it a gene only adds noise and halves the effective
        search resolution on the battery, which is the actual decision variable.

        Layout: ``[bat_factor₀, bat_factor₁, …, bat_factor_{n-1}]``

        * ``bat_factor`` bounds: ``[−1.0, +1.0]``
        """
        if self._num_steps is None:
            raise RuntimeError("Call setup_run() before genome_requirements().")
        n = self._num_steps
        return GenomeSlice(
            start=0,
            size=n,
            lower_bound=np.full(n, -1.0),
            upper_bound=np.full(n, +1.0),
        )

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
            genome_batch: Float array of shape ``(population_size, horizon)``
                with interleaved ``[bat_factor]`` genes per step.
                Mutated in place with repaired values (Lamarckian repair).

        Returns:
            The mutated ``genome_batch``.
        """
        pv_power_w = self._pv_power_w
        if self._step_interval_sec is None:
            raise RuntimeError("Call setup_run() before apply_genome_batch().")
        if pv_power_w is None and self.param.inverter_type in (
            InverterType.SOLAR,
            InverterType.HYBRID,
        ):
            raise RuntimeError("Call setup_run() before apply_genome_batch().")
        p = self.param

        if not hasattr(self, "_diagloggerged"):
            self._diagloggerged = True
            from loguru import logger as logger
            if pv_power_w is not None:
                nonzero_pv_steps = int((pv_power_w > 0).sum())
                logger.info(
                    "{}: apply_genome_batch — inverter_type={} capacity_wh={} "
                    "initial_soc_wh={} pv_steps={}/{} pv_max={:.0f}W",
                    self.device_id, p.inverter_type.name, p.battery_capacity_wh,
                    self._battery_initial_soc_wh, nonzero_pv_steps,
                    len(pv_power_w), float(pv_power_w.max()),
                )
            else:
                logger.info(
                    "{}: apply_genome_batch — inverter_type={} capacity_wh={} "
                    "initial_soc_wh={} no PV",
                    self.device_id, p.inverter_type.name,
                    p.battery_capacity_wh, self._battery_initial_soc_wh,
                )

        soc_wh = np.full(
            state.population_size,
            self._battery_initial_soc_wh if self._battery_initial_soc_wh is not None else 0.0,
        )

        for t in range(state.horizon):
            raw_bat = genome_batch[:, t]   # shape (pop,)

            if pv_power_w is None:
                pv_dc_avail_w = 0.0
            elif pv_power_w[t] < p.pv_min_power_w:
                pv_dc_avail_w = 0.0
            else:
                pv_dc_avail_w = float(np.clip(pv_power_w[t], 0.0, p.pv_max_power_w))

            ac_w, soc_wh, bat_t = self._compute_ac_power_and_soc_and_repair(
                raw_bat, soc_wh, pv_dc_avail_w
            )

            state.bat_factors[:, t] = bat_t
            state.pv_util_factors[:, t] = (
                pv_dc_avail_w / p.pv_max_power_w
                if np.isfinite(p.pv_max_power_w) and p.pv_max_power_w > 0
                else float(pv_dc_avail_w > 0.0)
            )
            state.ac_power_w[:, t] = ac_w
            state.soc_wh[:, t] = soc_wh

            # Lamarckian write-back
            genome_batch[:, t] = bat_t

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

    # ------------------------------------------------------------------
    # Cost accumulation
    # ------------------------------------------------------------------

    def compute_cost(self, state: HybridInverterBatchState) -> np.ndarray:
        """Compute per-individual energy cost including SoC correction, LCOS, and optional discharge reward.

        This method evaluates the economic performance of each individual in the
        population over the simulation horizon. The resulting cost matrix is used
        by the optimizer (e.g., genetic algorithm) for fitness evaluation.

        The total cost consists of three components:

        1. Terminal SoC correction
        2. Levelized Cost of Storage (LCOS)
        3. Optional battery discharge reward

        Terminal SoC correction
        -----------------------

        Without this correction, the optimizer can exploit "free" stored energy by
        discharging the battery within the horizon without accounting for the cost
        of recharging it later.

        The correction penalizes net energy drawn from the battery:

            delta_soc_wh = initial_soc_wh - terminal_soc_wh
            soc_cost     = max(0, delta_soc_wh) / 1000 * soc_price_per_kwh

        The SoC price is derived as:

            - Mean non-negative export price (if time series available), or
            - Static export price from configuration, or
            - Fallback default (0.08 EUR/kWh)

        Only net discharge (positive delta) is penalized. Net charging is not
        rewarded, as grid import costs already account for it.

        LCOS (Levelized Cost of Storage)
        --------------------------------

        If ``battery_lcos_amt_kwh > 0``, an additional cost proportional to total
        battery throughput is applied:

            cycled_energy_wh = charge_energy + discharge_energy
            lcos_cost        = cycled_energy_wh / 1000 * battery_lcos_amt_kwh

        This models battery degradation and efficiency-related lifecycle costs.

        Discharge reward (optional)
        ---------------------------

        If ``battery_discharge_reward_amt_kwh > 0``, a reward is applied for
        energy delivered from the battery to the AC bus:

            discharged_ac_wh = sum(discharge_power * efficiency * step_h)
            reward           = -discharged_ac_wh / 1000 * reward_per_kwh

        The reward is returned as a negative cost (benefit), encouraging the
        optimizer to utilize the battery for discharge.

        The reward is provided as a separate column in the output matrix, allowing
        multi-objective scalarization.

        Args:
            state: Batched inverter simulation state containing per-individual
                trajectories of state-of-charge and control variables.

        Returns:
            np.ndarray:
                Cost matrix of shape:

                - ``(population_size, 0)`` for SOLAR systems (no battery)
                - ``(population_size, 1)`` if no discharge reward is configured
                - ``(population_size, 2)`` if discharge reward is active

                Columns:
                    0: Total energy cost (SoC correction + LCOS)
                    1: Discharge reward (negative cost, optional)

        Raises:
            RuntimeError: If the simulation has not been initialized via
                ``setup_run()`` before calling this method.
        """
        p = self.param
        if p.inverter_type == InverterType.SOLAR:
            return np.zeros((state.population_size, 0))

        if self._step_interval_sec is None or self._battery_initial_soc_wh is None:
            raise RuntimeError("Call setup_run() before compute_cost().")

        step_h = self._step_interval_sec / 3600.0

        # --- SoC correction ---
        if self._export_price_per_kwh is not None:
            prices = self._export_price_per_kwh
            nonneg = prices[prices >= 0.0]
            soc_price_per_kwh = float(nonneg.mean()) if len(nonneg) > 0 else 0.0
        elif hasattr(p, "export_revenue_per_kwh"):
            soc_price_per_kwh = float(p.export_revenue_per_kwh)
        else:
            soc_price_per_kwh = 0.08

        initial_soc_wh = self._battery_initial_soc_wh
        terminal_soc_wh = state.soc_wh[:, -1]
        soc_delta_wh = initial_soc_wh - terminal_soc_wh
        soc_cost = np.maximum(0.0, soc_delta_wh) / 1000.0 * soc_price_per_kwh

        # --- LCOS ---
        if p.battery_lcos_amt_kwh > 0.0:
            bat_f = state.bat_factors
            charge_mask = bat_f > 0.0
            discharge_mask = bat_f < 0.0
            cycled_wh = (
                np.where(charge_mask, bat_f * p.battery_max_charge_rate, 0.0)
                + np.where(discharge_mask, -bat_f * p.battery_max_discharge_rate, 0.0)
            ) * p.battery_capacity_wh * step_h
            lcos = cycled_wh.sum(axis=1) / 1000.0 * p.battery_lcos_amt_kwh
        else:
            lcos = np.zeros(state.population_size)

        energy_cost = (soc_cost + lcos)  # shape (pop,)

        if p.battery_discharge_reward_amt_kwh == 0.0:
            return energy_cost[:, np.newaxis]  # (pop, 1)

        # --- Discharge reward ---
        # Compute AC Wh delivered by the battery per individual across the horizon.
        # Only discharge steps contribute (bat_f < 0).
        bat_f = state.bat_factors
        discharge_mask = bat_f < 0.0
        discharged_ac_wh = (
            np.where(discharge_mask, -bat_f * p.battery_max_discharge_rate, 0.0)
            * p.battery_capacity_wh
            * p.battery_to_ac_efficiency
            * step_h
        ).sum(axis=1)  # (pop,)

        # Negative cost = benefit. The GA minimises total cost, so this drives
        # it toward higher discharge.
        discharge_reward = -discharged_ac_wh / 1000.0 * p.battery_discharge_reward_amt_kwh

        return np.column_stack([energy_cost, discharge_reward])  # (pop, 2)

    # ------------------------------------------------------------------
    # S2 instruction extraction
    # ------------------------------------------------------------------

    def extract_instructions(
        self,
        state: HybridInverterBatchState,
        individual_index: int,
        instruction_context: InstructionContext | None = None,
    ) -> list[EnergyManagementInstruction]:
        """Emit a ``OMBCInstruction`` object per time step.

        The battery command:

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
                bat_factor_out = 1.0  # abs(float(bat_f))
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
        return instructions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ac_power_and_soc_and_repair(
        self,
        raw_bat: np.ndarray,
        soc_wh: np.ndarray,
        pv_dc_avail_w: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Repair bat_factor, compute net AC power, and advance SoC for one time step.

        Combines _repair_genes, _compute_ac_power, and _advance_soc into a single
        pass so that repair, physics, and SoC advance all share the same masks and
        intermediate values without recomputing them.

        Args:
            raw_bat:       Raw battery factor genes, shape ``(pop,)``, in ``[−1, +1]``.
                        Positive = charge intent, negative = discharge intent.
            soc_wh:        Current SoC per individual, shape ``(pop,)`` [Wh].
            pv_dc_avail_w: Clipped available DC PV power this step [W] (scalar).
                        Already bounded to ``[pv_min_power_w, pv_max_power_w]``
                        by the caller; pass 0.0 when PV is unavailable.

        Returns:
            ac_w:       Net AC port power, shape ``(pop,)`` [W].
                        Positive = consuming from bus, negative = injecting.
            new_soc_wh: Updated SoC, shape ``(pop,)`` [Wh], clamped to
                        ``[battery_min_soc_wh, battery_max_soc_wh]``.
            bat_repaired: Repaired battery factor, shape ``(pop,)``, in ``[−1, +1]``.
                        Written back into genome_batch by the caller (Lamarckian repair).
        """
        if self._step_interval_sec is None:
            raise RuntimeError("Step interval is None.")
        p = self.param
        step_h = self._step_interval_sec / 3600.0
        pop = len(raw_bat)

        # pv_util is always 1.0 — PV is never curtailed (see genome_requirements).
        pv_dc_used_w = pv_dc_avail_w  # scalar; per-individual: pv_util[i]=1.0 so used=avail

        # --- SOLAR: no battery at all — zero bat, pass PV through, return immediately ---
        if p.inverter_type == InverterType.SOLAR:
            ac_w = np.full(pop, -pv_dc_used_w * p.pv_to_ac_efficiency + p.on_state_power_consumption_w
                        if pv_dc_used_w > 0.0 else p.off_state_power_consumption_w)
            new_soc_wh = soc_wh.copy()
            bat_repaired = np.zeros(pop)
            return ac_w, new_soc_wh, bat_repaired

        # Working copies
        bat = raw_bat.copy()
        ac_w = np.zeros(pop)
        new_soc_wh = soc_wh.copy()

        # ----------------------------------------------------------------
        # CHARGING path
        # ----------------------------------------------------------------
        charge_mask = bat > 0.0
        if charge_mask.any():
            # --- REPAIR (charge) ----------------------------------------
            bf = bat[charge_mask]
            # Step 1: scale gene [0,+1] → absolute rate fraction [0, max_charge_rate]
            abs_rate = bf * p.battery_max_charge_rate

            # Step 2: snap to nearest discrete rate if configured
            if p.battery_charge_rates is not None:
                rates = np.array(p.battery_charge_rates)
                idx = np.argmin(
                    np.abs(abs_rate[:, np.newaxis] - rates[np.newaxis, :]), axis=1
                )
                abs_rate = rates[idx]

            # Step 3: cap by SoC headroom so battery cannot exceed max_soc_wh this step.
            # Conservative worst-case: all energy from AC (no PV assist).
            # stored_wh = abs_rate * capacity * step_h * ac_eff  ≤  headroom_wh
            headroom_wh = p.battery_max_soc_wh - soc_wh[charge_mask]
            headroom_wh = np.maximum(headroom_wh, 0.0)
            max_abs_rate = np.clip(
                headroom_wh / (p.battery_capacity_wh * step_h * p.ac_to_battery_efficiency),
                0.0,
                p.battery_max_charge_rate,
            )
            abs_rate = np.minimum(abs_rate, max_abs_rate)

            # Step 4: apply minimum deadband — below min_charge_rate, treat as idle
            abs_rate[abs_rate < p.battery_min_charge_rate] = 0.0

            # Step 5: write repaired rate back to genome domain [0,+1]
            bf_repaired = (
                abs_rate / p.battery_max_charge_rate
                if p.battery_max_charge_rate > 0 else np.zeros_like(abs_rate)
            )
            bat[charge_mask] = bf_repaired

            # Step 6: recalculate after repair
            charge_mask = bat > 0.0
            abs_rate = bat[charge_mask] * p.battery_max_charge_rate
            # --- END REPAIR (charge) ------------------------------------

            # Physics: battery charge demand [bat-side W]
            charge_bat_w = abs_rate * p.battery_capacity_wh

            if p.inverter_type == InverterType.HYBRID:
                pv_for_bat_avail_w = pv_dc_used_w * p.pv_to_battery_efficiency
                pv_for_bat_w = np.minimum(pv_for_bat_avail_w, charge_bat_w)
                pv_surplus_dc_w = pv_dc_used_w - pv_for_bat_w / p.pv_to_battery_efficiency
                pv_surplus_ac_w = pv_surplus_dc_w * p.pv_to_ac_efficiency
                ac_for_bat_w = (
                    np.maximum(0.0, charge_bat_w - pv_for_bat_w) / p.ac_to_battery_efficiency
                )
                ac_w[charge_mask] = (
                    ac_for_bat_w - pv_surplus_ac_w + p.on_state_power_consumption_w
                )
                stored_bat_wh = (pv_for_bat_w + ac_for_bat_w * p.ac_to_battery_efficiency) * step_h
            else:
                # BATTERY type: no PV
                ac_for_bat_w = charge_bat_w / p.ac_to_battery_efficiency
                ac_w[charge_mask] = ac_for_bat_w + p.on_state_power_consumption_w
                stored_bat_wh = charge_bat_w * step_h

            new_soc_wh[charge_mask] += stored_bat_wh

            #logger.info(f"ac_for_bat_w: {ac_for_bat_w}")

        # ----------------------------------------------------------------
        # DISCHARGING path
        # ----------------------------------------------------------------
        discharge_mask = bat < 0.0
        if discharge_mask.any():
            # --- REPAIR (discharge) -------------------------------------
            bf = bat[discharge_mask]  # negative values
            # Step 1: scale gene [-1,0) → absolute rate fraction [0, max_discharge_rate]
            abs_rate = (-bf) * p.battery_max_discharge_rate

            # Step 2: cap by available SoC so battery cannot go below min_soc_wh this step.
            # drawn_wh = abs_rate * capacity * step_h  ≤  available_wh
            available_wh = soc_wh[discharge_mask] - p.battery_min_soc_wh
            available_wh = np.maximum(available_wh, 0.0)
            max_abs_rate = np.clip(
                available_wh / (p.battery_capacity_wh * step_h),
                0.0,
                p.battery_max_discharge_rate,
            )
            abs_rate = np.minimum(abs_rate, max_abs_rate)

            # Step 3: apply minimum deadband — below min_discharge_rate, treat as idle
            abs_rate[abs_rate < p.battery_min_discharge_rate] = 0.0

            # Step 4: write repaired rate back to genome domain [-1,0]
            bf_repaired = (
                -(abs_rate / p.battery_max_discharge_rate)
                if p.battery_max_discharge_rate > 0 else np.zeros_like(abs_rate)
            )
            bat[discharge_mask] = bf_repaired

            # Step 5: recalculate after repair
            discharge_mask = bat < 0.0
            abs_rate = (-bat[discharge_mask]) * p.battery_max_discharge_rate
            # --- END REPAIR (discharge) ---------------------------------

            # Physics: battery discharge power [bat-side W]
            discharge_bat_w = abs_rate * p.battery_capacity_wh
            discharge_bat_ac_w = discharge_bat_w * p.battery_to_ac_efficiency

            if p.inverter_type == InverterType.HYBRID:
                pv_ac_w = pv_dc_used_w * p.pv_to_ac_efficiency
                ac_w[discharge_mask] = (
                    -(discharge_bat_ac_w + pv_ac_w) + p.on_state_power_consumption_w
                )
            else:
                # BATTERY type: no PV
                ac_w[discharge_mask] = (
                    -discharge_bat_ac_w + p.on_state_power_consumption_w
                )

            new_soc_wh[discharge_mask] -= discharge_bat_w * step_h

            #logger.info(f"discharge_bat_ac_w: {discharge_bat_ac_w}")

        # ----------------------------------------------------------------
        # IDLE path
        # ----------------------------------------------------------------
        idle_mask = ~(charge_mask | discharge_mask)
        if idle_mask.any():
            # --- REPAIR (idle) ------------------------------------------
            # No battery action requested; bat gene is already 0.0 or was
            # driven to 0 by the deadband in charge/discharge above.
            # Nothing to repair — bat[idle_mask] stays 0.0.
            # --- END REPAIR (idle) --------------------------------------

            if p.inverter_type == InverterType.HYBRID:
                pv_ac_w = pv_dc_used_w * p.pv_to_ac_efficiency
                if pv_ac_w > 0.0:
                    ac_w[idle_mask] = -pv_ac_w + p.on_state_power_consumption_w
                else:
                    ac_w[idle_mask] = p.off_state_power_consumption_w
            else:
                # BATTERY type: no PV, standby draw
                ac_w[idle_mask] = p.off_state_power_consumption_w

        # Clamp SoC to valid range
        # battery_min_soc_wh / max_soc_wh are soft operational targets.
        # Physical limits are enforced only by [0, capacity] clamp.
        new_soc_wh = np.clip(new_soc_wh, 0, p.battery_capacity_wh)

        #logger.info(f"ac_w: {ac_w}")

        return ac_w, new_soc_wh, bat

    def _compute_min_energy(
        self,
        state: HybridInverterBatchState,
        step_h: float,
    ) -> np.ndarray:
        """Compute minimum acceptable AC energy per individual per step.

        Used by the arbitrator to determine the tightest AC commitment.

        Notes:
            - Charging: minimum AC energy needed at min charge rate, no PV assist.
            - Discharging: minimum injection includes min discharge rate, plus PV already used.
            - Idle/PV-only: minimum is 0 unless PV is generating (must inject surplus).

        Returns:
            min_e: Minimum AC energy per individual per step [Wh].
                Positive = consuming from bus, negative = injecting.
        """
        p = self.param
        pv_power_w = self._pv_power_w  # None for BATTERY

        min_e = np.zeros_like(state.ac_power_w)

        # --- Conservative min AC energy for charging (no PV assist) ---
        min_charge_bat_w = p.battery_min_charge_rate * p.battery_max_charge_rate * p.battery_capacity_wh
        min_charge_ac_w = min_charge_bat_w / p.ac_to_battery_efficiency

        # --- Conservative min AC energy for discharging ---
        min_dis_bat_w = p.battery_min_discharge_rate * p.battery_max_discharge_rate * p.battery_capacity_wh
        min_dis_bat_ac_w = min_dis_bat_w * p.battery_to_ac_efficiency

        # Masks
        charge_mask = state.bat_factors > 0.0
        discharge_mask = state.bat_factors < 0.0
        idle_mask = state.bat_factors == 0.0

        # --- Charging: allocate min AC energy ---
        if charge_mask.any():
            if p.inverter_type == InverterType.HYBRID and pv_power_w is not None:
                pop_idx, step_idx = np.where(charge_mask)
                pv_clipped = np.clip(pv_power_w, 0.0, p.pv_max_power_w)
                pv_util_at_step = state.pv_util_factors[pop_idx, step_idx]
                pv_dc_used = pv_util_at_step * pv_clipped[step_idx]
                pv_for_bat_avail = pv_dc_used * p.pv_to_battery_efficiency
                pv_for_bat = np.minimum(pv_for_bat_avail, min_charge_bat_w)
                pv_surplus_dc = pv_dc_used - pv_for_bat / p.pv_to_battery_efficiency
                pv_surplus_ac = pv_surplus_dc * p.pv_to_ac_efficiency
                ac_for_bat = np.maximum(0.0, min_charge_bat_w - pv_for_bat) / p.ac_to_battery_efficiency
                net_min_ac_w = np.maximum(0.0, ac_for_bat - pv_surplus_ac)
                min_e[pop_idx, step_idx] = net_min_ac_w * step_h
            else:
                min_e[charge_mask] = min_charge_ac_w * step_h

        # --- Discharging: always allocate at least min discharge ---
        if discharge_mask.any():
            if p.inverter_type == InverterType.HYBRID and pv_power_w is not None:
                pop_idx, step_idx = np.where(discharge_mask)
                pv_clipped = np.clip(pv_power_w, 0.0, p.pv_max_power_w)
                pv_util_at_step = state.pv_util_factors[pop_idx, step_idx]
                pv_dc_used = pv_util_at_step * pv_clipped[step_idx]
                pv_inject_ac_w = pv_dc_used * p.pv_to_ac_efficiency
                # Minimum discharge injection plus PV already injecting
                total_min_inject = min_dis_bat_ac_w + pv_inject_ac_w
                min_e[pop_idx, step_idx] = -total_min_inject * step_h
            else:
                min_e[discharge_mask] = -min_dis_bat_ac_w * step_h

        # --- Idle: inject PV if generating ---
        if idle_mask.any() and p.inverter_type in (InverterType.SOLAR, InverterType.HYBRID) and pv_power_w is not None:
            pop_idx, step_idx = np.where(idle_mask)
            pv_clipped = np.clip(pv_power_w, 0.0, p.pv_max_power_w)
            pv_util_at_step = state.pv_util_factors[pop_idx, step_idx]
            pv_ac_w = pv_util_at_step * pv_clipped[step_idx] * p.pv_to_ac_efficiency
            pv_injection = -pv_ac_w * step_h
            min_e[pop_idx, step_idx] = np.minimum(min_e[pop_idx, step_idx], pv_injection)

        return min_e
