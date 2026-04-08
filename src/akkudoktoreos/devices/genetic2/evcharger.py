"""EV charger energy device for the genetic2 simulation framework.

Models a wall-box / EVSE (Electric Vehicle Supply Equipment) with a connected
EV battery.  The device is a pure AC sink — it draws power from the AC bus and
stores it in the EV battery, subject to:

- Whether the EV is physically plugged in at each simulation step.
- The EV battery's usable SoC window.
- Power limits of the charger hardware.
- A realistic multi-stage efficiency chain from the wall to the battery.

Genome structure
----------------
One gene per time step:

    ``charge_factor[t] ∈ [0, 1]``

where ``0`` means idle (or EV absent) and ``1`` means charging at
``max_charge_power_w``.  The gene is always repaired to ``0`` for steps
where the EV is not connected.

Connection state
----------------
At ``setup_run`` the device builds a ``(horizon,)`` binary array
``_connected`` using the following priority:

1. **Measurement** — ``ev_connected_measurement_key`` resolves to ``1.0``
   (connected) or ``0.0`` (absent).  Best for real-time dispatch where the
   EMS knows the current plug state.
2. **Time-window** — ``connection_time_window_key`` resolves a
   ``CycleTimeWindowSequence`` from the config.  Each step that falls inside
   any window is treated as connected.  Best for forward-planning / forecast
   runs.
3. **Always-connected fallback** — if neither source is available every step
   is treated as connected.

Loss model
----------
The efficiency chain from wall-plug to stored energy:

::

    AC_wall → [charger_efficiency] → DC_cable
             → [hold_time_efficiency × control_efficiency] → effective DC
             → [ev_charger_efficiency_{low|high}] → EV onboard charger output
             → [ev_battery_efficiency] → stored in EV battery

Additionally:

- ``standby_power_w``    — constant draw when EV is connected but not charging.
- ``deep_standby_power_w`` — constant draw when EV is absent (charger idle).
- The low/high current boundary is set by ``high_current_threshold_w``.

Cost contribution
-----------------
``EVChargerDevice`` contributes to the global ``"energy_cost_amt"`` objective
via:

1. **SoC shortfall penalty** — if the EV departs (end of horizon) below
   ``ev_target_soc_factor`` the shortfall is valued at the mean import price.
   This drives the GA to ensure the vehicle is charged by departure.
2. **LCOS** (optional) — lifecycle cost of battery cycling.

Sign convention
---------------
Consistent with the rest of the framework:

    positive energy_wh  → consuming from the AC bus (load, charging)
    negative energy_wh  → injecting into the AC bus  (not applicable here)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
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
# Immutable parameter dataclass
# ============================================================


@dataclass(frozen=True, slots=True)
class EVChargerParam(DeviceParam):
    """Immutable parameters for an EV charger device.

    Frozen, slotted, and hashable — safe as a dictionary or cache key
    inside the genetic algorithm.

    Attributes
    ----------
    device_id, ports:
        Inherited from ``DeviceParam``.

    Power limits
    ~~~~~~~~~~~~
    min_charge_power_w:
        Minimum non-zero charge power [W].  Charge commands below this
        value are rounded to zero (deadband).  Typical: 1 380 W (6 A × 230 V).
    max_charge_power_w:
        Maximum charge power at the AC wall socket [W].  ``charge_factor=1``
        maps here.  Typical: 11 000 W (16 A × 3-phase).
    high_current_threshold_w:
        Boundary between the low-current and high-current EV onboard charger
        efficiency regimes [W].  Typical: 3 680 W (single-phase 16 A).

    Loss model
    ~~~~~~~~~~
    deep_standby_power_w:
        AC draw of the charger hardware when the EV is *not* plugged in [W].
        Typical: 1–3 W.
    standby_power_w:
        AC draw when the EV *is* plugged in but charging is paused [W].
        Typical: 3–10 W.
    charger_efficiency:
        Efficiency of the charger's AC→DC conversion stage (0, 1].
        Typical: 0.95.
    ev_charger_efficiency_low:
        Efficiency of the EV's onboard charger at low AC current (0, 1].
        Typical single-phase 6 A: 0.77.
    ev_charger_efficiency_high:
        Efficiency of the EV's onboard charger at high AC current (0, 1].
        Typical three-phase 16 A: 0.90.
    ev_battery_efficiency:
        Round-trip efficiency of the EV's high-voltage battery when charging
        (DC in → stored kWh) (0, 1].  Typical: 0.96.
    hold_time_efficiency:
        Fraction of scheduled energy actually delivered when accounting for
        scheduling lag (the charger may stay in the wrong mode for part of a
        time step) (0, 1].  Typical: 0.98.
    control_efficiency:
        Fraction of available power that the charger actually tracks when the
        power source (e.g. solar) varies within a step (0, 1].  Typical: 0.97.

    EV battery
    ~~~~~~~~~~
    ev_battery_capacity_wh:
        Usable EV battery capacity [Wh].  Typical: 40 000–100 000 Wh.
    ev_min_soc_factor:
        Minimum allowed SoC as a fraction of ``ev_battery_capacity_wh``.
        Must be ≥ 0 and < ``ev_max_soc_factor``.
    ev_max_soc_factor:
        Maximum allowed SoC as a fraction of ``ev_battery_capacity_wh``.
        Must be ≤ 1 and > ``ev_min_soc_factor``.
    ev_target_soc_factor:
        Desired SoC at departure as a fraction of ``ev_battery_capacity_wh``.
        Used in the ``compute_cost`` SoC shortfall penalty.  Set equal to
        ``ev_max_soc_factor`` to always try to fill the battery.
    ev_initial_soc_factor_key:
        SimulationContext *measurement* key resolving to the EV's current SoC
        as a fraction of ``ev_battery_capacity_wh``.  An empty string means
        the device defaults to ``ev_min_soc_factor`` (empty battery).

    Connection state
    ~~~~~~~~~~~~~~~~
    ev_connected_measurement_key:
        SimulationContext *measurement* key resolving to ``1.0`` (EV plugged
        in) or ``0.0`` (EV absent).  Takes precedence over
        ``connection_time_window_key`` when a non-None value is returned.
        An empty string disables measurement-based connection detection.
    connection_time_window_key:
        ``/``-separated path into the EOS config tree pointing to a
        ``CycleTimeWindowSequence`` that defines when the EV is expected to be
        plugged in.  ``None`` disables time-window-based connection detection.

    Price keys
    ~~~~~~~~~~
    import_price_amt_kwh_key:
        Key for the per-step import price forecast, used to value the SoC
        shortfall penalty in ``compute_cost``.
    export_price_amt_kwh_key:
        Key for the per-step export price forecast (currently unused but
        reserved for future V2G support).

    Wear cost
    ~~~~~~~~~
    ev_lcos_amt_kwh:
        Levelized cost of EV battery storage [currency / kWh cycled].
        Added to the fitness to discourage unnecessary cycling.  Default 0.0.
    """

    # Power limits
    min_charge_power_w: float
    max_charge_power_w: float
    high_current_threshold_w: float

    # Loss model
    deep_standby_power_w: float
    standby_power_w: float
    charger_efficiency: float
    ev_charger_efficiency_low: float
    ev_charger_efficiency_high: float
    ev_battery_efficiency: float
    hold_time_efficiency: float
    control_efficiency: float

    # EV battery
    ev_battery_capacity_wh: float
    ev_min_soc_factor: float
    ev_max_soc_factor: float
    ev_target_soc_factor: float
    ev_initial_soc_factor_key: str

    # Connection state
    ev_connected_measurement_key: str
    connection_time_window_key: str | None

    # Price keys
    import_price_amt_kwh_key: str | None
    export_price_amt_kwh_key: str | None

    # Wear cost
    ev_lcos_amt_kwh: float = 0.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.min_charge_power_w < 0:
            raise ValueError(f"{self.device_id}: min_charge_power_w must be >= 0")
        if self.max_charge_power_w <= 0:
            raise ValueError(f"{self.device_id}: max_charge_power_w must be > 0")
        if self.min_charge_power_w > self.max_charge_power_w:
            raise ValueError(
                f"{self.device_id}: min_charge_power_w must be <= max_charge_power_w"
            )
        if self.high_current_threshold_w < 0:
            raise ValueError(f"{self.device_id}: high_current_threshold_w must be >= 0")
        if self.deep_standby_power_w < 0:
            raise ValueError(f"{self.device_id}: deep_standby_power_w must be >= 0")
        if self.standby_power_w < 0:
            raise ValueError(f"{self.device_id}: standby_power_w must be >= 0")
        for name, val in [
            ("charger_efficiency", self.charger_efficiency),
            ("ev_charger_efficiency_low", self.ev_charger_efficiency_low),
            ("ev_charger_efficiency_high", self.ev_charger_efficiency_high),
            ("ev_battery_efficiency", self.ev_battery_efficiency),
            ("hold_time_efficiency", self.hold_time_efficiency),
            ("control_efficiency", self.control_efficiency),
        ]:
            if not (0 < val <= 1):
                raise ValueError(f"{self.device_id}: {name} must be in (0, 1]")
        if self.ev_battery_capacity_wh <= 0:
            raise ValueError(f"{self.device_id}: ev_battery_capacity_wh must be > 0")
        if not (0 <= self.ev_min_soc_factor < self.ev_max_soc_factor <= 1):
            raise ValueError(
                f"{self.device_id}: SoC factors must satisfy "
                "0 <= ev_min_soc_factor < ev_max_soc_factor <= 1"
            )
        if not (self.ev_min_soc_factor <= self.ev_target_soc_factor <= self.ev_max_soc_factor):
            raise ValueError(
                f"{self.device_id}: ev_target_soc_factor must be in "
                "[ev_min_soc_factor, ev_max_soc_factor]"
            )

    # Derived convenience properties
    @property
    def ev_min_soc_wh(self) -> float:
        return self.ev_min_soc_factor * self.ev_battery_capacity_wh

    @property
    def ev_max_soc_wh(self) -> float:
        return self.ev_max_soc_factor * self.ev_battery_capacity_wh

    @property
    def ev_target_soc_wh(self) -> float:
        return self.ev_target_soc_factor * self.ev_battery_capacity_wh

    @property
    def combined_max_efficiency(self) -> float:
        """DC-cable-to-battery efficiency at full (high-current) power.

        Excludes ``charger_efficiency`` because that loss is captured in the
        AC wall-draw formula.  This value is used by ``compute_cost`` to
        estimate total energy stored in the LCOS calculation.
        """
        return (
            self.ev_charger_efficiency_high
            * self.ev_battery_efficiency
            * self.hold_time_efficiency
            * self.control_efficiency
        )


# ============================================================
# Mutable batch state
# ============================================================


@dataclass
class EVChargerBatchState:
    """Mutable batch state for ``EVChargerDevice``.

    Created fresh each generation by ``create_batch_state``.
    Never shared between devices or between generations.

    Attributes
    ----------
    charge_factors:
        Repaired charge power fraction in ``[0, 1]`` per individual per step,
        shape ``(population_size, horizon)``.  Zero at steps where the EV is
        absent.
    connected:
        Binary array of shape ``(horizon,)`` indicating EV presence (``1.0``)
        or absence (``0.0``) at each step.  Identical for all individuals in
        the population — connection state is not a genome decision.
    soc_wh:
        EV battery state of charge after each step [Wh],
        shape ``(population_size, horizon)``.
    ac_power_w:
        Net AC port power (wall draw including standby/deep-standby) per
        individual per step [W], shape ``(population_size, horizon)``.
        Always ≥ 0 (pure sink device).
    population_size:
        Number of individuals in this batch.
    horizon:
        Number of simulation time steps.
    step_times:
        Ordered ``DateTime`` timestamps, length == horizon.
    """

    charge_factors: np.ndarray  # (population_size, horizon)  float64
    connected: np.ndarray  # (horizon,)                   float64  0/1
    soc_wh: np.ndarray  # (population_size, horizon)  float64
    ac_power_w: np.ndarray  # (population_size, horizon)  float64
    population_size: int
    horizon: int
    step_times: tuple[DateTime, ...]


# ============================================================
# Device implementation
# ============================================================


class EVChargerDevice(EnergyDevice):
    """EV charger device — pure AC sink with a detailed efficiency model.

    Parameters
    ----------
    param:
        Immutable device parameters.
    device_index:
        Position of this device in the shared device list used by the
        arbitrator to route grants back to this device.
    port_index:
        Index of this device's AC port in the arbitrator's port-to-bus
        topology array.
    """

    def __init__(
        self,
        param: EVChargerParam,
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
        self._step_interval_sec: float | None = None
        self._connected: np.ndarray | None = None  # shape (horizon,) float64 0/1
        self._initial_soc_wh: float | None = None
        self._import_price_per_kwh: np.ndarray | None = None  # shape (horizon,)

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return self.param.ports

    @property
    def objective_names(self) -> list[str]:
        """Contributes ``"energy_cost_amt"`` for the SoC shortfall / LCOS penalty."""
        return ["energy_cost_amt"]

    # ------------------------------------------------------------------
    # Structure Phase
    # ------------------------------------------------------------------

    def setup_run(self, context: SimulationContext) -> None:
        """Resolve connection state, initial SoC, and price forecasts.

        Connection-state resolution order:
        1. Measurement ``ev_connected_measurement_key`` (``1.0``/``0.0``).
        2. Time-window ``connection_time_window_key`` (``CycleTimeWindowSequence``).
        3. Always-connected fallback.

        Args:
            context: Run-scoped simulation context.
        """
        p = self.param
        horizon = context.horizon
        self._step_times = context.step_times
        self._step_interval_sec = context.step_interval.total_seconds()

        # ---- Connection state ----------------------------------------
        connected = np.ones(horizon, dtype=np.float64)  # default: always present

        # Priority 2: time-window (resolved before measurement so measurement
        # can override individual steps below)
        if p.connection_time_window_key is not None:
            try:
                _cycles, matrix = context.resolve_config_cycle_time_windows(
                    p.connection_time_window_key
                )
                # matrix shape: (num_cycles, horizon); any cycle being active
                # means the EV is expected to be present.
                if matrix.shape[0] > 0:
                    connected = np.where(matrix.any(axis=0), 1.0, 0.0)
                    logger.debug(
                        "{}: connection time-window resolved, connected steps={}/{}",
                        self.device_id,
                        int(connected.sum()),
                        horizon,
                    )
            except Exception as exc:
                logger.warning(
                    "{}: could not resolve connection_time_window_key '{}': {}",
                    self.device_id,
                    p.connection_time_window_key,
                    exc,
                )

        # Priority 1: measurement overrides the time-window for the *current*
        # state.  A single scalar value (current plug state) is broadcast to
        # all horizon steps.  This is the correct behaviour for real-time
        # dispatch: if the EV is absent right now, no charging can start.
        if p.ev_connected_measurement_key:
            meas = context.resolve_measurement(p.ev_connected_measurement_key)
            if meas is not None:
                # The measurement is the ground truth for the *initial* step;
                # for future steps we fall back to the time-window (already in
                # `connected`).  As a practical simplification we apply the
                # measured state uniformly across the horizon — the operator
                # can always set up a time-window for finer control.
                broadcast_state = 1.0 if meas >= 0.5 else 0.0
                connected[:] = broadcast_state
                logger.info(
                    "{}: EV connection state from measurement: {}",
                    self.device_id,
                    "connected" if broadcast_state == 1.0 else "absent",
                )

        self._connected = connected

        # ---- Initial SoC ---------------------------------------------
        soc_factor: float | None = None
        if p.ev_initial_soc_factor_key:
            soc_factor = context.resolve_measurement(p.ev_initial_soc_factor_key)
        if soc_factor is None:
            soc_factor = p.ev_min_soc_factor
            logger.debug(
                "{}: no initial SoC measurement, defaulting to ev_min_soc_factor={}",
                self.device_id,
                soc_factor,
            )
        if not (0.0 <= soc_factor <= 1.0):
            raise ValueError(
                f"{self.device_id}: initial SoC factor {soc_factor} outside [0, 1]."
            )
        self._initial_soc_wh = soc_factor * p.ev_battery_capacity_wh
        logger.info(
            "{}: initial SoC {:.1f} Wh ({:.0f}% of {:.0f} Wh capacity)",
            self.device_id,
            self._initial_soc_wh,
            soc_factor * 100,
            p.ev_battery_capacity_wh,
        )

        # ---- Import price (for SoC shortfall penalty) ----------------
        if p.import_price_amt_kwh_key:
            try:
                prices = context.resolve_prediction(p.import_price_amt_kwh_key)
                if prices.shape == (horizon,):
                    self._import_price_per_kwh = prices
                else:
                    logger.warning(
                        "{}: import price array shape {} != ({},), ignoring.",
                        self.device_id,
                        prices.shape,
                        horizon,
                    )
            except Exception as exc:
                logger.warning(
                    "{}: could not resolve import_price_amt_kwh_key '{}': {}",
                    self.device_id,
                    p.import_price_amt_kwh_key,
                    exc,
                )

    def genome_requirements(self) -> GenomeSlice:
        """One ``charge_factor`` gene per simulation step, bounded ``[0, 1]``."""
        if self._step_times is None:
            raise RuntimeError("Call setup_run() before genome_requirements().")
        n = len(self._step_times)
        return GenomeSlice(
            start=0,
            size=n,
            lower_bound=np.zeros(n),
            upper_bound=np.ones(n),
        )

    # ------------------------------------------------------------------
    # Batch Lifecycle
    # ------------------------------------------------------------------

    def create_batch_state(
        self,
        population_size: int,
        horizon: int,
    ) -> EVChargerBatchState:
        """Allocate a fresh batch state for one generation."""
        if self._step_times is None or self._connected is None:
            raise RuntimeError("Call setup_run() before create_batch_state().")

        initial_soc = self._initial_soc_wh if self._initial_soc_wh is not None else 0.0
        return EVChargerBatchState(
            charge_factors=np.zeros((population_size, horizon), dtype=np.float64),
            connected=self._connected.copy(),
            soc_wh=np.full((population_size, horizon), initial_soc, dtype=np.float64),
            ac_power_w=np.zeros((population_size, horizon), dtype=np.float64),
            population_size=population_size,
            horizon=horizon,
            step_times=self._step_times,
        )

    def apply_genome_batch(
        self,
        state: EVChargerBatchState,
        genome_batch: np.ndarray,
    ) -> np.ndarray:
        """Decode, repair, simulate the genome for the full population.

        Repairs ``genome_batch`` in place (Lamarckian write-back):

        * Steps where EV is absent → gene forced to 0.
        * Charge power below deadband → rounded to 0.
        * Charge power exceeding SoC headroom → capped.

        Args:
            state: Pre-allocated batch state (mutated in place).
            genome_batch: Float array of shape ``(population_size, horizon)``.

        Returns:
            The mutated ``genome_batch`` with repaired charge factors.
        """
        if self._step_interval_sec is None or self._connected is None:
            raise RuntimeError("Call setup_run() before apply_genome_batch().")

        p = self.param
        step_h = self._step_interval_sec / 3600.0

        # Running SoC per individual, shape (pop,)
        soc_wh = np.full(
            state.population_size,
            self._initial_soc_wh if self._initial_soc_wh is not None else 0.0,
        )

        for t in range(state.horizon):
            connected_t = self._connected[t]

            if connected_t == 0.0:
                # EV absent — deep standby only, no charging possible
                state.charge_factors[:, t] = 0.0
                state.ac_power_w[:, t] = p.deep_standby_power_w
                state.soc_wh[:, t] = soc_wh
                genome_batch[:, t] = 0.0
                continue

            # --- Decode ---------------------------------------------------
            raw_cf = genome_batch[:, t]  # shape (pop,)
            charge_w = raw_cf * p.max_charge_power_w  # raw requested AC power

            # --- Repair 1: deadband ---------------------------------------
            below_min = (charge_w > 0.0) & (charge_w < p.min_charge_power_w)
            charge_w[below_min] = 0.0

            # --- Repair 2: clip to max ------------------------------------
            charge_w = np.clip(charge_w, 0.0, p.max_charge_power_w)

            # --- Repair 3: SoC headroom cap --------------------------------
            # Maximum energy storable this step (bat side):
            #   headroom_wh = ev_max_soc_wh - soc_wh
            # Energy stored per Watt of AC wall power:
            #   store_per_ac_w = _stored_per_wall_w(charge_w, p) * step_h
            # We use the combined efficiency at the *current* charge_w to
            # compute how much headroom is available, then cap.
            headroom_wh = np.maximum(0.0, p.ev_max_soc_wh - soc_wh)
            combined_eff = self._combined_efficiency(charge_w, p)
            # Avoid divide-by-zero when combined_eff ≈ 0 (shouldn't happen
            # because all efficiencies are validated > 0, but be safe)
            safe_eff = np.where(combined_eff > 0.0, combined_eff, 1.0)
            max_charge_from_headroom = headroom_wh / (safe_eff * step_h)
            charge_w = np.minimum(charge_w, max_charge_from_headroom)

            # Re-apply deadband after headroom cap (may have pushed below min)
            below_min = (charge_w > 0.0) & (charge_w < p.min_charge_power_w)
            charge_w[below_min] = 0.0

            # --- Physics: AC wall draw ------------------------------------
            # Model: charge_w is the DC power at the cable (after scheduling
            # losses have been accounted for via hold_time_efficiency and
            # control_efficiency reducing the effective stored energy).
            # The wall-box charger stage (charger_efficiency) converts AC→DC:
            #   AC_wall = charge_w / charger_eff  (+ standby overhead)
            # Scheduling losses (hold_time_eff, control_eff) are DC-side:
            # they mean only charge_w × hold_eff × control_eff of the DC
            # power actually reaches the EV onboard charger.
            charging_mask = charge_w > 0.0

            ac_w = np.where(
                charging_mask,
                # Active charging: wall draw = DC power / charger AC→DC eff + standby
                charge_w / p.charger_efficiency + p.standby_power_w,
                # EV connected but idle
                p.standby_power_w,
            )

            # --- Physics: energy stored in EV battery --------------------
            combined_eff_arr = self._combined_efficiency(charge_w, p)
            stored_wh = charge_w * combined_eff_arr * step_h

            # --- Advance SoC ---------------------------------------------
            soc_wh = soc_wh + stored_wh
            soc_wh = np.clip(soc_wh, 0.0, p.ev_battery_capacity_wh)

            # --- Lamarckian write-back ------------------------------------
            repaired_cf = np.where(
                p.max_charge_power_w > 0.0,
                charge_w / p.max_charge_power_w,
                0.0,
            )
            genome_batch[:, t] = repaired_cf

            # --- Store in state ------------------------------------------
            state.charge_factors[:, t] = repaired_cf
            state.ac_power_w[:, t] = ac_w
            state.soc_wh[:, t] = soc_wh

        return genome_batch

    # ------------------------------------------------------------------
    # Arbitration
    # ------------------------------------------------------------------

    def build_device_request(
        self,
        state: EVChargerBatchState,
    ) -> DeviceRequest:
        """Submit the AC wall draw as a sink request to the bus arbitrator.

        Args:
            state: Batch state after ``apply_genome_batch``.

        Returns:
            ``DeviceRequest`` with the per-individual AC energy demand.
        """
        if self._step_interval_sec is None:
            raise RuntimeError("Call setup_run() before build_device_request().")

        step_h = self._step_interval_sec / 3600.0
        energy_wh = state.ac_power_w * step_h  # (pop, horizon)

        return DeviceRequest(
            device_index=self._device_index,
            port_requests=(
                PortRequest(
                    port_index=self._port_index,
                    energy_wh=energy_wh,
                    # Accept partial supply (grid slack covers shortfall);
                    # min = 0 means the charger can be throttled to zero.
                    min_energy_wh=np.zeros_like(energy_wh),
                    is_slack=False,
                ),
            ),
        )

    def apply_device_grant(
        self,
        state: EVChargerBatchState,
        grant: DeviceGrant,
    ) -> None:
        """Record the arbitrated AC energy grant.

        Args:
            state: Mutable batch state.
            grant: Arbitrated energy grant from the bus arbitrator.
        """
        if self._step_interval_sec is None:
            raise RuntimeError("Call setup_run() before apply_device_grant().")
        step_h = self._step_interval_sec / 3600.0
        # Convert Wh back to W for consistent units in state
        state.ac_power_w[:] = grant.port_grants[0].granted_wh / step_h

    # ------------------------------------------------------------------
    # Cost Evaluation
    # ------------------------------------------------------------------

    def compute_cost(
        self,
        state: EVChargerBatchState,
    ) -> np.ndarray:
        """Compute SoC shortfall penalty and optional LCOS contribution.

        The cost has two components:

        1. **SoC shortfall** — the EV must leave (end of horizon) with at
           least ``ev_target_soc_wh``.  Any shortfall is valued at the mean
           import price (the user will need to charge elsewhere at full cost).
        2. **LCOS** (optional) — lifecycle cost proportional to total energy
           cycled through the EV battery.

        Args:
            state: Batch state after ``apply_device_grant``.

        Returns:
            Cost array of shape ``(population_size, 1)`` with ``"energy_cost_amt"``
            column.
        """
        if self._step_interval_sec is None or self._initial_soc_wh is None:
            raise RuntimeError("Call setup_run() before compute_cost().")

        p = self.param
        step_h = self._step_interval_sec / 3600.0

        # Determine import price for SoC shortfall valuation
        if self._import_price_per_kwh is not None:
            import_price = float(self._import_price_per_kwh.mean())
        else:
            import_price = 0.30  # sensible residential default [€/kWh]

        # --- SoC shortfall penalty ------------------------------------
        terminal_soc_wh = state.soc_wh[:, -1]  # (pop,)
        shortfall_wh = np.maximum(0.0, p.ev_target_soc_wh - terminal_soc_wh)
        soc_cost = shortfall_wh / 1000.0 * import_price

        # --- LCOS -------------------------------------------------------
        if p.ev_lcos_amt_kwh > 0.0:
            # Total energy stored in the EV battery across the horizon
            total_stored_wh = (
                state.charge_factors * p.max_charge_power_w * step_h
            ).sum(axis=1) * p.combined_max_efficiency
            lcos = total_stored_wh / 1000.0 * p.ev_lcos_amt_kwh
        else:
            lcos = np.zeros(state.population_size)

        total_cost = soc_cost + lcos  # (pop,)
        return total_cost[:, np.newaxis]  # (pop, 1)

    # ------------------------------------------------------------------
    # Instruction Extraction
    # ------------------------------------------------------------------

    def extract_instructions(
        self,
        state: EVChargerBatchState,
        individual_index: int,
        instruction_context: InstructionContext | None = None,
    ) -> list[EnergyManagementInstruction]:
        """Emit one ``OMBCInstruction`` per time step.

        Operation modes:
        - ``"CHARGE"``       — EV connected, active charging (``charge_factor > 0``).
        - ``"STANDBY"``      — EV connected but idle (``charge_factor == 0``).
        - ``"DEEP_STANDBY"`` — EV absent.

        The ``operation_mode_factor`` carries ``charge_factor`` so that
        downstream consumers (dashboards, logs) can see the scheduled
        power fraction.
        """
        instructions: list[EnergyManagementInstruction] = []

        for t in range(state.horizon):
            dt = state.step_times[t]
            cf = float(state.charge_factors[individual_index, t])
            connected_t = float(state.connected[t])

            if connected_t == 0.0:
                mode = "DEEP_STANDBY"
                factor = 0.0
            elif cf > 0.0:
                mode = "CHARGE"
                factor = cf
            else:
                mode = "STANDBY"
                factor = 0.0

            instructions.append(
                OMBCInstruction(
                    resource_id=self.device_id,
                    execution_time=dt,
                    operation_mode_id=mode,
                    operation_mode_factor=factor,
                )
            )

        return instructions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _combined_efficiency(
        charge_w: np.ndarray,
        p: EVChargerParam,
    ) -> np.ndarray:
        """Compute the DC-cable-to-battery combined efficiency per individual.

        ``charge_w`` represents the DC power at the cable (after the wall-box
        AC→DC conversion stage).  This function computes the fraction of that
        DC power that is ultimately stored in the EV battery, accounting for:

        - ``hold_time_efficiency`` — DC-side scheduling loss: the charger may
          operate at the wrong power level for part of the time step.
        - ``control_efficiency``   — DC-side tracking loss: the charger cannot
          perfectly follow a varying power source within a step.
        - ``ev_charger_efficiency_{low|high}`` — EV onboard charger loss
          (regime selected by ``high_current_threshold_w``).
        - ``ev_battery_efficiency`` — battery acceptance loss.

        Note: ``charger_efficiency`` (AC→DC at the wall box) is intentionally
        excluded here because it is applied in the AC wall-draw formula
        (``ac_w = charge_w / charger_efficiency + standby``).  Including it
        here would double-count the loss.

        Args:
            charge_w: DC cable power per individual [W], shape ``(pop,)``.
            p:        Immutable device parameters.

        Returns:
            Combined DC-to-battery efficiency array, shape ``(pop,)``.
        """
        ev_onboard_eff = np.where(
            charge_w >= p.high_current_threshold_w,
            p.ev_charger_efficiency_high,
            p.ev_charger_efficiency_low,
        )
        return (
            ev_onboard_eff
            * p.ev_battery_efficiency
            * p.hold_time_efficiency
            * p.control_efficiency
        )
