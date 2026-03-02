"""Hybrid inverter energy device for the genetic2 simulation framework.

Models three inverter topologies as a single device class:

- **Battery inverter** (``BATTERY``) — manages a battery pack on the AC bus. No PV input.
- **Solar inverter** (``SOLAR``) — converts DC PV power to AC. No battery.
- **Hybrid inverter** (``HYBRID``) — manages both a battery pack and PV input behind a single
    AC port.

All three types share one bidirectional AC port. Net AC power is positive
when consuming from the bus (charging) and negative when injecting into
the bus (PV production or battery discharging), following the load
convention used throughout the genetic2 framework.

Genome structure
----------------
Each time step is encoded in a **single gene** whose integer part selects
the operation mode and whose fractional part encodes the operation mode
factor:

    gene = mode_position + factor          where factor ∈ [0, 1)

``mode_position`` is the 0-based index into
``HybridInverterParam.valid_modes``.  ``factor`` is the same quantity as
before: a fraction of the 1C rate for charge/discharge, or irrelevant
(zeroed during repair) for OFF and PV modes.

This gives a genome of size ``horizon`` for all inverter types, with
lower bound ``0.0`` and upper bound ``n_modes`` (upper boundary value
decodes to the last valid mode with factor 0.0 after position clamping).

Decoding one gene:
    position = clip(floor(gene), 0, n_modes − 1)
    mode     = valid_modes[position]
    factor   = gene − floor(gene)           # fractional part ∈ [0, 1)

Lamarckian repair writes the repaired values back:
    genome[t] = position + repaired_factor

PV forecast handling
--------------------
PV production is not stored directly in the parameter object.
Instead, ``pv_power_w_key`` references a time-series in the
``SimulationContext``. The aligned PV forecast is resolved
during ``setup_run()``.

Initial SoC handling
--------------------
The initial battery state of charge is not stored as a fixed value.
Instead, ``battery_initial_soc_factor_key`` references a scalar
in the ``SimulationContext`` measurement store. The resolved value
is converted to Wh during ``setup_run()``.

Power and energy sign convention
---------------------------------
- Positive power/energy → consuming from the AC bus (load, charging).
- Negative power/energy → injecting into the AC bus (PV, discharging).

Factor-to-power convention
---------------------------
``factor`` is a fraction of the 1C rate:

    charge_power_w    = factor × battery_capacity_wh   [W]
    stored_wh         = charge_power_w × step_h × ac_to_battery_efficiency
    discharge_power_w = factor × battery_capacity_wh   [W]  (battery side)
    drawn_wh          = discharge_power_w × step_h              (battery side)
    delivered_ac_wh   = drawn_wh × battery_to_ac_efficiency

This is step-size-independent: the factor always represents a fraction
of the 1C rate, not a fraction of capacity per step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from akkudoktoreos.core.emplan import EnergyManagementInstruction, OMBCInstruction
from akkudoktoreos.devices.devicesabc import (
    EnergyDevice,
    EnergyPort,
    PortDirection,
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
# Enumerations
# ============================================================


class InverterType(IntEnum):
    """Physical topology of the inverter."""

    BATTERY = 0  # Battery only, no PV
    SOLAR = 1  # PV only, no battery
    HYBRID = 2  # PV + battery behind one AC port


class InverterMode(IntEnum):
    """Operation mode as stored in ``HybridInverterBatchState.modes``."""

    OFF = 0
    PV = 1
    CHARGE = 2
    DISCHARGE = 3


# ============================================================
# Immutable parameter dataclass
# ============================================================


@dataclass(frozen=True, slots=True)
class HybridInverterParam:
    """Immutable parameters for a hybrid inverter device."""

    device_id: str
    port_id: str
    bus_id: str
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
    battery_charge_rates: tuple[float, ...] | None

    # Continuous rate bounds
    battery_min_charge_rate: float
    battery_max_charge_rate: float
    battery_min_discharge_rate: float
    battery_max_discharge_rate: float

    # SoC constraints
    battery_min_soc_factor: float
    battery_max_soc_factor: float
    battery_initial_soc_factor_key: str

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
                        "use OFF mode to stop charging, not rate 0.0"
                    )

    @property
    def valid_modes(self) -> tuple[InverterMode, ...]:
        if self.inverter_type == InverterType.BATTERY:
            return (InverterMode.OFF, InverterMode.CHARGE, InverterMode.DISCHARGE)
        if self.inverter_type == InverterType.SOLAR:
            return (InverterMode.OFF, InverterMode.PV)
        return (InverterMode.OFF, InverterMode.PV, InverterMode.CHARGE, InverterMode.DISCHARGE)

    @property
    def n_modes(self) -> int:
        return len(self.valid_modes)

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
    """Mutable batch state for ``HybridInverterDevice``."""

    modes: np.ndarray  # (population_size, horizon)  int8
    factors: np.ndarray  # (population_size, horizon)  float64
    soc_wh: np.ndarray  # (population_size, horizon)  float64
    ac_power_w: np.ndarray  # (population_size, horizon)  float64
    population_size: int
    horizon: int
    step_times: tuple[DateTime, ...]


# ============================================================
# Device implementation
# ============================================================


class HybridInverterDevice(EnergyDevice):
    """Vectorised hybrid inverter energy device.

    Genome encoding
    ---------------
    Each time step is represented by a **single gene** in ``[0, n_modes)``.
    The integer part (floor) selects the mode position in
    ``param.valid_modes``; the fractional part is the operation mode
    factor (charge/discharge rate as fraction of 1C).

    OFF and PV modes ignore the factor (it is zeroed during repair).
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
        self._pv_power_w: np.ndarray | None = None  # set for SOLAR/HYBRID
        self._battery_initial_soc_wh: float | None = None  # set for BATTERY/HYBRID

        # Cached mode-value lookup array, built once in __init__
        self._mode_values: np.ndarray = np.array(
            [m.value for m in param.valid_modes], dtype=np.int8
        )

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return (
            EnergyPort(
                port_id=self.param.port_id,
                bus_id=self.param.bus_id,
                direction=PortDirection.BIDIRECTIONAL,
            ),
        )

    @property
    def objective_names(self) -> list[str]:
        return []

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
            soc_factor = context.resolve_measurement(self.param.battery_initial_soc_factor_key)
            if not (0.0 <= soc_factor <= 1.0):
                raise ValueError(
                    f"{self.device_id}: initial SoC factor {soc_factor} "
                    "outside allowed bounds [0.0, 1.0]."
                )
            self._battery_initial_soc_wh = soc_factor * self.param.battery_capacity_wh

    def genome_requirements(self) -> GenomeSlice:
        """Return genome slice descriptor.

        The genome has exactly ``horizon`` genes, one per time step.
        Each gene encodes both mode and factor:

            gene ∈ [0.0, n_modes)
            position = floor(gene)       → index into valid_modes
            factor   = gene − position   → fractional part ∈ [0, 1)
        """
        if self._num_steps is None:
            raise RuntimeError("Call setup_run() before genome_requirements().")
        n = self._num_steps
        p = self.param
        return GenomeSlice(
            start=0,
            size=n,
            lower_bound=np.zeros(n),
            upper_bound=np.full(n, float(p.n_modes)),
        )

    # ------------------------------------------------------------------
    # Batch Lifecycle
    # ------------------------------------------------------------------

    def create_batch_state(self, population_size: int, horizon: int) -> HybridInverterBatchState:
        """Allocate batch state; soc_wh pre-filled with resolved initial SoC (0 for SOLAR)."""
        if self._step_times is None:
            raise RuntimeError("Call setup_run() before create_batch_state().")
        initial_soc = (
            self._battery_initial_soc_wh if self._battery_initial_soc_wh is not None else 0.0
        )
        return HybridInverterBatchState(
            modes=np.zeros((population_size, horizon), dtype=np.int8),
            factors=np.zeros((population_size, horizon), dtype=np.float64),
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
            genome_batch: Float array of shape ``(population_size, horizon)``.
                Each column ``t`` is a single gene per individual encoding
                both mode position (integer part) and factor (fractional
                part).  The array is updated in place with the repaired
                values (Lamarckian repair).

        Returns:
            The mutated ``genome_batch`` with repaired gene values.
        """
        if self._step_interval_sec is None:
            raise RuntimeError("Call setup_run() before apply_genome_batch().")
        p = self.param

        # Initial SoC: 0.0 for SOLAR (no battery), resolved Wh for BATTERY/HYBRID.
        soc = np.full(
            state.population_size,
            self._battery_initial_soc_wh if self._battery_initial_soc_wh is not None else 0.0,
        )

        pv_power_w = self._pv_power_w  # None for BATTERY

        for t in range(state.horizon):
            gene_t = genome_batch[:, t]  # shape (pop,)

            mode_t, raw_factor_t = self._decode_gene(gene_t)
            state.modes[:, t] = mode_t

            factor_t = self._repair_factor(mode_t, raw_factor_t, soc)
            state.factors[:, t] = factor_t

            pv_dc_w = (
                float(np.clip(pv_power_w[t], p.pv_min_power_w, p.pv_max_power_w))
                if p.inverter_type in (InverterType.SOLAR, InverterType.HYBRID)
                else 0.0
            )

            state.ac_power_w[:, t] = self._compute_ac_power(mode_t, factor_t, pv_dc_w)

            soc = self._advance_soc(soc, mode_t, factor_t, pv_dc_w)
            state.soc_wh[:, t] = soc

            # Lamarckian write-back: integer part = mode position, frac = factor.
            position = self._mode_position(mode_t)  # 0-based index into valid_modes
            genome_batch[:, t] = position.astype(np.float64) + factor_t

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
        return np.zeros((state.population_size, 0))

    # ------------------------------------------------------------------
    # S2 instruction extraction
    # ------------------------------------------------------------------

    def extract_instructions(
        self,
        state: HybridInverterBatchState,
        individual_index: int,
    ) -> list[EnergyManagementInstruction]:
        instructions: list[EnergyManagementInstruction] = []
        for mode_int, factor, dt in zip(
            state.modes[individual_index],
            state.factors[individual_index],
            state.step_times,
        ):
            mode = InverterMode(int(mode_int))
            instructions.append(
                OMBCInstruction(
                    resource_id=self.device_id,
                    execution_time=dt,
                    operation_mode_id=mode.name,
                    operation_mode_factor=float(factor),
                )
            )
        return instructions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_gene(self, gene: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode a single-gene vector into mode integers and raw factors.

        Args:
            gene: Float array of shape ``(population_size,)``.
                The integer part (floor) is the mode position index into
                ``param.valid_modes``; the fractional part is the raw
                (unrepaired) factor.

        Returns:
            mode: ``int8`` array of shape ``(population_size,)`` containing
                the actual ``InverterMode`` integer values.
            factor: ``float64`` array of shape ``(population_size,)``
                containing the fractional part of the gene, in ``[0, 1)``.
        """
        floored = np.floor(gene)
        position = np.clip(floored, 0, self.param.n_modes - 1).astype(np.intp)
        mode = self._mode_values[position]
        factor = gene - floored
        # Clamp factor to [0, 1) to guard against floating-point overshoot at ub.
        factor = np.clip(factor, 0.0, 1.0 - np.finfo(float).eps)
        return mode, factor

    def _mode_position(self, mode: np.ndarray) -> np.ndarray:
        """Return the 0-based position index in ``valid_modes`` for each mode value.

        Args:
            mode: ``int8`` array of mode *values* (InverterMode integers).

        Returns:
            ``int64`` array of the same shape with the position indices.
        """
        # Build inverse lookup: mode_value → position_index.
        # valid_modes has at most 4 entries so a linear search is fine.
        positions = np.zeros_like(mode, dtype=np.int64)
        for pos, m in enumerate(self.param.valid_modes):
            positions[mode == m.value] = pos
        return positions

    def _repair_factor(
        self,
        modes: np.ndarray,
        raw_factors: np.ndarray,
        soc_wh: np.ndarray,
    ) -> np.ndarray:
        """Repair factor genes for one time step across all individuals."""
        if self._step_interval_sec is None:
            raise RuntimeError("Step interval is None.")
        p = self.param
        step_h = self._step_interval_sec / 3600.0
        factor = raw_factors.copy()

        off_or_pv = (modes == InverterMode.OFF) | (modes == InverterMode.PV)
        factor[off_or_pv] = 0.0

        charge_mask = modes == InverterMode.CHARGE
        if charge_mask.any():
            f = factor[charge_mask]
            f = np.clip(f, p.battery_min_charge_rate, p.battery_max_charge_rate)
            if p.battery_charge_rates is not None:
                rates = np.array(p.battery_charge_rates)
                idx = np.argmin(np.abs(f[:, np.newaxis] - rates[np.newaxis, :]), axis=1)
                f = rates[idx]
            headroom_wh = p.battery_max_soc_wh - soc_wh[charge_mask]
            max_factor = np.clip(
                headroom_wh / (p.battery_capacity_wh * step_h * p.ac_to_battery_efficiency),
                0.0,
                p.battery_max_charge_rate,
            )
            f = np.minimum(f, max_factor)
            f[f < p.battery_min_charge_rate] = 0.0
            factor[charge_mask] = f

        discharge_mask = modes == InverterMode.DISCHARGE
        if discharge_mask.any():
            f = factor[discharge_mask]
            f = np.clip(f, p.battery_min_discharge_rate, p.battery_max_discharge_rate)
            available_wh = soc_wh[discharge_mask] - p.battery_min_soc_wh
            max_factor = np.clip(
                available_wh / (p.battery_capacity_wh * step_h),
                0.0,
                p.battery_max_discharge_rate,
            )
            f = np.minimum(f, max_factor)
            f[f < p.battery_min_discharge_rate] = 0.0
            factor[discharge_mask] = f

        return factor

    def _compute_ac_power(
        self,
        modes: np.ndarray,
        factors: np.ndarray,
        pv_dc_w: float,
    ) -> np.ndarray:
        """Compute net AC port power for one step across all individuals."""
        p = self.param
        pop = len(modes)
        ac = np.zeros(pop)

        off_mask = modes == InverterMode.OFF
        ac[off_mask] = p.off_state_power_consumption_w

        pv_mask = modes == InverterMode.PV
        if pv_mask.any():
            ac[pv_mask] = -pv_dc_w * p.pv_to_ac_efficiency + p.on_state_power_consumption_w

        charge_mask = modes == InverterMode.CHARGE
        if charge_mask.any():
            f = factors[charge_mask]
            active = f > 0.0
            charge_power_w = f * p.battery_capacity_wh
            gross_ac_w = charge_power_w / p.ac_to_battery_efficiency

            if p.inverter_type == InverterType.HYBRID and pv_dc_w > 0.0:
                pv_for_battery_w = pv_dc_w * p.pv_to_battery_efficiency
                pv_used_w = np.minimum(pv_for_battery_w, charge_power_w)
                ac_offset_w = pv_used_w / p.ac_to_battery_efficiency
                net_ac_demand = gross_ac_w - ac_offset_w
                pv_surplus_dc = pv_dc_w - pv_used_w / p.pv_to_battery_efficiency
                pv_surplus_ac = pv_surplus_dc * p.pv_to_ac_efficiency
                ac_w = np.where(
                    active,
                    net_ac_demand - pv_surplus_ac + p.on_state_power_consumption_w,
                    p.off_state_power_consumption_w,
                )
            else:
                ac_w = np.where(
                    active,
                    gross_ac_w + p.on_state_power_consumption_w,
                    p.off_state_power_consumption_w,
                )
            ac[charge_mask] = ac_w

        discharge_mask = modes == InverterMode.DISCHARGE
        if discharge_mask.any():
            f = factors[discharge_mask]
            active = f > 0.0
            battery_ac_w = f * p.battery_capacity_wh * p.battery_to_ac_efficiency
            total_inject_w = np.where(active, battery_ac_w, 0.0)
            if p.inverter_type == InverterType.HYBRID:
                total_inject_w = total_inject_w + pv_dc_w * p.pv_to_ac_efficiency
            ac[discharge_mask] = -total_inject_w + p.on_state_power_consumption_w

        return ac

    def _advance_soc(
        self,
        soc_wh: np.ndarray,
        modes: np.ndarray,
        factors: np.ndarray,
        pv_dc_w: float,
    ) -> np.ndarray:
        """Advance SoC by one time step.

        For SOLAR type, no battery exists: soc stays at 0.0 and the clamp
        is a no-op because battery_min_soc_wh == battery_max_soc_wh == 0.
        """
        p = self.param
        if p.inverter_type == InverterType.SOLAR:
            # Solar inverter does not have battery
            return soc_wh
        if self._step_interval_sec is None:
            raise RuntimeError("Step interval is None.")
        step_h = self._step_interval_sec / 3600.0
        new_soc = soc_wh.copy()

        charge_mask = (modes == InverterMode.CHARGE) & (factors > 0.0)
        if charge_mask.any():
            charge_power_w = factors[charge_mask] * p.battery_capacity_wh
            if p.inverter_type == InverterType.HYBRID and pv_dc_w > 0.0:
                pv_for_battery_w = pv_dc_w * p.pv_to_battery_efficiency
                pv_used_w = np.minimum(pv_for_battery_w, charge_power_w)
                ac_used_w = np.maximum(0.0, charge_power_w - pv_used_w)
                stored_wh = (pv_used_w + ac_used_w * p.ac_to_battery_efficiency) * step_h
            else:
                stored_wh = charge_power_w * p.ac_to_battery_efficiency * step_h
            new_soc[charge_mask] += stored_wh

        discharge_mask = (modes == InverterMode.DISCHARGE) & (factors > 0.0)
        if discharge_mask.any():
            discharge_power_w = factors[discharge_mask] * p.battery_capacity_wh
            new_soc[discharge_mask] -= discharge_power_w * step_h

        # For SOLAR: min_soc_wh == max_soc_wh == 0.0, clamp is a no-op.
        return np.clip(new_soc, p.battery_min_soc_wh, p.battery_max_soc_wh)

    def _compute_min_energy(
        self,
        state: HybridInverterBatchState,
        step_h: float,
    ) -> np.ndarray:
        """Compute minimum acceptable AC energy per individual per step."""
        p = self.param
        pv_power_w = self._pv_power_w  # None for BATTERY

        min_e = np.zeros_like(state.ac_power_w)

        charge_mask = state.modes == InverterMode.CHARGE
        if charge_mask.any():
            min_charge_ac_w = (
                p.battery_min_charge_rate * p.battery_capacity_wh / p.ac_to_battery_efficiency
            )
            if p.inverter_type == InverterType.HYBRID and pv_power_w is not None:
                pv_clipped = np.clip(pv_power_w, p.pv_min_power_w, p.pv_max_power_w)
                pv_for_battery_w = pv_clipped * p.pv_to_battery_efficiency
                min_charge_battery_w = p.battery_min_charge_rate * p.battery_capacity_wh
                pv_used = np.minimum(pv_for_battery_w, min_charge_battery_w)
                ac_offset_w = pv_used / p.ac_to_battery_efficiency
                net_min_ac_w = np.maximum(0.0, min_charge_ac_w - ac_offset_w)
                pop_idx, step_idx = np.where(charge_mask)
                min_e[pop_idx, step_idx] = net_min_ac_w[step_idx] * step_h
            else:
                min_e[charge_mask] = min_charge_ac_w * step_h

        discharge_mask = state.modes == InverterMode.DISCHARGE
        if discharge_mask.any():
            min_inject_ac_w = (
                p.battery_min_discharge_rate * p.battery_capacity_wh * p.battery_to_ac_efficiency
            )
            if p.inverter_type == InverterType.HYBRID and pv_power_w is not None:
                pv_clipped = np.clip(pv_power_w, p.pv_min_power_w, p.pv_max_power_w)
                pv_inject_w = pv_clipped * p.pv_to_ac_efficiency
                total_min_inject_w = min_inject_ac_w + pv_inject_w
                pop_idx, step_idx = np.where(discharge_mask)
                min_e[pop_idx, step_idx] = -total_min_inject_w[step_idx] * step_h
            else:
                min_e[discharge_mask] = -min_inject_ac_w * step_h

        return min_e
