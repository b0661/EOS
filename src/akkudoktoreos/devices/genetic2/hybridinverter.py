"""Hybrid inverter energy device for the genetic2 simulation framework.

Models three inverter topologies as a single device class:

- **Battery inverter** (``BATTERY``) — manages a battery pack on the AC
  bus. No PV input.
- **Solar inverter** (``SOLAR``) — converts DC PV power to AC. No battery.
- **Hybrid inverter** (``HYBRID``) — manages both a battery pack and PV
  input behind a single AC port.

All three types share one bidirectional AC port. Net AC power is positive
when consuming from the bus (charging) and negative when injecting into
the bus (PV production or battery discharging), following the load
convention used throughout the genetic2 framework.

Genome structure
----------------
Genes are interleaved per time step as ``[mode₀, factor₀, mode₁, factor₁, …]``
for BATTERY and HYBRID types. SOLAR uses only ``[mode₀, mode₁, …]`` because
PV output is fully determined by the forecast — there is nothing to control.

Mode genes are float-valued in the global genome but semantically discrete.
``_repair_mode`` maps the float to the nearest *position index* in
``HybridInverterParam.valid_modes`` (0-based), then looks up the actual
``InverterMode`` integer at that position. This is necessary because
BATTERY type skips InverterMode.PV (value 1), so a raw round-and-clip
against the enum values would produce invalid modes.

Factor genes encode charge or discharge power as a fraction of the 1C
rate: ``factor = 1.0`` means power = ``battery_capacity_wh`` W.

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
from akkudoktoreos.utils.datetimeutil import DateTime


# ============================================================
# Enumerations
# ============================================================


class InverterType(IntEnum):
    """Physical topology of the inverter."""

    BATTERY = 0  # Battery only, no PV
    SOLAR = 1    # PV only, no battery
    HYBRID = 2   # PV + battery behind one AC port


class InverterMode(IntEnum):
    """Operation mode as stored in ``HybridInverterBatchState.modes``.

    Integer values are contiguous from 0 and shared across all inverter
    types. Not every mode is valid for every type — see
    ``HybridInverterParam.valid_modes``.

    The values stored in ``state.modes`` are always ``InverterMode``
    integers (not raw position indices). ``_repair_mode`` handles the
    mapping from genome float → position index → InverterMode value.
    """

    OFF = 0
    PV = 1
    CHARGE = 2
    DISCHARGE = 3


# ============================================================
# Immutable parameter dataclass
# ============================================================


@dataclass(frozen=True, slots=True)
class HybridInverterParam:
    """Immutable parameters for a hybrid inverter device.

    Frozen, slotted, and hashable — safe as a dictionary or cache key
    inside the genetic algorithm. Carries no mutable state or simulation
    logic.

    Battery-specific fields (``battery_*``, ``ac_to_battery_efficiency``,
    ``battery_to_ac_efficiency``) are only validated for BATTERY and HYBRID
    types. For SOLAR inverters pass ``battery_capacity_wh=0.0`` and
    ``battery_charge_rates=None``; those fields are ignored at runtime.

    PV-specific fields (``pv_*``) are only validated for SOLAR and HYBRID
    types. For BATTERY inverters the fields are present but unused.

    Attributes
    ----------
    device_id :
        Unique identifier for this device instance.
    port_id :
        Identifier of the single AC port. Referenced by the bus
        arbitrator's port-index map.
    bus_id :
        ID of the AC bus this inverter connects to.
    inverter_type :
        ``BATTERY``, ``SOLAR``, or ``HYBRID``.
    off_state_power_consumption_w :
        Standby power drawn from the AC bus in OFF mode [W]. ≥ 0.
    on_state_power_consumption_w :
        Internal auxiliary AC draw when the inverter is active (PV,
        CHARGE, or DISCHARGE mode) [W]. Represents control electronics
        and cooling, independent of the energy conversion flow. ≥ 0.
    pv_to_ac_efficiency :
        Fraction of DC PV power delivered as AC output (0–1].
        Used for SOLAR type and HYBRID PV/DISCHARGE modes.
    pv_to_battery_efficiency :
        Fraction of DC PV power stored in the battery when PV is routed
        through the battery path (0–1]. Used in HYBRID CHARGE mode.
    pv_max_power_w :
        Maximum DC PV power [W]. ``pv_prediction_w`` values are clipped
        to this limit.
    pv_min_power_w :
        Minimum DC PV contribution [W]. Values below this are treated as
        zero. Must be ≤ ``pv_max_power_w``.
    pv_prediction_w :
        Forecast DC PV power for each simulation step [W].
        Length must equal the horizon passed to ``setup_run``.
    ac_to_battery_efficiency :
        Fraction of AC input power stored in the battery during CHARGE
        mode (0–1]. Accounts for rectifier and charger losses.
    battery_to_ac_efficiency :
        Fraction of battery DC power delivered as AC output during
        DISCHARGE mode (0–1]. Accounts for inverter losses.
    battery_capacity_wh :
        Total usable battery capacity [Wh]. Must be > 0 for BATTERY and
        HYBRID types.
    battery_charge_rates :
        Optional tuple of discrete charge rate fractions (0–1] the
        battery supports. If set, CHARGE-mode factors are snapped to the
        nearest value. Must not contain 0.0 (use OFF mode instead).
        ``None`` means continuous control within
        [``battery_min_charge_rate``, ``battery_max_charge_rate``].
    battery_min_charge_rate :
        Minimum charge rate fraction when CHARGE is active [0–1].
        If the repaired factor falls below this, it is zeroed.
    battery_max_charge_rate :
        Maximum allowed charge rate fraction [0–1].
    battery_min_discharge_rate :
        Minimum discharge rate fraction when DISCHARGE is active [0–1].
        If the repaired factor falls below this, it is zeroed.
    battery_max_discharge_rate :
        Maximum allowed discharge rate fraction [0–1].
    battery_min_soc_factor :
        Minimum allowed SoC as a fraction of ``battery_capacity_wh``
        [0–1). The SoC is clamped to this floor after each step.
    battery_max_soc_factor :
        Maximum allowed SoC as a fraction of ``battery_capacity_wh``
        (0–1]. The SoC is clamped to this ceiling after each step.
    battery_initial_soc_factor :
        SoC fraction at the start of each simulation run.
        Must be in [``battery_min_soc_factor``, ``battery_max_soc_factor``].
    """

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
    pv_prediction_w: tuple[float, ...]

    # AC ↔ Battery conversion (BATTERY, HYBRID)
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
    battery_initial_soc_factor: float

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
                    "battery charge rates must satisfy "
                    "0 <= min_charge_rate <= max_charge_rate <= 1"
                )
            if not (
                0 <= self.battery_min_discharge_rate
                <= self.battery_max_discharge_rate <= 1
            ):
                raise ValueError(
                    "battery discharge rates must satisfy "
                    "0 <= min_discharge_rate <= max_discharge_rate <= 1"
                )
            if not (0 <= self.battery_min_soc_factor < self.battery_max_soc_factor <= 1):
                raise ValueError(
                    "SoC factors must satisfy 0 <= min_soc_factor < max_soc_factor <= 1"
                )
            if not (
                self.battery_min_soc_factor
                <= self.battery_initial_soc_factor
                <= self.battery_max_soc_factor
            ):
                raise ValueError(
                    "battery_initial_soc_factor must be in "
                    "[battery_min_soc_factor, battery_max_soc_factor]"
                )
            if self.battery_charge_rates is not None:
                if len(self.battery_charge_rates) == 0:
                    raise ValueError("battery_charge_rates must not be empty if set")
                if any(r <= 0 or r > 1 for r in self.battery_charge_rates):
                    raise ValueError(
                        "battery_charge_rates values must be in (0, 1] — "
                        "use OFF mode to stop charging, not rate 0.0"
                    )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def valid_modes(self) -> tuple[InverterMode, ...]:
        """Operation modes valid for this inverter type, in position order.

        The *position* in this tuple (0, 1, 2, …) is the value encoded in
        the genome's mode genes. ``_repair_mode`` converts from genome
        float → position index → ``InverterMode`` value.

        This indirection is necessary because BATTERY type omits
        InverterMode.PV (value 1), so genome positions 0/1/2 map to
        OFF/CHARGE/DISCHARGE (values 0/2/3), not 0/1/2.
        """
        if self.inverter_type == InverterType.BATTERY:
            return (InverterMode.OFF, InverterMode.CHARGE, InverterMode.DISCHARGE)
        if self.inverter_type == InverterType.SOLAR:
            return (InverterMode.OFF, InverterMode.PV)
        # HYBRID
        return (InverterMode.OFF, InverterMode.PV, InverterMode.CHARGE, InverterMode.DISCHARGE)

    @property
    def has_factor_gene(self) -> bool:
        """True for BATTERY and HYBRID (which have a battery to control)."""
        return self.inverter_type != InverterType.SOLAR

    @property
    def n_modes(self) -> int:
        """Number of valid operation modes for this inverter type."""
        return len(self.valid_modes)

    @property
    def battery_min_soc_wh(self) -> float:
        """Minimum SoC in Wh."""
        return self.battery_min_soc_factor * self.battery_capacity_wh

    @property
    def battery_max_soc_wh(self) -> float:
        """Maximum SoC in Wh."""
        return self.battery_max_soc_factor * self.battery_capacity_wh

    @property
    def battery_initial_soc_wh(self) -> float:
        """Initial SoC in Wh."""
        return self.battery_initial_soc_factor * self.battery_capacity_wh


# ============================================================
# Mutable batch state
# ============================================================


@dataclass
class HybridInverterBatchState:
    """Mutable batch state for ``HybridInverterDevice``.

    Created fresh each generation by ``create_batch_state``.
    Never shared between devices or between generations.

    Attributes
    ----------
    modes :
        Decoded ``InverterMode`` integers per individual per step,
        shape ``(population_size, horizon)``, dtype int8.
        Written by ``apply_genome_batch``.
    factors :
        Repaired factor per individual per step,
        shape ``(population_size, horizon)``, dtype float64.
        0.0 for OFF and PV modes.
        Written by ``apply_genome_batch``.
    soc_wh :
        Battery SoC in Wh *after* each step,
        shape ``(population_size, horizon)``, dtype float64.
        Column ``t`` holds the SoC after completing step ``t``.
        Written by ``apply_genome_batch``.
    ac_power_w :
        Net AC port power per individual per step [W],
        shape ``(population_size, horizon)``, dtype float64.
        Positive = consuming from the AC bus.
        Negative = injecting into the AC bus.
        Set by ``apply_genome_batch``, then updated by ``apply_device_grant``.
    population_size :
        Number of individuals in this batch.
    horizon :
        Number of simulation time steps.
    step_times :
        Ordered ``DateTime`` timestamps, length == horizon.
        Available to ``compute_cost`` and ``extract_instructions``.
    """

    modes: np.ndarray       # (population_size, horizon)  int8
    factors: np.ndarray     # (population_size, horizon)  float64
    soc_wh: np.ndarray      # (population_size, horizon)  float64
    ac_power_w: np.ndarray  # (population_size, horizon)  float64
    population_size: int
    horizon: int
    step_times: tuple[DateTime, ...]


# ============================================================
# Device implementation
# ============================================================


class HybridInverterDevice(EnergyDevice):
    """Vectorised hybrid inverter energy device.

    Implements the full ``EnergyDevice`` lifecycle directly — not via
    ``SingleStateEnergyDevice`` — because the genome encodes two genes per
    step (mode + factor) for BATTERY/HYBRID types, which is incompatible
    with the single-gene-per-step assumption of that base class.

    Parameters
    ----------
    param :
        Immutable device parameters. Validated at construction.
    device_index :
        Index used by the bus arbitrator to identify this device's port
        requests. Must be unique across all devices in the same engine.
    port_index :
        Index of this device's single AC port in the arbitrator's
        port-topology mapping.
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
        self._step_interval: float | None = None

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        """Single bidirectional AC port."""
        return (
            EnergyPort(
                port_id=self.param.port_id,
                bus_id=self.param.bus_id,
                direction=PortDirection.BIDIRECTIONAL,
            ),
        )

    @property
    def objective_names(self) -> list[str]:
        """No direct cost objectives.

        Energy cost is attributed to the grid connection device, which
        observes net AC bus flow after arbitration. Subclass and override
        to add battery degradation cost or other inverter-specific
        objectives; also extend ``objective_names`` to match.
        """
        return []

    # ------------------------------------------------------------------
    # Structure Phase
    # ------------------------------------------------------------------

    def setup_run(
        self,
        step_times: tuple[DateTime, ...],
        step_interval: float,
    ) -> None:
        """Store horizon and step interval; validate PV prediction length.

        Args:
            step_times: Ordered timestamps, length == horizon.
            step_interval: Fixed duration of each simulation step [s].

        Raises:
            ValueError: If ``pv_prediction_w`` length doesn't match the
                horizon for SOLAR or HYBRID inverters.
        """
        horizon = len(step_times)
        if self.param.inverter_type in (InverterType.SOLAR, InverterType.HYBRID):
            n_pv = len(self.param.pv_prediction_w)
            if n_pv != horizon:
                raise ValueError(
                    f"HybridInverterDevice '{self.device_id}': "
                    f"pv_prediction_w has {n_pv} entries but horizon is {horizon}."
                )
        self._step_times = step_times
        self._num_steps = horizon
        self._step_interval = step_interval

    def genome_requirements(self) -> GenomeSlice:
        """Declare the genome slice for this inverter.

        Gene layout per time step:

        - BATTERY / HYBRID: ``[mode_t, factor_t]`` → ``2 × horizon`` total.
        - SOLAR: ``[mode_t]`` → ``horizon`` total.

        Bounds:

        - Mode: ``[0.0, float(n_modes - 1)]`` — GA treats this as continuous;
          repair rounds to the nearest integer position index.
        - Factor: ``[0.0, 1.0]``.

        Returns:
            ``GenomeSlice`` with per-gene bounds set for the GA initialiser
            and mutation operators.
        """
        assert self._num_steps is not None, "Call setup_run() before genome_requirements()."
        n = self._num_steps
        p = self.param

        mode_lb = np.zeros(n)
        mode_ub = np.full(n, float(p.n_modes - 1))

        if not p.has_factor_gene:
            return GenomeSlice(
                start=0,
                size=n,
                lower_bound=mode_lb,
                upper_bound=mode_ub,
            )

        # Interleaved [mode_t, factor_t] layout
        lower = np.empty(2 * n)
        upper = np.empty(2 * n)
        lower[0::2] = mode_lb   # even positions: mode genes
        upper[0::2] = mode_ub
        lower[1::2] = 0.0       # odd positions: factor genes
        upper[1::2] = 1.0

        return GenomeSlice(
            start=0,
            size=2 * n,
            lower_bound=lower,
            upper_bound=upper,
        )

    # ------------------------------------------------------------------
    # Batch Lifecycle
    # ------------------------------------------------------------------

    def create_batch_state(
        self,
        population_size: int,
        horizon: int,
    ) -> HybridInverterBatchState:
        """Allocate batch state arrays.

        ``soc_wh`` is pre-filled with ``battery_initial_soc_wh`` so the
        first repair step sees the correct starting SoC. For SOLAR
        inverters this field is allocated but unused.

        Args:
            population_size: Number of individuals.
            horizon: Number of simulation time steps.
        """
        assert self._step_times is not None, "Call setup_run() before create_batch_state()."
        return HybridInverterBatchState(
            modes=np.zeros((population_size, horizon), dtype=np.int8),
            factors=np.zeros((population_size, horizon), dtype=np.float64),
            soc_wh=np.full(
                (population_size, horizon),
                self.param.battery_initial_soc_wh,
                dtype=np.float64,
            ),
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

        The outer loop iterates over time steps (sequential because SoC
        at step ``t`` depends on the SoC at ``t-1``). The population axis
        is fully vectorised — no Python loop over individuals.

        Repair is Lamarckian: the corrected gene values are written back
        into ``genome_batch`` so that the repaired phenotype becomes the
        individual's new genotype for the next generation.

        Args:
            state: Mutable batch state, modified in-place.
            genome_batch: Shape ``(population_size, genome_size)`` where
                ``genome_size`` is ``2 × horizon`` (BATTERY/HYBRID) or
                ``horizon`` (SOLAR).

        Returns:
            ``genome_batch`` modified in-place with repaired values.
        """
        assert self._step_interval is not None, "Call setup_run() before apply_genome_batch()."
        p = self.param

        # Views into the genome for BATTERY/HYBRID; synthesised zeros for SOLAR.
        if not p.has_factor_gene:
            raw_modes = genome_batch             # (pop, horizon) — view
            raw_factors = np.zeros_like(genome_batch)
        else:
            raw_modes = genome_batch[:, 0::2]    # (pop, horizon) — view
            raw_factors = genome_batch[:, 1::2]  # (pop, horizon) — view

        soc = np.full(state.population_size, p.battery_initial_soc_wh)

        for t in range(state.horizon):
            # Clip PV forecast to physical bounds (scalar).
            pv_dc_w = (
                float(np.clip(p.pv_prediction_w[t], p.pv_min_power_w, p.pv_max_power_w))
                if p.inverter_type in (InverterType.SOLAR, InverterType.HYBRID)
                else 0.0
            )

            mode_t = self._repair_mode(raw_modes[:, t])
            state.modes[:, t] = mode_t

            factor_t = self._repair_factor(mode_t, raw_factors[:, t], soc)
            state.factors[:, t] = factor_t

            state.ac_power_w[:, t] = self._compute_ac_power(mode_t, factor_t, pv_dc_w)

            soc = self._advance_soc(soc, mode_t, factor_t, pv_dc_w)
            state.soc_wh[:, t] = soc

            # Write repaired values back (Lamarckian repair).
            if not p.has_factor_gene:
                genome_batch[:, t] = mode_t.astype(np.float64)
            else:
                genome_batch[:, 2 * t] = mode_t.astype(np.float64)
                genome_batch[:, 2 * t + 1] = factor_t

        return genome_batch

    # ------------------------------------------------------------------
    # Arbitration
    # ------------------------------------------------------------------

    def build_device_request(
        self,
        state: HybridInverterBatchState,
    ) -> DeviceRequest:
        """Build the AC bus energy request from the simulated net AC power.

        ``min_energy_wh`` is the minimum feasible AC energy for the chosen
        mode after efficiency adjustment. The arbitrator uses this to
        prevent partial grants from creating infeasible battery states.

        Sign convention: positive = consuming; negative = injecting.

        Args:
            state: Batch state after ``apply_genome_batch``.

        Returns:
            ``DeviceRequest`` with a single ``PortRequest`` for the AC port.
        """
        assert self._step_interval is not None, "Call setup_run() before build_device_request()."
        step_h = self._step_interval / 3600.0
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

    def apply_device_grant(
        self,
        state: HybridInverterBatchState,
        grant: DeviceGrant,
    ) -> None:
        """Overwrite ``ac_power_w`` with the arbitrated grant.

        Args:
            state: Mutable batch state.
            grant: Arbitrated energy grant from the bus arbitrator.
        """
        assert self._step_interval is not None, "Call setup_run() before apply_device_grant()."
        step_h = self._step_interval / 3600.0
        state.ac_power_w[:] = grant.port_grants[0].granted_wh / step_h

    def compute_cost(
        self,
        state: HybridInverterBatchState,
    ) -> np.ndarray:
        """Return a zero-column cost matrix.

        Energy cost attribution belongs to the grid connection device.
        Subclass and override to add battery degradation or other
        inverter-specific objectives; also extend ``objective_names``.

        Returns:
            Shape ``(population_size, 0)``.
        """
        return np.zeros((state.population_size, 0))

    # ------------------------------------------------------------------
    # S2 instruction extraction
    # ------------------------------------------------------------------

    def extract_instructions(
        self,
        state: HybridInverterBatchState,
        individual_index: int,
    ) -> list[EnergyManagementInstruction]:
        """Extract one OMBC instruction per step for the specified individual.

        OMBC (Operation Mode Based Control) is the natural S2 control type
        here because the genome maps directly to named operation modes with
        a continuous factor — exactly the S2 OMBC semantic.

        Args:
            state: Batch state (typically from a single-individual
                re-evaluation by ``GeneticOptimizer.extract_best_instructions``).
            individual_index: Row index into the batch arrays.

        Returns:
            List of ``OMBCInstruction`` objects, one per step, ordered
            earliest to latest.
        """
        instructions: list[EnergyManagementInstruction] = []
        modes_row = state.modes[individual_index]     # (horizon,)
        factors_row = state.factors[individual_index] # (horizon,)

        for mode_int, factor, dt in zip(modes_row, factors_row, state.step_times):
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
    # Internal helpers — repair
    # ------------------------------------------------------------------

    def _repair_mode(self, raw_modes: np.ndarray) -> np.ndarray:
        """Map float mode genes to valid ``InverterMode`` integers.

        Steps:

        1. Clip to ``[0, n_modes - 1]``.
        2. Round to the nearest integer *position index*.
        3. Look up the ``InverterMode`` value at that position in
           ``param.valid_modes``.

        The position-index indirection is necessary because BATTERY type
        omits InverterMode.PV (value 1), so raw genome positions 0/1/2
        must map to OFF/CHARGE/DISCHARGE (values 0/2/3), not 0/1/2.

        Args:
            raw_modes: Float mode genes, shape ``(population_size,)``.

        Returns:
            ``InverterMode`` integers, shape ``(population_size,)``,
            dtype int8.
        """
        mode_values = np.array(
            [m.value for m in self.param.valid_modes], dtype=np.int8
        )
        pos = np.clip(np.round(raw_modes), 0, self.param.n_modes - 1).astype(np.intp)
        return mode_values[pos]

    def _repair_factor(
        self,
        modes: np.ndarray,
        raw_factors: np.ndarray,
        soc_wh: np.ndarray,
    ) -> np.ndarray:
        """Repair factor genes for one time step across all individuals.

        All operations are vectorised over the population axis.

        Rules
        -----
        OFF / PV
            Factor forced to 0.0 (no battery control in these modes).
        CHARGE
            1. Clip to ``[battery_min_charge_rate, battery_max_charge_rate]``.
            2. Snap to nearest discrete rate if ``battery_charge_rates`` set.
            3. SoC cap: reduce factor so that
               ``factor × capacity_wh × step_h × ac_to_battery_efficiency``
               does not exceed available SoC headroom.
            4. Zero out if resulting factor < ``battery_min_charge_rate``
               (headroom exhausted — treated as standby in
               ``_compute_ac_power``).
        DISCHARGE
            1. Clip to ``[battery_min_discharge_rate, battery_max_discharge_rate]``.
            2. SoC floor: reduce factor so that
               ``factor × capacity_wh × step_h`` does not exceed
               available SoC above the minimum.
            3. Zero out if resulting factor < ``battery_min_discharge_rate``
               (battery depleted).

        Args:
            modes: InverterMode integers, shape ``(population_size,)``.
            raw_factors: Float factor genes, shape ``(population_size,)``.
            soc_wh: Current SoC in Wh, shape ``(population_size,)``.

        Returns:
            Repaired factor array, shape ``(population_size,)``.
        """
        assert self._step_interval is not None
        p = self.param
        step_h = self._step_interval / 3600.0
        factor = raw_factors.copy()

        # OFF and PV: no battery control.
        off_or_pv = (modes == InverterMode.OFF) | (modes == InverterMode.PV)
        factor[off_or_pv] = 0.0

        # --- CHARGE ---
        charge_mask = modes == InverterMode.CHARGE
        if charge_mask.any():
            f = factor[charge_mask]

            # 1. Rate bounds.
            f = np.clip(f, p.battery_min_charge_rate, p.battery_max_charge_rate)

            # 2. Discrete rate snapping.
            if p.battery_charge_rates is not None:
                rates = np.array(p.battery_charge_rates)
                idx = np.argmin(
                    np.abs(f[:, np.newaxis] - rates[np.newaxis, :]), axis=1
                )
                f = rates[idx]

            # 3. SoC cap.
            # stored_wh = factor × capacity_wh × step_h × ac_to_battery_efficiency
            # Constraint: stored_wh ≤ headroom_wh
            # → factor ≤ headroom_wh / (capacity_wh × step_h × efficiency)
            headroom_wh = p.battery_max_soc_wh - soc_wh[charge_mask]
            max_factor = np.clip(
                headroom_wh
                / (p.battery_capacity_wh * step_h * p.ac_to_battery_efficiency),
                0.0,
                p.battery_max_charge_rate,
            )
            f = np.minimum(f, max_factor)

            # 4. Zero out if below minimum.
            f[f < p.battery_min_charge_rate] = 0.0

            factor[charge_mask] = f

        # --- DISCHARGE ---
        discharge_mask = modes == InverterMode.DISCHARGE
        if discharge_mask.any():
            f = factor[discharge_mask]

            # 1. Rate bounds.
            f = np.clip(f, p.battery_min_discharge_rate, p.battery_max_discharge_rate)

            # 2. SoC floor.
            # drawn_wh = factor × capacity_wh × step_h  (battery side, pre-efficiency)
            # Constraint: drawn_wh ≤ available_wh
            # → factor ≤ available_wh / (capacity_wh × step_h)
            available_wh = soc_wh[discharge_mask] - p.battery_min_soc_wh
            max_factor = np.clip(
                available_wh / (p.battery_capacity_wh * step_h),
                0.0,
                p.battery_max_discharge_rate,
            )
            f = np.minimum(f, max_factor)

            # 3. Zero out if below minimum.
            f[f < p.battery_min_discharge_rate] = 0.0

            factor[discharge_mask] = f

        return factor

    def _compute_ac_power(
        self,
        modes: np.ndarray,
        factors: np.ndarray,
        pv_dc_w: float,
    ) -> np.ndarray:
        """Compute net AC port power for one step across all individuals.

        Sign convention: positive = consuming, negative = injecting.
        ``on_state_power_consumption_w`` is added as a positive AC draw
        for all active modes (PV, CHARGE, DISCHARGE).

        HYBRID CHARGE
            The battery is charged at ``factor × capacity_wh`` W total.
            PV DC feeds the battery directly via ``pv_to_battery_efficiency``
            (reducing the AC demand). Any PV surplus beyond the battery's
            charge demand is injected onto the AC bus via ``pv_to_ac_efficiency``.

        Args:
            modes: InverterMode integers, shape ``(population_size,)``.
            factors: Repaired factors, shape ``(population_size,)``.
            pv_dc_w: Available DC PV power this step [W] (pre-clipped).

        Returns:
            Net AC power, shape ``(population_size,)`` [W].
        """
        p = self.param
        pop = len(modes)
        ac = np.zeros(pop)

        # OFF
        off_mask = modes == InverterMode.OFF
        ac[off_mask] = p.off_state_power_consumption_w

        # PV: inject all PV onto the AC bus; no battery.
        pv_mask = modes == InverterMode.PV
        if pv_mask.any():
            ac[pv_mask] = (
                -pv_dc_w * p.pv_to_ac_efficiency   # injection (negative)
                + p.on_state_power_consumption_w    # auxiliary draw (positive)
            )

        # CHARGE: consume AC to charge battery.
        charge_mask = modes == InverterMode.CHARGE
        if charge_mask.any():
            f = factors[charge_mask]
            active = f > 0.0

            # Gross battery charge power [W].
            charge_power_w = f * p.battery_capacity_wh
            # Gross AC demand before PV offset.
            gross_ac_w = charge_power_w / p.ac_to_battery_efficiency

            if p.inverter_type == InverterType.HYBRID and pv_dc_w > 0.0:
                # PV feeds battery first via pv_to_battery_efficiency.
                pv_for_battery_w = pv_dc_w * p.pv_to_battery_efficiency
                pv_used_w = np.minimum(pv_for_battery_w, charge_power_w)
                # AC demand is reduced by the PV portion (PV bypasses AC bus).
                ac_offset_w = pv_used_w / p.ac_to_battery_efficiency
                net_ac_demand = gross_ac_w - ac_offset_w
                # Remaining PV surplus injects onto AC bus.
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

        # DISCHARGE: inject battery (+ PV for HYBRID) into AC bus.
        discharge_mask = modes == InverterMode.DISCHARGE
        if discharge_mask.any():
            f = factors[discharge_mask]
            active = f > 0.0

            # AC output from battery.
            battery_ac_w = f * p.battery_capacity_wh * p.battery_to_ac_efficiency
            total_inject_w = np.where(active, battery_ac_w, 0.0)

            if p.inverter_type == InverterType.HYBRID:
                # PV always injects in DISCHARGE mode regardless of factor.
                total_inject_w = total_inject_w + pv_dc_w * p.pv_to_ac_efficiency

            ac[discharge_mask] = (
                -total_inject_w                    # injection (negative)
                + p.on_state_power_consumption_w   # auxiliary draw (positive)
            )

        return ac

    def _advance_soc(
        self,
        soc_wh: np.ndarray,
        modes: np.ndarray,
        factors: np.ndarray,
        pv_dc_w: float,
    ) -> np.ndarray:
        """Advance SoC by one time step for all individuals.

        CHARGE
            ``charge_power_w = factor × battery_capacity_wh`` [W].

            For BATTERY type:
                ``stored_wh = charge_power_w × step_h × ac_to_battery_efficiency``

            For HYBRID type:
                PV contributes first via ``pv_to_battery_efficiency``; AC
                covers the remainder via ``ac_to_battery_efficiency``::

                    pv_used_w    = min(pv_dc_w × pv_to_battery_efficiency,
                                       charge_power_w)
                    ac_used_w    = max(0, charge_power_w - pv_used_w)
                    stored_wh    = (pv_used_w + ac_used_w × ac_eff) × step_h

        DISCHARGE
            ``discharge_power_w = factor × battery_capacity_wh`` [W].
            ``drawn_wh = discharge_power_w × step_h`` (battery side, before
            AC conversion loss).

        Result is clamped to ``[battery_min_soc_wh, battery_max_soc_wh]``.

        Args:
            soc_wh: Current SoC, shape ``(population_size,)`` [Wh].
            modes: InverterMode integers, shape ``(population_size,)``.
            factors: Repaired factors, shape ``(population_size,)``.
            pv_dc_w: Available DC PV power this step [W] (pre-clipped).

        Returns:
            Updated SoC, shape ``(population_size,)`` [Wh].
        """
        assert self._step_interval is not None
        p = self.param
        step_h = self._step_interval / 3600.0
        new_soc = soc_wh.copy()

        # CHARGE (factor > 0 guaranteed by repair)
        charge_mask = (modes == InverterMode.CHARGE) & (factors > 0.0)
        if charge_mask.any():
            charge_power_w = factors[charge_mask] * p.battery_capacity_wh  # [W]

            if p.inverter_type == InverterType.HYBRID and pv_dc_w > 0.0:
                pv_for_battery_w = pv_dc_w * p.pv_to_battery_efficiency
                pv_used_w = np.minimum(pv_for_battery_w, charge_power_w)
                ac_used_w = np.maximum(0.0, charge_power_w - pv_used_w)
                stored_wh = (pv_used_w + ac_used_w * p.ac_to_battery_efficiency) * step_h
            else:
                stored_wh = charge_power_w * p.ac_to_battery_efficiency * step_h

            new_soc[charge_mask] += stored_wh

        # DISCHARGE (factor > 0 guaranteed by repair)
        discharge_mask = (modes == InverterMode.DISCHARGE) & (factors > 0.0)
        if discharge_mask.any():
            discharge_power_w = factors[discharge_mask] * p.battery_capacity_wh  # [W]
            drawn_wh = discharge_power_w * step_h  # battery side, pre-efficiency
            new_soc[discharge_mask] -= drawn_wh

        return np.clip(new_soc, p.battery_min_soc_wh, p.battery_max_soc_wh)

    def _compute_min_energy(
        self,
        state: HybridInverterBatchState,
        step_h: float,
    ) -> np.ndarray:
        """Compute minimum acceptable AC energy per individual per step.

        Used for ``PortRequest.min_energy_wh``. The arbitrator uses this
        to avoid granting less than the minimum feasible power.

        CHARGE
            ``min_energy_wh = min_charge_power_ac × step_h``
            where ``min_charge_power_ac = min_charge_rate × capacity_wh
            / ac_to_battery_efficiency``.
            For HYBRID, PV reduces the minimum AC demand.

        DISCHARGE
            ``min_energy_wh = -(min_discharge_power_ac × step_h)``
            where ``min_discharge_power_ac = min_discharge_rate × capacity_wh
            × battery_to_ac_efficiency``.
            For HYBRID, PV adds to the minimum injection.

        OFF / PV
            0 (no minimum constraint).

        Args:
            state: Batch state after ``apply_genome_batch``.
            step_h: Step duration in hours.

        Returns:
            Shape ``(population_size, horizon)`` [Wh].
        """
        p = self.param
        min_e = np.zeros_like(state.ac_power_w)

        # CHARGE minimum
        charge_mask = state.modes == InverterMode.CHARGE
        if charge_mask.any():
            min_charge_ac_w = (
                p.battery_min_charge_rate * p.battery_capacity_wh
                / p.ac_to_battery_efficiency
            )

            if p.inverter_type == InverterType.HYBRID:
                # PV DC offsets the AC demand (same routing as _compute_ac_power).
                pv_clipped = np.array(
                    [
                        float(np.clip(pv, p.pv_min_power_w, p.pv_max_power_w))
                        for pv in p.pv_prediction_w
                    ]
                )  # (horizon,)
                pv_for_battery_w = pv_clipped * p.pv_to_battery_efficiency
                min_charge_battery_w = p.battery_min_charge_rate * p.battery_capacity_wh
                pv_used = np.minimum(pv_for_battery_w, min_charge_battery_w)
                ac_offset_w = pv_used / p.ac_to_battery_efficiency
                net_min_ac_w = np.maximum(0.0, min_charge_ac_w - ac_offset_w)  # (horizon,)
                pop_idx, step_idx = np.where(charge_mask)
                min_e[pop_idx, step_idx] = net_min_ac_w[step_idx] * step_h
            else:
                min_e[charge_mask] = min_charge_ac_w * step_h

        # DISCHARGE minimum
        discharge_mask = state.modes == InverterMode.DISCHARGE
        if discharge_mask.any():
            min_inject_ac_w = (
                p.battery_min_discharge_rate * p.battery_capacity_wh
                * p.battery_to_ac_efficiency
            )

            if p.inverter_type == InverterType.HYBRID:
                pv_clipped = np.array(
                    [
                        float(np.clip(pv, p.pv_min_power_w, p.pv_max_power_w))
                        for pv in p.pv_prediction_w
                    ]
                )  # (horizon,)
                pv_inject_w = pv_clipped * p.pv_to_ac_efficiency
                total_min_inject_w = min_inject_ac_w + pv_inject_w  # (horizon,)
                pop_idx, step_idx = np.where(discharge_mask)
                min_e[pop_idx, step_idx] = -total_min_inject_w[step_idx] * step_h
            else:
                min_e[discharge_mask] = -min_inject_ac_w * step_h

        return min_e
