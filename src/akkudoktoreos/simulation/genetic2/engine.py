"""Energy simulation engine.

The engine coordinates the per-step simulation loop across all registered
devices.  It is intentionally stateless between ``simulate()`` calls —
all mutable state lives inside the devices themselves.

The engine never accesses device internals.  It only calls the public
device methods defined in ``EnergyDevice``:

* ``reset()``           — restore physical state before a run.
* ``apply_genome()``    — decode genome into schedule (once per run).
* ``request()``         — Phase 1: declare resource needs per step.
* ``simulate_step()``   — Phase 2: act on arbitrated grant per step.
* ``repair_genome()``   — post-run: optionally propose genome corrections.

Genome dispatch vs apply_genome()
----------------------------------
Splitting genome handling across two calls gives each collaborator a clear
responsibility:

``GenomeAssembler.dispatch(genome, registry)``
    Slices the flat optimizer genome, validates bounds, and writes each raw
    slice into the device via ``EnergyDevice.store_genome(slice)``.  This
    is called by the **optimizer** before ``simulate()``; the engine does not
    participate.

``EnergySimulationEngine.simulate(inputs)``
    Calls ``EnergyDevice.apply_genome(stored_slice, step_times)`` during its
    setup phase.  The device reads its stored raw slice and the step schedule
    together, enabling time-dependent decoding (e.g. "only schedule charging
    during steps whose ``step_time`` falls in off-peak hours").

This separation means the engine never holds genome values and the optimizer
never needs to know the step schedule.

Step-based design
-----------------
The simulation operates on uniform time steps of arbitrary duration
(e.g. 15 min, 30 min, 1 h).  ``SimulationInput.step_times`` defines the
wall-clock schedule; ``step_interval`` defines the duration of each step.
Both are forwarded to ``apply_genome()`` so devices can scale energy
quantities correctly (``energy_wh = power_w * step_interval.total_seconds() / 3600``).

Genome repair
-------------
After all steps complete, the engine calls ``repair_genome()`` on every
device.  Each proposal is validated against the device's cached ``GenomeSlice``
and defensively copied before being stored in ``SimulationResult.repairs``.
Invalid proposals are discarded with a logged warning — a misbehaving device
must never abort an optimization run.  The genetic optimizer reads ``repairs``
after fitness evaluation and decides whether to apply them to the population.
"""

import logging
import warnings
from datetime import datetime

import numpy as np

from akkudoktoreos.devices.genetic2.base import EnergyDevice, GenomeSlice
from akkudoktoreos.simulation.genetic2.arbitrator import (
    ArbitratorBase,
    PriorityArbitrator,
)
from akkudoktoreos.simulation.genetic2.flows import ResourceGrant
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.result import (
    DeviceStepState,
    GenomeRepairRecord,
    SimulationResult,
    StepResult,
)
from akkudoktoreos.simulation.genetic2.timeseries import SimulationInput

logger = logging.getLogger(__name__)


class EnergySimulationEngine:
    """Stateless energy flow simulation engine supporting multiple devices of each type.

    The engine runs a three-phase protocol per step:

    **Phase 1 — Request:** every device declares what resources it wants
    for the current step via ``request(step_index)``.

    **Phase 2 — Arbitrate:** the arbitrator sees all requests simultaneously
    and returns a ``ResourceGrant`` for each device.

    **Phase 3 — Simulate:** each device calls ``simulate_step(grant)`` and
    returns its actual ``EnergyFlows`` for the step.

    After all steps, the engine calls ``repair_genome()`` on every device,
    validates each proposal, and collects accepted ones in
    ``SimulationResult.repairs``.

    Adding a new device type requires no engine changes — implement the
    ``EnergyDevice`` interface, register it, and the engine picks it up
    automatically.

    Args:
        registry (DeviceRegistry): All devices participating in simulation.
        arbitrator (ArbitratorBase | None): Resource allocation strategy.
            Defaults to ``PriorityArbitrator`` if None.

    Example::

        # Optimizer side: write genome values into devices
        assembler = GenomeAssembler(registry)
        assembler.dispatch(genome, registry)   # calls device.store_genome() per device

        # Engine side: decode schedules and run simulation
        engine = EnergySimulationEngine(registry)
        result = engine.simulate(inputs)       # calls device.apply_genome() internally
        print(result.net_balance_eur)

        # Genetic engine: apply any device-proposed repairs to the population
        for repair in result.repairs:
            assembler.set_slice(repair.device_id, genome, repair.repaired_slice)
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        arbitrator: ArbitratorBase | None = None,
    ) -> None:
        self._registry = registry
        self._arbitrator = arbitrator or PriorityArbitrator()
        # Genome requirements cached per simulate() call, used by _collect_repairs().
        # Keyed by device_id; only devices with non-None requirements are included.
        self._genome_reqs: dict[str, GenomeSlice] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self, inputs: SimulationInput) -> SimulationResult:
        """Run a full simulation over all steps defined in *inputs*.

        Lifecycle per call
        ------------------
        1. **reset_all()** — restore every device to its initial physical
           state.  Schedule arrays survive because ``reset()`` must not clear
           them (see ``EnergyDevice`` docstring).
        2. **_build_genome_req_cache()** — query genome requirements from all
           devices for later use in repair validation.
        3. **_apply_genomes()** — call ``apply_genome(stored_slice, step_times)``
           on every device, triggering schedule decoding.  Each device reads
           the raw slice stored by the preceding ``GenomeAssembler.dispatch()``
           call and uses ``step_times`` for time-dependent scheduling.
        4. **Step loop** — for each step: request → arbitrate → simulate.
        5. **_collect_repairs()** — validate and copy genome repair proposals.

        Args:
            inputs (SimulationInput): Step schedule, time-series data, and
                optional external device schedules for this run.

        Returns:
            SimulationResult: Per-step detail, aggregate metrics, and any
                genome repair proposals from devices.
        """
        # 1. Restore physical state. Schedule arrays are repopulated in step 3.
        self._registry.reset_all()

        # 2. Cache genome requirements for repair validation later.
        self._genome_reqs = self._build_genome_req_cache(inputs)

        # 3. Trigger schedule decoding in all devices.
        self._apply_genomes(inputs)

        # 4. Step loop — state carries forward between steps.
        result = SimulationResult()
        for step_index, step_time in enumerate(inputs.step_times):
            result.steps.append(self._simulate_step(step_index, step_time, inputs))

        # 5. Collect repair proposals.
        result.repairs = self._collect_repairs()

        return result

    # ------------------------------------------------------------------
    # Private helpers — lifecycle
    # ------------------------------------------------------------------

    def _build_genome_req_cache(self, inputs: SimulationInput) -> dict[str, GenomeSlice]:
        """Query and cache genome requirements for all genome-bearing devices.

        Called once at the start of ``simulate()`` so that ``_collect_repairs()``
        can validate repair proposals without re-querying devices (whose state
        may have changed after the simulation loop).

        Only devices whose ``genome_requirements()`` returns a non-None
        ``GenomeSlice`` are included.

        Args:
            inputs (SimulationInput): Provides ``num_steps`` and
                ``step_interval`` for the ``genome_requirements()`` call.

        Returns:
            dict[str, GenomeSlice]: Device IDs mapped to their GenomeSlice.
        """
        cache: dict[str, GenomeSlice] = {}
        for device in self._registry.all_of_type(EnergyDevice):
            req = device.genome_requirements(
                num_steps=inputs.num_steps,
                step_interval=inputs.step_interval,
            )
            if req is not None:
                cache[device.device_id] = req
        return cache

    def _apply_genomes(self, inputs: SimulationInput) -> None:
        """Trigger schedule decoding in every device via ``apply_genome()``.

        The raw genome slices were written into devices by
        ``GenomeAssembler.dispatch()`` (which calls ``device.store_genome(slice)``).
        This method calls ``device.apply_genome(stored_slice, step_times)``
        so each device can decode its stored raw values into a per-step
        schedule, using ``step_times`` for time-of-day constraints.

        Devices with no genome (not in ``self._genome_reqs``) receive an
        empty array for the slice.  Their ``apply_genome()`` implementation
        must handle ``len(genome_slice) == 0`` gracefully (typically a no-op
        on the slice, but still using ``step_times`` if needed).

        Args:
            inputs (SimulationInput): Provides ``step_times`` to forward.
        """
        empty_slice = np.empty(0)
        for device in self._registry.all_of_type(EnergyDevice):
            # Retrieve the stored raw slice.  genome_slice is empty for
            # devices that have no genome requirement.
            req = self._genome_reqs.get(device.device_id)
            genome_slice = device.get_stored_genome() if req is not None else empty_slice
            device.apply_genome(genome_slice, inputs.step_times)

    # ------------------------------------------------------------------
    # Private helpers — step loop
    # ------------------------------------------------------------------

    def _simulate_step(
        self,
        step_index: int,
        step_time: datetime,
        inputs: SimulationInput,
    ) -> StepResult:
        """Simulate a single step across all devices.

        Implements the request → arbitrate → simulate protocol for one step.

        Args:
            step_index (int): Zero-based step index.
            step_time (datetime): Wall-clock time at the start of this step.
            inputs (SimulationInput): Full simulation inputs.

        Returns:
            StepResult: Aggregated flows and device states for this step.
        """
        devices = list(self._registry.all_of_type(EnergyDevice))

        # Phase 1: all devices declare their needs simultaneously
        requests = [device.request(step_index) for device in devices]

        # Phase 2: arbitrator allocates resources
        grants = self._arbitrator.arbitrate(requests)

        # Phase 3: each device acts on its grant and updates internal state
        device_states: dict[str, DeviceStepState] = {}
        for device in devices:
            grant = grants.get(
                device.device_id,
                ResourceGrant.idle(device.device_id),
            )
            flows = device.simulate_step(grant)
            device_states[device.device_id] = DeviceStepState(
                device_id=device.device_id,
                flows=flows,
                curtailed=grant.curtailed,
            )

        return self._aggregate(step_index, step_time, device_states, inputs)

    def _aggregate(
        self,
        step_index: int,
        step_time: datetime,
        device_states: dict[str, DeviceStepState],
        inputs: SimulationInput,
    ) -> StepResult:
        """Aggregate per-device flows into system-level totals for one step.

        The grid acts as the balancing element: net surplus becomes feed-in,
        net deficit becomes grid import.

        AC power sign convention (from ``EnergyFlows``):
            Positive = device provides AC to system (source, reduces import).
            Negative = device consumes AC from system (sink, increases import).

        Args:
            step_index (int): Current step index.
            step_time (datetime): Wall-clock time for this step.
            device_states (dict[str, DeviceStepState]): All device states.
            inputs (SimulationInput): Provides base load and price arrays.

        Returns:
            StepResult: Aggregated result for this step.
        """
        base_load_wh = inputs.load_wh[step_index]
        total_load_wh = base_load_wh
        total_losses_wh = 0.0
        net_ac_wh = 0.0  # positive = surplus (feed-in), negative = deficit

        for state in device_states.values():
            flows = state.flows
            # Accumulate device-reported fixed load contributions
            total_load_wh += flows.load_wh
            total_losses_wh += flows.losses_wh
            # Sum AC flows: sources positive, sinks negative
            net_ac_wh += flows.ac_power_wh

        # Base household load is a sink — subtract from net AC balance
        net_ac_wh -= base_load_wh

        # Grid absorbs the imbalance
        feedin_wh = max(net_ac_wh, 0.0)
        grid_import_wh = max(-net_ac_wh, 0.0)

        # Self-consumption: generation consumed directly within the system
        total_generation_wh = sum(
            s.flows.ac_power_wh for s in device_states.values() if s.flows.is_ac_source
        )
        self_consumption_wh = max(total_generation_wh - feedin_wh, 0.0)

        price = inputs.electricity_price[step_index]
        tariff = inputs.feed_in_tariff[step_index]

        return StepResult(
            step_index=step_index,
            step_time=step_time,
            total_load_wh=total_load_wh,
            feedin_wh=feedin_wh,
            grid_import_wh=grid_import_wh,
            total_losses_wh=total_losses_wh,
            self_consumption_wh=self_consumption_wh,
            cost_eur=grid_import_wh * price,
            revenue_eur=feedin_wh * tariff,
            devices=device_states,
        )

    # ------------------------------------------------------------------
    # Private helpers — genome repair
    # ------------------------------------------------------------------

    def _collect_repairs(self) -> list[GenomeRepairRecord]:
        """Call ``repair_genome()`` on every device and collect valid proposals.

        Four validation rules are applied to each proposal:

        1. **Skip absent or unchanged:** ``None`` or ``changed=False`` → no-op.
        2. **Skip genome-less devices:** a device whose
           ``genome_requirements()`` returned ``None`` should not return a
           repair proposal; warn and discard rather than raise.
        3. **Validate length and bounds:** call ``GenomeSlice.validate()`` on
           ``repaired_slice``; discard invalid proposals with a warning.
        4. **Defensive copy:** ``repaired_slice.copy()`` before storing —
           the device may reuse its internal buffer on the next call.

        A logged warning rather than an exception is used for rules 2 and 3
        so that a single misbehaving device cannot abort an entire
        optimization run.

        Returns:
            list[GenomeRepairRecord]: Validated, copied repair records ready
                for the genetic engine to write back into the population.
        """
        records: list[GenomeRepairRecord] = []

        for device in self._registry.all_of_type(EnergyDevice):
            proposal = device.repair_genome()

            # Rule 1: skip absent or unchanged proposals
            if proposal is None or not proposal.changed:
                continue

            # Rule 2: skip genome-less devices
            genome_req = self._genome_reqs.get(device.device_id)
            if genome_req is None:
                warnings.warn(
                    f"Device '{device.device_id}' proposed a genome repair but "
                    "genome_requirements() returned None — this device has no "
                    "genome.  Proposal discarded.",
                    stacklevel=2,
                )
                continue

            # Rule 3: validate length and element bounds
            try:
                genome_req.validate(proposal.repaired_slice)
            except ValueError as exc:
                warnings.warn(
                    f"Device '{device.device_id}' returned an invalid genome "
                    f"repair proposal (discarded): {exc}",
                    stacklevel=2,
                )
                continue

            # Rule 4: defensive copy
            records.append(
                GenomeRepairRecord(
                    device_id=device.device_id,
                    repaired_slice=proposal.repaired_slice.copy(),
                )
            )
            logger.debug(
                "Accepted genome repair from '%s' (%d slots).",
                device.device_id,
                len(proposal.repaired_slice),
            )

        return records
