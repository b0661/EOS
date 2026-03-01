"""Vectorized multi-bus arbitrator for rolling-horizon simulation.

Design principles:
- Port-based resource exchange
- Bus-level vector balancing over the full population batch
- Immutable request/grant objects
- No carrier logic inside devices
- Optimized for GA evaluation loops

Array shape conventions
-----------------------
All energy arrays carry a leading ``population_size`` axis so the
arbitrator can process an entire population in a single call without
any Python loop over individuals:

    energy_wh:    (population_size, horizon)
    granted_wh:   (population_size, horizon)

Internally the arbitrator stacks port arrays along a leading port axis,
giving working tensors of shape ``(num_ports_on_bus, population_size,
horizon)``. Bus-level sums reduce over ``axis=0``, collapsing the port
dimension and leaving ``(population_size, horizon)`` results. The
``supply_ratio`` broadcast therefore requires a ``newaxis`` insertion
so that ``(population_size, horizon)`` expands to match
``(num_ports_on_bus, population_size, horizon)``.

Slack ports
-----------
A port marked ``is_slack=True`` on its ``PortRequest`` is treated as the
last-resort balancing resource for its bus. The arbitrator uses a
two-phase algorithm per bus:

**Phase 1 — slack injection pre-computation**
    The raw deficit (``max(0, total_non_slack_demand − total_supply)``) is
    computed and the slack injects up to its capacity (``|min_energy_wh|``)
    to cover it. The effective supply available to non-slack consumers is
    ``total_supply + slack_injection``.

**Phase 2 — non-slack proportional settlement**
    Non-slack consumers are served proportionally from the effective
    supply. ``min_energy_wh`` floors are applied after scaling. Producers
    always receive their full requested injection.

**Phase 3 — slack absorb**
    Any remaining surplus (``effective_supply − non_slack_grant_sum``) is
    absorbed by the slack up to its absorb capacity (``energy_wh``). The
    net slack grant is ``slack_absorb − slack_injection``, which is
    positive (absorbing surplus) or negative (injecting to cover deficit).

At most one slack port per bus is supported. If no slack port is present,
the bus is balanced by proportional scaling alone (identical to the
pre-slack algorithm).

Sign convention for slack ports
    ``energy_wh``     (positive) — maximum absorb capacity [Wh]
    ``min_energy_wh`` (negative) — maximum inject capacity [Wh] (stored negative)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np


# ============================================================
# Arbitration Enums
# ============================================================


class ArbitrationPriority(IntEnum):
    """Arbitration priority for resource allocation.

    Lower value means higher priority. Consumers with higher priority
    are served first when supply is insufficient.
    """

    CRITICAL = 0  # Life-safety or contractual must-run loads
    HIGH = 1      # Must-charge windows, heating demand
    NORMAL = 2    # Standard battery cycling
    LOW = 3       # Opportunistic charging, deferrable loads


# ============================================================
# Request / Grant Data Structures
# ============================================================


@dataclass(frozen=True, slots=True)
class PortRequest:
    """Energy request for a single port over the full population and horizon.

    Sign convention:
        positive value  → consume from bus (sink)
        negative value  → inject into bus  (source)

    Attributes:
        port_index: Index identifying this port in the topology mapping.
        energy_wh: Requested energy, shape ``(population_size, horizon)``.
            For a slack port this is the maximum import energy [Wh].
        min_energy_wh: Minimum acceptable energy, shape
            ``(population_size, horizon)``. For non-slack consumers,
            a proportional floor applied after proportional scaling.
            For a slack port this is the maximum export energy [Wh]
            expressed as a negative number — i.e. the most negative
            value the grant may take.
        is_slack: When ``True`` this port is treated as the last-resort
            balancing resource for its bus. It receives the bus residual
            after all non-slack ports have been settled proportionally,
            clamped to ``[min_energy_wh, energy_wh]``. At most one slack
            port per bus is supported. Default ``False``.
    """

    port_index: int
    energy_wh: np.ndarray        # shape: (population_size, horizon)
    min_energy_wh: np.ndarray    # shape: (population_size, horizon)
    is_slack: bool = False


@dataclass(frozen=True, slots=True)
class DeviceRequest:
    """All port requests for one device, covering the full population batch.

    Attributes:
        device_index: Index identifying this device.
        port_requests: Tuple of per-port requests.
    """

    device_index: int
    port_requests: tuple[PortRequest, ...]


@dataclass(frozen=True, slots=True)
class PortGrant:
    """Granted energy for a single port over the full population and horizon.

    Attributes:
        port_index: Index identifying this port.
        granted_wh: Granted energy, shape ``(population_size, horizon)``.
    """

    port_index: int
    granted_wh: np.ndarray       # shape: (population_size, horizon)


@dataclass(frozen=True, slots=True)
class DeviceGrant:
    """All granted energies for one device, covering the full population batch.

    Attributes:
        device_index: Index identifying this device.
        port_grants: Tuple of per-port grants.
    """

    device_index: int
    port_grants: tuple[PortGrant, ...]


# ============================================================
# Bus Topology Mapping
# ============================================================


@dataclass(frozen=True, slots=True)
class BusTopology:
    """Static mapping from port indices to bus indices.

    Attributes:
        port_to_bus: Integer array of shape ``(num_ports,)`` where
            ``port_to_bus[i]`` is the bus index that port ``i``
            connects to.
        num_buses: Total number of buses in the simulation.
    """

    port_to_bus: np.ndarray  # shape: (num_ports,)
    num_buses: int


# ============================================================
# Vectorized Bus Arbitrator
# ============================================================


class VectorizedBusArbitrator:
    """Vectorized per-bus energy balancing across the full population and horizon.

    Processes an entire GA population in a single ``arbitrate()`` call.
    No Python loop over individuals is required.

    Args:
        topology: Static port-to-bus mapping.
        horizon: Number of simulation time steps.
    """

    def __init__(self, topology: BusTopology, horizon: int) -> None:
        self._topology = topology
        self._horizon = horizon

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def arbitrate(
        self,
        device_requests: Sequence[DeviceRequest],
    ) -> tuple[DeviceGrant, ...]:
        """Perform bus arbitration across the full population and horizon.

        Uses a three-phase algorithm per bus when a slack port is present,
        or simple proportional scaling when no slack port exists.

        **No slack port**
            Non-slack consumers are served proportionally from total
            producer supply. ``min_energy_wh`` floors are applied after
            scaling. Producers always receive their full injection.

        **With slack port (three phases)**

        1. *Pre-compute slack injection*: compute the raw deficit
           (``max(0, non_slack_demand − producer_supply)``) and inject
           up to ``|slack.min_energy_wh|`` to cover it.
        2. *Proportional non-slack settlement*: run the proportional
           algorithm using ``producer_supply + slack_injection`` as the
           effective supply. Apply ``min_energy_wh`` floors.
        3. *Slack absorb*: absorb any remaining surplus (``effective_supply
           − non_slack_grants``) up to ``slack.energy_wh``. The net slack
           grant is ``absorb − injection``.

        Args:
            device_requests: All device requests for the current generation.
                Each ``PortRequest.energy_wh`` must have shape
                ``(population_size, horizon)``.

        Returns:
            Tuple of ``DeviceGrant`` objects, one per requesting device.
            Each ``PortGrant.granted_wh`` has shape
            ``(population_size, horizon)``.
            Returns an empty tuple if ``device_requests`` is empty.
        """
        # ----------------------------------------------------
        # 1. Flatten all port requests into indexed arrays
        # ----------------------------------------------------

        all_ports: list[PortRequest] = []
        port_owner: list[int] = []

        for device in device_requests:
            for pr in device.port_requests:
                all_ports.append(pr)
                port_owner.append(device.device_index)

        if not all_ports:
            return ()

        # energy:     (num_ports, population_size, horizon)
        # min_energy: (num_ports, population_size, horizon)
        energy = np.stack([p.energy_wh for p in all_ports])
        min_energy = np.stack([p.min_energy_wh for p in all_ports])
        is_slack = np.array([p.is_slack for p in all_ports], dtype=bool)

        port_indices = np.array([p.port_index for p in all_ports])
        bus_indices = self._topology.port_to_bus[port_indices]

        # ----------------------------------------------------
        # 2. Allocate result tensor — same shape as energy
        # ----------------------------------------------------

        granted = np.zeros_like(energy)

        # ----------------------------------------------------
        # 3. Bus-level balancing (two-phase)
        #
        # All array operations broadcast over (population_size, horizon).
        # axis=0 always reduces the port dimension.
        # ----------------------------------------------------

        for bus in range(self._topology.num_buses):
            bus_mask = bus_indices == bus        # (num_ports,) bool
            if not np.any(bus_mask):
                continue

            # Identify slack and non-slack port positions within bus_mask.
            # bus_mask selects the relevant rows from the flat port arrays.
            bus_is_slack = is_slack[bus_mask]    # (n_bus_ports,) bool
            has_slack = np.any(bus_is_slack)

            bus_energy = energy[bus_mask]        # (n_bus_ports, pop, horizon)
            bus_min = min_energy[bus_mask]

            # ---- Phase 1: non-slack proportional settlement ----

            # Boolean masks over all bus ports — same shape as bus_energy.
            producers = bus_energy < 0           # (n_bus_ports, pop, horizon)

            # Non-slack consumers only participate in phase 1.
            non_slack_rows = ~bus_is_slack       # (n_bus_ports,) bool
            non_slack_consumers = (
                (bus_energy > 0)
                & non_slack_rows[:, np.newaxis, np.newaxis]
            )

            # Total supply from all producers on this bus → (pop, horizon)
            total_supply = -np.sum(np.where(producers, bus_energy, 0.0), axis=0)

            # Total non-slack demand → (pop, horizon)
            total_non_slack_demand = np.sum(
                np.where(non_slack_consumers, bus_energy, 0.0), axis=0
            )

            # Supply ratio for non-slack consumers.
            # Default to 1.0 where demand is zero (no over-grant risk).
            supply_ratio = np.divide(
                total_supply,
                total_non_slack_demand,
                out=np.ones_like(total_supply),
                where=total_non_slack_demand > 0,
            )
            supply_ratio = np.clip(supply_ratio, 0.0, 1.0)

            # Broadcast ratio over the port axis → (1, pop, horizon)
            ratio = supply_ratio[np.newaxis]

            # Non-slack consumer grants — proportionally scaled.
            non_slack_consumer_grant = np.where(
                non_slack_consumers, bus_energy * ratio, 0.0
            )

            # Apply min_energy floor to non-slack consumers only.
            non_slack_consumer_grant = np.where(
                non_slack_consumers,
                np.maximum(non_slack_consumer_grant, bus_min),
                non_slack_consumer_grant,
            )

            # Producer grants — always full injection.
            producer_grant = np.where(producers, bus_energy, 0.0)

            # Accumulate phase 1 grants into the bus result.
            # Slack port rows stay at zero until phase 2.
            bus_granted = non_slack_consumer_grant + producer_grant

            # ---- Phase 2: slack residual assignment ----

            if has_slack:
                # The slack port balances the bus after phase 1.
                #
                # Bus energy balance (all quantities ≥ 0 here for clarity):
                #   surplus  = max(0, total_supply − total_non_slack_demand)
                #   deficit  = max(0, total_non_slack_demand − total_supply)
                #
                # A surplus means producers generated more than non-slack
                # consumers needed → slack absorbs the surplus (positive grant,
                # consuming from the bus in device convention).
                #
                # A deficit means local supply is insufficient → slack injects
                # the shortfall into the bus (negative grant, sourcing onto the
                # bus) so that non-slack consumers can be fully served.
                #
                # After establishing the slack's net contribution we recompute
                # the effective total supply (producers + slack injection) and
                # re-run the non-slack proportional settlement so that consumers
                # benefit from the slack's injection.
                #
                # Slack port bounds from its request:
                #   energy_wh     (positive) = max absorb capacity (e.g. max grid export Wh)
                #   min_energy_wh (negative) = max inject capacity (e.g. max grid import Wh)
                #
                # All arrays are shape (pop, horizon); operations are fully
                # vectorised with no Python loop over individuals.

                slack_idx = np.where(bus_is_slack)[0][0]
                slack_max_absorb = bus_energy[slack_idx]    # (pop, horizon), ≥ 0
                slack_max_inject = -bus_min[slack_idx]      # (pop, horizon), ≥ 0  (sign flip)

                # Raw deficit: how much local producers are short of non-slack demand.
                deficit = np.maximum(0.0, total_non_slack_demand - total_supply)

                # Slack injection to cover deficit, clamped to connection capacity.
                slack_injection = np.minimum(deficit, slack_max_inject)  # (pop, horizon) ≥ 0

                # Effective supply seen by non-slack consumers.
                effective_supply = total_supply + slack_injection        # (pop, horizon)

                # Re-run proportional settlement with effective supply.
                eff_ratio = np.divide(
                    effective_supply,
                    total_non_slack_demand,
                    out=np.ones_like(effective_supply),
                    where=total_non_slack_demand > 0,
                )
                eff_ratio = np.clip(eff_ratio, 0.0, 1.0)

                non_slack_consumer_grant = np.where(
                    non_slack_consumers,
                    bus_energy * eff_ratio[np.newaxis],
                    0.0,
                )
                # Re-apply min_energy floor.
                non_slack_consumer_grant = np.where(
                    non_slack_consumers,
                    np.maximum(non_slack_consumer_grant, bus_min),
                    non_slack_consumer_grant,
                )

                # Surplus: effective supply remaining after serving non-slack consumers.
                non_slack_grant_sum = np.sum(non_slack_consumer_grant, axis=0)
                surplus = np.maximum(0.0, effective_supply - non_slack_grant_sum)

                # Slack absorbs the surplus, clamped to connection capacity.
                slack_absorb = np.minimum(surplus, slack_max_absorb)

                # Net slack grant: absorb is positive, injection is negative.
                slack_grant = slack_absorb - slack_injection             # (pop, horizon)

                # Update bus_granted with the corrected non-slack and slack grants.
                bus_granted = np.where(
                    non_slack_consumers,
                    non_slack_consumer_grant,
                    0.0,
                )
                bus_granted += producer_grant
                bus_granted[slack_idx] = slack_grant

            granted[bus_mask] = bus_granted

        # ----------------------------------------------------
        # 4. Reassemble per-device grants
        # ----------------------------------------------------

        device_map: dict[int, list[PortGrant]] = {}

        for idx, owner in enumerate(port_owner):
            port_grant = PortGrant(
                port_index=port_indices[idx],
                granted_wh=granted[idx],   # (population_size, horizon)
            )
            device_map.setdefault(owner, []).append(port_grant)

        return tuple(
            DeviceGrant(
                device_index=device_idx,
                port_grants=tuple(grants),
            )
            for device_idx, grants in device_map.items()
        )
