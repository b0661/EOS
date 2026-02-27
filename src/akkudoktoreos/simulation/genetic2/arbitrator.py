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
    HIGH = 1  # Must-charge windows, heating demand
    NORMAL = 2  # Standard battery cycling
    LOW = 3  # Opportunistic charging, deferrable loads


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
        min_energy_wh: Minimum acceptable energy, shape
            ``(population_size, horizon)``. Only applied to consumers
            (positive values). Ignored for producers.
    """

    port_index: int
    energy_wh: np.ndarray  # shape: (population_size, horizon)
    min_energy_wh: np.ndarray  # shape: (population_size, horizon)


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
    granted_wh: np.ndarray  # shape: (population_size, horizon)


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

        For each bus, computes a per-individual, per-timestep supply ratio
        and scales consumer grants proportionally. Producers always receive
        their full requested amount. The ``min_energy_wh`` floor is applied
        to consumers after proportional scaling.

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

        port_indices = np.array([p.port_index for p in all_ports])
        bus_indices = self._topology.port_to_bus[port_indices]

        # ----------------------------------------------------
        # 2. Allocate result tensor — same shape as energy
        # ----------------------------------------------------

        granted = np.zeros_like(energy)

        # ----------------------------------------------------
        # 3. Bus-level balancing
        #
        # All array operations broadcast over (population_size, horizon).
        # axis=0 always reduces the port dimension.
        # ----------------------------------------------------

        for bus in range(self._topology.num_buses):
            mask = bus_indices == bus  # (num_ports,) bool
            if not np.any(mask):
                continue

            # bus_energy: (n_bus_ports, population_size, horizon)
            bus_energy = energy[mask]
            bus_min = min_energy[mask]

            # Boolean masks — same shape as bus_energy
            consumers = bus_energy > 0  # (n_bus_ports, population_size, horizon)
            producers = bus_energy < 0

            # Sum over the port axis → (population_size, horizon)
            total_demand = np.sum(np.where(consumers, bus_energy, 0.0), axis=0)
            total_supply = -np.sum(np.where(producers, bus_energy, 0.0), axis=0)

            # supply_ratio: (population_size, horizon)
            # Default to 1.0 where demand is zero to avoid division warnings.
            supply_ratio = np.divide(
                total_supply,
                total_demand,
                out=np.ones_like(total_demand),
                where=total_demand > 0,
            )
            supply_ratio = np.clip(supply_ratio, 0.0, 1.0)

            # Broadcast supply_ratio over the port axis:
            # (population_size, horizon) → (1, population_size, horizon)
            ratio = supply_ratio[np.newaxis]

            # Consumer grants scaled proportionally
            consumer_grant = np.where(consumers, bus_energy * ratio, 0.0)

            # Producer grants — always full injection
            producer_grant = np.where(producers, bus_energy, 0.0)

            bus_granted = consumer_grant + producer_grant

            # Apply min_energy floor to consumers only
            bus_granted = np.where(
                consumers,
                np.maximum(bus_granted, bus_min),
                bus_granted,
            )

            granted[mask] = bus_granted

        # ----------------------------------------------------
        # 4. Reassemble per-device grants
        # ----------------------------------------------------

        device_map: dict[int, list[PortGrant]] = {}

        for idx, owner in enumerate(port_owner):
            port_grant = PortGrant(
                port_index=port_indices[idx],
                granted_wh=granted[idx],  # (population_size, horizon)
            )
            device_map.setdefault(owner, []).append(port_grant)

        return tuple(
            DeviceGrant(
                device_index=device_idx,
                port_grants=tuple(grants),
            )
            for device_idx, grants in device_map.items()
        )
