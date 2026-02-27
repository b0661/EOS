"""Topology validation for the energy simulation.

This module validates the structural consistency of a device-bus topology
before any simulation runs. Validation is performed once at engine
initialisation time via ``TopologyValidator.validate()``.

What is validated
-----------------
**Per device:**
    - All ``port_id`` values are unique within the device.
    - Every port's ``bus_id`` references a bus that exists in the
      provided bus list.

**Per bus:**
    - Every bus that has at least one connected port has at least one
      source (or bidirectional) port and at least one sink (or
      bidirectional) port. A bus with no connected ports is not
      validated — it is legal to register buses that are not yet used.
    - If the bus declares a ``constraint``, the actual number of sink
      ports and source ports must not exceed ``max_sinks`` and
      ``max_sources`` respectively.

What is NOT validated
---------------------
- Carrier matching between ports and buses. Carrier type is a property
  of the bus; ports connect to a bus by ``bus_id`` and inherit its
  carrier implicitly. No per-port carrier field exists.
- Power balance or flow feasibility — that is the responsibility of the
  arbitrator at runtime.
- Whether all registered buses are reachable — unused buses are ignored.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from akkudoktoreos.devices.devicesabc import EnergyBus, EnergyPort, PortDirection


class TopologyValidationError(Exception):
    """Raised when the device-bus topology is structurally invalid.

    The error message identifies the offending device, port, or bus and
    describes the specific constraint that was violated.
    """


class TopologyValidator:
    """Static validator for energy device-bus topologies.

    All methods are static — no instantiation is required or useful.
    Call ``TopologyValidator.validate(devices, buses)`` directly.
    """

    @staticmethod
    def validate(
        devices: Iterable,
        buses: Iterable[EnergyBus],
    ) -> bool:
        """Validate the structural consistency of a device-bus topology.

        Iterates over all devices and their ports, checks bus references
        and port uniqueness, then verifies bus-level source/sink balance
        and optional connection constraints.

        Args:
            devices: Iterable of device objects. Each device must expose
                ``device_id: str`` and ``ports: Iterable[EnergyPort]``.
            buses: Iterable of ``EnergyBus`` instances representing all
                buses in the simulation.

        Returns:
            ``True`` if the topology is valid.

        Raises:
            TopologyValidationError: On the first structural violation
                found. The message identifies the offending element and
                the violated constraint.
        """
        bus_map: dict[str, EnergyBus] = {b.bus_id: b for b in buses}

        # Accumulate ports per bus for bus-level checks after device iteration.
        # Only buses that receive at least one port end up in this mapping.
        ports_per_bus: dict[str, list[EnergyPort]] = defaultdict(list)

        # ------------------------------------------------------------------
        # Per-device and per-port validation
        # ------------------------------------------------------------------
        for device in devices:
            seen_port_ids: set[str] = set()

            for port in device.ports:
                # Unique port_id within this device
                if port.port_id in seen_port_ids:
                    raise TopologyValidationError(
                        f"Device '{device.device_id}': duplicate port_id "
                        f"'{port.port_id}'. Port IDs must be unique per device."
                    )
                seen_port_ids.add(port.port_id)

                # Referenced bus must exist
                if port.bus_id not in bus_map:
                    raise TopologyValidationError(
                        f"Device '{device.device_id}', port '{port.port_id}': "
                        f"references unknown bus '{port.bus_id}'. "
                        f"Known buses: {sorted(bus_map.keys())}"
                    )

                ports_per_bus[port.bus_id].append(port)

        # ------------------------------------------------------------------
        # Bus-level validation
        # Only buses with at least one connected port are checked.
        # ------------------------------------------------------------------
        for bus_id, ports in ports_per_bus.items():
            sources = [
                p
                for p in ports
                if p.direction in (PortDirection.SOURCE, PortDirection.BIDIRECTIONAL)
            ]
            sinks = [
                p for p in ports if p.direction in (PortDirection.SINK, PortDirection.BIDIRECTIONAL)
            ]

            if not sources:
                raise TopologyValidationError(
                    f"Bus '{bus_id}' has no source ports. "
                    "At least one port with direction SOURCE or BIDIRECTIONAL "
                    "is required."
                )
            if not sinks:
                raise TopologyValidationError(
                    f"Bus '{bus_id}' has no sink ports. "
                    "At least one port with direction SINK or BIDIRECTIONAL "
                    "is required."
                )

            bus = bus_map[bus_id]
            if bus.constraint:
                if bus.constraint.max_sinks is not None and len(sinks) > bus.constraint.max_sinks:
                    raise TopologyValidationError(
                        f"Bus '{bus_id}' exceeds max_sinks="
                        f"{bus.constraint.max_sinks} "
                        f"(found {len(sinks)} sink ports)."
                    )
                if (
                    bus.constraint.max_sources is not None
                    and len(sources) > bus.constraint.max_sources
                ):
                    raise TopologyValidationError(
                        f"Bus '{bus_id}' exceeds max_sources="
                        f"{bus.constraint.max_sources} "
                        f"(found {len(sources)} source ports)."
                    )

        return True
