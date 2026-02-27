"""Tests for TopologyValidator.

Covers:
    - Valid topologies pass without error
    - Device port referencing an unknown bus raises
    - Duplicate port_id on the same device raises
    - Bus with no source ports raises
    - Bus with no sink ports raises
    - Bus max_sinks constraint violation raises
    - Bus max_sources constraint violation raises
    - Bus constraints at exact limit pass
    - Bidirectional ports satisfy both source and sink requirements
    - Empty device list passes
    - Device with no ports does not affect bus validation
    - Duplicate port_id on different devices is legal
    - Unconstrained buses accept any number of ports
    - Buses with no connected ports are not validated
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from akkudoktoreos.devices.devicesabc import (
    EnergyBus,
    EnergyBusConstraint,
    EnergyCarrier,
    EnergyPort,
    PortDirection,
)
from akkudoktoreos.simulation.genetic2.topology import (
    TopologyValidationError,
    TopologyValidator,
)

# ============================================================
# Helpers
# ============================================================

def make_device(device_id: str, ports: list[EnergyPort]) -> SimpleNamespace:
    """Create a minimal device stub with a device_id and ports sequence.

    Using SimpleNamespace keeps test devices decoupled from the full
    EnergyDevice ABC — topology validation only inspects device_id and ports.
    """
    return SimpleNamespace(device_id=device_id, ports=ports)


# ============================================================
# Reusable bus definitions
# ============================================================

AC_BUS = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
DC_BUS = EnergyBus(bus_id="bus_dc", carrier=EnergyCarrier.DC)
HEAT_BUS = EnergyBus(bus_id="bus_heat", carrier=EnergyCarrier.HEAT)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture()
def ac_source_port() -> EnergyPort:
    return EnergyPort(port_id="p_src", bus_id="bus_ac", direction=PortDirection.SOURCE)


@pytest.fixture()
def ac_sink_port() -> EnergyPort:
    return EnergyPort(port_id="p_snk", bus_id="bus_ac", direction=PortDirection.SINK)


@pytest.fixture()
def ac_bidi_port() -> EnergyPort:
    return EnergyPort(port_id="p_bidi", bus_id="bus_ac", direction=PortDirection.BIDIRECTIONAL)


@pytest.fixture()
def minimal_valid_topology(ac_source_port, ac_sink_port):
    """Simplest valid topology: one source and one sink device on one AC bus."""
    source_device = make_device("pv", [ac_source_port])
    sink_device = make_device("load", [ac_sink_port])
    return [source_device, sink_device], [AC_BUS]


# ============================================================
# TestTopologyValidatorValidTopologies
# ============================================================

class TestTopologyValidatorValidTopologies:
    def test_minimal_source_and_sink_passes(self, minimal_valid_topology):
        devices, buses = minimal_valid_topology
        assert TopologyValidator.validate(devices, buses) is True

    def test_bidirectional_port_satisfies_source_and_sink(self, ac_bidi_port):
        """A single BIDIRECTIONAL port should satisfy both source and sink."""
        device = make_device("battery", [ac_bidi_port])
        assert TopologyValidator.validate([device], [AC_BUS]) is True

    def test_multiple_devices_same_bus_passes(self):
        pv = make_device("pv", [
            EnergyPort(port_id="p1", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        load_a = make_device("load_a", [
            EnergyPort(port_id="p2", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        load_b = make_device("load_b", [
            EnergyPort(port_id="p3", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        assert TopologyValidator.validate([pv, load_a, load_b], [AC_BUS]) is True

    def test_multiple_independent_buses_pass(self):
        """Devices on separate buses are validated independently."""
        pv = make_device("pv", [
            EnergyPort(port_id="p_dc_src", bus_id="bus_dc", direction=PortDirection.SOURCE)
        ])
        inv_dc_side = make_device("inv_dc", [
            EnergyPort(port_id="p_dc_snk", bus_id="bus_dc", direction=PortDirection.SINK)
        ])
        inv_ac_side = make_device("inv_ac", [
            EnergyPort(port_id="p_ac_src", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        load = make_device("load", [
            EnergyPort(port_id="p_ac_snk", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        assert TopologyValidator.validate(
            [pv, inv_dc_side, inv_ac_side, load], [DC_BUS, AC_BUS]
        ) is True

    def test_device_with_ports_on_multiple_buses_passes(self):
        """An inverter-style device with one DC sink port and one AC source port."""
        inverter = make_device("inverter", [
            EnergyPort(port_id="p_dc", bus_id="bus_dc", direction=PortDirection.SINK),
            EnergyPort(port_id="p_ac", bus_id="bus_ac", direction=PortDirection.SOURCE),
        ])
        pv = make_device("pv", [
            EnergyPort(port_id="p_pv", bus_id="bus_dc", direction=PortDirection.SOURCE)
        ])
        load = make_device("load", [
            EnergyPort(port_id="p_load", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        assert TopologyValidator.validate([pv, inverter, load], [DC_BUS, AC_BUS]) is True

    def test_empty_device_list_passes(self):
        """No devices means no ports — valid by definition."""
        assert TopologyValidator.validate([], [AC_BUS]) is True

    def test_device_with_no_ports_does_not_affect_bus_validation(
        self, ac_source_port, ac_sink_port
    ):
        """A portless device contributes nothing to any bus and must not block validation."""
        portless = make_device("controller", [])
        source = make_device("pv", [ac_source_port])
        sink = make_device("load", [ac_sink_port])
        assert TopologyValidator.validate([portless, source, sink], [AC_BUS]) is True

    def test_unreferenced_bus_not_validated(self):
        """A registered bus with no connected ports should not require source/sink."""
        pv = make_device("pv", [
            EnergyPort(port_id="p_src", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        load = make_device("load", [
            EnergyPort(port_id="p_snk", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        # DC_BUS is registered but no device connects to it
        assert TopologyValidator.validate([pv, load], [AC_BUS, DC_BUS]) is True

    def test_constraint_at_exact_max_sinks_passes(self):
        constrained_bus = EnergyBus(
            bus_id="bus_ac",
            carrier=EnergyCarrier.AC,
            constraint=EnergyBusConstraint(max_sinks=2),
        )
        pv = make_device("pv", [
            EnergyPort(port_id="p_src", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        load_a = make_device("load_a", [
            EnergyPort(port_id="p_snk1", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        load_b = make_device("load_b", [
            EnergyPort(port_id="p_snk2", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        assert TopologyValidator.validate([pv, load_a, load_b], [constrained_bus]) is True

    def test_constraint_at_exact_max_sources_passes(self):
        constrained_bus = EnergyBus(
            bus_id="bus_ac",
            carrier=EnergyCarrier.AC,
            constraint=EnergyBusConstraint(max_sources=2),
        )
        pv_a = make_device("pv_a", [
            EnergyPort(port_id="p_src1", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        pv_b = make_device("pv_b", [
            EnergyPort(port_id="p_src2", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        load = make_device("load", [
            EnergyPort(port_id="p_snk", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        assert TopologyValidator.validate([pv_a, pv_b, load], [constrained_bus]) is True

    def test_unconstrained_bus_accepts_many_sinks(self):
        pv = make_device("pv", [
            EnergyPort(port_id="p_src", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        loads = [
            make_device(f"load_{i}", [
                EnergyPort(port_id=f"p_snk_{i}", bus_id="bus_ac", direction=PortDirection.SINK)
            ])
            for i in range(10)
        ]
        assert TopologyValidator.validate([pv, *loads], [AC_BUS]) is True

    def test_unconstrained_bus_accepts_many_sources(self):
        pvs = [
            make_device(f"pv_{i}", [
                EnergyPort(port_id=f"p_src_{i}", bus_id="bus_ac", direction=PortDirection.SOURCE)
            ])
            for i in range(10)
        ]
        load = make_device("load", [
            EnergyPort(port_id="p_snk", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        assert TopologyValidator.validate([*pvs, load], [AC_BUS]) is True


# ============================================================
# TestTopologyValidatorPortErrors
# ============================================================

class TestTopologyValidatorPortErrors:
    def test_port_referencing_unknown_bus_raises(self):
        device = make_device("pv", [
            EnergyPort(port_id="p1", bus_id="bus_nonexistent", direction=PortDirection.SOURCE)
        ])
        with pytest.raises(TopologyValidationError, match="bus_nonexistent"):
            TopologyValidator.validate([device], [AC_BUS])

    def test_unknown_bus_error_includes_device_id(self):
        device = make_device("my_device", [
            EnergyPort(port_id="p1", bus_id="bus_missing", direction=PortDirection.SOURCE)
        ])
        with pytest.raises(TopologyValidationError, match="my_device"):
            TopologyValidator.validate([device], [AC_BUS])

    def test_unknown_bus_error_includes_port_id(self):
        device = make_device("pv", [
            EnergyPort(port_id="my_port", bus_id="bus_missing", direction=PortDirection.SOURCE)
        ])
        with pytest.raises(TopologyValidationError, match="my_port"):
            TopologyValidator.validate([device], [AC_BUS])

    def test_duplicate_port_id_on_same_device_raises(self):
        device = make_device("pv", [
            EnergyPort(port_id="p_dup", bus_id="bus_ac", direction=PortDirection.SOURCE),
            EnergyPort(port_id="p_dup", bus_id="bus_ac", direction=PortDirection.SINK),
        ])
        with pytest.raises(TopologyValidationError, match="p_dup"):
            TopologyValidator.validate([device], [AC_BUS])

    def test_duplicate_port_id_error_includes_device_id(self):
        device = make_device("my_inverter", [
            EnergyPort(port_id="p_dup", bus_id="bus_ac", direction=PortDirection.SOURCE),
            EnergyPort(port_id="p_dup", bus_id="bus_ac", direction=PortDirection.SINK),
        ])
        with pytest.raises(TopologyValidationError, match="my_inverter"):
            TopologyValidator.validate([device], [AC_BUS])

    def test_duplicate_port_id_on_different_devices_is_legal(self):
        """port_id uniqueness is per-device only, not global."""
        device_a = make_device("dev_a", [
            EnergyPort(port_id="p1", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        device_b = make_device("dev_b", [
            EnergyPort(port_id="p1", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        assert TopologyValidator.validate([device_a, device_b], [AC_BUS]) is True


# ============================================================
# TestTopologyValidatorBusStructureErrors
# ============================================================

class TestTopologyValidatorBusStructureErrors:
    def test_bus_with_only_sink_ports_raises(self):
        sink_only = make_device("load", [
            EnergyPort(port_id="p_snk", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        with pytest.raises(TopologyValidationError, match="no source"):
            TopologyValidator.validate([sink_only], [AC_BUS])

    def test_bus_with_only_source_ports_raises(self):
        source_only = make_device("pv", [
            EnergyPort(port_id="p_src", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        with pytest.raises(TopologyValidationError, match="no sink"):
            TopologyValidator.validate([source_only], [AC_BUS])

    def test_no_source_error_includes_bus_id(self):
        sink_only = make_device("load", [
            EnergyPort(port_id="p_snk", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        with pytest.raises(TopologyValidationError, match="bus_ac"):
            TopologyValidator.validate([sink_only], [AC_BUS])

    def test_no_sink_error_includes_bus_id(self):
        source_only = make_device("pv", [
            EnergyPort(port_id="p_src", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        with pytest.raises(TopologyValidationError, match="bus_ac"):
            TopologyValidator.validate([source_only], [AC_BUS])

    def test_max_sinks_exceeded_raises(self):
        constrained_bus = EnergyBus(
            bus_id="bus_ac",
            carrier=EnergyCarrier.AC,
            constraint=EnergyBusConstraint(max_sinks=1),
        )
        pv = make_device("pv", [
            EnergyPort(port_id="p_src", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        load_a = make_device("load_a", [
            EnergyPort(port_id="p_snk1", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        load_b = make_device("load_b", [
            EnergyPort(port_id="p_snk2", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        with pytest.raises(TopologyValidationError, match="max_sinks"):
            TopologyValidator.validate([pv, load_a, load_b], [constrained_bus])

    def test_max_sources_exceeded_raises(self):
        constrained_bus = EnergyBus(
            bus_id="bus_ac",
            carrier=EnergyCarrier.AC,
            constraint=EnergyBusConstraint(max_sources=1),
        )
        pv_a = make_device("pv_a", [
            EnergyPort(port_id="p_src1", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        pv_b = make_device("pv_b", [
            EnergyPort(port_id="p_src2", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        load = make_device("load", [
            EnergyPort(port_id="p_snk", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        with pytest.raises(TopologyValidationError, match="max_sources"):
            TopologyValidator.validate([pv_a, pv_b, load], [constrained_bus])

    def test_max_sinks_error_includes_bus_id(self):
        constrained_bus = EnergyBus(
            bus_id="bus_ac",
            carrier=EnergyCarrier.AC,
            constraint=EnergyBusConstraint(max_sinks=1),
        )
        pv = make_device("pv", [
            EnergyPort(port_id="p_src", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        load_a = make_device("load_a", [
            EnergyPort(port_id="p1", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        load_b = make_device("load_b", [
            EnergyPort(port_id="p2", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        with pytest.raises(TopologyValidationError, match="bus_ac"):
            TopologyValidator.validate([pv, load_a, load_b], [constrained_bus])

    def test_max_sources_error_includes_bus_id(self):
        constrained_bus = EnergyBus(
            bus_id="bus_ac",
            carrier=EnergyCarrier.AC,
            constraint=EnergyBusConstraint(max_sources=1),
        )
        pv_a = make_device("pv_a", [
            EnergyPort(port_id="p_src1", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        pv_b = make_device("pv_b", [
            EnergyPort(port_id="p_src2", bus_id="bus_ac", direction=PortDirection.SOURCE)
        ])
        load = make_device("load", [
            EnergyPort(port_id="p_snk", bus_id="bus_ac", direction=PortDirection.SINK)
        ])
        with pytest.raises(TopologyValidationError, match="bus_ac"):
            TopologyValidator.validate([pv_a, pv_b, load], [constrained_bus])
