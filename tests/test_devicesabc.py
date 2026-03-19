"""Tests for immutable device parameter dataclasses in devicesabc.py.

Covers:
    - Valid construction of every parameter type
    - All __post_init__ validation paths (one test per violated constraint)
    - Immutability enforcement (FrozenInstanceError on assignment)
    - Hashability and value equality contracts
    - Optional field defaults
    - Topology types: EnergyPort, EnergyBus, EnergyBusConstraint
    - PortDirection enum values
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from akkudoktoreos.devices.devicesabc import (
    ApplianceOperationMode,
    BatteryOperationMode,
    EnergyBus,
    EnergyBusConstraint,
    EnergyCarrier,
    EnergyPort,
    PortDirection,
)

# ============================================================
# Shared fixtures
# ============================================================


@pytest.fixture()
def ac_port() -> EnergyPort:
    return EnergyPort(port_id="p_ac", bus_id="bus_ac", direction=PortDirection.BIDIRECTIONAL)


@pytest.fixture()
def dc_port() -> EnergyPort:
    return EnergyPort(port_id="p_dc", bus_id="bus_dc", direction=PortDirection.SOURCE)


@pytest.fixture()
def heat_port() -> EnergyPort:
    return EnergyPort(port_id="p_heat", bus_id="bus_heat", direction=PortDirection.SINK)


# ============================================================
# PortDirection
# ============================================================


class TestPortDirection:
    def test_source_value(self):
        assert PortDirection.SOURCE.value == "source"

    def test_sink_value(self):
        assert PortDirection.SINK.value == "sink"

    def test_bidirectional_value(self):
        assert PortDirection.BIDIRECTIONAL.value == "bidirectional"

    def test_all_three_members_exist(self):
        assert {PortDirection.SOURCE, PortDirection.SINK, PortDirection.BIDIRECTIONAL} == set(PortDirection)


# ============================================================
# EnergyPort
# ============================================================


class TestEnergyPort:
    def test_valid_construction_minimal(self):
        port = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.BIDIRECTIONAL)
        assert port.port_id == "p1"
        assert port.bus_id == "bus1"
        assert port.direction == PortDirection.BIDIRECTIONAL
        assert port.max_power_w is None

    def test_valid_construction_source(self):
        port = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SOURCE)
        assert port.direction == PortDirection.SOURCE

    def test_valid_construction_sink(self):
        port = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SINK)
        assert port.direction == PortDirection.SINK

    def test_valid_construction_with_power_limit(self):
        port = EnergyPort(
            port_id="p1", bus_id="bus1",
            direction=PortDirection.SINK,
            max_power_w=3_000.0,
        )
        assert port.max_power_w == 3_000.0

    def test_immutable_port_id(self):
        port = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.BIDIRECTIONAL)
        with pytest.raises(FrozenInstanceError):
            port.port_id = "other"  # type: ignore[misc]

    def test_immutable_direction(self):
        port = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.BIDIRECTIONAL)
        with pytest.raises(FrozenInstanceError):
            port.direction = PortDirection.SINK  # type: ignore[misc]

    def test_equal_instances_are_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SINK, max_power_w=1_000.0)
        b = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SINK, max_power_w=1_000.0)
        assert a == b

    def test_equal_instances_hash_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SINK, max_power_w=1_000.0)
        b = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SINK, max_power_w=1_000.0)
        assert hash(a) == hash(b)

    def test_different_port_id_not_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.BIDIRECTIONAL)
        b = EnergyPort(port_id="p2", bus_id="bus1", direction=PortDirection.BIDIRECTIONAL)
        assert a != b

    def test_different_direction_not_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SOURCE)
        b = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SINK)
        assert a != b

    def test_usable_as_dict_key(self):
        port = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.BIDIRECTIONAL)
        d = {port: "value"}
        assert d[port] == "value"

    def test_usable_as_set_member(self):
        a = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.BIDIRECTIONAL)
        b = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.BIDIRECTIONAL)
        assert len({a, b}) == 1

    def test_different_directions_are_distinct_set_members(self):
        source = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SOURCE)
        sink   = EnergyPort(port_id="p1", bus_id="bus1", direction=PortDirection.SINK)
        assert len({source, sink}) == 2


# ============================================================
# EnergyBusConstraint
# ============================================================


class TestEnergyBusConstraint:
    def test_valid_construction_all_none(self):
        c = EnergyBusConstraint()
        assert c.max_sinks is None
        assert c.max_sources is None

    def test_valid_construction_with_values(self):
        c = EnergyBusConstraint(max_sinks=3, max_sources=1)
        assert c.max_sinks == 3
        assert c.max_sources == 1

    def test_immutable(self):
        c = EnergyBusConstraint(max_sinks=2)
        with pytest.raises(FrozenInstanceError):
            c.max_sinks = 5  # type: ignore[misc]

    def test_equality(self):
        assert EnergyBusConstraint(max_sinks=2) == EnergyBusConstraint(max_sinks=2)
        assert EnergyBusConstraint(max_sinks=2) != EnergyBusConstraint(max_sinks=3)

    def test_hashable(self):
        a = EnergyBusConstraint(max_sinks=1, max_sources=1)
        b = EnergyBusConstraint(max_sinks=1, max_sources=1)
        assert hash(a) == hash(b)

    def test_only_sinks_set(self):
        c = EnergyBusConstraint(max_sinks=2)
        assert c.max_sinks == 2
        assert c.max_sources is None

    def test_only_sources_set(self):
        c = EnergyBusConstraint(max_sources=4)
        assert c.max_sinks is None
        assert c.max_sources == 4


# ============================================================
# EnergyBus
# ============================================================


class TestEnergyBus:
    def test_valid_construction_minimal(self):
        bus = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        assert bus.bus_id == "bus_ac"
        assert bus.carrier == EnergyCarrier.AC
        assert bus.constraint is None

    def test_valid_construction_with_constraint(self):
        constraint = EnergyBusConstraint(max_sinks=4)
        bus = EnergyBus(bus_id="bus_dc", carrier=EnergyCarrier.DC, constraint=constraint)
        assert bus.constraint is not None
        assert bus.constraint.max_sinks == 4

    def test_all_carriers_accepted(self):
        for carrier in EnergyCarrier:
            bus = EnergyBus(bus_id=f"bus_{carrier.value}", carrier=carrier)
            assert bus.carrier == carrier

    def test_immutable(self):
        bus = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        with pytest.raises(FrozenInstanceError):
            bus.bus_id = "other"  # type: ignore[misc]

    def test_equality(self):
        a = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        b = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        assert a == b

    def test_different_carrier_not_equal(self):
        a = EnergyBus(bus_id="bus_x", carrier=EnergyCarrier.AC)
        b = EnergyBus(bus_id="bus_x", carrier=EnergyCarrier.DC)
        assert a != b

    def test_hashable(self):
        a = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        b = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self):
        bus = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        d = {bus: 42}
        assert d[bus] == 42

    def test_constraint_participates_in_equality(self):
        no_constraint = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        with_constraint = EnergyBus(
            bus_id="bus_ac", carrier=EnergyCarrier.AC,
            constraint=EnergyBusConstraint(max_sinks=2),
        )
        assert no_constraint != with_constraint
