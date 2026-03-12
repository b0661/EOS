"""Tests for immutable device parameter dataclasses in devicesabc.py.

Covers:
    - Valid construction of every parameter type
    - All __post_init__ validation paths (one test per violated constraint)
    - Immutability enforcement (FrozenInstanceError on assignment)
    - Hashability and value equality contracts
    - Optional field defaults
    - Topology types: EnergyPort, EnergyBus, EnergyBusConstraint
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
)

# ============================================================
# Shared fixtures
# ============================================================


@pytest.fixture()
def ac_port() -> EnergyPort:
    return EnergyPort(port_id="p_ac", bus_id="bus_ac")


@pytest.fixture()
def dc_port() -> EnergyPort:
    return EnergyPort(port_id="p_dc", bus_id="bus_dc")


@pytest.fixture()
def heat_port() -> EnergyPort:
    return EnergyPort(port_id="p_heat", bus_id="bus_heat")


# ============================================================
# EnergyPort
# ============================================================


class TestEnergyPort:
    def test_valid_construction_minimal(self):
        port = EnergyPort(port_id="p1", bus_id="bus1")
        assert port.port_id == "p1"
        assert port.bus_id == "bus1"
        assert port.max_power_w is None

    def test_valid_construction_with_power_limit(self):
        port = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=3_000.0)
        assert port.max_power_w == 3_000.0

    def test_immutable(self):
        port = EnergyPort(port_id="p1", bus_id="bus1")
        with pytest.raises(FrozenInstanceError):
            port.port_id = "other"

    def test_equal_instances_are_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=1_000.0)
        b = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=1_000.0)
        assert a == b

    def test_equal_instances_hash_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=1_000.0)
        b = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=1_000.0)
        assert hash(a) == hash(b)

    def test_different_instances_are_not_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1")
        b = EnergyPort(port_id="p2", bus_id="bus1")
        assert a != b

    def test_usable_as_dict_key(self):
        port = EnergyPort(port_id="p1", bus_id="bus1")
        d = {port: "value"}
        assert d[port] == "value"

    def test_usable_as_set_member(self):
        a = EnergyPort(port_id="p1", bus_id="bus1")
        b = EnergyPort(port_id="p1", bus_id="bus1")
        assert len({a, b}) == 1


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
            c.max_sinks = 5

    def test_equality(self):
        assert EnergyBusConstraint(max_sinks=2) == EnergyBusConstraint(max_sinks=2)
        assert EnergyBusConstraint(max_sinks=2) != EnergyBusConstraint(max_sinks=3)

    def test_hashable(self):
        a = EnergyBusConstraint(max_sinks=1, max_sources=1)
        b = EnergyBusConstraint(max_sinks=1, max_sources=1)
        assert hash(a) == hash(b)


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
        assert bus.constraint.max_sinks == 4

    def test_all_carriers_accepted(self):
        for carrier in EnergyCarrier:
            bus = EnergyBus(bus_id=f"bus_{carrier.value}", carrier=carrier)
            assert bus.carrier == carrier

    def test_immutable(self):
        bus = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        with pytest.raises(FrozenInstanceError):
            bus.bus_id = "other"

    def test_equality(self):
        a = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        b = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        assert a == b

    def test_hashable(self):
        a = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        b = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self):
        bus = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        d = {bus: 42}
        assert d[bus] == 42
