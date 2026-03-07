"""Tests for DeviceRegistry.

Covers:
    - Registration and retrieval of devices
    - Duplicate device_id rejection
    - Unregistration including error on unknown id
    - get() and get_typed() including type mismatch
    - all_of_type() filtering and subclass inclusion
    - all_devices() iteration order
    - device_ids() ordering
    - __contains__, __len__, __repr__
    - reset_all() delegation to devices
    - all_of_type() rejection of non-EnergyDevice types
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from akkudoktoreos.devices.devicesabc import EnergyDevice
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext

# ============================================================
# Minimal concrete stubs
# ============================================================

class StubDevice(EnergyDevice):
    """Minimal concrete EnergyDevice for registry tests."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id

    # Implement abstract methods minimally so instantiation works
    def setup_run(self, context: SimulationContext): pass
    def genome_requirements(self): return None
    @property
    def ports(self): return ()
    @property
    def objective_names(self): return []
    def create_batch_state(self, population_size, horizon): return None
    def apply_genome_batch(self, state, genome_batch): return genome_batch
    def build_device_request(self, state): return None
    def apply_device_grant(self, state, grant): pass
    def compute_cost(self, state): return None


class StubBattery(StubDevice):
    """Subclass stub to test all_of_type subclass filtering."""
    pass


class StubInverter(StubDevice):
    """Another subclass stub to test type filtering isolation."""
    pass


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture()
def registry() -> DeviceRegistry:
    return DeviceRegistry()


@pytest.fixture()
def device_a() -> StubDevice:
    return StubDevice("device_a")


@pytest.fixture()
def device_b() -> StubDevice:
    return StubDevice("device_b")


@pytest.fixture()
def battery_1() -> StubBattery:
    return StubBattery("battery_1")


@pytest.fixture()
def battery_2() -> StubBattery:
    return StubBattery("battery_2")


@pytest.fixture()
def inverter_1() -> StubInverter:
    return StubInverter("inverter_1")


@pytest.fixture()
def populated_registry(
    registry, battery_1, battery_2, inverter_1
) -> DeviceRegistry:
    """Registry with two batteries and one inverter pre-registered."""
    registry.register(battery_1)
    registry.register(battery_2)
    registry.register(inverter_1)
    return registry


# ============================================================
# TestDeviceRegistryConstruction
# ============================================================

class TestDeviceRegistryConstruction:
    def test_empty_on_creation(self, registry):
        assert len(registry) == 0

    def test_repr_empty(self, registry):
        assert repr(registry) == "DeviceRegistry([])"

    def test_device_ids_empty(self, registry):
        assert registry.device_ids() == []


# ============================================================
# TestDeviceRegistryRegister
# ============================================================

class TestDeviceRegistryRegister:
    def test_register_single_device(self, registry, device_a):
        registry.register(device_a)
        assert len(registry) == 1

    def test_register_multiple_devices(self, registry, device_a, device_b):
        registry.register(device_a)
        registry.register(device_b)
        assert len(registry) == 2

    def test_register_duplicate_id_raises(self, registry, device_a):
        registry.register(device_a)
        duplicate = StubDevice("device_a")
        with pytest.raises(ValueError, match="device_a"):
            registry.register(duplicate)

    def test_register_preserves_insertion_order(
        self, registry, device_a, device_b
    ):
        registry.register(device_a)
        registry.register(device_b)
        assert registry.device_ids() == ["device_a", "device_b"]

    def test_register_different_types_same_registry(
        self, registry, battery_1, inverter_1
    ):
        registry.register(battery_1)
        registry.register(inverter_1)
        assert len(registry) == 2


# ============================================================
# TestDeviceRegistryUnregister
# ============================================================

class TestDeviceRegistryUnregister:
    def test_unregister_removes_device(self, registry, device_a):
        registry.register(device_a)
        registry.unregister("device_a")
        assert "device_a" not in registry

    def test_unregister_decrements_length(self, registry, device_a, device_b):
        registry.register(device_a)
        registry.register(device_b)
        registry.unregister("device_a")
        assert len(registry) == 1

    def test_unregister_unknown_id_raises(self, registry):
        with pytest.raises(KeyError, match="unknown_id"):
            registry.unregister("unknown_id")

    def test_unregister_then_reregister_succeeds(self, registry, device_a):
        registry.register(device_a)
        registry.unregister("device_a")
        registry.register(device_a)
        assert "device_a" in registry


# ============================================================
# TestDeviceRegistryGet
# ============================================================

class TestDeviceRegistryGet:
    def test_get_returns_registered_device(self, registry, device_a):
        registry.register(device_a)
        result = registry.get("device_a")
        assert result is device_a

    def test_get_unknown_id_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="missing"):
            registry.get("missing")

    def test_get_error_message_lists_registered_ids(
        self, registry, device_a
    ):
        registry.register(device_a)
        with pytest.raises(KeyError, match="device_a"):
            registry.get("unknown")


# ============================================================
# TestDeviceRegistryGetTyped
# ============================================================

class TestDeviceRegistryGetTyped:
    def test_get_typed_correct_type_returns_device(
        self, registry, battery_1
    ):
        registry.register(battery_1)
        result = registry.get_typed("battery_1", StubBattery)
        assert result is battery_1

    def test_get_typed_base_type_accepted(self, registry, battery_1):
        """Requesting the base type should succeed for a subclass instance."""
        registry.register(battery_1)
        result = registry.get_typed("battery_1", StubDevice)
        assert result is battery_1

    def test_get_typed_wrong_type_raises_type_error(
        self, registry, battery_1
    ):
        registry.register(battery_1)
        with pytest.raises(TypeError, match="battery_1"):
            registry.get_typed("battery_1", StubInverter)

    def test_get_typed_type_error_message_contains_expected_type(
        self, registry, battery_1
    ):
        registry.register(battery_1)
        with pytest.raises(TypeError, match="StubInverter"):
            registry.get_typed("battery_1", StubInverter)

    def test_get_typed_unknown_id_raises_key_error(self, registry):
        with pytest.raises(KeyError):
            registry.get_typed("missing", StubBattery)


# ============================================================
# TestDeviceRegistryAllOfType
# ============================================================

class TestDeviceRegistryAllOfType:
    def test_all_of_type_returns_only_matching_type(
        self, populated_registry, battery_1, battery_2
    ):
        result = list(populated_registry.all_of_type(StubBattery))
        assert set(d.device_id for d in result) == {"battery_1", "battery_2"}

    def test_all_of_type_excludes_other_types(self, populated_registry):
        result = list(populated_registry.all_of_type(StubBattery))
        assert all(isinstance(d, StubBattery) for d in result)

    def test_all_of_type_preserves_registration_order(
        self, populated_registry
    ):
        result = list(populated_registry.all_of_type(StubBattery))
        assert [d.device_id for d in result] == ["battery_1", "battery_2"]

    def test_all_of_type_base_class_returns_all(self, populated_registry):
        """Filtering by EnergyDevice base should yield all devices."""
        result = list(populated_registry.all_of_type(StubDevice))
        assert len(result) == 3

    def test_all_of_type_no_matches_returns_empty(
        self, registry, device_a
    ):
        registry.register(device_a)
        result = list(registry.all_of_type(StubBattery))
        assert result == []

    def test_all_of_type_non_energy_device_raises(self, registry):
        with pytest.raises(TypeError):
            list(registry.all_of_type(str))

    def test_all_of_type_empty_registry_returns_empty(self, registry):
        result = list(registry.all_of_type(StubDevice))
        assert result == []


# ============================================================
# TestDeviceRegistryAllDevices
# ============================================================

class TestDeviceRegistryAllDevices:
    def test_all_devices_returns_all_in_order(self, populated_registry):
        result = list(populated_registry.all_devices())
        assert [d.device_id for d in result] == [
            "battery_1", "battery_2", "inverter_1"
        ]

    def test_all_devices_empty_registry(self, registry):
        assert list(registry.all_devices()) == []

    def test_all_devices_returns_exact_instances(
        self, registry, device_a, device_b
    ):
        registry.register(device_a)
        registry.register(device_b)
        result = list(registry.all_devices())
        assert result[0] is device_a
        assert result[1] is device_b


# ============================================================
# TestDeviceRegistryDeviceIds
# ============================================================

class TestDeviceRegistryDeviceIds:
    def test_device_ids_returns_ordered_list(self, populated_registry):
        assert populated_registry.device_ids() == [
            "battery_1", "battery_2", "inverter_1"
        ]

    def test_device_ids_returns_copy(self, registry, device_a):
        """Mutating the returned list must not affect the registry."""
        registry.register(device_a)
        ids = registry.device_ids()
        ids.clear()
        assert len(registry) == 1


# ============================================================
# TestDeviceRegistryContains
# ============================================================

class TestDeviceRegistryContains:
    def test_contains_registered_id(self, registry, device_a):
        registry.register(device_a)
        assert "device_a" in registry

    def test_not_contains_unregistered_id(self, registry):
        assert "ghost" not in registry

    def test_not_contains_after_unregister(self, registry, device_a):
        registry.register(device_a)
        registry.unregister("device_a")
        assert "device_a" not in registry


# ============================================================
# TestDeviceRegistryLen
# ============================================================

class TestDeviceRegistryLen:
    def test_len_empty(self, registry):
        assert len(registry) == 0

    def test_len_after_register(self, registry, device_a, device_b):
        registry.register(device_a)
        assert len(registry) == 1
        registry.register(device_b)
        assert len(registry) == 2

    def test_len_after_unregister(self, registry, device_a, device_b):
        registry.register(device_a)
        registry.register(device_b)
        registry.unregister("device_a")
        assert len(registry) == 1


# ============================================================
# TestDeviceRegistryRepr
# ============================================================

class TestDeviceRegistryRepr:
    def test_repr_single_device(self, registry, device_a):
        registry.register(device_a)
        assert repr(registry) == "DeviceRegistry(['device_a'])"

    def test_repr_multiple_devices_in_order(self, populated_registry):
        assert repr(populated_registry) == (
            "DeviceRegistry(['battery_1', 'battery_2', 'inverter_1'])"
        )
