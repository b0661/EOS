"""Tests for genetic2.simulation.registry.

Covers: DeviceRegistry registration, retrieval, type filtering,
reset_all, and error handling.

Changes from previous version:
- SimpleBattery() constructor: prediction_hours -> num_steps
"""

import numpy as np
import pytest
from fixtures.geneti2fixtures import FixedLoad, SimpleBattery, SimplePV

from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry


class TestDeviceRegistryRegistration:
    """Registering and unregistering devices."""

    def test_register_single_device(self):
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1")
        reg.register(bat)
        assert "bat1" in reg

    def test_register_multiple_devices_different_types(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        reg.register(SimplePV("pv1", np.zeros(24)))
        assert "bat1" in reg
        assert "pv1" in reg
        assert len(reg) == 2

    def test_register_multiple_batteries(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        reg.register(SimpleBattery("bat2"))
        assert len(reg) == 2

    def test_duplicate_id_raises(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        with pytest.raises(ValueError, match="bat1"):
            reg.register(SimpleBattery("bat1"))

    def test_unregister_removes_device(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        reg.unregister("bat1")
        assert "bat1" not in reg
        assert len(reg) == 0

    def test_unregister_unknown_raises(self):
        reg = DeviceRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            reg.unregister("nonexistent")

    def test_len_reflects_count(self):
        reg = DeviceRegistry()
        assert len(reg) == 0
        reg.register(SimpleBattery("bat1"))
        assert len(reg) == 1
        reg.register(SimpleBattery("bat2"))
        assert len(reg) == 2


class TestDeviceRegistryRetrieval:
    """Retrieving devices by ID and type."""

    def test_get_returns_correct_device(self):
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1")
        reg.register(bat)
        assert reg.get("bat1") is bat

    def test_get_unknown_raises_key_error(self):
        reg = DeviceRegistry()
        with pytest.raises(KeyError, match="unknown"):
            reg.get("unknown")

    def test_get_typed_returns_correct_type(self):
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1")
        reg.register(bat)
        result = reg.get_typed("bat1", SimpleBattery)
        assert result is bat

    def test_get_typed_wrong_type_raises(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        with pytest.raises(TypeError, match="SimpleBattery"):
            reg.get_typed("bat1", SimplePV)

    def test_device_ids_returns_all(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        reg.register(SimpleBattery("bat2"))
        reg.register(SimplePV("pv1", np.zeros(24)))
        assert set(reg.device_ids()) == {"bat1", "bat2", "pv1"}

    def test_device_ids_preserves_registration_order(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        reg.register(SimplePV("pv1", np.zeros(24)))
        reg.register(SimpleBattery("bat2"))
        assert reg.device_ids() == ["bat1", "pv1", "bat2"]


class TestDeviceRegistryTypeFiltering:
    """all_of_type() iteration."""

    def test_all_of_type_batteries_only(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        reg.register(SimpleBattery("bat2"))
        reg.register(SimplePV("pv1", np.zeros(24)))
        batteries = list(reg.all_of_type(SimpleBattery))
        assert len(batteries) == 2
        assert all(isinstance(b, SimpleBattery) for b in batteries)

    def test_all_of_type_pv_only(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        reg.register(SimplePV("pv1", np.zeros(24)))
        pvs = list(reg.all_of_type(SimplePV))
        assert len(pvs) == 1
        assert pvs[0].device_id == "pv1"

    def test_all_of_type_empty_if_none_match(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        assert list(reg.all_of_type(SimplePV)) == []

    def test_all_of_type_includes_subclasses(self):
        """all_of_type(StorageDevice) yields SimpleBattery instances."""
        from akkudoktoreos.devices.base import StorageDevice
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        reg.register(SimplePV("pv1", np.zeros(24)))
        storage = list(reg.all_of_type(StorageDevice))
        assert len(storage) == 1
        assert storage[0].device_id == "bat1"

    def test_all_devices_yields_all(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1"))
        reg.register(SimplePV("pv1", np.zeros(24)))
        assert len(list(reg.all_devices())) == 2


class TestDeviceRegistryReset:
    """reset_all() resets every device's physical state."""

    def test_reset_all_restores_battery_soc(self):
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1", initial_soc_pct=20.0)
        reg.register(bat)

        bat._soc_wh = 0.0
        assert bat.current_soc_percentage() == pytest.approx(0.0)

        reg.reset_all()
        assert bat.current_soc_percentage() == pytest.approx(20.0)

    def test_reset_all_resets_multiple_devices(self):
        reg = DeviceRegistry()
        bat1 = SimpleBattery("bat1", initial_soc_pct=30.0)
        bat2 = SimpleBattery("bat2", initial_soc_pct=80.0)
        reg.register(bat1)
        reg.register(bat2)

        bat1._soc_wh = 0.0
        bat2._soc_wh = 0.0
        reg.reset_all()

        assert bat1.current_soc_percentage() == pytest.approx(30.0)
        assert bat2.current_soc_percentage() == pytest.approx(80.0)

    def test_reset_does_not_clear_stored_genome(self):
        """store_genome() must survive reset() so the schedule can be re-decoded."""
        from datetime import timedelta
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1", num_steps=4)
        reg.register(bat)

        raw = np.array([2.0, 2.0, 1.0, 0.0])
        bat.store_genome(raw)
        reg.reset_all()

        # Stored genome must still be retrievable after reset
        np.testing.assert_array_equal(bat.get_stored_genome(), raw)
