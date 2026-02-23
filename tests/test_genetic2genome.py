"""Tests for genetic2.optimization.genome.

Covers: GenomeAssembler construction, dispatch, bounds, get_slice, set_slice,
random_genome, describe, and error handling.
"""

import numpy as np
import pytest
from fixtures.genetic2fixtures import SimpleBattery, SimplePV

from akkudoktoreos.devices.genetic2.base import GenomeSlice
from akkudoktoreos.optimization.genetic2.genome import GenomeAssembler
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry


class TestGenomeAssemblerConstruction:
    """GenomeAssembler correctly maps device requirements to slices."""

    def test_total_size_single_device(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=24))
        assembler = GenomeAssembler(reg)
        assert assembler.total_size == 24

    def test_total_size_two_batteries(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=24))
        reg.register(SimpleBattery("bat2", prediction_hours=24))
        assembler = GenomeAssembler(reg)
        assert assembler.total_size == 48

    def test_pv_does_not_contribute_to_genome(self):
        """Devices returning None from genome_requirements are skipped."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.zeros(24)))
        reg.register(SimpleBattery("bat1", prediction_hours=24))
        assembler = GenomeAssembler(reg)
        assert assembler.total_size == 24  # only battery

    def test_empty_registry_total_size_zero(self):
        reg = DeviceRegistry()
        assembler = GenomeAssembler(reg)
        assert assembler.total_size == 0

    def test_only_pv_total_size_zero(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.zeros(24)))
        assembler = GenomeAssembler(reg)
        assert assembler.total_size == 0

    def test_slices_are_non_overlapping(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=12))
        reg.register(SimpleBattery("bat2", prediction_hours=12))
        assembler = GenomeAssembler(reg)

        slice1 = assembler._slices["bat1"]
        slice2 = assembler._slices["bat2"]

        # bat1 must end where bat2 starts
        assert slice1 == (0, 12)
        assert slice2 == (12, 24)

    def test_registration_order_determines_slice_order(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat_a", prediction_hours=10))
        reg.register(SimpleBattery("bat_b", prediction_hours=5))
        assembler = GenomeAssembler(reg)

        assert assembler._slices["bat_a"] == (0, 10)
        assert assembler._slices["bat_b"] == (10, 15)


class TestGenomeAssemblerBounds:
    """bounds() returns correct lower and upper bound arrays."""

    def test_bounds_length_equals_total_size(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=24))
        assembler = GenomeAssembler(reg)
        lows, highs = assembler.bounds()
        assert len(lows) == 24
        assert len(highs) == 24

    def test_bounds_values_from_genome_slice(self):
        """SimpleBattery declares low=0, high=2."""
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=6))
        assembler = GenomeAssembler(reg)
        lows, highs = assembler.bounds()
        assert all(l == 0.0 for l in lows)
        assert all(h == 2.0 for h in highs)

    def test_bounds_concatenated_for_multiple_devices(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=3))
        reg.register(SimpleBattery("bat2", prediction_hours=3))
        assembler = GenomeAssembler(reg)
        lows, highs = assembler.bounds()
        assert len(lows) == 6
        assert len(highs) == 6

    def test_empty_assembler_bounds_empty(self):
        reg = DeviceRegistry()
        assembler = GenomeAssembler(reg)
        lows, highs = assembler.bounds()
        assert lows == []
        assert highs == []


class TestGenomeAssemblerDispatch:
    """dispatch() routes genome slices to correct devices."""

    def test_dispatch_single_battery_receives_its_slice(self):
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1", prediction_hours=4)
        reg.register(bat)
        assembler = GenomeAssembler(reg)

        genome = np.array([0, 1, 2, 1], dtype=float)
        assembler.dispatch(genome, reg)

        assert bat.last_genome_slice is not None
        np.testing.assert_array_equal(bat.last_genome_slice, [0, 1, 2, 1])

    def test_dispatch_two_batteries_independent_slices(self):
        reg = DeviceRegistry()
        bat1 = SimpleBattery("bat1", prediction_hours=3)
        bat2 = SimpleBattery("bat2", prediction_hours=3)
        reg.register(bat1)
        reg.register(bat2)
        assembler = GenomeAssembler(reg)

        genome = np.array([2, 1, 0, 0, 2, 1], dtype=float)
        assembler.dispatch(genome, reg)

        np.testing.assert_array_equal(bat1.last_genome_slice, [2, 1, 0])
        np.testing.assert_array_equal(bat2.last_genome_slice, [0, 2, 1])

    def test_pv_apply_genome_not_called_with_slice(self):
        """PV has no genome slice — apply_genome receives empty call only if at all."""
        reg = DeviceRegistry()
        pv = SimplePV("pv1", np.zeros(4))
        bat = SimpleBattery("bat1", prediction_hours=4)
        reg.register(pv)
        reg.register(bat)
        assembler = GenomeAssembler(reg)

        genome = np.array([0, 1, 2, 0], dtype=float)
        assembler.dispatch(genome, reg)

        # Battery got its slice; PV has no recorded genome slice
        assert bat.last_genome_slice is not None

    def test_dispatch_wrong_genome_length_raises(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=24))
        assembler = GenomeAssembler(reg)

        with pytest.raises(ValueError, match="24"):
            assembler.dispatch(np.zeros(10), reg)

    def test_dispatch_out_of_bounds_value_raises(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=4))
        assembler = GenomeAssembler(reg)

        # SimpleBattery declares high=2.0; value of 5 is out of bounds
        genome = np.array([0.0, 1.0, 5.0, 2.0])
        with pytest.raises(ValueError, match="bat1"):
            assembler.dispatch(genome, reg)

    def test_dispatch_empty_assembler_empty_genome(self):
        reg = DeviceRegistry()
        assembler = GenomeAssembler(reg)
        assembler.dispatch(np.array([]), reg)  # should not raise


class TestGenomeAssemblerSliceAccess:
    """get_slice() and set_slice() for targeted genome inspection."""

    def test_get_slice_returns_correct_view(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=3))
        reg.register(SimpleBattery("bat2", prediction_hours=3))
        assembler = GenomeAssembler(reg)

        genome = np.array([1.0, 2.0, 0.0, 0.0, 1.0, 2.0])
        slice_bat1 = assembler.get_slice("bat1", genome)
        slice_bat2 = assembler.get_slice("bat2", genome)

        np.testing.assert_array_equal(slice_bat1, [1.0, 2.0, 0.0])
        np.testing.assert_array_equal(slice_bat2, [0.0, 1.0, 2.0])

    def test_get_slice_unknown_device_raises(self):
        reg = DeviceRegistry()
        assembler = GenomeAssembler(reg)
        with pytest.raises(KeyError, match="nonexistent"):
            assembler.get_slice("nonexistent", np.zeros(10))

    def test_get_slice_pv_raises_no_genome(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.zeros(4)))
        assembler = GenomeAssembler(reg)
        with pytest.raises(KeyError):
            assembler.get_slice("pv1", np.zeros(10))

    def test_set_slice_modifies_genome_in_place(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=3))
        assembler = GenomeAssembler(reg)

        genome = np.zeros(3)
        assembler.set_slice("bat1", genome, np.array([1.0, 2.0, 0.0]))
        np.testing.assert_array_equal(genome, [1.0, 2.0, 0.0])

    def test_set_slice_out_of_bounds_raises(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=3))
        assembler = GenomeAssembler(reg)
        genome = np.zeros(3)
        with pytest.raises(ValueError):
            assembler.set_slice("bat1", genome, np.array([5.0, 5.0, 5.0]))


class TestGenomeAssemblerRandomGenome:
    """random_genome() generates valid genomes within bounds."""

    def test_random_genome_correct_length(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=24))
        assembler = GenomeAssembler(reg)
        genome = assembler.random_genome(np.random.default_rng(42))
        assert len(genome) == 24

    def test_random_genome_within_bounds(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=100))
        assembler = GenomeAssembler(reg)
        genome = assembler.random_genome(np.random.default_rng(0))
        assert np.all(genome >= 0)
        assert np.all(genome <= 2)

    def test_random_genome_reproducible_with_seed(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=24))
        assembler = GenomeAssembler(reg)
        g1 = assembler.random_genome(np.random.default_rng(42))
        g2 = assembler.random_genome(np.random.default_rng(42))
        np.testing.assert_array_equal(g1, g2)

    def test_random_genome_empty_assembler(self):
        reg = DeviceRegistry()
        assembler = GenomeAssembler(reg)
        genome = assembler.random_genome()
        assert len(genome) == 0

    def test_random_genome_dispatches_without_error(self):
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1", prediction_hours=24)
        reg.register(bat)
        assembler = GenomeAssembler(reg)
        genome = assembler.random_genome(np.random.default_rng(1))
        # A random genome must always be dispatchable
        assembler.dispatch(genome, reg)


class TestGenomeAssemblerRepair:
    """repair_genome() collects repaired slices from devices into a new genome."""

    class MockDeviceWithRepair(SimpleBattery):
        """Battery-like device that proposes a repaired genome slice."""

        def __init__(self, device_id: str, prediction_hours: int, repair_slice: np.ndarray | None):
            super().__init__(device_id, prediction_hours)
            self._repair_slice = repair_slice
            self.repair_called = False

        def repair_genome(self):
            self.repair_called = True
            if self._repair_slice is None:
                return None
            from akkudoktoreos.devices.genetic2.base import GenomeRepairResult
            return GenomeRepairResult(repaired_slice=self._repair_slice, changed=True)

    def test_repair_applies_only_changed_slices(self):
        reg = DeviceRegistry()
        device1 = self.MockDeviceWithRepair(
            "bat1", prediction_hours=24, repair_slice=np.full(24, 2)
        )
        device2 = self.MockDeviceWithRepair(
            "bat2", prediction_hours=24, repair_slice=None
        )
        reg.register(device1)
        reg.register(device2)

        assembler = GenomeAssembler(reg)

        # Original genome must match total_size
        genome = np.arange(48, dtype=float)

        # Repair genome
        repaired = assembler.repair_genome(genome)

        # bat1 should be replaced with 2s, bat2 unchanged
        np.testing.assert_array_equal(
            repaired[:24], np.full(24, 2)
        )
        np.testing.assert_array_equal(
            repaired[24:], genome[24:]
        )

    def test_repair_genome_no_changes_returns_copy(self):
        reg = DeviceRegistry()
        device = self.MockDeviceWithRepair("bat1", prediction_hours=3, repair_slice=None)
        reg.register(device)
        assembler = GenomeAssembler(reg)

        genome = np.array([0.0, 1.0, 2.0])
        repaired = assembler.repair_genome(genome)

        # Should produce a new array but identical content
        assert np.all(repaired == genome)
        assert repaired is not genome  # must be a copy

    def test_repair_genome_empty_assembler(self):
        reg = DeviceRegistry()
        assembler = GenomeAssembler(reg)
        genome = np.array([], dtype=float)
        repaired = assembler.repair_genome(genome)
        assert repaired.shape == (0,)


class TestGenomeAssemblerDescribe:
    """describe() produces human-readable layout output."""

    def test_describe_contains_device_id(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("battery_main", prediction_hours=24))
        assembler = GenomeAssembler(reg)
        desc = assembler.describe()
        assert "battery_main" in desc

    def test_describe_contains_total_size(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=24))
        assembler = GenomeAssembler(reg)
        desc = assembler.describe()
        assert "total_size=24" in desc

    def test_describe_contains_slice_indices(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", prediction_hours=12))
        reg.register(SimpleBattery("bat2", prediction_hours=12))
        assembler = GenomeAssembler(reg)
        desc = assembler.describe()
        assert "0:" in desc
        assert "12:" in desc

    def test_describe_empty_assembler(self):
        reg = DeviceRegistry()
        assembler = GenomeAssembler(reg)
        desc = assembler.describe()
        assert "no devices" in desc.lower()
