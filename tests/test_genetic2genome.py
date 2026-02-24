"""Tests for genetic2.optimization.genome.

Covers: GenomeAssembler construction, dispatch, bounds, get_slice, set_slice,
random_genome, describe, and error handling.

Changes from previous version:
- GenomeAssembler(registry) -> GenomeAssembler(registry, num_steps=N, step_interval=td)
- dispatch() calls store_genome(), not apply_genome()
  → tests check bat.get_stored_genome() instead of bat.last_genome_slice
- genome_requirements(num_steps, step_interval) now takes arguments
"""

from datetime import timedelta

import numpy as np
import pytest
from fixtures.genetic2fixtures import (
    STEP_INTERVAL,
    SimpleBattery,
    SimplePV,
    make_assembler,
)

from akkudoktoreos.devices.genetic2.base import GenomeSlice
from akkudoktoreos.optimization.genetic2.genome import GenomeAssembler
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestGenomeAssemblerConstruction:
    """GenomeAssembler correctly maps device requirements to genome slices."""

    def test_total_size_single_device(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=24))
        assembler = make_assembler(reg, n=24)
        assert assembler.total_size == 24

    def test_total_size_two_batteries(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=24))
        reg.register(SimpleBattery("bat2", num_steps=24))
        assembler = make_assembler(reg, n=24)
        assert assembler.total_size == 48

    def test_pv_does_not_contribute_to_genome(self):
        """Devices returning None from genome_requirements are skipped."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.zeros(24)))
        reg.register(SimpleBattery("bat1", num_steps=24))
        assembler = make_assembler(reg, n=24)
        assert assembler.total_size == 24  # only the battery

    def test_empty_registry_total_size_zero(self):
        reg = DeviceRegistry()
        assembler = make_assembler(reg, n=24)
        assert assembler.total_size == 0

    def test_only_pv_total_size_zero(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.zeros(24)))
        assembler = make_assembler(reg, n=24)
        assert assembler.total_size == 0

    def test_slices_are_non_overlapping(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=12))
        reg.register(SimpleBattery("bat2", num_steps=12))
        assembler = make_assembler(reg, n=12)
        assert assembler._slices["bat1"] == (0, 12)
        assert assembler._slices["bat2"] == (12, 24)

    def test_registration_order_determines_slice_order(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat_a", num_steps=10))
        reg.register(SimpleBattery("bat_b", num_steps=5))
        assembler = make_assembler(reg, n=10)
        assert assembler._slices["bat_a"] == (0, 10)
        assert assembler._slices["bat_b"] == (10, 20)

    def test_num_steps_propagated_to_genome_size(self):
        """GenomeSlice.size equals num_steps passed to the assembler."""
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=96))
        assembler = make_assembler(reg, n=96)
        assert assembler.total_size == 96

    def test_step_interval_stored_on_assembler(self):
        reg = DeviceRegistry()
        interval = timedelta(minutes=15)
        assembler = GenomeAssembler(DeviceRegistry(), num_steps=96, step_interval=interval)
        assert assembler._step_interval == interval


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

class TestGenomeAssemblerBounds:
    """bounds() returns correct lower and upper bound arrays."""

    def test_bounds_length_equals_total_size(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=24))
        assembler = make_assembler(reg, n=24)
        lows, highs = assembler.bounds()
        assert len(lows) == 24
        assert len(highs) == 24

    def test_bounds_values_from_genome_slice(self):
        """SimpleBattery declares low=0, high=2."""
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=6))
        assembler = make_assembler(reg, n=6)
        lows, highs = assembler.bounds()
        assert all(l == 0.0 for l in lows)
        assert all(h == 2.0 for h in highs)

    def test_bounds_concatenated_for_multiple_devices(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=3))
        reg.register(SimpleBattery("bat2", num_steps=3))
        assembler = make_assembler(reg, n=3)
        lows, highs = assembler.bounds()
        assert len(lows) == 6
        assert len(highs) == 6

    def test_empty_assembler_bounds_empty(self):
        assembler = make_assembler(DeviceRegistry(), n=24)
        lows, highs = assembler.bounds()
        assert lows == []
        assert highs == []


# ---------------------------------------------------------------------------
# Dispatch — now calls store_genome(), not apply_genome()
# ---------------------------------------------------------------------------

class TestGenomeAssemblerDispatch:
    """dispatch() writes slices into devices via store_genome()."""

    def test_dispatch_single_battery_stores_its_slice(self):
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1", num_steps=4)
        reg.register(bat)
        assembler = make_assembler(reg, n=4)

        genome = np.array([0.0, 1.0, 2.0, 1.0])
        assembler.dispatch(genome, reg)

        stored = bat.get_stored_genome()
        np.testing.assert_array_equal(stored, [0.0, 1.0, 2.0, 1.0])

    def test_dispatch_two_batteries_independent_slices(self):
        reg = DeviceRegistry()
        bat1 = SimpleBattery("bat1", num_steps=3)
        bat2 = SimpleBattery("bat2", num_steps=3)
        reg.register(bat1)
        reg.register(bat2)
        assembler = make_assembler(reg, n=3)

        genome = np.array([2.0, 1.0, 0.0, 0.0, 2.0, 1.0])
        assembler.dispatch(genome, reg)

        np.testing.assert_array_equal(bat1.get_stored_genome(), [2.0, 1.0, 0.0])
        np.testing.assert_array_equal(bat2.get_stored_genome(), [0.0, 2.0, 1.0])

    def test_dispatch_stores_defensive_copy(self):
        """Mutating the genome after dispatch must not alter stored slice."""
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1", num_steps=3)
        reg.register(bat)
        assembler = make_assembler(reg, n=3)

        genome = np.array([0.0, 1.0, 2.0])
        assembler.dispatch(genome, reg)
        genome[:] = 99.0  # mutate original

        stored = bat.get_stored_genome()
        assert np.all(stored != 99.0), "store_genome must copy, not reference"

    def test_pv_has_no_stored_genome_after_dispatch(self):
        """PV returns None from genome_requirements — dispatch skips it."""
        reg = DeviceRegistry()
        pv = SimplePV("pv1", np.zeros(4))
        bat = SimpleBattery("bat1", num_steps=4)
        reg.register(pv)
        reg.register(bat)
        assembler = make_assembler(reg, n=4)

        assembler.dispatch(np.array([0.0, 1.0, 2.0, 0.0]), reg)

        # Battery has its slice; PV has no genome (empty array by default)
        assert len(bat.get_stored_genome()) == 4
        assert len(pv.get_stored_genome()) == 0

    def test_dispatch_wrong_genome_length_raises(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=24))
        assembler = make_assembler(reg, n=24)
        with pytest.raises(ValueError, match="24"):
            assembler.dispatch(np.zeros(10), reg)

    def test_dispatch_out_of_bounds_value_raises(self):
        """SimpleBattery declares high=2.0; value of 5 is out of bounds."""
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=4))
        assembler = make_assembler(reg, n=4)
        with pytest.raises(ValueError, match="bat1"):
            assembler.dispatch(np.array([0.0, 1.0, 5.0, 2.0]), reg)

    def test_dispatch_empty_assembler_empty_genome(self):
        assembler = make_assembler(DeviceRegistry(), n=24)
        assembler.dispatch(np.array([]), DeviceRegistry())  # must not raise


# ---------------------------------------------------------------------------
# Slice access
# ---------------------------------------------------------------------------

class TestGenomeAssemblerSliceAccess:
    """get_slice() and set_slice() for targeted genome inspection and repair."""

    def test_get_slice_returns_correct_view(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=3))
        reg.register(SimpleBattery("bat2", num_steps=3))
        assembler = make_assembler(reg, n=3)

        genome = np.array([1.0, 2.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(assembler.get_slice("bat1", genome), [1.0, 2.0, 0.0])
        np.testing.assert_array_equal(assembler.get_slice("bat2", genome), [0.0, 1.0, 2.0])

    def test_get_slice_unknown_device_raises(self):
        assembler = make_assembler(DeviceRegistry(), n=24)
        with pytest.raises(KeyError, match="nonexistent"):
            assembler.get_slice("nonexistent", np.zeros(10))

    def test_get_slice_pv_raises_no_genome(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.zeros(4)))
        assembler = make_assembler(reg, n=4)
        with pytest.raises(KeyError):
            assembler.get_slice("pv1", np.zeros(10))

    def test_set_slice_modifies_genome_in_place(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=3))
        assembler = make_assembler(reg, n=3)

        genome = np.zeros(3)
        assembler.set_slice("bat1", genome, np.array([1.0, 2.0, 0.0]))
        np.testing.assert_array_equal(genome, [1.0, 2.0, 0.0])

    def test_set_slice_out_of_bounds_raises(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=3))
        assembler = make_assembler(reg, n=3)
        genome = np.zeros(3)
        with pytest.raises(ValueError):
            assembler.set_slice("bat1", genome, np.array([5.0, 5.0, 5.0]))


# ---------------------------------------------------------------------------
# Random genome
# ---------------------------------------------------------------------------

class TestGenomeAssemblerRandomGenome:
    """random_genome() generates valid genomes within declared bounds."""

    def test_random_genome_correct_length(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=24))
        assembler = make_assembler(reg, n=24)
        assert len(assembler.random_genome(np.random.default_rng(42))) == 24

    def test_random_genome_within_bounds(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=100))
        assembler = make_assembler(reg, n=100)
        genome = assembler.random_genome(np.random.default_rng(0))
        assert np.all(genome >= 0)
        assert np.all(genome <= 2)

    def test_random_genome_reproducible_with_seed(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=24))
        assembler = make_assembler(reg, n=24)
        g1 = assembler.random_genome(np.random.default_rng(42))
        g2 = assembler.random_genome(np.random.default_rng(42))
        np.testing.assert_array_equal(g1, g2)

    def test_random_genome_empty_assembler(self):
        assembler = make_assembler(DeviceRegistry(), n=24)
        assert len(assembler.random_genome()) == 0

    def test_random_genome_is_dispatchable(self):
        """A random genome must always satisfy declared bounds → dispatch OK."""
        reg = DeviceRegistry()
        bat = SimpleBattery("bat1", num_steps=24)
        reg.register(bat)
        assembler = make_assembler(reg, n=24)
        genome = assembler.random_genome(np.random.default_rng(1))
        assembler.dispatch(genome, reg)  # must not raise


# ---------------------------------------------------------------------------
# Describe
# ---------------------------------------------------------------------------

class TestGenomeAssemblerDescribe:
    """describe() produces a human-readable genome layout."""

    def test_describe_contains_device_id(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("battery_main", num_steps=24))
        assembler = make_assembler(reg, n=24)
        assert "battery_main" in assembler.describe()

    def test_describe_contains_total_size(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=24))
        assembler = make_assembler(reg, n=24)
        assert "total_size=24" in assembler.describe()

    def test_describe_contains_slice_indices(self):
        reg = DeviceRegistry()
        reg.register(SimpleBattery("bat1", num_steps=12))
        reg.register(SimpleBattery("bat2", num_steps=12))
        assembler = make_assembler(reg, n=12)
        desc = assembler.describe()
        assert "0:" in desc
        assert "12:" in desc

    def test_describe_empty_assembler(self):
        assembler = make_assembler(DeviceRegistry(), n=24)
        assert "no devices" in assembler.describe().lower()
