"""Tests for GenomeSlice, AssembledGenome, and GenomeAssembler.

Covers:
    GenomeSlice
        - Construction with and without bounds
        - Immutability (FrozenInstanceError)
        - end property
        - extract() on 1-D genome (flat vector)
        - extract() on 2-D genome (population batch)
        - extract() returns a view, not a copy

    AssembledGenome
        - Construction
        - Immutability

    GenomeAssembler.assemble()
        - Empty device list produces zero-size genome
        - Device returning None is skipped
        - Device returning size-zero slice is skipped
        - Single device gets start=0
        - Multiple devices get non-overlapping sequential slices
        - Slices are keyed by device_id
        - Devices registered in order determine gene ordering
        - lower_bounds and upper_bounds filled correctly
        - Genes with no declared bound default to -inf / +inf
        - Mixed bounded and unbounded devices in same genome

    GenomeAssembler.extract_device_genome()
        - Returns correct slice from flat 1-D genome
        - Returns correct slice from 2-D population batch
        - Returns empty array for device with no slice (1-D)
        - Returns empty array for device with no slice (2-D)
        - Returned array is a view into the original genome
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from akkudoktoreos.optimization.genetic2.genome import (
    AssembledGenome,
    GenomeAssembler,
    GenomeSlice,
)

# ============================================================
# Stub helpers
# ============================================================

def make_device(
    device_id: str,
    size: int,
    lower: float | None = None,
    upper: float | None = None,
) -> SimpleNamespace:
    """Create a minimal device stub whose genome_requirements() returns
    a GenomeSlice of the requested size with optional uniform bounds.

    Uses SimpleNamespace to avoid depending on the full EnergyDevice ABC.
    The assembler only calls device.device_id and device.genome_requirements().
    """
    lb = np.full(size, lower) if lower is not None else None
    ub = np.full(size, upper) if upper is not None else None
    req = GenomeSlice(start=0, size=size, lower_bound=lb, upper_bound=ub)

    return SimpleNamespace(
        device_id=device_id,
        genome_requirements=lambda: req,
    )


def make_device_no_genome(device_id: str) -> SimpleNamespace:
    """Device that returns None from genome_requirements."""
    return SimpleNamespace(
        device_id=device_id,
        genome_requirements=lambda: None,
    )


def make_device_zero_size(device_id: str) -> SimpleNamespace:
    """Device that returns a size-zero GenomeSlice."""
    req = GenomeSlice(start=0, size=0, lower_bound=None, upper_bound=None)
    return SimpleNamespace(
        device_id=device_id,
        genome_requirements=lambda: req,
    )


# ============================================================
# TestGenomeSliceConstruction
# ============================================================

class TestGenomeSliceConstruction:
    def test_minimal_construction(self):
        slc = GenomeSlice(start=0, size=4)
        assert slc.start == 0
        assert slc.size == 4
        assert slc.lower_bound is None
        assert slc.upper_bound is None

    def test_construction_with_bounds(self):
        lb = np.full(4, -100.0)
        ub = np.full(4, 100.0)
        slc = GenomeSlice(start=2, size=4, lower_bound=lb, upper_bound=ub)
        np.testing.assert_array_equal(slc.lower_bound, lb)
        np.testing.assert_array_equal(slc.upper_bound, ub)

    def test_end_property(self):
        slc = GenomeSlice(start=3, size=5)
        assert slc.end == 8

    def test_end_property_start_zero(self):
        slc = GenomeSlice(start=0, size=10)
        assert slc.end == 10

    def test_immutable_start(self):
        slc = GenomeSlice(start=0, size=4)
        with pytest.raises(FrozenInstanceError):
            slc.start = 1

    def test_immutable_size(self):
        slc = GenomeSlice(start=0, size=4)
        with pytest.raises(FrozenInstanceError):
            slc.size = 8

    def test_immutable_lower_bound_reference(self):
        slc = GenomeSlice(start=0, size=2, lower_bound=np.zeros(2))
        with pytest.raises(FrozenInstanceError):
            slc.lower_bound = np.ones(2)

    def test_immutable_upper_bound_reference(self):
        slc = GenomeSlice(start=0, size=2, upper_bound=np.ones(2))
        with pytest.raises(FrozenInstanceError):
            slc.upper_bound = np.zeros(2)


# ============================================================
# TestGenomeSliceExtract1D
# ============================================================

class TestGenomeSliceExtract1D:
    def test_extract_from_start(self):
        slc = GenomeSlice(start=0, size=3)
        genome = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(slc.extract(genome), [1.0, 2.0, 3.0])

    def test_extract_from_middle(self):
        slc = GenomeSlice(start=2, size=3)
        genome = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(slc.extract(genome), [3.0, 4.0, 5.0])

    def test_extract_single_gene(self):
        slc = GenomeSlice(start=4, size=1)
        genome = np.arange(10.0)
        np.testing.assert_array_equal(slc.extract(genome), [4.0])

    def test_extract_full_genome(self):
        genome = np.arange(5.0)
        slc = GenomeSlice(start=0, size=5)
        np.testing.assert_array_equal(slc.extract(genome), genome)

    def test_extract_returns_view(self):
        """extract() must return a view so in-place edits affect the original."""
        slc = GenomeSlice(start=1, size=2)
        genome = np.array([0.0, 1.0, 2.0, 3.0])
        view = slc.extract(genome)
        view[0] = 99.0
        assert genome[1] == 99.0


# ============================================================
# TestGenomeSliceExtract2D
# ============================================================

class TestGenomeSliceExtract2D:
    def test_extract_batch_shape(self):
        """2-D genome (pop_size, total) should return (pop_size, size)."""
        slc = GenomeSlice(start=1, size=2)
        genome = np.arange(12.0).reshape(3, 4)
        result = slc.extract(genome)
        assert result.shape == (3, 2)

    def test_extract_batch_values(self):
        slc = GenomeSlice(start=0, size=2)
        genome = np.array([
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ])
        result = slc.extract(genome)
        np.testing.assert_array_equal(result, [[10.0, 20.0], [40.0, 50.0]])

    def test_extract_batch_middle_slice(self):
        slc = GenomeSlice(start=1, size=2)
        genome = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ])
        result = slc.extract(genome)
        np.testing.assert_array_equal(result, [[2.0, 3.0], [6.0, 7.0]])

    def test_extract_batch_returns_view(self):
        slc = GenomeSlice(start=0, size=2)
        genome = np.zeros((2, 4))
        view = slc.extract(genome)
        view[0, 0] = 99.0
        assert genome[0, 0] == 99.0


# ============================================================
# TestAssembledGenome
# ============================================================

class TestAssembledGenome:
    def test_construction(self):
        slc = GenomeSlice(start=0, size=3)
        ag = AssembledGenome(
            total_size=3,
            slices={"dev_0": slc},
            lower_bounds=np.full(3, -np.inf),
            upper_bounds=np.full(3, np.inf),
        )
        assert ag.total_size == 3
        assert "dev_0" in ag.slices

    def test_immutable_total_size(self):
        ag = AssembledGenome(
            total_size=0,
            slices={},
            lower_bounds=np.array([]),
            upper_bounds=np.array([]),
        )
        with pytest.raises(FrozenInstanceError):
            ag.total_size = 10

    def test_immutable_slices_reference(self):
        ag = AssembledGenome(
            total_size=0,
            slices={},
            lower_bounds=np.array([]),
            upper_bounds=np.array([]),
        )
        with pytest.raises(FrozenInstanceError):
            ag.slices = {"new": GenomeSlice(start=0, size=1)}


# ============================================================
# TestGenomeAssemblerAssemble
# ============================================================

class TestGenomeAssemblerAssemble:
    def test_empty_device_list_produces_zero_size_genome(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([])
        assert result.total_size == 0
        assert result.slices == {}
        assert result.lower_bounds.shape == (0,)
        assert result.upper_bounds.shape == (0,)

    def test_device_returning_none_is_skipped(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([make_device_no_genome("dev_0")])
        assert result.total_size == 0
        assert "dev_0" not in result.slices

    def test_device_returning_zero_size_is_skipped(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([make_device_zero_size("dev_0")])
        assert result.total_size == 0
        assert "dev_0" not in result.slices

    def test_single_device_gets_start_zero(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([make_device("dev_0", size=5)])
        assert result.slices["dev_0"].start == 0
        assert result.slices["dev_0"].size == 5
        assert result.total_size == 5

    def test_two_devices_get_sequential_non_overlapping_slices(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([
            make_device("dev_0", size=4),
            make_device("dev_1", size=6),
        ])
        slc_0 = result.slices["dev_0"]
        slc_1 = result.slices["dev_1"]

        assert slc_0.start == 0
        assert slc_0.end == 4
        assert slc_1.start == 4
        assert slc_1.end == 10
        assert result.total_size == 10

    def test_three_devices_sequential_layout(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([
            make_device("a", size=3),
            make_device("b", size=2),
            make_device("c", size=5),
        ])
        assert result.slices["a"].start == 0
        assert result.slices["b"].start == 3
        assert result.slices["c"].start == 5
        assert result.total_size == 10

    def test_slices_keyed_by_device_id(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([make_device("my_battery", size=4)])
        assert "my_battery" in result.slices

    def test_skipped_device_does_not_shift_subsequent_slices(self):
        """A device with no genome must not consume any index space."""
        assembler = GenomeAssembler()
        result = assembler.assemble([
            make_device("dev_0", size=3),
            make_device_no_genome("dev_skip"),
            make_device("dev_1", size=4),
        ])
        assert result.slices["dev_0"].start == 0
        assert result.slices["dev_1"].start == 3
        assert result.total_size == 7

    def test_registration_order_determines_gene_ordering(self):
        """Swapping device order must swap their slice positions."""
        assembler = GenomeAssembler()

        result_ab = assembler.assemble([
            make_device("a", size=3),
            make_device("b", size=5),
        ])
        result_ba = assembler.assemble([
            make_device("b", size=5),
            make_device("a", size=3),
        ])

        assert result_ab.slices["a"].start == 0
        assert result_ab.slices["b"].start == 3
        assert result_ba.slices["b"].start == 0
        assert result_ba.slices["a"].start == 5

    def test_lower_bounds_filled_from_device_requirements(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([make_device("dev_0", size=3, lower=-50.0)])
        np.testing.assert_array_equal(result.lower_bounds, [-50.0, -50.0, -50.0])

    def test_upper_bounds_filled_from_device_requirements(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([make_device("dev_0", size=3, upper=100.0)])
        np.testing.assert_array_equal(result.upper_bounds, [100.0, 100.0, 100.0])

    def test_genes_without_declared_bounds_default_to_inf(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([make_device("dev_0", size=3)])
        assert np.all(result.lower_bounds == -np.inf)
        assert np.all(result.upper_bounds == np.inf)

    def test_mixed_bounded_and_unbounded_devices(self):
        """Bounded and unbounded devices in the same genome must not interfere."""
        assembler = GenomeAssembler()
        result = assembler.assemble([
            make_device("bounded", size=2, lower=-10.0, upper=10.0),
            make_device("unbounded", size=3),
        ])
        # First 2 genes: bounded
        np.testing.assert_array_equal(result.lower_bounds[:2], [-10.0, -10.0])
        np.testing.assert_array_equal(result.upper_bounds[:2], [10.0, 10.0])
        # Next 3 genes: unbounded
        assert np.all(result.lower_bounds[2:] == -np.inf)
        assert np.all(result.upper_bounds[2:] == np.inf)

    def test_two_devices_both_bounded_correct_positions(self):
        assembler = GenomeAssembler()
        result = assembler.assemble([
            make_device("dev_0", size=2, lower=0.0, upper=5.0),
            make_device("dev_1", size=3, lower=-1.0, upper=1.0),
        ])
        np.testing.assert_array_equal(result.lower_bounds, [0.0, 0.0, -1.0, -1.0, -1.0])
        np.testing.assert_array_equal(result.upper_bounds, [5.0, 5.0, 1.0, 1.0, 1.0])


# ============================================================
# TestGenomeAssemblerExtractDeviceGenome
# ============================================================

class TestGenomeAssemblerExtractDeviceGenome:
    @pytest.fixture()
    def two_device_assembly(self) -> AssembledGenome:
        return GenomeAssembler().assemble([
            make_device("dev_0", size=3),
            make_device("dev_1", size=4),
        ])

    def test_extract_first_device_1d(self, two_device_assembly):
        genome = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = GenomeAssembler.extract_device_genome(genome, "dev_0", two_device_assembly)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_extract_second_device_1d(self, two_device_assembly):
        genome = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = GenomeAssembler.extract_device_genome(genome, "dev_1", two_device_assembly)
        np.testing.assert_array_equal(result, [4.0, 5.0, 6.0, 7.0])

    def test_extract_first_device_2d(self, two_device_assembly):
        genome = np.arange(14.0).reshape(2, 7)
        result = GenomeAssembler.extract_device_genome(genome, "dev_0", two_device_assembly)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, [[0.0, 1.0, 2.0], [7.0, 8.0, 9.0]])

    def test_extract_second_device_2d(self, two_device_assembly):
        genome = np.arange(14.0).reshape(2, 7)
        result = GenomeAssembler.extract_device_genome(genome, "dev_1", two_device_assembly)
        assert result.shape == (2, 4)
        np.testing.assert_array_equal(result, [[3.0, 4.0, 5.0, 6.0], [10.0, 11.0, 12.0, 13.0]])

    def test_unknown_device_id_returns_empty_1d(self, two_device_assembly):
        genome = np.zeros(7)
        result = GenomeAssembler.extract_device_genome(genome, "ghost", two_device_assembly)
        assert result.shape == (0,)

    def test_unknown_device_id_returns_empty_2d(self, two_device_assembly):
        genome = np.zeros((3, 7))
        result = GenomeAssembler.extract_device_genome(genome, "ghost", two_device_assembly)
        assert result.shape == (3, 0)

    def test_extract_1d_returns_view(self, two_device_assembly):
        genome = np.zeros(7)
        result = GenomeAssembler.extract_device_genome(genome, "dev_0", two_device_assembly)
        result[0] = 99.0
        assert genome[0] == 99.0

    def test_extract_2d_returns_view(self, two_device_assembly):
        genome = np.zeros((2, 7))
        result = GenomeAssembler.extract_device_genome(genome, "dev_1", two_device_assembly)
        result[0, 0] = 99.0
        assert genome[0, 3] == 99.0
