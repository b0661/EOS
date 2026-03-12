"""Genome definitions for the genetic optimisation framework.

This module defines immutable metadata structures describing how a device
maps into the global optimiser genome vector, and the assembler that
builds the global layout from per-device requirements.

``GenomeSlice``
    Describes where a device's genes live inside the global genome and
    what bounds apply. Contains no actual gene values.

``AssembledGenome``
    The immutable result of assembly: total genome size, per-device slice
    definitions keyed by ``device_id``, and global bound vectors.

``GenomeAssembler``
    Runs exactly once during optimiser setup:

    1. Collects genome requirements from all devices via
       ``device.genome_requirements()``.
    2. Assigns non-overlapping index ranges in device registration order.
    3. Builds global lower/upper bound vectors (default ``-inf`` / ``+inf``
       where a device declares no bound).
    4. Exposes helpers to extract per-device slices from a global genome
       batch of shape ``(population_size, horizon)``.

Slices are keyed by ``device_id`` (``str``) throughout, consistent with
``DeviceRegistry`` and ``BatchSimulationState``, and avoiding the need
for ``EnergyDevice`` instances to be hashable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from akkudoktoreos.devices.devicesabc import EnergyDevice


@dataclass(frozen=True, slots=True)
class GenomeSlice:
    """Immutable descriptor for one device's segment of the global genome.

    A ``GenomeSlice`` carries only structural metadata — it never holds
    actual gene values.

    Attributes:
        start: Inclusive start index inside the global genome vector.
        size: Number of genes assigned to this device.
        lower_bound: Per-gene lower bounds, shape ``(size,)``, or ``None``
            if no lower bound constraint applies.
        upper_bound: Per-gene upper bounds, shape ``(size,)``, or ``None``
            if no upper bound constraint applies.
    """

    start: int
    size: int
    lower_bound: np.ndarray | None = None
    upper_bound: np.ndarray | None = None

    @property
    def end(self) -> int:
        """Exclusive end index of this slice (``start + size``)."""
        return self.start + self.size

    def extract(self, genome: np.ndarray) -> np.ndarray:
        """Extract this device's genes from a genome array.

        Supports both a flat global genome vector and a population batch:

        - 1-D input ``(total_genome_size,)`` → returns ``(size,)``
        - 2-D input ``(population_size, total_genome_size)`` → returns
          ``(population_size, size)``

        Args:
            genome: Global genome array, either 1-D or 2-D.

        Returns:
            View into ``genome`` corresponding to this slice.
        """
        if genome.ndim == 1:
            return genome[self.start : self.end]
        return genome[:, self.start : self.end]


@dataclass(frozen=True, slots=True)
class AssembledGenome:
    """Immutable result of genome assembly.

    Produced once by ``GenomeAssembler.assemble()`` and reused for the
    entire optimisation run.

    Attributes:
        total_size: Total number of genes in the global genome vector.
        slices: Mapping of ``device_id`` to the assigned ``GenomeSlice``.
            Devices that returned ``None`` or size-zero requirements from
            ``genome_requirements()`` are absent from this mapping.
        lower_bounds: Global lower bound vector, shape ``(total_size,)``.
            Genes with no declared bound are set to ``-inf``.
        upper_bounds: Global upper bound vector, shape ``(total_size,)``.
            Genes with no declared bound are set to ``+inf``.
    """

    total_size: int
    slices: dict[str, GenomeSlice]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray


class GenomeAssembler:
    """Assembles per-device genome requirements into a single global genome.

    Instantiate once and call ``assemble()`` during optimiser setup.
    The resulting ``AssembledGenome`` is immutable and safe to reuse
    across all generations of the same run.
    """

    def assemble(self, devices: Iterable[EnergyDevice]) -> AssembledGenome:
        """Assemble genome slices for all participating devices.

        Iterates over ``devices`` in order, calls ``genome_requirements()``
        on each, and assigns non-overlapping index ranges. Devices that
        return ``None`` or a zero-size slice are skipped and will be absent
        from ``AssembledGenome.slices``.

        Args:
            devices: Devices participating in the optimisation. Iterated
                in registration order, which determines gene ordering in
                the global genome.

        Returns:
            Immutable ``AssembledGenome`` describing the full genome layout.
        """
        slices: dict[str, GenomeSlice] = {}
        cursor = 0

        for device in devices:
            requirement = device.genome_requirements()

            if requirement is None or requirement.size == 0:
                continue

            slices[device.device_id] = GenomeSlice(
                start=cursor,
                size=requirement.size,
                lower_bound=requirement.lower_bound,
                upper_bound=requirement.upper_bound,
            )
            cursor += requirement.size

        total_size = cursor

        # Build global bound vectors; default to unconstrained.
        lower_bounds = np.full(total_size, -np.inf)
        upper_bounds = np.full(total_size, np.inf)

        for slc in slices.values():
            if slc.lower_bound is not None:
                lower_bounds[slc.start : slc.end] = slc.lower_bound
            if slc.upper_bound is not None:
                upper_bounds[slc.start : slc.end] = slc.upper_bound

        return AssembledGenome(
            total_size=total_size,
            slices=slices,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    @staticmethod
    def extract_device_genome(
        genome: np.ndarray,
        device_id: str,
        assembled: AssembledGenome,
    ) -> np.ndarray:
        """Extract a device's genome slice from a global genome array.

        Supports both flat and population-batch genomes — see
        ``GenomeSlice.extract`` for shape details.

        Args:
            genome: Global genome array, shape ``(total_size,)`` or
                ``(population_size, total_size)``.
            device_id: ``device_id`` of the device whose genes to extract.
            assembled: Assembly result from ``GenomeAssembler.assemble()``.

        Returns:
            The device's gene array. Returns an empty array of shape
            ``(0,)`` or ``(population_size, 0)`` if the device has no
            slice in the assembled genome.
        """
        slc = assembled.slices.get(device_id)

        if slc is None:
            if genome.ndim == 1:
                return np.empty(0, dtype=float)
            return np.empty((genome.shape[0], 0), dtype=float)

        return slc.extract(genome)
