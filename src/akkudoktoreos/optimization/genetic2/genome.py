"""Genome assembly and dispatch for the optimizer.

The GenomeAssembler is the contract between the optimizer and the devices.
Devices declare their genome needs via ``genome_requirements()``; the assembler
assigns non-overlapping slices and handles dispatch back to devices before
each simulation run.

This means:
- The optimizer only sees a flat array of numbers.
- Devices only see their own slice.
- Adding a new device automatically extends the genome — no optimizer changes needed.
"""

import numpy as np

from akkudoktoreos.devices.genetic2.base import EnergyDevice, GenomeSlice
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry


class GenomeAssembler:
    """Builds, validates, and dispatches optimizer genomes from device requirements.

    At construction, the assembler queries every registered device for its
    genome requirements and assigns contiguous, non-overlapping slices.
    The total genome is the concatenation of all device slices in device
    registration order.

    This class is load-bearing — a slicing bug here corrupts every device's
    schedule simultaneously and is very hard to diagnose. The design therefore
    prioritizes validation and clear error messages over performance.

    Args:
        registry (DeviceRegistry): Registry of all devices participating
            in optimization. Devices with ``genome_requirements() == None``
            are registered but get no genome slice.

    Example:
        >>> assembler = GenomeAssembler(registry)
        >>> print(assembler.total_size)
        96
        >>> lows, highs = assembler.bounds()
        >>> genome = np.random.randint(lows, highs)
        >>> assembler.dispatch(genome, registry)  # devices receive their slices
    """

    def __init__(self, registry: DeviceRegistry) -> None:
        # Map device_id -> (start_index, end_index) in the flat genome
        self._slices: dict[str, tuple[int, int]] = {}
        # Ordered list of GenomeSlice declarations (for bounds and validation)
        self._requirements: list[GenomeSlice] = []
        # Devices that have genome requirements (in registration order)
        self._genome_devices: list[str] = []

        offset = 0
        for device in registry.all_of_type(EnergyDevice):
            req = device.genome_requirements()
            if req is None:
                continue
            if req.size <= 0:
                raise ValueError(
                    f"Device '{device.device_id}': genome_requirements().size "
                    f"must be > 0, got {req.size}."
                )
            self._slices[device.device_id] = (offset, offset + req.size)
            self._requirements.append(req)
            self._genome_devices.append(device.device_id)
            offset += req.size

        self.total_size: int = offset
        """Total number of genome slots across all devices."""

    def dispatch(self, genome: np.ndarray, registry: DeviceRegistry) -> None:
        """Slice the genome and send each slice to its device.

        Called once per simulation run before any ``simulate_hour()`` call.
        Each device receives exactly the slice it declared in
        ``genome_requirements()``.

        Args:
            genome (np.ndarray): Flat genome array of length ``total_size``.
            registry (DeviceRegistry): Registry containing all devices.
                Must be the same registry used to construct this assembler.

        Raises:
            ValueError: If genome length does not match ``total_size``.
            ValueError: If any slice violates the device's declared bounds.
            KeyError: If a device that declared requirements is no longer
                in the registry.
        """
        if len(genome) != self.total_size:
            raise ValueError(
                f"Genome length {len(genome)} does not match expected "
                f"{self.total_size} (sum of all device requirements)."
            )

        for device_id, (start, end) in self._slices.items():
            device = registry.get(device_id)
            genome_slice = genome[start:end]

            # Validate slice against declared bounds before dispatching
            req = self._requirement_for(device_id)
            req.validate(genome_slice)

            device.apply_genome(genome_slice)

    def get_slice(self, device_id: str, genome: np.ndarray) -> np.ndarray:
        """Extract a specific device's slice from a genome array.

        Useful for inspecting or mutating a single device's genome portion
        without dispatching the full genome.

        Args:
            device_id (str): Device whose slice to extract.
            genome (np.ndarray): Full genome array.

        Returns:
            np.ndarray: View into genome for this device's slice.

        Raises:
            KeyError: If device has no genome slice (not registered or no requirements).
            ValueError: If genome is too short.
        """
        if device_id not in self._slices:
            raise KeyError(
                f"Device '{device_id}' has no genome slice. "
                "Either it is not registered or genome_requirements() returned None."
            )
        start, end = self._slices[device_id]
        if len(genome) < end:
            raise ValueError(
                f"Genome length {len(genome)} is too short to contain "
                f"slice [{start}:{end}] for device '{device_id}'."
            )
        return genome[start:end]

    def set_slice(self, device_id: str, genome: np.ndarray, values: np.ndarray) -> None:
        """Write values into a specific device's slice of the genome.

        Args:
            device_id (str): Target device.
            genome (np.ndarray): Full genome array to modify in place.
            values (np.ndarray): Values to write. Must match slice length.

        Raises:
            KeyError: If device has no genome slice.
            ValueError: If values length or bounds are incorrect.
        """
        if device_id not in self._slices:
            raise KeyError(f"Device '{device_id}' has no genome slice.")
        req = self._requirement_for(device_id)
        req.validate(values)
        start, end = self._slices[device_id]
        genome[start:end] = values

    def bounds(self) -> tuple[list[float], list[float]]:
        """Return lower and upper bounds for the full genome.

        Used by the optimizer to set up mutation and initialization operators.
        Bounds are repeated once per genome slot, so length equals ``total_size``.

        Returns:
            tuple[list[float], list[float]]: (lower_bounds, upper_bounds).
                Both lists have length ``total_size``.
        """
        lows: list[float] = []
        highs: list[float] = []
        for req in self._requirements:
            lows.extend([req.low] * req.size)
            highs.extend([req.high] * req.size)
        return lows, highs

    def random_genome(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate a random valid genome respecting all device bounds.

        Useful for initializing optimizer populations.

        Args:
            rng (np.random.Generator | None): Optional random generator for
                reproducibility. Uses ``np.random.default_rng()`` if None.

        Returns:
            np.ndarray: Random genome of length ``total_size``.
        """
        if rng is None:
            rng = np.random.default_rng()
        lows, highs = self.bounds()
        genome = np.zeros(self.total_size)
        for req in self._requirements:
            start, end = self._slices[req.device_id]
            if req.dtype == int:
                genome[start:end] = rng.integers(int(req.low), int(req.high) + 1, size=req.size)
            else:
                genome[start:end] = rng.uniform(req.low, req.high, size=req.size)
        return genome

    def describe(self) -> str:
        """Return a human-readable description of the genome layout.

        Useful for debugging genome structure and verifying device slice
        assignments are correct.

        Returns:
            str: Multi-line description of each device's genome slice.
        """
        if not self._requirements:
            return "GenomeAssembler: no devices with genome requirements."
        lines = [f"GenomeAssembler: total_size={self.total_size}"]
        for req in self._requirements:
            start, end = self._slices[req.device_id]
            lines.append(
                f"  [{start:4d}:{end:4d}] {req.device_id} "
                f"(size={req.size}, dtype={req.dtype.__name__}, "
                f"range=[{req.low}, {req.high}])"
                + (f"\n           {req.description}" if req.description else "")
            )
        return "\n".join(lines)

    def _requirement_for(self, device_id: str) -> GenomeSlice:
        """Look up the GenomeSlice declaration for a device.

        Args:
            device_id (str): Device ID to look up.

        Returns:
            GenomeSlice: The declaration for this device.

        Raises:
            KeyError: If no requirement is found.
        """
        for req in self._requirements:
            if req.device_id == device_id:
                return req
        raise KeyError(f"No genome requirement found for device '{device_id}'.")

    def __repr__(self) -> str:
        return f"GenomeAssembler(total_size={self.total_size}, devices={self._genome_devices})"
