"""Simulation context and run-scoped infrastructure for genetic2 framework.

This module defines the class `SimulationContext`, which represents the
complete run-scoped environment for a single optimization or simulation
execution within EOS.

The context is injected into all devices during ``setup_run()`` and provides:

    * Time structure (start time, step times, horizon)
    * Access to aligned time-series prediction data
    * Access to measurement data
    * Scenario isolation

The context is immutable after construction and must not be mutated during
simulation. Devices may extract required data during ``setup_run()`` but
must not retain references to the full context object.

Typical usage:
    context = SimulationContext(...)
    device.setup_run(context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Do not use datetimemutil - we do not need the pydantic model
from pendulum import DateTime, Duration

from akkudoktoreos.core.cache import cache_energy_management
from akkudoktoreos.core.coreabc import get_measurement, get_prediction


@dataclass(slots=True)
class SimulationContext:
    """Run-scoped immutable context injected into all devices during simulation setup.

    The SimulationContext encapsulates all information that is specific to
    a single simulation or optimization run.

    It provides:
        * Time structure definition
        * Cached forecast and signal resolution
        * Scenario isolation

    Devices must extract required data during ``setup_run()`` and must not
    retain a reference to the full context object to avoid unintended coupling.

    Attributes:
        step_times: Ordered sequence of ``DateTime`` timestamps defining
            the simulation horizon. Length determines the genome horizon
            size. Passed directly to ``device.setup_run()`` and forwarded
            into ``SingleStateBatchState.step_times`` so that every
            simulation lifecycle method has access to real timestamps —
            useful for time-of-use pricing in ``compute_cost`` and for
            building S2 ``execution_time`` fields in ``extract_instructions``.
        step_interval: Fixed time delta between consecutive steps, in seconds.
        horizon: Number of time steps in the simulation horizon.
    """

    # --- Time structure ---
    step_times: tuple[DateTime, ...]
    step_interval: Duration

    # --- Derived ---
    horizon: int = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived attributes after construction."""
        self.horizon = len(self.step_times)

    # ------------------------------------------------------------------
    # Generic resolution
    # ------------------------------------------------------------------

    @cache_energy_management
    def resolve_prediction(self, key: str) -> np.ndarray:
        """Resolve an arbitrary prediction key aligned to this run.

        This method is intended to be called by devices during
        ``setup_run()`` to extract required forecast data.

        Args:
            key: Unique identifier of the prediction time-series.

        Returns:
            NumPy array of shape (horizon,) aligned to the simulation horizon.

        Raises:
            KeyError: If the key does not exist in the store.
            ValueError: If alignment to the horizon fails.
        """
        return get_prediction().key_to_array(
            key=key,
            start_datetime=self.step_times[0],
            end_datetime=self.step_times[-1] + self.step_interval,  # exclusive
            interval=self.step_interval,
            dropna=True,
            boundary="context",
            align_to_interval=True,
        )

    @cache_energy_management
    def resolve_measurement(self, key: str) -> Optional[float]:
        """Resolve an arbitrary measurement key aligned to this run.

        This method is intended to be called by devices during
        ``setup_run()`` to extract required measurement data.

        Args:
            key: Unique identifier of the measurement.

        Returns:
            Float aligned to the simulation start or None if not found.

        Raises:
            KeyError: If the key does not exist in the store.
        """
        return get_measurement().key_to_value(
            key=key,
            target_datetime=self.step_times[0],
            time_window=self.step_interval * self.horizon,
        )
