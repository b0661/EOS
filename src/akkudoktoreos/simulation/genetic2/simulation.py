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
from typing import Any, Optional

import numpy as np
import pandas as pd

# Do not use datetimemutil - we do not need the pydantic model
from pendulum import DateTime, Duration

from akkudoktoreos.core.cache import cache_energy_management
from akkudoktoreos.core.coreabc import get_config, get_measurement, get_prediction

# Track the prediction keys that were accessed during the simulation run
_prediction_key_tracker: set[str] = set()


def reset_prediction_key_tracker() -> None:
    _prediction_key_tracker.clear()


@dataclass(slots=True, frozen=True)
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
        object.__setattr__(self, "horizon", len(self.step_times))
        reset_prediction_key_tracker()

    # ------------------------------------------------------------------
    # Generic resolution
    # ------------------------------------------------------------------

    def resolve_prediction(self, key: str) -> np.ndarray:
        """Resolve an arbitrary prediction key aligned to this run.

        This method is intended to be called by devices during
        ``setup_run()`` to extract required forecast data.

        The key is recorded to later be able to get all predictions that
        were used in the simulation

        Args:
            key: Unique identifier of the prediction time-series.

        Returns:
            NumPy array of shape (horizon,) aligned to the simulation horizon.

        Raises:
            KeyError: If the key does not exist in the store.
            ValueError: If alignment to the horizon fails.
        """
        # prediction may already be cached, even before start of simulation
        # Ensure the key is recorded.
        _prediction_key_tracker.add(key)
        return self._resolve_prediction_cached(key)

    @cache_energy_management
    def _resolve_prediction_cached(self, key: str) -> np.ndarray:
        return get_prediction().key_to_array(
            key=key,
            start_datetime=self.step_times[0],
            end_datetime=self.step_times[-1] + self.step_interval,  # exclusive
            interval=self.step_interval,
            dropna=True,
            boundary="context",
            align_to_interval=True,
        )

    def resolved_predictions(self) -> pd.DataFrame:
        """Return all prediction arrays that were requested during this run."""
        tz = self.step_times[0].timezone_name
        reference_index = pd.date_range(
            start=self.step_times[0].replace(tzinfo=None),
            end=(self.step_times[-1] + self.step_interval).replace(tzinfo=None),
            freq=self.step_interval,
            inclusive="left",
            tz=tz,
        )
        keys = _prediction_key_tracker
        data: dict[str, Any] = {
            "date_time": reference_index,
        }
        for key in keys:
            try:
                array = self.resolve_prediction(key)
                if len(array) != len(reference_index):
                    raise ValueError(
                        f"Array length mismatch for key '{key}' (expected {len(reference_index)}, got {len(array)})"
                    )

                data[key] = self.resolve_prediction(key)
            except KeyError as e:
                raise KeyError(f"Failed to retrieve data for key '{key}': {e}")

        if not data:
            raise KeyError(f"No valid data found for the requested keys {keys}.")

        df = pd.DataFrame(data, index=reference_index)
        return df

    @cache_energy_management
    def resolve_measurement(self, key: str) -> Optional[float]:
        """Resolve an arbitrary measurement key aligned to this run.

        This method is intended to be called by devices during
        ``setup_run()`` to extract required measurement data.

        Args:
            key: Unique identifier of the measurement. An empty or blank
                string is treated as "not configured" and returns ``None``
                immediately without querying the measurement store.

        Returns:
            Float aligned to the simulation start or None if not found.

        Raises:
            KeyError: If the key does not exist in the store.
        """
        if not key or not key.strip():
            return None
        return get_measurement().key_to_value(
            key=key,
            target_datetime=self.step_times[0],
            time_window=self.step_interval * self.horizon,
        )

    @cache_energy_management
    def resolve_config_cycle_time_windows(self, config_path: str) -> tuple[list[int], np.ndarray]:
        """Resolve a ``CycleTimeWindowSequence`` from the config by path.

        Retrieves the ``CycleTimeWindowSequence`` stored at ``config_path`` in
        the EOS config tree (using ``/``-separated path notation as defined by
        ``PydanticModelNestedValueMixin.get_nested_value``) and converts it into
        a ``(cycles, matrix)`` pair aligned to the simulation horizon.

        The matrix rows correspond to the sorted cycle indices present in the
        sequence.  Each cell ``matrix[k, t]`` is ``1.0`` when step ``t`` falls
        inside any window belonging to cycle ``cycles[k]``, ``0.0`` otherwise.
        The same time-grid parameters as ``resolve_prediction`` are used —
        identical ``start_datetime``, ``end_datetime``, and ``interval`` — so
        window arrays are directly comparable with prediction arrays.

        Results are cached by ``config_path`` string (via
        ``@cache_energy_management``) so repeated calls within a single run pay
        the resolution cost only once.

        Args:
            config_path: ``/``-separated path into the EOS config tree, e.g.
                ``"devices/home_appliances/0/cycle_time_windows"``.

        Returns:
            ``(cycle_indices, matrix)`` where

            * ``cycle_indices`` is a sorted ``list[int]`` of cycle numbers found
              in the sequence (e.g. ``[0, 1, 2]``).
            * ``matrix`` is a ``np.ndarray`` of shape
              ``(len(cycle_indices), horizon)`` with ``dtype=float64``.

        Raises:
            KeyError: If ``config_path`` does not resolve to a value in the
                config tree.
            TypeError: If the resolved value is not a ``CycleTimeWindowSequence``.
        """
        from akkudoktoreos.config.configabc import CycleTimeWindowSequence

        seq = get_config().get_nested_value(config_path)
        if not isinstance(seq, CycleTimeWindowSequence):
            raise TypeError(
                f"Config path '{config_path}' resolved to {type(seq).__name__}, "
                "expected CycleTimeWindowSequence."
            )
        cycle_indices, matrix = seq.cycles_to_matrix(
            start_datetime=self.step_times[0],
            end_datetime=self.step_times[-1] + self.step_interval,  # exclusive
            interval=self.step_interval,
        )
        return cycle_indices, matrix
