"""Batch simulation state container.

This module defines the top-level mutable state container used during
population evaluation in the simulation engine.

``BatchSimulationState`` holds all per-device state objects for a single
generation's evaluation. It is created fresh at the start of each
``evaluate_population()`` call and never shared between generations or
between concurrent evaluations.

Device-specific state objects (e.g. ``SingleStateBatchState``) are stored
by ``device_id`` and retrieved by the engine when orchestrating the
per-device lifecycle steps. Their concrete types are defined alongside
the device classes that own them — see ``devicesabc.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatchSimulationState:
    """Mutable top-level state container for one generation's batch evaluation.

    Created once per ``evaluate_population()`` call. Each device's mutable
    runtime data lives in a device-specific state object stored here by
    ``device_id``. The engine retrieves these objects by key when calling
    per-device lifecycle methods.

    Attributes:
        device_states: Mapping of ``device_id`` to the device-specific
            batch state object. Values are typed as ``object`` because
            each device defines its own state type — callers are
            responsible for casting to the appropriate concrete type.
        population_size: Number of individuals in the current population.
            All device state arrays must have a leading axis of this size.
    """

    device_states: dict[str, object]
    population_size: int
