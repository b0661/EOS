"""Tests for BatchSimulationState.

BatchSimulationState is a plain mutable dataclass — a thin container with
no logic of its own. The tests therefore focus on the container contract:

    - Construction with valid arguments
    - Field access and mutation (it is intentionally mutable)
    - Default dataclass behaviour: equality, repr
    - The population_size field correctly reflects the batch dimension
    - device_states accepts any object as a value (typed as object)
    - Structural integrity after modification
"""

from __future__ import annotations

import pytest

from akkudoktoreos.simulation.genetic2.state import BatchSimulationState

# ============================================================
# Minimal stub state objects
# ============================================================

class _StubDeviceState:
    """Minimal stand-in for a device-specific batch state."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StubDeviceState) and self.label == other.label


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture()
def empty_state() -> BatchSimulationState:
    """BatchSimulationState with no devices."""
    return BatchSimulationState(device_states={}, population_size=10)


@pytest.fixture()
def populated_state() -> BatchSimulationState:
    """BatchSimulationState with two device states."""
    return BatchSimulationState(
        device_states={
            "battery_0": _StubDeviceState("battery_0"),
            "pv_0": _StubDeviceState("pv_0"),
        },
        population_size=32,
    )


# ============================================================
# TestBatchSimulationStateConstruction
# ============================================================

class TestBatchSimulationStateConstruction:
    def test_construction_with_empty_device_states(self, empty_state):
        assert empty_state.device_states == {}
        assert empty_state.population_size == 10

    def test_construction_with_device_states(self, populated_state):
        assert len(populated_state.device_states) == 2
        assert populated_state.population_size == 32

    def test_population_size_one(self):
        state = BatchSimulationState(device_states={}, population_size=1)
        assert state.population_size == 1

    def test_device_states_accepts_any_value_type(self):
        """device_states values are typed as object — any type must be accepted."""
        import numpy as np
        state = BatchSimulationState(
            device_states={
                "dev_a": _StubDeviceState("a"),
                "dev_b": {"schedule": np.zeros((10, 24))},
                "dev_c": 42,
                "dev_d": None,
            },
            population_size=10,
        )
        assert len(state.device_states) == 4


# ============================================================
# TestBatchSimulationStateFieldAccess
# ============================================================

class TestBatchSimulationStateFieldAccess:
    def test_device_state_retrieved_by_device_id(self, populated_state):
        result = populated_state.device_states["battery_0"]
        assert isinstance(result, _StubDeviceState)
        assert result.label == "battery_0"

    def test_missing_device_id_raises_key_error(self, populated_state):
        with pytest.raises(KeyError):
            _ = populated_state.device_states["nonexistent"]

    def test_population_size_accessible(self, populated_state):
        assert populated_state.population_size == 32


# ============================================================
# TestBatchSimulationStateMutability
# ============================================================

class TestBatchSimulationStateMutability:
    def test_device_state_can_be_added(self, empty_state):
        """The device_states dict is mutable — new entries can be added."""
        empty_state.device_states["new_device"] = _StubDeviceState("new_device")
        assert "new_device" in empty_state.device_states

    def test_device_state_can_be_replaced(self, populated_state):
        new_state = _StubDeviceState("battery_0_new")
        populated_state.device_states["battery_0"] = new_state
        assert populated_state.device_states["battery_0"].label == "battery_0_new"

    def test_device_state_can_be_removed(self, populated_state):
        del populated_state.device_states["pv_0"]
        assert "pv_0" not in populated_state.device_states
        assert len(populated_state.device_states) == 1

    def test_population_size_can_be_reassigned(self, empty_state):
        """population_size is mutable — the engine may update it between runs."""
        empty_state.population_size = 64
        assert empty_state.population_size == 64


# ============================================================
# TestBatchSimulationStateEquality
# ============================================================

class TestBatchSimulationStateEquality:
    def test_equal_instances_are_equal(self):
        state_a = BatchSimulationState(
            device_states={"dev": _StubDeviceState("dev")},
            population_size=8,
        )
        state_b = BatchSimulationState(
            device_states={"dev": _StubDeviceState("dev")},
            population_size=8,
        )
        assert state_a == state_b

    def test_different_population_size_not_equal(self):
        state_a = BatchSimulationState(device_states={}, population_size=8)
        state_b = BatchSimulationState(device_states={}, population_size=16)
        assert state_a != state_b

    def test_different_device_states_not_equal(self):
        state_a = BatchSimulationState(
            device_states={"dev": _StubDeviceState("x")},
            population_size=8,
        )
        state_b = BatchSimulationState(
            device_states={"dev": _StubDeviceState("y")},
            population_size=8,
        )
        assert state_a != state_b


# ============================================================
# TestBatchSimulationStateRepr
# ============================================================

class TestBatchSimulationStateRepr:
    def test_repr_contains_class_name(self, empty_state):
        assert "BatchSimulationState" in repr(empty_state)

    def test_repr_contains_population_size(self, empty_state):
        assert "10" in repr(empty_state)
