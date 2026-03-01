"""Tests for SingleStateEnergyDevice.

Focus areas (as agreed):
    1. apply_genome_batch + _simulate_batch interaction
    2. genome_requirements bounds

Secondary coverage:
    - setup_run stores horizon and step_interval correctly
    - create_batch_state allocates correct shapes and initial values
    - repair_batch default clamp behaviour
    - State-dependent repair via repair_batch override
    - State evolution across the full horizon
    - State reset at the start of each _simulate_batch call
    - Population-axis independence (different individuals evolve independently)

Concrete stub devices
---------------------
Two minimal concrete subclasses are defined here:

``AccumulatorDevice``
    State = running sum of applied power * step_interval (energy accumulator).
    Simple, analytically verifiable physics.
    Uses default repair (clamp to power_bounds).
    power_bounds: (-100.0, 200.0)

``CapacityLimitedDevice``
    Same accumulator physics but overrides repair_batch to enforce a
    capacity ceiling: power is clamped so state never exceeds ``capacity``.
    Inherits power_bounds (-100.0, 200.0) W from AccumulatorDevice.
    capacity: 1000.0 Wh
"""

from __future__ import annotations

import numpy as np
import pytest

from akkudoktoreos.devices.devicesabc import (
    EnergyPort,
    PortDirection,
    SingleStateBatchState,
    SingleStateEnergyDevice,
)
from akkudoktoreos.optimization.genetic2.genome import GenomeSlice
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import to_datetime

# ============================================================
# Concrete stub devices
# ============================================================

class AccumulatorDevice(SingleStateEnergyDevice):
    """Minimal concrete device.

    State = cumulative energy [Wh].
    state[t+1] = state[t] + power[t] * step_interval / 3600
    power_bounds: (-100.0, 200.0) W
    initial_state: 0.0 Wh
    """

    LOWER = -100.0
    UPPER = 200.0

    def __init__(self, device_id: str = "acc_0") -> None:
        super().__init__()
        self.device_id = device_id

    @property
    def ports(self) -> tuple[EnergyPort, ...]:
        return (EnergyPort(
            port_id="p0", bus_id="bus_ac",
            direction=PortDirection.BIDIRECTIONAL,
        ),)

    @property
    def objective_names(self) -> list[str]:
        return ["energy_wh"]

    def initial_state(self) -> float:
        return 0.0

    def state_transition_batch(
        self,
        state: np.ndarray,
        power: np.ndarray,
        step_interval: float,
    ) -> np.ndarray:
        return state + power * step_interval / 3600.0

    def power_bounds(self) -> tuple[float, float]:
        return (self.LOWER, self.UPPER)

    # Arbitration and cost left abstract-ish — not under test here
    def build_device_request(self, state):
        return None

    def apply_device_grant(self, state, grant):
        pass

    def compute_cost(self, state) -> np.ndarray:
        return np.zeros((state.population_size, 1))


class CapacityLimitedDevice(AccumulatorDevice):
    """AccumulatorDevice with state-dependent repair.

    Inherits power_bounds (-100.0, 200.0) W from AccumulatorDevice.
    Overrides repair_batch to additionally clamp charge power so that
    the accumulated state never exceeds ``capacity``.
    capacity: 1000.0 Wh
    """

    CAPACITY = 1000.0

    def __init__(self, device_id: str = "cap_0") -> None:
        super().__init__(device_id)

    def repair_batch(
        self,
        step: int,
        requested_power: np.ndarray,
        current_state: np.ndarray,
    ) -> np.ndarray:
        # First apply static bounds
        power = np.clip(requested_power, self.LOWER, self.UPPER)
        # Then cap charging so state does not exceed capacity
        # max_charge = (capacity - current_state) * 3600 / step_interval
        # We need step_interval here — stored on self after setup_run
        max_charge = (self.CAPACITY - current_state) * 3600.0 / self._step_interval
        power = np.minimum(power, max_charge)
        return power


# ============================================================
# FakeContext and helpers
# ============================================================

STEP_INTERVAL = 3600.0   # 1 hour in seconds
STEP_TIMES = tuple(to_datetime(i * 3600) for i in range(24))  # 24-step horizon, hourly


def make_step_times(n: int) -> tuple:
    return tuple(to_datetime(i * 3600) for i in range(n))


class FakeContext:
    """Minimal SimulationContext stand-in for unit tests."""

    def __init__(self, step_times: tuple, step_interval: float) -> None:
        self.step_times = step_times
        self.step_interval = step_interval
        self.horizon = len(step_times)


def make_context(n: int = 24, step_interval: float = STEP_INTERVAL) -> FakeContext:
    return FakeContext(make_step_times(n), step_interval)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture()
def device() -> AccumulatorDevice:
    dev = AccumulatorDevice()
    dev.setup_run(make_context(24))
    return dev


@pytest.fixture()
def cap_device() -> CapacityLimitedDevice:
    dev = CapacityLimitedDevice()
    dev.setup_run(make_context(24))
    return dev


@pytest.fixture()
def batch_state(device) -> SingleStateBatchState:
    return device.create_batch_state(population_size=4, horizon=len(STEP_TIMES))


# ============================================================
# TestSetupRun
# ============================================================

class TestSetupRun:
    def test_stores_num_steps(self, device):
        assert device._num_steps == len(STEP_TIMES)

    def test_stores_step_interval(self, device):
        assert device._step_interval == STEP_INTERVAL

    def test_stores_step_times(self, device):
        assert device._step_times == STEP_TIMES

    def test_can_be_reconfigured_with_different_horizon(self):
        dev = AccumulatorDevice()
        dev.setup_run(make_context(48))
        assert dev._num_steps == 48


# ============================================================
# TestGenomeRequirements
# ============================================================

class TestGenomeRequirements:
    def test_returns_genome_slice(self, device):
        req = device.genome_requirements()
        assert isinstance(req, GenomeSlice)

    def test_size_equals_horizon(self, device):
        req = device.genome_requirements()
        assert req.size == len(STEP_TIMES)

    def test_start_is_zero(self, device):
        """start=0 before assembler re-indexes."""
        req = device.genome_requirements()
        assert req.start == 0

    def test_lower_bound_shape(self, device):
        req = device.genome_requirements()
        assert req.lower_bound.shape == (len(STEP_TIMES),)

    def test_upper_bound_shape(self, device):
        req = device.genome_requirements()
        assert req.upper_bound.shape == (len(STEP_TIMES),)

    def test_lower_bound_values_match_power_bounds(self, device):
        req = device.genome_requirements()
        np.testing.assert_array_equal(req.lower_bound, np.full(len(STEP_TIMES), -100.0))

    def test_upper_bound_values_match_power_bounds(self, device):
        req = device.genome_requirements()
        np.testing.assert_array_equal(req.upper_bound, np.full(len(STEP_TIMES), 200.0))

    def test_bounds_are_uniform_across_all_steps(self, device):
        """All steps must have the same bound — no step-varying bounds here."""
        req = device.genome_requirements()
        assert np.all(req.lower_bound == req.lower_bound[0])
        assert np.all(req.upper_bound == req.upper_bound[0])

    def test_raises_if_setup_run_not_called(self):
        dev = AccumulatorDevice()
        with pytest.raises(RuntimeError):
            dev.genome_requirements()

    def test_genome_requirements_reflects_new_horizon_after_reconfigure(self):
        dev = AccumulatorDevice()
        dev.setup_run(make_context(48))
        req = dev.genome_requirements()
        assert req.size == 48


# ============================================================
# TestCreateBatchState
# ============================================================

class TestCreateBatchState:
    def test_schedule_shape(self, batch_state):
        assert batch_state.schedule.shape == (4, len(STEP_TIMES))

    def test_state_shape(self, batch_state):
        assert batch_state.state.shape == (4,)

    def test_schedule_initialised_to_zero(self, batch_state):
        np.testing.assert_array_equal(batch_state.schedule, 0.0)

    def test_state_initialised_to_initial_state(self, device, batch_state):
        np.testing.assert_array_equal(batch_state.state, device.initial_state())

    def test_population_size_stored(self, batch_state):
        assert batch_state.population_size == 4

    def test_horizon_stored(self, batch_state):
        assert batch_state.horizon == len(STEP_TIMES)

    def test_step_times_stored(self, batch_state):
        assert batch_state.step_times == STEP_TIMES

    def test_step_times_length_matches_horizon(self, batch_state):
        assert len(batch_state.step_times) == batch_state.horizon


# ============================================================
# TestApplyGenomeBatch
# ============================================================

class TestApplyGenomeBatch:
    def test_returns_array_of_correct_shape(self, device, batch_state):
        genome = np.zeros((4, len(STEP_TIMES)))
        result = device.apply_genome_batch(batch_state, genome)
        assert result.shape == (4, len(STEP_TIMES))

    def test_returned_array_is_state_schedule(self, device, batch_state):
        """Return value must be the same object as state.schedule."""
        genome = np.zeros((4, len(STEP_TIMES)))
        result = device.apply_genome_batch(batch_state, genome)
        assert result is batch_state.schedule

    def test_feasible_genome_unchanged_after_apply(self, device, batch_state):
        """A genome within bounds must pass through unmodified."""
        genome = np.full((4, len(STEP_TIMES)), 50.0)   # within (-100, 200)
        result = device.apply_genome_batch(batch_state, genome)
        np.testing.assert_array_equal(result, 50.0)

    def test_genome_above_upper_bound_clamped(self, device, batch_state):
        genome = np.full((4, len(STEP_TIMES)), 999.0)   # above 200 W
        result = device.apply_genome_batch(batch_state, genome)
        np.testing.assert_array_equal(result, 200.0)

    def test_genome_below_lower_bound_clamped(self, device, batch_state):
        genome = np.full((4, len(STEP_TIMES)), -999.0)  # below -100 W
        result = device.apply_genome_batch(batch_state, genome)
        np.testing.assert_array_equal(result, -100.0)

    def test_mixed_feasible_and_infeasible_values_repaired_correctly(
        self, device, batch_state
    ):
        genome = np.zeros((4, len(STEP_TIMES)))
        genome[0, :] = 150.0     # feasible
        genome[1, :] = 300.0     # above upper → clamped to 200
        genome[2, :] = -50.0     # feasible
        genome[3, :] = -200.0    # below lower → clamped to -100
        result = device.apply_genome_batch(batch_state, genome)
        np.testing.assert_array_equal(result[0], 150.0)
        np.testing.assert_array_equal(result[1], 200.0)
        np.testing.assert_array_equal(result[2], -50.0)
        np.testing.assert_array_equal(result[3], -100.0)

    def test_state_updated_after_apply(self, device, batch_state):
        """After apply_genome_batch, state must reflect the simulated final state."""
        # 24 steps * 100 W * 1 h = 2400 Wh accumulated
        genome = np.full((4, len(STEP_TIMES)), 100.0)
        device.apply_genome_batch(batch_state, genome)
        expected_final_state = 24 * 100.0 * 1.0  # 2400 Wh
        np.testing.assert_array_almost_equal(batch_state.state, expected_final_state)

    def test_schedule_in_state_updated_in_place(self, device, batch_state):
        genome = np.full((4, len(STEP_TIMES)), 50.0)
        device.apply_genome_batch(batch_state, genome)
        np.testing.assert_array_equal(batch_state.schedule, 50.0)


# ============================================================
# TestSimulateBatch
# ============================================================

class TestSimulateBatch:
    def test_state_reset_to_initial_at_start_of_each_simulate(
        self, device, batch_state
    ):
        """Calling apply_genome_batch twice must produce identical results."""
        genome = np.full((4, len(STEP_TIMES)), 100.0)
        device.apply_genome_batch(batch_state, genome)
        state_after_first = batch_state.state.copy()

        device.apply_genome_batch(batch_state, genome)
        state_after_second = batch_state.state.copy()

        np.testing.assert_array_equal(state_after_first, state_after_second)

    def test_state_evolves_step_by_step_correctly(self, device):
        """Verify state at each step matches analytical accumulation."""
        pop_size, horizon = 1, 5
        dev = AccumulatorDevice()
        dev.setup_run(make_context(horizon))
        state = dev.create_batch_state(pop_size, horizon)

        power = 100.0  # W
        genome = np.full((pop_size, horizon), power)
        dev.apply_genome_batch(state, genome)

        # After each step: state += 100 * 3600 / 3600 = 100 Wh
        # Final state after 5 steps: 500 Wh
        np.testing.assert_array_almost_equal(state.state, [500.0])

    def test_population_individuals_evolve_independently(self, device):
        """Different individuals must not affect each other's state."""
        pop_size, horizon = 3, 4
        dev = AccumulatorDevice()
        dev.setup_run(make_context(horizon))
        state = dev.create_batch_state(pop_size, horizon)

        # Three different constant power schedules
        genome = np.array([
            [50.0] * horizon,     # ind 0: 50 W → 200 Wh after 4 steps
            [100.0] * horizon,    # ind 1: 100 W → 400 Wh
            [200.0] * horizon,    # ind 2: 200 W → 800 Wh
        ])
        dev.apply_genome_batch(state, genome)

        np.testing.assert_array_almost_equal(
            state.state, [200.0, 400.0, 800.0]
        )

    def test_repair_applied_before_state_transition(self, device):
        """Infeasible values must be repaired before feeding into state_transition."""
        pop_size, horizon = 1, 2
        dev = AccumulatorDevice()
        dev.setup_run(make_context(horizon))
        state = dev.create_batch_state(pop_size, horizon)

        # Request 999 W (above 200 W bound) for both steps
        # Should be clamped to 200 W → state = 200 + 200 = 400 Wh
        genome = np.full((pop_size, horizon), 999.0)
        dev.apply_genome_batch(state, genome)

        np.testing.assert_array_almost_equal(state.state, [400.0])

    def test_varying_power_schedule_accumulates_correctly(self, device):
        """Non-uniform power schedule must accumulate step by step."""
        pop_size, horizon = 1, 4
        dev = AccumulatorDevice()
        dev.setup_run(make_context(horizon))
        state = dev.create_batch_state(pop_size, horizon)

        # Powers: 100, -50, 100, -50 → net = 100 Wh
        genome = np.array([[100.0, -50.0, 100.0, -50.0]])
        dev.apply_genome_batch(state, genome)

        np.testing.assert_array_almost_equal(state.state, [100.0])


# ============================================================
# TestDefaultRepairBatch
# ============================================================

class TestDefaultRepairBatch:
    def test_clamps_above_upper(self, device):
        power = np.array([300.0, 150.0, -50.0])
        state = np.zeros(3)
        result = device.repair_batch(0, power, state)
        np.testing.assert_array_equal(result, [200.0, 150.0, -50.0])

    def test_clamps_below_lower(self, device):
        power = np.array([-200.0, -100.0, 0.0])
        state = np.zeros(3)
        result = device.repair_batch(0, power, state)
        np.testing.assert_array_equal(result, [-100.0, -100.0, 0.0])

    def test_feasible_values_unchanged(self, device):
        power = np.array([0.0, 100.0, -50.0])
        state = np.zeros(3)
        result = device.repair_batch(0, power, state)
        np.testing.assert_array_equal(result, [0.0, 100.0, -50.0])

    def test_step_argument_does_not_affect_default_repair(self, device):
        """Default repair is step-invariant."""
        power = np.array([999.0])
        state = np.zeros(1)
        for step in range(5):
            result = device.repair_batch(step, power.copy(), state)
            np.testing.assert_array_equal(result, [200.0])


# ============================================================
# TestStateDependentRepair
# ============================================================

class TestStateDependentRepair:
    def test_charge_power_clamped_when_near_capacity(self, cap_device):
        """When state is near capacity, charge power must be reduced."""
        pop_size, horizon = 1, 1
        dev = CapacityLimitedDevice()
        dev.setup_run(make_context(1))
        state = dev.create_batch_state(pop_size, horizon)

        # Manually set state to 900 Wh (100 Wh below capacity)
        state.state[:] = 900.0

        # Request 500 W for 1 hour = 500 Wh, but only 100 Wh headroom
        # max_charge = (1000 - 900) * 3600 / 3600 = 100 W
        power = np.array([500.0])
        repaired = dev.repair_batch(0, power, state.state)
        np.testing.assert_array_almost_equal(repaired, [100.0])

    def test_charge_power_not_clamped_when_headroom_is_ample(self, cap_device):
        """When state is well below capacity, charge power passes through."""
        dev = CapacityLimitedDevice()
        dev.setup_run(make_context(1))
        state = dev.create_batch_state(1, 1)

        state.state[:] = 0.0
        power = np.array([200.0])   # within static bounds, well below capacity
        repaired = dev.repair_batch(0, power, state.state)
        np.testing.assert_array_almost_equal(repaired, [200.0])

    def test_state_never_exceeds_capacity_across_full_horizon(self, cap_device):
        """State must never exceed capacity when state-dependent repair is active."""
        pop_size, horizon = 3, 24
        dev = CapacityLimitedDevice()
        dev.setup_run(make_context(24))
        state = dev.create_batch_state(pop_size, horizon)

        # Request maximum charge power every step for all individuals
        genome = np.full((pop_size, horizon), 500.0)
        dev.apply_genome_batch(state, genome)

        # Final state must never exceed capacity for any individual
        assert np.all(state.state <= CapacityLimitedDevice.CAPACITY + 1e-6)

    def test_state_dependent_repair_is_independent_per_individual(self):
        """Each individual's repair must use its own current state."""
        dev = CapacityLimitedDevice()
        horizon = 2
        dev.setup_run(make_context(horizon))
        state = dev.create_batch_state(2, horizon)

        # ind 0: state=0    → 500 W headroom → gets full 200 W (static upper)
        # ind 1: state=950  → 50 W headroom  → clamped to 50 W
        state.state[:] = np.array([0.0, 950.0])
        power = np.full(2, 500.0)
        repaired = dev.repair_batch(0, power, state.state)

        np.testing.assert_array_almost_equal(repaired[0], 200.0)
        np.testing.assert_array_almost_equal(repaired[1], 50.0)
