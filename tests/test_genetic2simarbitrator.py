"""Tests for VectorizedBusArbitrator with population-vectorized shapes.

All energy arrays now have shape (population_size, horizon). Tests are
structured at two levels:

    pop_size=1  — verifies the core arbitration logic in a shape that
                  is easy to reason about numerically.

    pop_size=3  — verifies true vectorization: each individual in the
                  population has independent supply/demand ratios and
                  receives independent grants in the same call.

Covers:
    PortRequest / DeviceRequest / PortGrant / DeviceGrant
        - Immutability
        - Shape contract
        - is_slack field default and immutability

    BusTopology
        - Immutability
        - port_to_bus mapping

    ArbitrationPriority
        - Ordering contract

    VectorizedBusArbitrator.arbitrate()
        - Empty request list
        - Single producer — full grant
        - Single consumer, no supply — zero grant
        - Fully supplied consumer — full grant
        - Proportional allocation when supply < demand
        - Unequal requests split proportionally
        - supply_ratio clamped to [0, 1]
        - Per-timestep ratio is independent across timesteps
        - Producer never curtailed
        - min_energy floor applied to consumers after proportional scaling
        - min_energy floor not applied when proportional grant already above
        - min_energy floor ignored for producers
        - Multiple buses arbitrated independently
        - Bus isolation — surplus on one bus does not flow to another
        - Grant reassembly preserves device_index and port_index
        - Multi-port device receives grants for all ports
        - Granted array shape is (population_size, horizon)
        - Population individuals receive independent grants

    Slack port (is_slack=True)
        - Slack receives bus residual after non-slack settlement
        - Slack does not distort non-slack proportional split
        - Slack export covers non-slack demand that producers cannot meet
        - Slack import capped at max_import (energy_wh)
        - Slack export capped at max_export (|min_energy_wh|)
        - No slack port → behaviour identical to original proportional algorithm
        - Slack grant is independent per individual and per timestep
        - Slack on fully balanced bus gets zero grant
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from akkudoktoreos.simulation.genetic2.arbitrator import (
    ArbitrationPriority,
    BusTopology,
    DeviceGrant,
    DeviceRequest,
    PortGrant,
    PortRequest,
    VectorizedBusArbitrator,
)

# ============================================================
# Helpers
# ============================================================

def make_energy(values: list[list[float]]) -> np.ndarray:
    """Build a (population_size, horizon) array from a nested list."""
    return np.array(values, dtype=float)


def make_request(
    device_index: int,
    port_index: int,
    energy: list[list[float]],
    min_energy: list[list[float]] | None = None,
    is_slack: bool = False,
) -> DeviceRequest:
    """Build a single-port DeviceRequest.

    Args:
        energy: Nested list of shape (population_size, horizon).
        min_energy: Optional floor, same shape. Defaults to all zeros.
        is_slack: Whether the port is a slack (last-resort) port.
    """
    e = make_energy(energy)
    m = np.zeros_like(e) if min_energy is None else make_energy(min_energy)
    return DeviceRequest(
        device_index=device_index,
        port_requests=(
            PortRequest(port_index=port_index, energy_wh=e, min_energy_wh=m,
                        is_slack=is_slack),
        ),
    )


def get_grant(grants: tuple[DeviceGrant, ...], device_index: int) -> DeviceGrant:
    for g in grants:
        if g.device_index == device_index:
            return g
    raise KeyError(f"No grant for device_index={device_index}")


def get_port_grant(
    grants: tuple[DeviceGrant, ...],
    device_index: int,
    port_index: int,
) -> np.ndarray:
    """Return granted_wh array of shape (population_size, horizon)."""
    device_grant = get_grant(grants, device_index)
    for pg in device_grant.port_grants:
        if pg.port_index == port_index:
            return pg.granted_wh
    raise KeyError(f"No port grant for device={device_index}, port={port_index}")


def single_bus_topology(num_ports: int) -> BusTopology:
    return BusTopology(
        port_to_bus=np.zeros(num_ports, dtype=int),
        num_buses=1,
    )


def two_bus_topology(
    ports_bus0: list[int],
    ports_bus1: list[int],
) -> BusTopology:
    total = len(ports_bus0) + len(ports_bus1)
    mapping = np.empty(total, dtype=int)
    for p in ports_bus0:
        mapping[p] = 0
    for p in ports_bus1:
        mapping[p] = 1
    return BusTopology(port_to_bus=mapping, num_buses=2)


# Convenience wrappers for pop_size=1 tests.
# Wrap a flat list into [[...]] so shape is (1, horizon).

def req1(device_index, port_index, energy: list[float], min_energy=None,
         is_slack: bool = False):
    """Single-individual request."""
    return make_request(
        device_index, port_index,
        [energy],
        None if min_energy is None else [min_energy],
        is_slack=is_slack,
    )


def grant1(grants, device_index, port_index) -> np.ndarray:
    """Extract the single individual's granted array as 1-D."""
    return get_port_grant(grants, device_index, port_index)[0]


# ============================================================
# TestPortRequest
# ============================================================

class TestPortRequest:
    def test_construction_shape(self):
        e = np.zeros((3, 4))
        m = np.zeros((3, 4))
        pr = PortRequest(port_index=0, energy_wh=e, min_energy_wh=m)
        assert pr.energy_wh.shape == (3, 4)
        assert pr.min_energy_wh.shape == (3, 4)

    def test_is_slack_defaults_to_false(self):
        pr = PortRequest(
            port_index=0,
            energy_wh=np.zeros((2, 4)),
            min_energy_wh=np.zeros((2, 4)),
        )
        assert pr.is_slack is False

    def test_is_slack_can_be_set_true(self):
        pr = PortRequest(
            port_index=0,
            energy_wh=np.zeros((2, 4)),
            min_energy_wh=np.zeros((2, 4)),
            is_slack=True,
        )
        assert pr.is_slack is True

    def test_immutable_port_index(self):
        pr = PortRequest(
            port_index=0,
            energy_wh=np.zeros((2, 4)),
            min_energy_wh=np.zeros((2, 4)),
        )
        with pytest.raises(FrozenInstanceError):
            pr.port_index = 1 # type: ignore

    def test_immutable_energy_reference(self):
        pr = PortRequest(
            port_index=0,
            energy_wh=np.zeros((2, 4)),
            min_energy_wh=np.zeros((2, 4)),
        )
        with pytest.raises(FrozenInstanceError):
            pr.energy_wh = np.ones((2, 4)) # type: ignore

    def test_immutable_is_slack(self):
        pr = PortRequest(
            port_index=0,
            energy_wh=np.zeros((2, 4)),
            min_energy_wh=np.zeros((2, 4)),
            is_slack=True,
        )
        with pytest.raises(FrozenInstanceError):
            pr.is_slack = False # type: ignore


# ============================================================
# TestDeviceRequest
# ============================================================

class TestDeviceRequest:
    def test_construction(self):
        pr = PortRequest(
            port_index=0,
            energy_wh=np.zeros((2, 4)),
            min_energy_wh=np.zeros((2, 4)),
        )
        dr = DeviceRequest(device_index=5, port_requests=(pr,))
        assert dr.device_index == 5
        assert len(dr.port_requests) == 1

    def test_immutable(self):
        dr = DeviceRequest(device_index=0, port_requests=())
        with pytest.raises(FrozenInstanceError):
            dr.device_index = 1 # type: ignore


# ============================================================
# TestPortGrant
# ============================================================

class TestPortGrant:
    def test_construction_shape(self):
        g = PortGrant(port_index=2, granted_wh=np.ones((3, 6)))
        assert g.port_index == 2
        assert g.granted_wh.shape == (3, 6)

    def test_immutable(self):
        g = PortGrant(port_index=0, granted_wh=np.zeros((2, 4)))
        with pytest.raises(FrozenInstanceError):
            g.port_index = 1 # type: ignore


# ============================================================
# TestDeviceGrant
# ============================================================

class TestDeviceGrant:
    def test_construction(self):
        pg = PortGrant(port_index=0, granted_wh=np.zeros((2, 4)))
        dg = DeviceGrant(device_index=3, port_grants=(pg,))
        assert dg.device_index == 3

    def test_immutable(self):
        dg = DeviceGrant(device_index=0, port_grants=())
        with pytest.raises(FrozenInstanceError):
            dg.device_index = 99 # type: ignore


# ============================================================
# TestBusTopology
# ============================================================

class TestBusTopology:
    def test_construction(self):
        mapping = np.array([0, 0, 1])
        topo = BusTopology(port_to_bus=mapping, num_buses=2)
        assert topo.num_buses == 2
        np.testing.assert_array_equal(topo.port_to_bus, [0, 0, 1])

    def test_immutable_num_buses(self):
        topo = BusTopology(port_to_bus=np.zeros(2, dtype=int), num_buses=1)
        with pytest.raises(FrozenInstanceError):
            topo.num_buses = 2 # type: ignore

    def test_immutable_array_reference(self):
        topo = BusTopology(port_to_bus=np.zeros(2, dtype=int), num_buses=1)
        with pytest.raises(FrozenInstanceError):
            topo.port_to_bus = np.ones(2, dtype=int) # type: ignore


# ============================================================
# TestArbitrationPriority
# ============================================================

class TestArbitrationPriority:
    def test_ordering(self):
        assert ArbitrationPriority.CRITICAL < ArbitrationPriority.HIGH
        assert ArbitrationPriority.HIGH < ArbitrationPriority.NORMAL
        assert ArbitrationPriority.NORMAL < ArbitrationPriority.LOW

    def test_int_values(self):
        assert ArbitrationPriority.CRITICAL == 0
        assert ArbitrationPriority.LOW == 3


# ============================================================
# TestArbitratorEmptyAndTrivial  (pop_size=1)
# ============================================================

class TestArbitratorEmptyAndTrivial:
    def test_empty_requests_returns_empty_tuple(self):
        arb = VectorizedBusArbitrator(single_bus_topology(0), horizon=4)
        assert arb.arbitrate([]) == ()

    def test_single_producer_gets_full_grant(self):
        arb = VectorizedBusArbitrator(single_bus_topology(1), horizon=4)
        producer = req1(0, 0, [-10.0, -20.0, -15.0, -5.0])
        grants = arb.arbitrate([producer])
        np.testing.assert_array_almost_equal(
            grant1(grants, 0, 0), [-10.0, -20.0, -15.0, -5.0]
        )

    def test_single_consumer_no_supply_gets_zero(self):
        arb = VectorizedBusArbitrator(single_bus_topology(1), horizon=4)
        consumer = req1(0, 0, [10.0, 20.0, 15.0, 5.0])
        grants = arb.arbitrate([consumer])
        np.testing.assert_array_almost_equal(grant1(grants, 0, 0), [0.0] * 4)

    def test_consumer_fully_supplied_gets_full_request(self):
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=4)
        producer = req1(0, 0, [-50.0] * 4)
        consumer = req1(1, 1, [10.0] * 4)
        grants = arb.arbitrate([producer, consumer])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [10.0] * 4)


# ============================================================
# TestArbitratorProportionalAllocation  (pop_size=1)
# ============================================================

class TestArbitratorProportionalAllocation:
    def test_two_equal_consumers_split_evenly(self):
        # supply=10, each consumer wants 10 → each gets 5
        arb = VectorizedBusArbitrator(single_bus_topology(3), horizon=1)
        producer = req1(0, 0, [-10.0])
        consumer_a = req1(1, 1, [10.0])
        consumer_b = req1(2, 2, [10.0])
        grants = arb.arbitrate([producer, consumer_a, consumer_b])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [5.0])
        np.testing.assert_array_almost_equal(grant1(grants, 2, 2), [5.0])

    def test_unequal_consumers_split_proportionally(self):
        # supply=6, a wants 6, b wants 3 → ratio=6/9
        # a gets 4.0, b gets 2.0
        arb = VectorizedBusArbitrator(single_bus_topology(3), horizon=1)
        producer = req1(0, 0, [-6.0])
        consumer_a = req1(1, 1, [6.0])
        consumer_b = req1(2, 2, [3.0])
        grants = arb.arbitrate([producer, consumer_a, consumer_b])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [4.0])
        np.testing.assert_array_almost_equal(grant1(grants, 2, 2), [2.0])

    def test_supply_ratio_clamped_excess_supply_does_not_over_grant(self):
        # supply=100, demand=10 → ratio clamped to 1.0 → grant=10
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)
        producer = req1(0, 0, [-100.0])
        consumer = req1(1, 1, [10.0])
        grants = arb.arbitrate([producer, consumer])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [10.0])

    def test_per_timestep_ratio_is_independent(self):
        # t=0: supply=20, demand=10 → grant=10
        # t=1: supply=5,  demand=10 → grant=5
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=2)
        producer = req1(0, 0, [-20.0, -5.0])
        consumer = req1(1, 1, [10.0, 10.0])
        grants = arb.arbitrate([producer, consumer])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [10.0, 5.0])

    def test_producer_always_gets_full_grant(self):
        arb = VectorizedBusArbitrator(single_bus_topology(1), horizon=3)
        producer = req1(0, 0, [-5.0, -15.0, -25.0])
        grants = arb.arbitrate([producer])
        np.testing.assert_array_almost_equal(
            grant1(grants, 0, 0), [-5.0, -15.0, -25.0]
        )


# ============================================================
# TestArbitratorMinEnergy  (pop_size=1)
# ============================================================

class TestArbitratorMinEnergy:
    def test_floor_applied_when_proportional_below_minimum(self):
        # supply=2, demand=10 → proportional=2 < min=5 → grant=5
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)
        producer = req1(0, 0, [-2.0])
        consumer = req1(1, 1, [10.0], min_energy=[5.0])
        grants = arb.arbitrate([producer, consumer])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [5.0])

    def test_floor_not_applied_when_proportional_already_above(self):
        # supply=20, demand=10 → grant=10 > min=3 → unchanged
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)
        producer = req1(0, 0, [-20.0])
        consumer = req1(1, 1, [10.0], min_energy=[3.0])
        grants = arb.arbitrate([producer, consumer])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [10.0])

    def test_floor_not_applied_to_producers(self):
        arb = VectorizedBusArbitrator(single_bus_topology(1), horizon=1)
        pr = PortRequest(
            port_index=0,
            energy_wh=np.array([[-5.0]]),
            min_energy_wh=np.array([[99.0]]),
        )
        dr = DeviceRequest(device_index=0, port_requests=(pr,))
        grants = arb.arbitrate([dr])
        np.testing.assert_array_almost_equal(grant1(grants, 0, 0), [-5.0])

    def test_floor_zero_has_no_effect(self):
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)
        producer = req1(0, 0, [-4.0])
        consumer = req1(1, 1, [10.0], min_energy=[0.0])
        grants = arb.arbitrate([producer, consumer])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [4.0])


# ============================================================
# TestArbitratorMultiBus  (pop_size=1)
# ============================================================

class TestArbitratorMultiBus:
    def test_two_buses_arbitrated_independently(self):
        topo = two_bus_topology(ports_bus0=[0, 1], ports_bus1=[2, 3])
        arb = VectorizedBusArbitrator(topo, horizon=1)

        producer_0 = req1(0, 0, [-100.0])
        consumer_0 = req1(1, 1, [10.0])
        consumer_1a = req1(2, 2, [10.0])
        consumer_1b = req1(3, 3, [10.0])

        grants = arb.arbitrate([producer_0, consumer_0, consumer_1a, consumer_1b])

        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [10.0])
        np.testing.assert_array_almost_equal(grant1(grants, 2, 2), [0.0])
        np.testing.assert_array_almost_equal(grant1(grants, 3, 3), [0.0])

    def test_surplus_on_one_bus_does_not_flow_to_tight_bus(self):
        topo = two_bus_topology(ports_bus0=[0, 1], ports_bus1=[2, 3])
        arb = VectorizedBusArbitrator(topo, horizon=1)

        producer_0 = req1(0, 0, [-5.0])
        consumer_0 = req1(1, 1, [10.0])
        producer_1 = req1(2, 2, [-100.0])
        consumer_1 = req1(3, 3, [10.0])

        grants = arb.arbitrate([producer_0, consumer_0, producer_1, consumer_1])

        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [5.0])
        np.testing.assert_array_almost_equal(grant1(grants, 3, 3), [10.0])


# ============================================================
# TestArbitratorGrantReassembly  (pop_size=1)
# ============================================================

class TestArbitratorGrantReassembly:
    def test_one_grant_per_requesting_device(self):
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=2)
        producer = req1(0, 0, [-10.0] * 2)
        consumer = req1(1, 1, [5.0] * 2)
        grants = arb.arbitrate([producer, consumer])
        assert len(grants) == 2

    def test_port_index_preserved(self):
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)
        producer = req1(0, 0, [-10.0])
        consumer = req1(1, 1, [5.0])
        grants = arb.arbitrate([producer, consumer])
        device_grant = get_grant(grants, device_index=1)
        assert device_grant.port_grants[0].port_index == 1

    def test_device_index_preserved(self):
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)
        producer = req1(7, 0, [-10.0])
        consumer = req1(42, 1, [5.0])
        grants = arb.arbitrate([producer, consumer])
        assert {g.device_index for g in grants} == {7, 42}

    def test_multi_port_device_receives_all_grants(self):
        topo = two_bus_topology(ports_bus0=[0, 2], ports_bus1=[1, 3])
        arb = VectorizedBusArbitrator(topo, horizon=1)

        pr0 = PortRequest(
            port_index=0,
            energy_wh=np.array([[-10.0]]),
            min_energy_wh=np.array([[0.0]]),
        )
        pr1 = PortRequest(
            port_index=1,
            energy_wh=np.array([[5.0]]),
            min_energy_wh=np.array([[0.0]]),
        )
        dr = DeviceRequest(device_index=0, port_requests=(pr0, pr1))
        supplier_bus1 = req1(1, 3, [-20.0])
        consumer_bus0 = req1(2, 2, [3.0])

        grants = arb.arbitrate([dr, supplier_bus1, consumer_bus0])
        device_grant = get_grant(grants, device_index=0)
        assert len(device_grant.port_grants) == 2
        assert {pg.port_index for pg in device_grant.port_grants} == {0, 1}

    def test_granted_shape_is_population_size_by_horizon(self):
        pop_size, horizon = 3, 6
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=horizon)
        producer = make_request(0, 0, [[-10.0] * horizon] * pop_size)
        consumer = make_request(1, 1, [[5.0] * horizon] * pop_size)
        grants = arb.arbitrate([producer, consumer])
        granted = get_port_grant(grants, 1, 1)
        assert granted.shape == (pop_size, horizon)


# ============================================================
# TestArbitratorVectorizedPopulation  (pop_size=3)
#
# These tests verify that individuals in the population are truly
# independent. A single arbitrate() call handles different
# supply/demand conditions for each individual simultaneously.
# ============================================================

class TestArbitratorVectorizedPopulation:
    def test_each_individual_gets_independent_proportional_grant(self):
        """Three individuals with different supply levels get different grants."""
        # ind 0: supply=20, demand=10 → ratio=1.0 → grant=10
        # ind 1: supply=5,  demand=10 → ratio=0.5 → grant=5
        # ind 2: supply=0,  demand=10 → ratio=0.0 → grant=0
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)

        producer = make_request(0, 0, [[-20.0], [-5.0], [0.0]])
        consumer = make_request(1, 1, [[10.0], [10.0], [10.0]])
        grants = arb.arbitrate([producer, consumer])

        granted = get_port_grant(grants, 1, 1)   # (3, 1)
        np.testing.assert_array_almost_equal(granted[:, 0], [10.0, 5.0, 0.0])

    def test_producers_all_get_full_grant_across_population(self):
        arb = VectorizedBusArbitrator(single_bus_topology(1), horizon=2)

        producer = make_request(0, 0, [
            [-10.0, -20.0],
            [-5.0,  -15.0],
            [-1.0,  -100.0],
        ])
        grants = arb.arbitrate([producer])

        granted = get_port_grant(grants, 0, 0)   # (3, 2)
        np.testing.assert_array_almost_equal(granted, [
            [-10.0, -20.0],
            [-5.0,  -15.0],
            [-1.0,  -100.0],
        ])

    def test_min_energy_floor_applied_per_individual(self):
        """min_energy floor operates independently for each individual."""
        # ind 0: supply=8,  demand=10 → proportional=8  > min=3  → grant=8
        # ind 1: supply=2,  demand=10 → proportional=2  < min=5  → grant=5
        # ind 2: supply=15, demand=10 → proportional=10 > min=7  → grant=10
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)

        producer = make_request(0, 0, [[-8.0], [-2.0], [-15.0]])
        consumer = make_request(
            1, 1,
            energy=[[10.0], [10.0], [10.0]],
            min_energy=[[3.0], [5.0], [7.0]],
        )
        grants = arb.arbitrate([producer, consumer])

        granted = get_port_grant(grants, 1, 1)   # (3, 1)
        np.testing.assert_array_almost_equal(granted[:, 0], [8.0, 5.0, 10.0])

    def test_per_timestep_and_per_individual_independence(self):
        """Supply ratio varies independently by both individual and timestep."""
        # ind 0: t0 supply=10 demand=10 → 10.0 | t1 supply=5  demand=10 → 5.0
        # ind 1: t0 supply=0  demand=10 → 0.0  | t1 supply=20 demand=10 → 10.0
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=2)

        producer = make_request(0, 0, [
            [-10.0, -5.0],
            [0.0,   -20.0],
        ])
        consumer = make_request(1, 1, [
            [10.0, 10.0],
            [10.0, 10.0],
        ])
        grants = arb.arbitrate([producer, consumer])

        granted = get_port_grant(grants, 1, 1)   # (2, 2)
        np.testing.assert_array_almost_equal(granted, [
            [10.0, 5.0],
            [0.0,  10.0],
        ])

    def test_two_buses_independent_per_individual(self):
        """Bus isolation holds independently for every individual."""
        topo = two_bus_topology(ports_bus0=[0, 1], ports_bus1=[2, 3])
        arb = VectorizedBusArbitrator(topo, horizon=1)

        # Bus 0: ind 0 supply=10, ind 1 supply=0
        # Bus 1: ind 0 supply=0,  ind 1 supply=10
        producer_bus0 = make_request(0, 0, [[-10.0], [0.0]])
        consumer_bus0 = make_request(1, 1, [[10.0], [10.0]])
        producer_bus1 = make_request(2, 2, [[0.0], [-10.0]])
        consumer_bus1 = make_request(3, 3, [[10.0], [10.0]])

        grants = arb.arbitrate([
            producer_bus0, consumer_bus0,
            producer_bus1, consumer_bus1,
        ])

        granted_bus0 = get_port_grant(grants, 1, 1)   # (2, 1)
        granted_bus1 = get_port_grant(grants, 3, 3)   # (2, 1)

        # Bus 0: ind 0 full grant, ind 1 zero
        np.testing.assert_array_almost_equal(granted_bus0[:, 0], [10.0, 0.0])
        # Bus 1: ind 0 zero, ind 1 full grant
        np.testing.assert_array_almost_equal(granted_bus1[:, 0], [0.0, 10.0])


# ============================================================
# TestArbitratorSlack  (pop_size=1 and pop_size=3)
#
# Verifies two-phase slack behaviour: non-slack ports settle first
# proportionally; the slack port absorbs the bus residual.
# ============================================================

class TestArbitratorSlack:
    # ------------------------------------------------------------------
    # Basic residual assignment (pop_size=1)
    # ------------------------------------------------------------------

    def test_slack_absorbs_surplus_when_supply_exceeds_non_slack_demand(self):
        """Surplus supply is absorbed by the slack port (positive grant).

        Supply=20, non-slack demand=10 → non-slack gets 10,
        surplus=10 → slack absorbs +10.
        """
        arb = VectorizedBusArbitrator(single_bus_topology(3), horizon=1)
        producer = req1(0, 0, [-20.0])
        consumer = req1(1, 1, [10.0])
        slack = req1(2, 2, [100.0], min_energy=[-100.0], is_slack=True)
        grants = arb.arbitrate([producer, consumer, slack])

        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [10.0])
        np.testing.assert_array_almost_equal(grant1(grants, 2, 2), [10.0])

    def test_slack_imports_to_cover_non_slack_demand_not_met_by_producers(self):
        """No local supply → slack injects to fully cover non-slack demand.

        No producer, non-slack demand=10.
        Deficit=10 → slack injects 10 into bus → slack grant = −10
        (negative = injecting, i.e. grid supplies household).
        Consumer receives full 10 from the slack's injection.
        """
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)
        consumer = req1(0, 0, [10.0])
        slack = req1(1, 1, [100.0], min_energy=[-100.0], is_slack=True)
        grants = arb.arbitrate([consumer, slack])

        np.testing.assert_array_almost_equal(grant1(grants, 0, 0), [10.0])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [-10.0])

    def test_slack_gets_zero_on_perfectly_balanced_bus(self):
        """When supply exactly meets non-slack demand, slack grant is zero."""
        arb = VectorizedBusArbitrator(single_bus_topology(3), horizon=1)
        producer = req1(0, 0, [-10.0])
        consumer = req1(1, 1, [10.0])
        slack = req1(2, 2, [100.0], min_energy=[-100.0], is_slack=True)
        grants = arb.arbitrate([producer, consumer, slack])

        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [10.0])
        np.testing.assert_array_almost_equal(grant1(grants, 2, 2), [0.0])

    def test_slack_does_not_distort_non_slack_proportional_split(self):
        """Non-slack consumers still split supply proportionally when slack is present.

        Supply=12, non-slack a wants 6, b wants 3 → total demand=9.
        Surplus=12−9=3. Slack absorbs +3. No deficit so no injection.
        a gets 6 (full), b gets 3 (full). Slack = +3.

        Use an exact-supply scenario to isolate proportional behaviour:
        supply=6, demand=6 (a=4, b=2) → balanced, slack=0.
        """
        arb = VectorizedBusArbitrator(single_bus_topology(4), horizon=1)
        # supply=6 exactly meets demand=6 (a=4, b=2 after proportional)
        producer = req1(0, 0, [-6.0])
        consumer_a = req1(1, 1, [4.0])
        consumer_b = req1(2, 2, [2.0])
        slack = req1(3, 3, [100.0], min_energy=[-100.0], is_slack=True)
        grants = arb.arbitrate([producer, consumer_a, consumer_b, slack])

        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [4.0])
        np.testing.assert_array_almost_equal(grant1(grants, 2, 2), [2.0])
        np.testing.assert_array_almost_equal(grant1(grants, 3, 3), [0.0])

    # ------------------------------------------------------------------
    # Capacity clamping
    # ------------------------------------------------------------------

    def test_slack_inject_clamped_to_max_inject(self):
        """Slack injection is capped at |min_energy_wh| when deficit is larger.

        No producer, non-slack demand=10, slack max inject=6.
        Deficit=10, capped to 6 → slack injects 6, consumer gets 6.
        Slack net grant = −6 (injecting into bus).
        """
        arb = VectorizedBusArbitrator(single_bus_topology(2), horizon=1)
        consumer = req1(0, 0, [10.0])
        slack = req1(1, 1, [100.0], min_energy=[-6.0], is_slack=True)
        grants = arb.arbitrate([consumer, slack])

        np.testing.assert_array_almost_equal(grant1(grants, 0, 0), [6.0])
        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [-6.0])

    def test_slack_absorb_clamped_to_max_absorb(self):
        """Slack absorb is capped at energy_wh when surplus exceeds capacity.

        Supply=20, non-slack demand=5, surplus=15.
        Slack max absorb (energy_wh)=3 → slack grant = +3.
        """
        arb = VectorizedBusArbitrator(single_bus_topology(3), horizon=1)
        producer = req1(0, 0, [-20.0])
        consumer = req1(1, 1, [5.0])
        slack = req1(2, 2, [3.0], min_energy=[-100.0], is_slack=True)
        grants = arb.arbitrate([producer, consumer, slack])

        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [5.0])
        np.testing.assert_array_almost_equal(grant1(grants, 2, 2), [3.0])

    # ------------------------------------------------------------------
    # No-slack bus — regression: behaviour unchanged
    # ------------------------------------------------------------------

    def test_no_slack_port_behaves_identically_to_original_algorithm(self):
        """Without any slack port the algorithm is identical to the original.

        supply=6, a wants 6, b wants 3 → proportional split: a=4, b=2.
        """
        arb = VectorizedBusArbitrator(single_bus_topology(3), horizon=1)
        producer = req1(0, 0, [-6.0])
        consumer_a = req1(1, 1, [6.0])
        consumer_b = req1(2, 2, [3.0])
        grants = arb.arbitrate([producer, consumer_a, consumer_b])

        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [4.0])
        np.testing.assert_array_almost_equal(grant1(grants, 2, 2), [2.0])

    # ------------------------------------------------------------------
    # Per-step independence (pop_size=1, horizon=2)
    # ------------------------------------------------------------------

    def test_slack_grant_is_independent_per_timestep(self):
        """Slack grant is computed step-by-step, independently.

        t=0: supply=15, demand=10 → surplus=5  → slack absorbs +5
        t=1: supply=8,  demand=10 → deficit=2  → slack injects −2,
             consumer gets full 10 (effective supply = 8+2 = 10).
        """
        arb = VectorizedBusArbitrator(single_bus_topology(3), horizon=2)
        producer = req1(0, 0, [-15.0, -8.0])
        consumer = req1(1, 1, [10.0, 10.0])
        slack = req1(2, 2, [100.0, 100.0], min_energy=[-100.0, -100.0], is_slack=True)
        grants = arb.arbitrate([producer, consumer, slack])

        np.testing.assert_array_almost_equal(grant1(grants, 1, 1), [10.0, 10.0])
        np.testing.assert_array_almost_equal(grant1(grants, 2, 2), [5.0, -2.0])

    # ------------------------------------------------------------------
    # Per-individual independence (pop_size=3)
    # ------------------------------------------------------------------

    def test_slack_grant_is_independent_per_individual(self):
        """Each individual's slack grant reflects its own bus balance.

        ind 0: supply=20, demand=10 → surplus=10  → slack absorbs +10
        ind 1: supply=10, demand=10 → balanced    → slack = 0
        ind 2: supply=5,  demand=10 → deficit=5   → slack injects −5,
               consumer gets full 10.
        """
        arb = VectorizedBusArbitrator(single_bus_topology(3), horizon=1)

        producer = make_request(0, 0, [[-20.0], [-10.0], [-5.0]])
        consumer = make_request(1, 1, [[10.0], [10.0], [10.0]])
        slack = make_request(2, 2,
                             energy=[[100.0], [100.0], [100.0]],
                             min_energy=[[-100.0], [-100.0], [-100.0]],
                             is_slack=True)
        grants = arb.arbitrate([producer, consumer, slack])

        granted_consumer = get_port_grant(grants, 1, 1)   # (3, 1)
        granted_slack = get_port_grant(grants, 2, 2)       # (3, 1)

        np.testing.assert_array_almost_equal(
            granted_consumer[:, 0], [10.0, 10.0, 10.0]
        )
        np.testing.assert_array_almost_equal(
            granted_slack[:, 0], [10.0, 0.0, -5.0]
        )
