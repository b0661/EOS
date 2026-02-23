"""Tests for genetic2.simulation.arbitrator.

Covers: PriorityArbitrator and ProportionalArbitrator allocation policies,
edge cases (empty supply, curtailment, cross-domain requests).
"""

import pytest

from akkudoktoreos.simulation.genetic2.arbitrator import (
    PriorityArbitrator,
    ProportionalArbitrator,
)
from akkudoktoreos.simulation.genetic2.flows import (
    Priority,
    ResourceGrant,
    ResourceRequest,
)


def make_request(
    device_id: str,
    ac: float = 0.0,
    heat: float = 0.0,
    min_ac: float = 0.0,
    min_heat: float = 0.0,
    priority: Priority = Priority.NORMAL,
    hour: int = 0,
) -> ResourceRequest:
    """Helper to build ResourceRequest instances concisely in tests."""
    return ResourceRequest(
        device_id=device_id,
        hour=hour,
        priority=priority,
        ac_power_wh=ac,
        heat_wh=heat,
        min_ac_power_wh=min_ac,
        min_heat_wh=min_heat,
    )


class TestPriorityArbitratorProducers:
    """Producers are always granted in full, regardless of supply."""

    def test_producer_always_granted(self):
        arb = PriorityArbitrator()
        requests = [make_request("pv", ac=-1000.0)]
        grants = arb.arbitrate(requests)
        assert "pv" in grants
        assert grants["pv"].ac_power_wh == pytest.approx(-1000.0)
        assert grants["pv"].curtailed is False

    def test_multiple_producers_all_granted(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("pv1", ac=-1000.0),
            make_request("pv2", ac=-500.0),
        ]
        grants = arb.arbitrate(requests)
        assert grants["pv1"].ac_power_wh == pytest.approx(-1000.0)
        assert grants["pv2"].ac_power_wh == pytest.approx(-500.0)

    def test_idle_device_granted_zero(self):
        arb = PriorityArbitrator()
        requests = [make_request("idle_bat")]  # ac=0, heat=0 → idle
        grants = arb.arbitrate(requests)
        assert "idle_bat" in grants
        assert grants["idle_bat"].ac_power_wh == pytest.approx(0.0)
        assert grants["idle_bat"].curtailed is False


class TestPriorityArbitratorConsumers:
    """Consumers are served in priority order from available supply."""

    def test_single_consumer_gets_full_request_when_sufficient(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-2000.0),      # producer: 2000 Wh available
            make_request("bat", ac=500.0),         # consumer: wants 500 Wh
        ]
        grants = arb.arbitrate(requests)
        assert grants["bat"].ac_power_wh == pytest.approx(500.0)
        assert grants["bat"].curtailed is False

    def test_consumer_capped_by_available_supply(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-300.0),        # only 300 Wh available
            make_request("bat", ac=1000.0),        # wants 1000 Wh
        ]
        grants = arb.arbitrate(requests)
        assert grants["bat"].ac_power_wh == pytest.approx(300.0)

    def test_high_priority_served_before_low(self):
        arb = PriorityArbitrator()
        supply = 500.0
        requests = [
            make_request("pv", ac=-supply),
            make_request("low_bat",  ac=400.0, priority=Priority.LOW),
            make_request("high_bat", ac=400.0, priority=Priority.HIGH),
        ]
        grants = arb.arbitrate(requests)
        # High priority gets its full 400 Wh; only 100 Wh remains for low
        assert grants["high_bat"].ac_power_wh == pytest.approx(400.0)
        assert grants["low_bat"].ac_power_wh == pytest.approx(100.0)

    def test_critical_takes_all_before_normal(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-600.0),
            make_request("normal", ac=500.0, priority=Priority.NORMAL),
            make_request("critical", ac=600.0, priority=Priority.CRITICAL),
        ]
        grants = arb.arbitrate(requests)
        assert grants["critical"].ac_power_wh == pytest.approx(600.0)
        assert grants["normal"].ac_power_wh == pytest.approx(0.0)

    def test_all_consumers_served_when_supply_sufficient(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-3000.0),
            make_request("bat1", ac=1000.0),
            make_request("bat2", ac=1000.0),
            make_request("bat3", ac=500.0),
        ]
        grants = arb.arbitrate(requests)
        assert grants["bat1"].ac_power_wh == pytest.approx(1000.0)
        assert grants["bat2"].ac_power_wh == pytest.approx(1000.0)
        assert grants["bat3"].ac_power_wh == pytest.approx(500.0)

    def test_no_supply_all_consumers_get_zero(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("bat1", ac=500.0),
            make_request("bat2", ac=300.0),
        ]
        grants = arb.arbitrate(requests)
        assert grants["bat1"].ac_power_wh == pytest.approx(0.0)
        assert grants["bat2"].ac_power_wh == pytest.approx(0.0)


class TestPriorityArbitratorCurtailment:
    """Devices below their minimum are curtailed entirely."""

    def test_device_curtailed_when_below_minimum(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-100.0),         # only 100 Wh
            make_request("bat", ac=500.0, min_ac=200.0),  # needs at least 200 Wh
        ]
        grants = arb.arbitrate(requests)
        assert grants["bat"].curtailed is True
        assert grants["bat"].ac_power_wh == pytest.approx(0.0)

    def test_device_not_curtailed_when_above_minimum(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-500.0),
            make_request("bat", ac=500.0, min_ac=200.0),
        ]
        grants = arb.arbitrate(requests)
        assert grants["bat"].curtailed is False
        assert grants["bat"].ac_power_wh == pytest.approx(500.0)

    def test_min_zero_never_curtailed(self):
        """min_ac_power_wh=0 means any amount is acceptable — never curtailed."""
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-50.0),
            make_request("bat", ac=500.0, min_ac=0.0),
        ]
        grants = arb.arbitrate(requests)
        assert grants["bat"].curtailed is False
        assert grants["bat"].ac_power_wh == pytest.approx(50.0)

    def test_curtailed_device_frees_supply_for_others(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-500.0),
            # high priority, high minimum — will be curtailed (only 500 available)
            make_request("expensive", ac=600.0, min_ac=600.0, priority=Priority.HIGH),
            # low priority, no minimum — should get remaining 500 Wh
            make_request("flexible", ac=500.0, min_ac=0.0, priority=Priority.LOW),
        ]
        grants = arb.arbitrate(requests)
        assert grants["expensive"].curtailed is True
        # Note: priority ordering means expensive is checked first and curtailed,
        # leaving the supply intact for flexible
        assert grants["flexible"].ac_power_wh == pytest.approx(500.0)


class TestPriorityArbitratorHeat:
    """Heat resource arbitration works independently of AC."""

    def test_heat_producer_always_granted(self):
        arb = PriorityArbitrator()
        requests = [make_request("heatpump", heat=-3000.0)]
        grants = arb.arbitrate(requests)
        assert grants["heatpump"].heat_wh == pytest.approx(-3000.0)

    def test_heat_consumer_served_from_heat_pool(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("heatpump", heat=-3000.0),
            make_request("tank", heat=1000.0),
        ]
        grants = arb.arbitrate(requests)
        assert grants["tank"].heat_wh == pytest.approx(1000.0)

    def test_ac_and_heat_arbitrated_independently(self):
        """A device curtailed for AC can still receive heat."""
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-100.0),              # 100 Wh AC
            make_request("heatpump_src", heat=-3000.0), # 3000 Wh heat
            ResourceRequest(
                device_id="combined",
                hour=0,
                ac_power_wh=500.0,     # wants 500 AC (only 100 available → curtailed)
                heat_wh=1000.0,        # wants 1000 heat (3000 available → granted)
                min_ac_power_wh=500.0, # minimum AC = 500 → curtailed
                min_heat_wh=0.0,
            ),
        ]
        grants = arb.arbitrate(requests)
        # AC curtailed (below minimum), heat granted
        assert grants["combined"].ac_power_wh == pytest.approx(0.0)
        assert grants["combined"].heat_wh == pytest.approx(1000.0)

    def test_empty_requests_returns_empty(self):
        arb = PriorityArbitrator()
        grants = arb.arbitrate([])
        assert grants == {}

    def test_all_devices_have_grant_entry(self):
        arb = PriorityArbitrator()
        requests = [
            make_request("pv", ac=-1000.0),
            make_request("bat1", ac=400.0),
            make_request("idle"),
        ]
        grants = arb.arbitrate(requests)
        assert set(grants.keys()) == {"pv", "bat1", "idle"}


class TestProportionalArbitrator:
    """ProportionalArbitrator distributes supply proportionally."""

    def test_equal_requests_split_equally(self):
        arb = ProportionalArbitrator()
        requests = [
            make_request("pv", ac=-1000.0),
            make_request("bat1", ac=500.0),
            make_request("bat2", ac=500.0),
        ]
        grants = arb.arbitrate(requests)
        assert grants["bat1"].ac_power_wh == pytest.approx(500.0)
        assert grants["bat2"].ac_power_wh == pytest.approx(500.0)

    def test_unequal_requests_split_proportionally(self):
        arb = ProportionalArbitrator()
        requests = [
            make_request("pv", ac=-600.0),   # 600 Wh available
            make_request("big", ac=400.0),    # wants 400
            make_request("small", ac=200.0),  # wants 200 → total 600, exactly met
        ]
        grants = arb.arbitrate(requests)
        assert grants["big"].ac_power_wh == pytest.approx(400.0)
        assert grants["small"].ac_power_wh == pytest.approx(200.0)

    def test_undersupply_proportional_split(self):
        arb = ProportionalArbitrator()
        requests = [
            make_request("pv", ac=-300.0),   # only 300 Wh, but 600 requested
            make_request("bat1", ac=400.0),
            make_request("bat2", ac=200.0),
        ]
        grants = arb.arbitrate(requests)
        # bat1 gets 400/600 * 300 = 200, bat2 gets 200/600 * 300 = 100
        assert grants["bat1"].ac_power_wh == pytest.approx(200.0, rel=1e-3)
        assert grants["bat2"].ac_power_wh == pytest.approx(100.0, rel=1e-3)

    def test_device_below_minimum_curtailed(self):
        arb = ProportionalArbitrator()
        requests = [
            make_request("pv", ac=-100.0),              # only 100 Wh
            make_request("strict", ac=500.0, min_ac=300.0),  # needs 300 min
        ]
        grants = arb.arbitrate(requests)
        assert grants["strict"].curtailed is True

    def test_producer_always_granted(self):
        arb = ProportionalArbitrator()
        requests = [make_request("pv", ac=-2000.0)]
        grants = arb.arbitrate(requests)
        assert grants["pv"].ac_power_wh == pytest.approx(-2000.0)
