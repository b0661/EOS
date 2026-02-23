"""Tests for genetic2.simulation.flows.

Covers: EnergyFlows, ResourceRequest, ResourceGrant, Priority.
"""

import pytest

from akkudoktoreos.simulation.genetic2.flows import (
    EnergyFlows,
    Priority,
    ResourceGrant,
    ResourceRequest,
)


class TestPriority:
    """Priority enum ordering and values."""

    def test_ordering_critical_lowest(self):
        assert Priority.CRITICAL < Priority.HIGH

    def test_ordering_full_chain(self):
        assert Priority.CRITICAL < Priority.HIGH < Priority.NORMAL < Priority.LOW

    def test_sortable(self):
        priorities = [Priority.LOW, Priority.CRITICAL, Priority.NORMAL, Priority.HIGH]
        assert sorted(priorities) == [
            Priority.CRITICAL,
            Priority.HIGH,
            Priority.NORMAL,
            Priority.LOW,
        ]

    def test_integer_values(self):
        assert int(Priority.CRITICAL) == 0
        assert int(Priority.LOW) == 3


class TestEnergyFlows:
    """EnergyFlows dataclass and its computed properties."""

    def test_default_all_zero(self):
        flows = EnergyFlows()
        assert flows.ac_power_wh == 0.0
        assert flows.dc_power_wh == 0.0
        assert flows.heat_provided_wh == 0.0
        assert flows.heat_consumed_wh == 0.0
        assert flows.losses_wh == 0.0
        assert flows.soc_pct is None
        assert flows.generation_wh == 0.0
        assert flows.load_wh == 0.0

    def test_is_ac_source_positive(self):
        flows = EnergyFlows(ac_power_wh=100.0)
        assert flows.is_ac_source is True
        assert flows.is_ac_sink is False

    def test_is_ac_sink_negative(self):
        flows = EnergyFlows(ac_power_wh=-100.0)
        assert flows.is_ac_source is False
        assert flows.is_ac_sink is True

    def test_is_neither_source_nor_sink_when_zero(self):
        flows = EnergyFlows(ac_power_wh=0.0)
        assert flows.is_ac_source is False
        assert flows.is_ac_sink is False

    def test_is_heat_source(self):
        flows = EnergyFlows(heat_provided_wh=500.0)
        assert flows.is_heat_source is True

    def test_is_not_heat_source_when_zero(self):
        flows = EnergyFlows(heat_provided_wh=0.0)
        assert flows.is_heat_source is False

    def test_soc_pct_stored(self):
        flows = EnergyFlows(soc_pct=75.5)
        assert flows.soc_pct == pytest.approx(75.5)

    def test_independent_dc_and_ac(self):
        flows = EnergyFlows(ac_power_wh=100.0, dc_power_wh=-50.0)
        assert flows.is_ac_source is True
        assert flows.ac_power_wh == 100.0
        assert flows.dc_power_wh == -50.0


class TestResourceRequest:
    """ResourceRequest dataclass, properties, and sign convention."""

    def test_idle_request_default(self):
        req = ResourceRequest(device_id="dev", hour=0)
        assert req.is_idle is True
        assert req.is_producer is False
        assert req.is_consumer is False

    def test_consumer_positive_ac(self):
        req = ResourceRequest(device_id="dev", hour=0, ac_power_wh=500.0)
        assert req.is_consumer is True
        assert req.is_producer is False
        assert req.is_idle is False

    def test_producer_negative_ac(self):
        req = ResourceRequest(device_id="dev", hour=0, ac_power_wh=-1000.0)
        assert req.is_producer is True
        assert req.is_consumer is False
        assert req.is_idle is False

    def test_producer_negative_heat(self):
        req = ResourceRequest(device_id="dev", hour=0, heat_wh=-300.0)
        assert req.is_producer is True

    def test_consumer_positive_heat(self):
        req = ResourceRequest(device_id="dev", hour=0, heat_wh=300.0)
        assert req.is_consumer is True

    def test_default_priority_is_normal(self):
        req = ResourceRequest(device_id="dev", hour=0)
        assert req.priority == Priority.NORMAL

    def test_custom_priority(self):
        req = ResourceRequest(device_id="dev", hour=0, priority=Priority.HIGH)
        assert req.priority == Priority.HIGH

    def test_minimum_fields_stored(self):
        req = ResourceRequest(
            device_id="dev",
            hour=5,
            ac_power_wh=1000.0,
            min_ac_power_wh=200.0,
        )
        assert req.min_ac_power_wh == 200.0
        assert req.hour == 5

    def test_cross_domain_producer_and_consumer(self):
        """A device offering heat while consuming AC is both producer and consumer."""
        req = ResourceRequest(
            device_id="heatpump",
            hour=0,
            ac_power_wh=1000.0,   # consuming AC
            heat_wh=-3000.0,      # offering heat
        )
        assert req.is_consumer is True
        assert req.is_producer is True
        assert req.is_idle is False


class TestResourceGrant:
    """ResourceGrant factory methods and attributes."""

    def test_idle_factory(self):
        grant = ResourceGrant.idle("dev")
        assert grant.device_id == "dev"
        assert grant.ac_power_wh == 0.0
        assert grant.dc_power_wh == 0.0
        assert grant.heat_wh == 0.0
        assert grant.curtailed is False

    def test_curtailed_factory(self):
        grant = ResourceGrant.curtailed_grant("dev")
        assert grant.device_id == "dev"
        assert grant.curtailed is True
        assert grant.ac_power_wh == 0.0

    def test_explicit_values(self):
        grant = ResourceGrant(
            device_id="battery",
            ac_power_wh=500.0,
            heat_wh=100.0,
            curtailed=False,
        )
        assert grant.ac_power_wh == 500.0
        assert grant.heat_wh == 100.0
        assert grant.curtailed is False

    def test_curtailed_flag_independent_of_values(self):
        """Curtailed flag can coexist with non-zero values (engine sets both)."""
        grant = ResourceGrant(device_id="dev", ac_power_wh=0.0, curtailed=True)
        assert grant.curtailed is True
