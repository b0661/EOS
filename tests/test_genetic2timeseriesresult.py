"""Tests for genetic2.simulation.timeseries and genetic2.simulation.result.

Covers: SimulationInput validation, DeviceSchedule validation,
SimulationResult aggregate properties, per-device queries, and to_dict export.
"""

import numpy as np
import pytest

from akkudoktoreos.simulation.genetic2.flows import EnergyFlows
from akkudoktoreos.simulation.genetic2.result import (
    DeviceHourlyState,
    HourlyResult,
    SimulationResult,
)
from akkudoktoreos.simulation.genetic2.timeseries import DeviceSchedule, SimulationInput

# ---------------------------------------------------------------------------
# DeviceSchedule
# ---------------------------------------------------------------------------

class TestDeviceSchedule:
    """DeviceSchedule validation."""

    def test_valid_schedule_creates_without_error(self):
        schedule = DeviceSchedule(
            device_id="bat1",
            charge_factors=np.array([0.0, 1.0, 0.5]),
            discharge_factors=np.array([1.0, 0.0, 0.5]),
        )
        assert schedule.device_id == "bat1"

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="bat1"):
            DeviceSchedule(
                device_id="bat1",
                charge_factors=np.array([0.0, 1.0]),
                discharge_factors=np.array([1.0, 0.0, 0.5]),
            )

    def test_charge_factor_above_one_raises(self):
        with pytest.raises(ValueError, match="charge_factors"):
            DeviceSchedule(
                device_id="bat1",
                charge_factors=np.array([0.0, 1.5]),
                discharge_factors=np.array([0.0, 0.0]),
            )

    def test_charge_factor_negative_raises(self):
        with pytest.raises(ValueError, match="charge_factors"):
            DeviceSchedule(
                device_id="bat1",
                charge_factors=np.array([-0.1, 1.0]),
                discharge_factors=np.array([0.0, 0.0]),
            )

    def test_discharge_factor_above_one_raises(self):
        with pytest.raises(ValueError, match="discharge_factors"):
            DeviceSchedule(
                device_id="bat1",
                charge_factors=np.array([0.0, 0.0]),
                discharge_factors=np.array([0.0, 1.2]),
            )

    def test_all_zeros_valid(self):
        schedule = DeviceSchedule(
            device_id="bat1",
            charge_factors=np.zeros(24),
            discharge_factors=np.zeros(24),
        )
        assert len(schedule.charge_factors) == 24


# ---------------------------------------------------------------------------
# SimulationInput
# ---------------------------------------------------------------------------

class TestSimulationInput:
    """SimulationInput construction and validation."""

    def _make_input(self, n: int = 24) -> SimulationInput:
        return SimulationInput(
            start_hour=0,
            end_hour=n,
            load_wh=np.full(n, 500.0),
            electricity_price=np.full(n, 0.30),
            feed_in_tariff=np.full(n, 0.08),
        )

    def test_valid_input_creates(self):
        inp = self._make_input(24)
        assert inp.total_hours == 24

    def test_total_hours_property(self):
        inp = SimulationInput(
            start_hour=6, end_hour=18,
            load_wh=np.zeros(18),
            electricity_price=np.zeros(18),
            feed_in_tariff=np.zeros(18),
        )
        assert inp.total_hours == 12

    def test_load_too_short_raises(self):
        with pytest.raises(ValueError, match="load_wh"):
            SimulationInput(
                start_hour=0, end_hour=24,
                load_wh=np.zeros(10),           # too short
                electricity_price=np.zeros(24),
                feed_in_tariff=np.zeros(24),
            )

    def test_price_too_short_raises(self):
        with pytest.raises(ValueError, match="electricity_price"):
            SimulationInput(
                start_hour=0, end_hour=24,
                load_wh=np.zeros(24),
                electricity_price=np.zeros(5),  # too short
                feed_in_tariff=np.zeros(24),
            )

    def test_add_and_get_schedule(self):
        inp = self._make_input()
        schedule = DeviceSchedule("bat1", np.zeros(24), np.zeros(24))
        inp.add_schedule(schedule)
        assert inp.get_schedule("bat1") is schedule

    def test_get_schedule_missing_returns_none(self):
        inp = self._make_input()
        assert inp.get_schedule("nonexistent") is None

    def test_add_schedule_overwrites_existing(self):
        inp = self._make_input()
        s1 = DeviceSchedule("bat1", np.zeros(24), np.zeros(24))
        s2 = DeviceSchedule("bat1", np.ones(24), np.zeros(24))
        inp.add_schedule(s1)
        inp.add_schedule(s2)
        assert inp.get_schedule("bat1") is s2


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

def _make_hour(
    hour: int,
    feedin: float = 0.0,
    grid_import: float = 0.0,
    losses: float = 0.0,
    cost: float = 0.0,
    revenue: float = 0.0,
    load: float = 500.0,
    device_states: dict | None = None,
) -> HourlyResult:
    return HourlyResult(
        hour=hour,
        total_load_wh=load,
        feedin_wh=feedin,
        grid_import_wh=grid_import,
        total_losses_wh=losses,
        self_consumption_wh=0.0,
        cost_eur=cost,
        revenue_eur=revenue,
        devices=device_states or {},
    )


class TestSimulationResultAggregates:
    """Aggregate property calculations."""

    def test_total_cost_sums_hours(self):
        result = SimulationResult(hours=[
            _make_hour(0, cost=10.0),
            _make_hour(1, cost=20.0),
            _make_hour(2, cost=5.0),
        ])
        assert result.total_cost_eur == pytest.approx(35.0)

    def test_total_revenue_sums_hours(self):
        result = SimulationResult(hours=[
            _make_hour(0, revenue=3.0),
            _make_hour(1, revenue=7.0),
        ])
        assert result.total_revenue_eur == pytest.approx(10.0)

    def test_net_balance_cost_minus_revenue(self):
        result = SimulationResult(hours=[
            _make_hour(0, cost=50.0, revenue=20.0),
        ])
        assert result.net_balance_eur == pytest.approx(30.0)

    def test_total_losses(self):
        result = SimulationResult(hours=[
            _make_hour(0, losses=10.0),
            _make_hour(1, losses=5.0),
        ])
        assert result.total_losses_wh == pytest.approx(15.0)

    def test_total_feedin(self):
        result = SimulationResult(hours=[
            _make_hour(0, feedin=100.0),
            _make_hour(1, feedin=200.0),
        ])
        assert result.total_feedin_wh == pytest.approx(300.0)

    def test_total_grid_import(self):
        result = SimulationResult(hours=[
            _make_hour(0, grid_import=50.0),
            _make_hour(1, grid_import=75.0),
        ])
        assert result.total_grid_import_wh == pytest.approx(125.0)

    def test_empty_result_all_zeros(self):
        result = SimulationResult()
        assert result.total_cost_eur == 0.0
        assert result.total_revenue_eur == 0.0
        assert result.net_balance_eur == 0.0
        assert result.total_losses_wh == 0.0


class TestSimulationResultPerDevice:
    """Per-device query methods."""

    def _make_result_with_device(self) -> SimulationResult:
        flows_h0 = EnergyFlows(soc_pct=80.0, ac_power_wh=500.0)
        flows_h1 = EnergyFlows(soc_pct=60.0, ac_power_wh=500.0)
        state_h0 = DeviceHourlyState("bat1", flows=flows_h0)
        state_h1 = DeviceHourlyState("bat1", flows=flows_h1)
        return SimulationResult(hours=[
            _make_hour(0, device_states={"bat1": state_h0}),
            _make_hour(1, device_states={"bat1": state_h1}),
        ])

    def test_soc_per_hour_returns_correct_values(self):
        result = self._make_result_with_device()
        soc = result.soc_per_hour("bat1")
        assert soc == [pytest.approx(80.0), pytest.approx(60.0)]

    def test_soc_per_hour_missing_device_returns_none(self):
        result = self._make_result_with_device()
        soc = result.soc_per_hour("nonexistent")
        assert all(s is None for s in soc)

    def test_flows_per_hour_returns_flows(self):
        result = self._make_result_with_device()
        flows = result.flows_per_hour("bat1")
        assert len(flows) == 2
        assert flows[0] is not None
        assert flows[0].ac_power_wh == pytest.approx(500.0)

    def test_all_device_ids_collected(self):
        flows = EnergyFlows()
        result = SimulationResult(hours=[
            _make_hour(0, device_states={
                "bat1": DeviceHourlyState("bat1", flows=flows),
                "pv1":  DeviceHourlyState("pv1",  flows=flows),
            }),
            _make_hour(1, device_states={
                "bat1": DeviceHourlyState("bat1", flows=flows),
            }),
        ])
        assert result.all_device_ids() == {"bat1", "pv1"}

    def test_hourly_result_device_accessor(self):
        flows = EnergyFlows(soc_pct=50.0)
        state = DeviceHourlyState("bat1", flows=flows)
        hour = _make_hour(0, device_states={"bat1": state})
        assert hour.device("bat1") is state
        assert hour.device("missing") is None


class TestSimulationResultToDict:
    """Backwards-compatible to_dict() export."""

    def test_to_dict_contains_required_keys(self):
        result = SimulationResult(hours=[_make_hour(0)])
        d = result.to_dict()
        required = {
            "Last_Wh_pro_Stunde",
            "Netzeinspeisung_Wh_pro_Stunde",
            "Netzbezug_Wh_pro_Stunde",
            "Kosten_Euro_pro_Stunde",
            "Einnahmen_Euro_pro_Stunde",
            "Verluste_Pro_Stunde",
            "Gesamtbilanz_Euro",
            "Gesamtkosten_Euro",
            "Gesamteinnahmen_Euro",
            "Gesamt_Verluste",
            "soc_per_hour",
        }
        assert required.issubset(d.keys())

    def test_to_dict_values_match_properties(self):
        result = SimulationResult(hours=[
            _make_hour(0, cost=10.0, revenue=5.0, feedin=100.0, grid_import=50.0),
        ])
        d = result.to_dict()
        assert d["Gesamtkosten_Euro"] == pytest.approx(10.0)
        assert d["Gesamteinnahmen_Euro"] == pytest.approx(5.0)
        assert d["Gesamtbilanz_Euro"] == pytest.approx(5.0)

    def test_to_dict_per_hour_lists_correct_length(self):
        result = SimulationResult(hours=[_make_hour(h) for h in range(24)])
        d = result.to_dict()
        assert len(d["Last_Wh_pro_Stunde"]) == 24
        assert len(d["Kosten_Euro_pro_Stunde"]) == 24
