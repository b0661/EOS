"""Tests for genetic2.simulation.timeseries and genetic2.simulation.result.

Covers: SimulationInput validation, DeviceSchedule validation,
SimulationResult aggregate properties, per-device step queries,
genome repair collection, and to_dict export.

Changes from previous version:
- SimulationInput: start_hour/end_hour -> step_times/step_interval
- total_hours -> num_steps, total_duration added
- Array validation: exact length match (not >=)
- HourlyResult -> StepResult, DeviceHourlyState -> DeviceStepState
- hours -> steps, hour -> step_index, step_time added
- soc_per_hour -> soc_per_step, flows_per_hour -> flows_per_step
- SimulationResult gains .repairs and .step_times()
- to_dict gains step_times and soc_per_step keys; soc_per_hour removed
- GenomeRepairRecord is a new dataclass in result.py
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
from fixtures.genetic2fixtures import START_TIME, STEP_INTERVAL, make_step_times

from akkudoktoreos.simulation.genetic2.flows import EnergyFlows
from akkudoktoreos.simulation.genetic2.result import (
    DeviceStepState,
    GenomeRepairRecord,
    SimulationResult,
    StepResult,
)
from akkudoktoreos.simulation.genetic2.timeseries import DeviceSchedule, SimulationInput

# ---------------------------------------------------------------------------
# DeviceSchedule
# ---------------------------------------------------------------------------

class TestDeviceSchedule:
    """DeviceSchedule per-step factor validation."""

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
    """SimulationInput construction, validation, and properties."""

    def _make_input(self, n: int = 24) -> SimulationInput:
        return SimulationInput(
            step_times=make_step_times(n),
            step_interval=STEP_INTERVAL,
            load_wh=np.full(n, 500.0),
            electricity_price=np.full(n, 0.30),
            feed_in_tariff=np.full(n, 0.08),
        )

    def test_valid_input_creates(self):
        inp = self._make_input(24)
        assert inp.num_steps == 24

    def test_num_steps_property(self):
        inp = self._make_input(12)
        assert inp.num_steps == 12

    def test_total_duration_property(self):
        inp = self._make_input(6)
        assert inp.total_duration == timedelta(hours=6)

    def test_total_duration_sub_hourly(self):
        interval = timedelta(minutes=15)
        inp = SimulationInput(
            step_times=make_step_times(96, interval),
            step_interval=interval,
            load_wh=np.zeros(96),
            electricity_price=np.zeros(96),
            feed_in_tariff=np.zeros(96),
        )
        assert inp.total_duration == timedelta(hours=24)

    def test_empty_step_times_raises(self):
        with pytest.raises(ValueError):
            SimulationInput(
                step_times=[],
                step_interval=STEP_INTERVAL,
                load_wh=np.zeros(0),
                electricity_price=np.zeros(0),
                feed_in_tariff=np.zeros(0),
            )

    def test_negative_step_interval_raises(self):
        with pytest.raises(ValueError, match="step_interval"):
            SimulationInput(
                step_times=make_step_times(4),
                step_interval=timedelta(seconds=-1),
                load_wh=np.zeros(4),
                electricity_price=np.zeros(4),
                feed_in_tariff=np.zeros(4),
            )

    def test_load_wrong_length_raises(self):
        with pytest.raises(ValueError, match="load_wh"):
            SimulationInput(
                step_times=make_step_times(24),
                step_interval=STEP_INTERVAL,
                load_wh=np.zeros(10),          # wrong length
                electricity_price=np.zeros(24),
                feed_in_tariff=np.zeros(24),
            )

    def test_price_wrong_length_raises(self):
        with pytest.raises(ValueError, match="electricity_price"):
            SimulationInput(
                step_times=make_step_times(24),
                step_interval=STEP_INTERVAL,
                load_wh=np.zeros(24),
                electricity_price=np.zeros(5),  # wrong length
                feed_in_tariff=np.zeros(24),
            )

    def test_tariff_wrong_length_raises(self):
        with pytest.raises(ValueError, match="feed_in_tariff"):
            SimulationInput(
                step_times=make_step_times(24),
                step_interval=STEP_INTERVAL,
                load_wh=np.zeros(24),
                electricity_price=np.zeros(24),
                feed_in_tariff=np.zeros(10),   # wrong length
            )

    def test_add_and_get_schedule(self):
        inp = self._make_input()
        schedule = DeviceSchedule("bat1", np.zeros(24), np.zeros(24))
        inp.add_schedule(schedule)
        assert inp.get_schedule("bat1") is schedule

    def test_get_schedule_missing_returns_none(self):
        assert self._make_input().get_schedule("nonexistent") is None

    def test_add_schedule_overwrites_existing(self):
        inp = self._make_input()
        s1 = DeviceSchedule("bat1", np.zeros(24), np.zeros(24))
        s2 = DeviceSchedule("bat1", np.ones(24), np.zeros(24))
        inp.add_schedule(s1)
        inp.add_schedule(s2)
        assert inp.get_schedule("bat1") is s2

    def test_step_times_are_stored(self):
        times = make_step_times(6)
        inp = SimulationInput(
            step_times=times,
            step_interval=STEP_INTERVAL,
            load_wh=np.zeros(6),
            electricity_price=np.zeros(6),
            feed_in_tariff=np.zeros(6),
        )
        assert inp.step_times == times


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _make_step(
    step_index: int,
    step_time: datetime | None = None,
    feedin: float = 0.0,
    grid_import: float = 0.0,
    losses: float = 0.0,
    cost: float = 0.0,
    revenue: float = 0.0,
    load: float = 500.0,
    device_states: dict | None = None,
) -> StepResult:
    return StepResult(
        step_index=step_index,
        step_time=step_time or (START_TIME + timedelta(hours=step_index)),
        total_load_wh=load,
        feedin_wh=feedin,
        grid_import_wh=grid_import,
        total_losses_wh=losses,
        self_consumption_wh=0.0,
        cost_eur=cost,
        revenue_eur=revenue,
        devices=device_states or {},
    )


# ---------------------------------------------------------------------------
# SimulationResult aggregate properties
# ---------------------------------------------------------------------------

class TestSimulationResultAggregates:
    """Aggregate properties sum correctly across all steps."""

    def test_total_cost_sums_steps(self):
        result = SimulationResult(steps=[
            _make_step(0, cost=10.0),
            _make_step(1, cost=20.0),
            _make_step(2, cost=5.0),
        ])
        assert result.total_cost_eur == pytest.approx(35.0)

    def test_total_revenue_sums_steps(self):
        result = SimulationResult(steps=[
            _make_step(0, revenue=3.0),
            _make_step(1, revenue=7.0),
        ])
        assert result.total_revenue_eur == pytest.approx(10.0)

    def test_net_balance_cost_minus_revenue(self):
        result = SimulationResult(steps=[_make_step(0, cost=50.0, revenue=20.0)])
        assert result.net_balance_eur == pytest.approx(30.0)

    def test_net_balance_negative_when_revenue_dominates(self):
        result = SimulationResult(steps=[_make_step(0, cost=5.0, revenue=30.0)])
        assert result.net_balance_eur == pytest.approx(-25.0)

    def test_total_losses(self):
        result = SimulationResult(steps=[
            _make_step(0, losses=10.0),
            _make_step(1, losses=5.0),
        ])
        assert result.total_losses_wh == pytest.approx(15.0)

    def test_total_feedin(self):
        result = SimulationResult(steps=[
            _make_step(0, feedin=100.0),
            _make_step(1, feedin=200.0),
        ])
        assert result.total_feedin_wh == pytest.approx(300.0)

    def test_total_grid_import(self):
        result = SimulationResult(steps=[
            _make_step(0, grid_import=50.0),
            _make_step(1, grid_import=75.0),
        ])
        assert result.total_grid_import_wh == pytest.approx(125.0)

    def test_empty_result_all_zeros(self):
        result = SimulationResult()
        assert result.total_cost_eur == 0.0
        assert result.total_revenue_eur == 0.0
        assert result.net_balance_eur == 0.0
        assert result.total_losses_wh == 0.0


# ---------------------------------------------------------------------------
# Per-device queries
# ---------------------------------------------------------------------------

class TestSimulationResultPerDevice:
    """soc_per_step(), flows_per_step(), all_device_ids(), step_times()."""

    def _make_result_with_device(self) -> SimulationResult:
        flows_s0 = EnergyFlows(soc_pct=80.0, ac_power_wh=500.0)
        flows_s1 = EnergyFlows(soc_pct=60.0, ac_power_wh=500.0)
        return SimulationResult(steps=[
            _make_step(0, device_states={"bat1": DeviceStepState("bat1", flows=flows_s0)}),
            _make_step(1, device_states={"bat1": DeviceStepState("bat1", flows=flows_s1)}),
        ])

    def test_soc_per_step_returns_correct_values(self):
        result = self._make_result_with_device()
        soc = result.soc_per_step("bat1")
        assert soc == [pytest.approx(80.0), pytest.approx(60.0)]

    def test_soc_per_step_missing_device_returns_none(self):
        result = self._make_result_with_device()
        assert all(s is None for s in result.soc_per_step("nonexistent"))

    def test_flows_per_step_returns_flows(self):
        result = self._make_result_with_device()
        flows = result.flows_per_step("bat1")
        assert len(flows) == 2
        assert flows[0].ac_power_wh == pytest.approx(500.0)

    def test_flows_per_step_missing_device_returns_none(self):
        result = self._make_result_with_device()
        assert all(f is None for f in result.flows_per_step("nonexistent"))

    def test_all_device_ids_collected_across_steps(self):
        flows = EnergyFlows()
        result = SimulationResult(steps=[
            _make_step(0, device_states={
                "bat1": DeviceStepState("bat1", flows=flows),
                "pv1":  DeviceStepState("pv1",  flows=flows),
            }),
            _make_step(1, device_states={
                "bat1": DeviceStepState("bat1", flows=flows),
            }),
        ])
        assert result.all_device_ids() == {"bat1", "pv1"}

    def test_step_times_method_returns_ordered_timestamps(self):
        t0 = START_TIME
        t1 = START_TIME + timedelta(hours=1)
        result = SimulationResult(steps=[
            _make_step(0, step_time=t0),
            _make_step(1, step_time=t1),
        ])
        assert result.step_times() == [t0, t1]

    def test_step_result_device_accessor(self):
        flows = EnergyFlows(soc_pct=50.0)
        state = DeviceStepState("bat1", flows=flows)
        step = _make_step(0, device_states={"bat1": state})
        assert step.device("bat1") is state
        assert step.device("missing") is None

    def test_device_step_state_soc_pct_property(self):
        flows = EnergyFlows(soc_pct=42.0)
        state = DeviceStepState("bat1", flows=flows)
        assert state.soc_pct == pytest.approx(42.0)

    def test_device_step_state_soc_none_for_non_storage(self):
        flows = EnergyFlows()  # soc_pct defaults to None
        state = DeviceStepState("pv1", flows=flows)
        assert state.soc_pct is None


# ---------------------------------------------------------------------------
# Genome repair collection
# ---------------------------------------------------------------------------

class TestSimulationResultRepairs:
    """SimulationResult.repairs stores validated genome repair records."""

    def test_repairs_empty_by_default(self):
        result = SimulationResult()
        assert result.repairs == []

    def test_genome_repair_record_stores_device_id_and_slice(self):
        record = GenomeRepairRecord(
            device_id="bat1",
            repaired_slice=np.array([0.0, 1.0, 2.0]),
        )
        assert record.device_id == "bat1"
        np.testing.assert_array_equal(record.repaired_slice, [0.0, 1.0, 2.0])

    def test_repairs_list_contains_multiple_records(self):
        records = [
            GenomeRepairRecord("bat1", np.array([1.0, 2.0])),
            GenomeRepairRecord("bat2", np.array([0.0, 0.0])),
        ]
        result = SimulationResult(repairs=records)
        assert len(result.repairs) == 2
        assert result.repairs[0].device_id == "bat1"
        assert result.repairs[1].device_id == "bat2"


# ---------------------------------------------------------------------------
# to_dict export
# ---------------------------------------------------------------------------

class TestSimulationResultToDict:
    """Backwards-compatible to_dict() export."""

    def test_to_dict_contains_required_keys(self):
        result = SimulationResult(steps=[_make_step(0)])
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
            "step_times",
            "soc_per_step",
        }
        assert required.issubset(d.keys())

    def test_to_dict_does_not_contain_legacy_hour_key(self):
        """soc_per_hour key has been replaced by soc_per_step."""
        result = SimulationResult(steps=[_make_step(0)])
        assert "soc_per_hour" not in result.to_dict()

    def test_to_dict_values_match_properties(self):
        result = SimulationResult(steps=[
            _make_step(0, cost=10.0, revenue=5.0, feedin=100.0, grid_import=50.0),
        ])
        d = result.to_dict()
        assert d["Gesamtkosten_Euro"] == pytest.approx(10.0)
        assert d["Gesamteinnahmen_Euro"] == pytest.approx(5.0)
        assert d["Gesamtbilanz_Euro"] == pytest.approx(5.0)

    def test_to_dict_per_step_lists_correct_length(self):
        result = SimulationResult(steps=[_make_step(i) for i in range(24)])
        d = result.to_dict()
        assert len(d["Last_Wh_pro_Stunde"]) == 24
        assert len(d["Kosten_Euro_pro_Stunde"]) == 24
        assert len(d["step_times"]) == 24

    def test_to_dict_step_times_are_iso_strings(self):
        t = START_TIME
        result = SimulationResult(steps=[_make_step(0, step_time=t)])
        d = result.to_dict()
        assert d["step_times"][0] == t.isoformat()
