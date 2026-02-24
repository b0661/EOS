"""Tests for genetic2.simulation.engine.

Covers: EnergySimulationEngine simulation loop, grid balancing,
multi-device scenarios, reset between runs, genome repair collection,
step-time forwarding, and arbitrator injection.

Changes from previous version:
- make_input() uses step_times/step_interval instead of start_hour/end_hour
- result.hours -> result.steps, h.hour -> s.step_index / s.step_time
- flows_per_hour -> flows_per_step, soc_per_hour -> soc_per_step
- dispatch() calls store_genome(); engine then calls apply_genome()
- GenomeAssembler requires num_steps and step_interval arguments
- New TestEngineGenomeRepair class
- New TestEngineStepTimes class (replaces TestEngineHourRange)
- call_count check: once per step (not hour)
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
from fixtures.genetic2fixtures import (
    START_TIME,
    STEP_INTERVAL,
    FixedLoad,
    SimpleBattery,
    SimplePV,
    make_assembler,
    make_input,
    make_step_times,
)

from akkudoktoreos.simulation.genetic2.arbitrator import PriorityArbitrator
from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.timeseries import SimulationInput

# ---------------------------------------------------------------------------
# Grid balancing
# ---------------------------------------------------------------------------

class TestEngineGridBalancing:
    """Grid acts as the balancing element — surplus feeds in, deficit imports."""

    def test_pure_pv_surplus_feeds_into_grid(self):
        """PV 1000 Wh, load 500 Wh → 500 Wh feed-in every step."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 1000.0)))
        result = EnergySimulationEngine(reg).simulate(make_input(load=500.0))

        for step in result.steps:
            assert step.feedin_wh == pytest.approx(500.0)
            assert step.grid_import_wh == pytest.approx(0.0)

    def test_no_pv_all_load_from_grid(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(24)))
        result = EnergySimulationEngine(reg).simulate(make_input(load=500.0))

        for step in result.steps:
            assert step.grid_import_wh == pytest.approx(500.0)
            assert step.feedin_wh == pytest.approx(0.0)

    def test_pv_exactly_matches_load_no_grid_interaction(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 500.0)))
        result = EnergySimulationEngine(reg).simulate(make_input(load=500.0))

        for step in result.steps:
            assert step.feedin_wh == pytest.approx(0.0)
            assert step.grid_import_wh == pytest.approx(0.0)

    def test_zero_load_all_pv_feeds_in(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 1000.0)))
        result = EnergySimulationEngine(reg).simulate(make_input(load=0.0))

        for step in result.steps:
            assert step.feedin_wh == pytest.approx(1000.0)
            assert step.grid_import_wh == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------

class TestEngineFinancials:
    """Cost and revenue calculations."""

    def test_cost_equals_grid_import_times_price(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(24)))
        result = EnergySimulationEngine(reg).simulate(make_input(load=1000.0, price=0.30))

        for step in result.steps:
            assert step.cost_eur == pytest.approx(1000.0 * 0.30)

    def test_revenue_equals_feedin_times_tariff(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 2000.0)))
        result = EnergySimulationEngine(reg).simulate(make_input(load=500.0, tariff=0.08))

        for step in result.steps:
            assert step.revenue_eur == pytest.approx(1500.0 * 0.08)

    def test_net_balance_positive_when_costs_exceed_revenue(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(24)))
        result = EnergySimulationEngine(reg).simulate(make_input(load=500.0, price=0.30, tariff=0.08))
        assert result.net_balance_eur > 0

    def test_net_balance_negative_when_revenue_exceeds_costs(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 5000.0)))
        result = EnergySimulationEngine(reg).simulate(make_input(load=100.0, price=0.30, tariff=0.10))
        assert result.net_balance_eur < 0


# ---------------------------------------------------------------------------
# Battery behaviour
# ---------------------------------------------------------------------------

class TestEngineBatteryBehaviour:
    """Battery charge, discharge, and SoC tracking in full simulation runs."""

    def _dispatch_and_run(self, reg, genome_array, n, **input_kwargs):
        """Helper: dispatch genome, simulate, return result."""
        assembler = make_assembler(reg, n=n)
        assembler.dispatch(genome_array, reg)
        return EnergySimulationEngine(reg).simulate(make_input(n=n, **input_kwargs))

    def test_battery_discharges_when_scheduled(self):
        reg = DeviceRegistry()
        pv = SimplePV("pv", np.zeros(24))
        bat = SimpleBattery("bat", initial_soc_pct=100.0, num_steps=24)
        reg.register(pv)
        reg.register(bat)

        result = self._dispatch_and_run(reg, np.ones(24), 24, load=500.0)

        # Battery should have provided AC power every step
        bat_flows = result.flows_per_step("bat")
        assert all(f.ac_power_wh >= 0 for f in bat_flows if f is not None)

    def test_battery_soc_decreases_during_discharge(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(4)))
        bat = SimpleBattery("bat", capacity_wh=10_000, initial_soc_pct=100.0, num_steps=4)
        reg.register(bat)

        result = self._dispatch_and_run(reg, np.ones(4), 4, load=500.0)

        soc = result.soc_per_step("bat")
        assert soc[0] == pytest.approx(100.0)
        for i in range(1, len(soc)):
            if soc[i] is not None and soc[i - 1] is not None:
                assert soc[i] <= soc[i - 1]

    def test_battery_charges_when_surplus_available(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(4, 3000.0)))
        bat = SimpleBattery("bat", capacity_wh=10_000, initial_soc_pct=0.0, num_steps=4)
        reg.register(bat)

        result = self._dispatch_and_run(reg, np.full(4, 2.0), 4, load=500.0)

        soc = result.soc_per_step("bat")
        assert soc[-1] > soc[0]

    def test_empty_battery_cannot_discharge(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(4)))
        bat = SimpleBattery("bat", capacity_wh=10_000, initial_soc_pct=0.0, num_steps=4)
        reg.register(bat)

        result = self._dispatch_and_run(reg, np.ones(4), 4, load=500.0)

        for f in result.flows_per_step("bat"):
            if f is not None:
                assert f.ac_power_wh == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Multiple devices
# ---------------------------------------------------------------------------

class TestEngineMultipleDevices:
    """Engine handles multiple batteries and PV arrays correctly."""

    def test_two_batteries_both_tracked(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(4, 5000.0)))
        bat1 = SimpleBattery("bat1", capacity_wh=5_000, initial_soc_pct=0.0, num_steps=4)
        bat2 = SimpleBattery("bat2", capacity_wh=5_000, initial_soc_pct=0.0, num_steps=4)
        reg.register(bat1)
        reg.register(bat2)

        assembler = make_assembler(reg, n=4)
        assembler.dispatch(np.full(assembler.total_size, 2.0), reg)

        result = EnergySimulationEngine(reg).simulate(make_input(n=4, load=100.0))
        assert "bat1" in result.all_device_ids()
        assert "bat2" in result.all_device_ids()

    def test_two_pv_arrays_generation_summed(self):
        """Two PV arrays each 500 Wh → 1000 Wh total feed-in with zero load."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.full(4, 500.0)))
        reg.register(SimplePV("pv2", np.full(4, 500.0)))

        result = EnergySimulationEngine(reg).simulate(make_input(n=4, load=0.0))

        for step in result.steps:
            assert step.feedin_wh == pytest.approx(1000.0)

    def test_result_contains_all_device_ids(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.zeros(4)))
        reg.register(SimpleBattery("bat1", num_steps=4))
        reg.register(SimpleBattery("bat2", num_steps=4))

        result = EnergySimulationEngine(reg).simulate(make_input(n=4))
        assert {"pv1", "bat1", "bat2"}.issubset(result.all_device_ids())


# ---------------------------------------------------------------------------
# Reset between runs
# ---------------------------------------------------------------------------

class TestEngineReset:
    """Engine resets device physical state between runs correctly."""

    def test_second_run_starts_from_initial_soc(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(4)))
        bat = SimpleBattery("bat", capacity_wh=10_000, initial_soc_pct=50.0, num_steps=4)
        reg.register(bat)

        assembler = make_assembler(reg, n=4)
        genome = np.ones(4)  # discharge

        engine = EnergySimulationEngine(reg)

        assembler.dispatch(genome, reg)
        result1 = engine.simulate(make_input(n=4, load=500.0))
        soc_end_run1 = result1.soc_per_step("bat")[-1]

        assembler.dispatch(genome, reg)
        result2 = engine.simulate(make_input(n=4, load=500.0))
        soc_start_run2 = result2.soc_per_step("bat")[0]

        # Run 2 must start at initial SoC (50%), not where run 1 ended
        assert soc_start_run2 == pytest.approx(50.0)
        # Sanity: run 1 actually drained the battery
        assert soc_end_run1 < 50.0

    def test_results_identical_for_same_genome(self):
        """Same genome → identical results across two separate runs."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(6, 1000.0)))
        bat = SimpleBattery("bat", capacity_wh=5_000, initial_soc_pct=30.0, num_steps=6)
        reg.register(bat)

        assembler = make_assembler(reg, n=6)
        genome = np.array([1.0, 1.0, 2.0, 2.0, 0.0, 0.0])
        engine = EnergySimulationEngine(reg)

        assembler.dispatch(genome, reg)
        r1 = engine.simulate(make_input(n=6, load=300.0))

        assembler.dispatch(genome, reg)
        r2 = engine.simulate(make_input(n=6, load=300.0))

        assert r1.net_balance_eur == pytest.approx(r2.net_balance_eur)
        assert r1.total_losses_wh == pytest.approx(r2.total_losses_wh)


# ---------------------------------------------------------------------------
# Step times
# ---------------------------------------------------------------------------

class TestEngineStepTimes:
    """Engine produces results with correct step counts and timestamps."""

    def test_result_has_correct_number_of_steps(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(12, 1000.0)))
        result = EnergySimulationEngine(reg).simulate(make_input(n=12))
        assert len(result.steps) == 12

    def test_step_indices_are_zero_based(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(6, 1000.0)))
        result = EnergySimulationEngine(reg).simulate(make_input(n=6))
        assert [s.step_index for s in result.steps] == [0, 1, 2, 3, 4, 5]

    def test_step_times_match_input_step_times(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(4, 0.0)))
        start = datetime(2024, 6, 1, 8, 0)
        inp = make_input(n=4, start=start)
        result = EnergySimulationEngine(reg).simulate(inp)

        for i, step in enumerate(result.steps):
            assert step.step_time == start + i * STEP_INTERVAL

    def test_sub_hourly_steps_run_correctly(self):
        """Engine runs at 15-minute resolution without any changes."""
        interval = timedelta(minutes=15)
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(96, 250.0)))  # 250 Wh per 15-min step
        inp = make_input(n=96, load=0.0, interval=interval)
        result = EnergySimulationEngine(reg).simulate(inp)

        assert len(result.steps) == 96
        for step in result.steps:
            assert step.feedin_wh == pytest.approx(250.0)

    def test_empty_registry_runs_without_error(self):
        result = EnergySimulationEngine(DeviceRegistry()).simulate(make_input(n=4, load=0.0))
        assert len(result.steps) == 4
        for step in result.steps:
            assert step.feedin_wh == pytest.approx(0.0)
            assert step.grid_import_wh == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_genome receives step_times
# ---------------------------------------------------------------------------

class TestEngineApplyGenomeForwarding:
    """Engine forwards step_times to devices via apply_genome()."""

    def test_battery_receives_step_times_on_apply_genome(self):
        reg = DeviceRegistry()
        bat = SimpleBattery("bat", num_steps=4)
        reg.register(bat)

        start = datetime(2024, 3, 15, 6, 0)
        inp = make_input(n=4, start=start)
        EnergySimulationEngine(reg).simulate(inp)

        assert bat.last_step_times is not None
        assert len(bat.last_step_times) == 4
        assert bat.last_step_times[0] == start

    def test_pv_receives_step_times_on_apply_genome(self):
        """Genome-less devices also receive step_times via apply_genome()."""
        step_times_received = []

        from collections.abc import Sequence as Seq
        from datetime import datetime as DT

        class TrackingPV(SimplePV):
            def apply_genome(self, genome_slice, step_times: Seq[DT]) -> None:
                step_times_received.extend(step_times)
                super().apply_genome(genome_slice, step_times)

        reg = DeviceRegistry()
        reg.register(TrackingPV("pv", np.zeros(3)))
        inp = make_input(n=3)
        EnergySimulationEngine(reg).simulate(inp)

        assert len(step_times_received) == 3

    def test_apply_genome_called_once_per_simulate(self):
        """apply_genome() is called exactly once per simulate() call."""
        apply_count = 0

        class CountingBattery(SimpleBattery):
            def apply_genome(self, genome_slice, step_times) -> None:
                nonlocal apply_count
                apply_count += 1
                super().apply_genome(genome_slice, step_times)

        reg = DeviceRegistry()
        bat = CountingBattery("bat", num_steps=4)
        reg.register(bat)

        engine = EnergySimulationEngine(reg)
        engine.simulate(make_input(n=4))
        assert apply_count == 1

        engine.simulate(make_input(n=4))
        assert apply_count == 2  # called again on the second run


# ---------------------------------------------------------------------------
# Genome repair collection
# ---------------------------------------------------------------------------

class TestEngineGenomeRepair:
    """Engine collects, validates, and copies repair proposals after a run."""

    def test_no_repairs_when_no_device_proposes(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(4)))
        result = EnergySimulationEngine(reg).simulate(make_input(n=4))
        assert result.repairs == []

    def test_valid_repair_is_collected(self):
        """A device returning a valid GenomeRepairResult has it accepted."""
        from akkudoktoreos.devices.genetic2.base import GenomeRepairResult

        class RepairingBattery(SimpleBattery):
            def repair_genome(self):
                return GenomeRepairResult(
                    repaired_slice=np.array([0.0, 0.0, 0.0, 0.0]),
                    changed=True,
                )

        reg = DeviceRegistry()
        bat = RepairingBattery("bat", num_steps=4)
        reg.register(bat)

        assembler = make_assembler(reg, n=4)
        assembler.dispatch(np.zeros(4), reg)

        result = EnergySimulationEngine(reg).simulate(make_input(n=4))
        assert len(result.repairs) == 1
        assert result.repairs[0].device_id == "bat"
        np.testing.assert_array_equal(result.repairs[0].repaired_slice, [0.0, 0.0, 0.0, 0.0])

    def test_repair_slice_is_defensively_copied(self):
        """Mutating the device's internal buffer after simulate() must not alter the stored repair."""
        from akkudoktoreos.devices.genetic2.base import GenomeRepairResult

        internal_buf = np.array([1.0, 1.0, 1.0, 1.0])

        class MutatingBattery(SimpleBattery):
            def repair_genome(self):
                return GenomeRepairResult(repaired_slice=internal_buf, changed=True)

        reg = DeviceRegistry()
        bat = MutatingBattery("bat", num_steps=4)
        reg.register(bat)

        assembler = make_assembler(reg, n=4)
        assembler.dispatch(np.zeros(4), reg)

        result = EnergySimulationEngine(reg).simulate(make_input(n=4))
        stored = result.repairs[0].repaired_slice.copy()

        internal_buf[:] = 99.0  # mutate the device's buffer after the run

        np.testing.assert_array_equal(result.repairs[0].repaired_slice, stored)

    def test_invalid_repair_bounds_discarded_with_warning(self):
        """Out-of-bounds repair proposal is discarded; no exception raised."""
        from akkudoktoreos.devices.genetic2.base import GenomeRepairResult

        class BadBattery(SimpleBattery):
            def repair_genome(self):
                return GenomeRepairResult(
                    repaired_slice=np.array([99.0, 99.0, 99.0, 99.0]),  # high=2 violated
                    changed=True,
                )

        reg = DeviceRegistry()
        bat = BadBattery("bat", num_steps=4)
        reg.register(bat)

        assembler = make_assembler(reg, n=4)
        assembler.dispatch(np.zeros(4), reg)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = EnergySimulationEngine(reg).simulate(make_input(n=4))

        assert result.repairs == []
        assert any("bat" in str(warning.message) for warning in w)

    def test_genome_less_device_repair_discarded_with_warning(self):
        """PV returning a repair proposal is warned and discarded."""
        from akkudoktoreos.devices.genetic2.base import GenomeRepairResult

        class RepairingPV(SimplePV):
            def repair_genome(self):
                return GenomeRepairResult(
                    repaired_slice=np.array([1.0]),
                    changed=True,
                )

        reg = DeviceRegistry()
        reg.register(RepairingPV("pv", np.zeros(4)))

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = EnergySimulationEngine(reg).simulate(make_input(n=4))

        assert result.repairs == []
        assert any("pv" in str(warning.message) for warning in w)

    def test_unchanged_repair_not_collected(self):
        """GenomeRepairResult(changed=False) is silently skipped."""
        from akkudoktoreos.devices.genetic2.base import GenomeRepairResult

        class NoChangeBattery(SimpleBattery):
            def repair_genome(self):
                return GenomeRepairResult(
                    repaired_slice=np.zeros(4),
                    changed=False,
                )

        reg = DeviceRegistry()
        bat = NoChangeBattery("bat", num_steps=4)
        reg.register(bat)

        assembler = make_assembler(reg, n=4)
        assembler.dispatch(np.zeros(4), reg)

        result = EnergySimulationEngine(reg).simulate(make_input(n=4))
        assert result.repairs == []


# ---------------------------------------------------------------------------
# Custom arbitrator
# ---------------------------------------------------------------------------

class TestEngineCustomArbitrator:
    """Engine respects an injected arbitrator."""

    def test_custom_arbitrator_is_called_once_per_step(self):
        from akkudoktoreos.simulation.genetic2.arbitrator import ArbitratorBase

        call_count = 0

        class CountingArbitrator(ArbitratorBase):
            def arbitrate(self, requests):
                nonlocal call_count
                call_count += 1
                return PriorityArbitrator().arbitrate(requests)

        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(4, 1000.0)))
        engine = EnergySimulationEngine(reg, arbitrator=CountingArbitrator())
        engine.simulate(make_input(n=4))

        assert call_count == 4  # exactly once per step
