"""Tests for genetic2.simulation.engine.

Covers: EnergySimulationEngine simulation loop, grid balancing,
multi-device scenarios, reset between runs, and arbitrator integration.
"""

import numpy as np
import pytest
from fixtures.genetic2fixtures import FixedLoad, SimpleBattery, SimplePV

from akkudoktoreos.optimization.genetic2.genome import GenomeAssembler
from akkudoktoreos.simulation.genetic2.arbitrator import PriorityArbitrator
from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.timeseries import SimulationInput


def make_input(
    hours: int = 24,
    load: float = 500.0,
    price: float = 0.30,
    tariff: float = 0.08,
    start: int = 0,
) -> SimulationInput:
    return SimulationInput(
        start_hour=start,
        end_hour=start + hours,
        load_wh=np.full(start + hours, load),
        electricity_price=np.full(start + hours, price),
        feed_in_tariff=np.full(start + hours, tariff),
    )


class TestEngineGridBalancing:
    """Grid acts as the balancing element — surplus feeds in, deficit imports."""

    def test_pure_pv_surplus_feeds_into_grid(self):
        """PV generates 1000 Wh, load is 500 Wh → 500 Wh feed-in."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 1000.0)))
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(load=500.0))

        for hour in result.hours:
            assert hour.feedin_wh == pytest.approx(500.0)
            assert hour.grid_import_wh == pytest.approx(0.0)

    def test_no_pv_all_load_from_grid(self):
        """No generation → all load must come from grid."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(24)))
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(load=500.0))

        for hour in result.hours:
            assert hour.grid_import_wh == pytest.approx(500.0)
            assert hour.feedin_wh == pytest.approx(0.0)

    def test_pv_exactly_matches_load_no_grid_interaction(self):
        """PV = load → no feed-in, no grid import."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 500.0)))
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(load=500.0))

        for hour in result.hours:
            assert hour.feedin_wh == pytest.approx(0.0)
            assert hour.grid_import_wh == pytest.approx(0.0)

    def test_zero_load_all_pv_feeds_in(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 1000.0)))
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(load=0.0))

        for hour in result.hours:
            assert hour.feedin_wh == pytest.approx(1000.0)
            assert hour.grid_import_wh == pytest.approx(0.0)


class TestEngineFinancials:
    """Cost and revenue calculations."""

    def test_cost_equals_grid_import_times_price(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(24)))
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(load=1000.0, price=0.30))

        for hour in result.hours:
            assert hour.cost_eur == pytest.approx(1000.0 * 0.30)

    def test_revenue_equals_feedin_times_tariff(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 2000.0)))
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(load=500.0, tariff=0.08))

        for hour in result.hours:
            assert hour.revenue_eur == pytest.approx(1500.0 * 0.08)

    def test_net_balance_positive_when_costs_exceed_revenue(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(24)))  # no generation
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(load=500.0, price=0.30, tariff=0.08))

        assert result.net_balance_eur > 0

    def test_net_balance_negative_when_revenue_exceeds_costs(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(24, 5000.0)))  # massive generation
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(load=100.0, price=0.30, tariff=0.10))

        assert result.net_balance_eur < 0


class TestEngineBatteryBehaviour:
    """Battery charge, discharge, and SoC tracking in full simulation runs."""

    def test_battery_discharges_when_scheduled(self):
        reg = DeviceRegistry()
        pv = SimplePV("pv", np.zeros(24))
        bat = SimpleBattery("bat", initial_soc_pct=100.0, prediction_hours=24)
        reg.register(pv)
        reg.register(bat)

        # Set discharge schedule for all hours via genome (1 = discharge)
        assembler = GenomeAssembler(reg)
        genome = np.ones(24)  # all hours = discharge
        assembler.dispatch(genome, reg)

        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(load=500.0))

        # Battery should have contributed AC power every hour
        bat_flows = result.flows_per_hour("bat")
        assert all(f.ac_power_wh >= 0 for f in bat_flows if f is not None)

    def test_battery_soc_decreases_during_discharge(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(4)))
        bat = SimpleBattery("bat", capacity_wh=10_000, initial_soc_pct=100.0,
                             prediction_hours=4)
        reg.register(bat)

        assembler = GenomeAssembler(reg)
        genome = np.ones(4)  # discharge all hours
        assembler.dispatch(genome, reg)

        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(hours=4, load=500.0))

        soc = result.soc_per_hour("bat")
        # SoC at start of each hour — should decrease progressively
        assert soc[0] == pytest.approx(100.0)
        for i in range(1, len(soc)):
            current = soc[i]
            previous = soc[i - 1]
            if current is not None and previous is not None:
                assert current <= previous

    def test_battery_charges_when_surplus_available(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(4, 3000.0)))
        bat = SimpleBattery("bat", capacity_wh=10_000, initial_soc_pct=0.0,
                             prediction_hours=4)
        reg.register(bat)

        # genome 2 = charge all hours
        assembler = GenomeAssembler(reg)
        genome = np.full(4, 2.0)
        assembler.dispatch(genome, reg)

        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(hours=4, load=500.0))

        soc = result.soc_per_hour("bat")
        # Battery should gain SoC over time
        first = soc[0]
        last = soc[-1]
        assert first is not None
        assert last is not None
        assert last > first

    def test_empty_battery_cannot_discharge(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(4)))
        bat = SimpleBattery("bat", capacity_wh=10_000, initial_soc_pct=0.0,
                             prediction_hours=4)
        reg.register(bat)

        assembler = GenomeAssembler(reg)
        genome = np.ones(4)  # discharge all hours
        assembler.dispatch(genome, reg)

        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(hours=4, load=500.0))

        bat_flows = result.flows_per_hour("bat")
        for f in bat_flows:
            if f is not None:
                assert f.ac_power_wh == pytest.approx(0.0)


class TestEngineMultipleDevices:
    """Engine handles multiple batteries, PV arrays, and device types correctly."""

    def test_two_batteries_both_tracked(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(4, 5000.0)))
        bat1 = SimpleBattery("bat1", capacity_wh=5_000, initial_soc_pct=0.0,
                              prediction_hours=4)
        bat2 = SimpleBattery("bat2", capacity_wh=5_000, initial_soc_pct=0.0,
                              prediction_hours=4)
        reg.register(bat1)
        reg.register(bat2)

        assembler = GenomeAssembler(reg)
        # Both batteries charge (genome=2 for all hours of both)
        genome = np.full(assembler.total_size, 2.0)
        assembler.dispatch(genome, reg)

        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(hours=4, load=100.0))

        assert "bat1" in result.all_device_ids()
        assert "bat2" in result.all_device_ids()

    def test_two_pv_arrays_generation_summed(self):
        """Two PV arrays each generating 500 Wh → 1000 Wh total AC supply."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.full(4, 500.0)))
        reg.register(SimplePV("pv2", np.full(4, 500.0)))

        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(hours=4, load=0.0))

        for hour in result.hours:
            assert hour.feedin_wh == pytest.approx(1000.0)

    def test_result_contains_all_device_ids(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv1", np.zeros(4)))
        bat1 = SimpleBattery("bat1", prediction_hours=4)
        bat2 = SimpleBattery("bat2", prediction_hours=4)
        reg.register(bat1)
        reg.register(bat2)

        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(hours=4))

        device_ids = result.all_device_ids()
        assert "pv1" in device_ids
        assert "bat1" in device_ids
        assert "bat2" in device_ids


class TestEngineReset:
    """Engine resets device state between runs correctly."""

    def test_second_run_starts_from_initial_soc(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.zeros(4)))
        bat = SimpleBattery("bat", capacity_wh=10_000, initial_soc_pct=50.0,
                             prediction_hours=4)
        reg.register(bat)

        # Schedule discharge
        assembler = GenomeAssembler(reg)
        genome = np.ones(4)
        assembler.dispatch(genome, reg)

        engine = EnergySimulationEngine(reg)
        result1 = engine.simulate(make_input(hours=4, load=500.0))
        soc_end_run1 = result1.soc_per_hour("bat")[-1]

        # Re-dispatch same genome and run again
        assembler.dispatch(genome, reg)
        result2 = engine.simulate(make_input(hours=4, load=500.0))
        soc_start_run2 = result2.soc_per_hour("bat")[0]

        # Second run must start at initial SoC, not where run 1 ended
        assert soc_start_run2 == pytest.approx(50.0)

    def test_results_identical_for_same_genome(self):
        """Deterministic: same genome → same result across two runs."""
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(6, 1000.0)))
        bat = SimpleBattery("bat", capacity_wh=5_000, initial_soc_pct=30.0,
                             prediction_hours=6)
        reg.register(bat)

        assembler = GenomeAssembler(reg)
        genome = np.array([1.0, 1.0, 2.0, 2.0, 0.0, 0.0])

        engine = EnergySimulationEngine(reg)

        assembler.dispatch(genome, reg)
        r1 = engine.simulate(make_input(hours=6, load=300.0))

        assembler.dispatch(genome, reg)
        r2 = engine.simulate(make_input(hours=6, load=300.0))

        assert r1.net_balance_eur == pytest.approx(r2.net_balance_eur)
        assert r1.total_losses_wh == pytest.approx(r2.total_losses_wh)


class TestEngineHourRange:
    """Engine correctly simulates only the requested hour range."""

    def test_result_has_correct_number_of_hours(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(48, 1000.0)))
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(hours=12))
        assert len(result.hours) == 12

    def test_result_hour_indices_are_correct(self):
        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(48, 1000.0)))
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(hours=6, start=6))
        hour_indices = [h.hour for h in result.hours]
        assert hour_indices == [6, 7, 8, 9, 10, 11]

    def test_empty_registry_runs_without_error(self):
        reg = DeviceRegistry()
        engine = EnergySimulationEngine(reg)
        result = engine.simulate(make_input(hours=4, load=0.0))
        assert len(result.hours) == 4
        for hour in result.hours:
            assert hour.feedin_wh == pytest.approx(0.0)
            assert hour.grid_import_wh == pytest.approx(0.0)


class TestEngineCustomArbitrator:
    """Engine respects injected arbitrator."""

    def test_custom_arbitrator_is_called(self):
        from akkudoktoreos.simulation.genetic2.arbitrator import ArbitratorBase
        from akkudoktoreos.simulation.genetic2.flows import ResourceGrant

        call_count = 0

        class CountingArbitrator(ArbitratorBase):
            def arbitrate(self, requests):
                nonlocal call_count
                call_count += 1
                # Delegate to PriorityArbitrator for correct grants
                return PriorityArbitrator().arbitrate(requests)

        reg = DeviceRegistry()
        reg.register(SimplePV("pv", np.full(4, 1000.0)))
        engine = EnergySimulationEngine(reg, arbitrator=CountingArbitrator())
        engine.simulate(make_input(hours=4))

        assert call_count == 4  # once per hour
