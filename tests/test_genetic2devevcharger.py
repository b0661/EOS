"""Tests for EVChargerDevice and EVChargerParam.

Test scope
----------
Unit tests for every significant code path in
``akkudoktoreos.devices.genetic2.evcharger``:

- ``EVChargerParam`` construction and validation.
- ``EVChargerCommonSettings`` construction, validation, and
  ``to_genetic2_param()`` conversion.
- ``EVChargerDevice`` full simulation lifecycle:
  - ``setup_run`` with measurement / time-window / fallback connection state.
  - ``genome_requirements`` slice shape and bounds.
  - ``create_batch_state`` initial SoC population.
  - ``apply_genome_batch`` repair, physics, Lamarckian write-back, SoC advance.
  - ``build_device_request`` energy shape and values.
  - ``apply_device_grant`` write-back.
  - ``compute_cost`` SoC shortfall penalty and LCOS.
  - ``extract_instructions`` mode labels and factors.
- Integration test: end-to-end arbitration with a grid connection device.
- Settings integration: ``EVChargerCommonSettings`` -> ``to_genetic2_param()``.

Test helpers
------------
``FakeContext`` replaces ``SimulationContext`` so that the tests run without
a live EOS configuration, measurement store, or prediction store.
``FakeContext.resolve_prediction`` returns a flat array of a configurable value;
``FakeContext.resolve_measurement`` returns a configurable scalar or ``None``;
``FakeContext.resolve_config_cycle_time_windows`` raises ``KeyError`` by default
(time-window path not configured) unless overridden per-test.
"""

from __future__ import annotations

from typing import Optional, cast
from unittest.mock import MagicMock

import numpy as np
import pendulum
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from akkudoktoreos.devices.genetic2.evcharger import (
    EVChargerBatchState,
    EVChargerDevice,
    EVChargerParam,
)
from akkudoktoreos.devices.settings.evchargersettings import EVChargerCommonSettings
from akkudoktoreos.devices.settings.devicebasesettings import PortConfig
from akkudoktoreos.devices.devicesabc import (
    EnergyCarrier,
    EnergyPort,
    PortDirection,
    InstructionContext,
)
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

HORIZON = 24  # one day at 1 h resolution
STEP_INTERVAL_SEC = 3600  # 1 hour
BATTERY_CAPACITY_WH = 60_000.0
MAX_CHARGE_POWER_W = 11_000.0
MIN_CHARGE_POWER_W = 1_380.0
HIGH_CURRENT_THRESHOLD_W = 3_680.0

# ---------------------------------------------------------------------------
# FakeContext — lightweight SimulationContext stub
# ---------------------------------------------------------------------------


class FakeContext:
    """Minimal stub for SimulationContext that avoids EOS singletons.

    Attributes
    ----------
    predictions : dict[str, np.ndarray]
        Keyed prediction arrays returned by ``resolve_prediction``.
        If a key is absent a flat array of zeros is returned.
    measurements : dict[str, float | None]
        Scalar values returned by ``resolve_measurement``.
        If a key is absent ``None`` is returned.
    time_windows : dict[str, tuple[list[int], np.ndarray]]
        Pre-built ``(cycle_indices, matrix)`` pairs returned by
        ``resolve_config_cycle_time_windows``.  If a key is absent
        a ``KeyError`` is raised (simulating "not configured").
    """

    def __init__(
        self,
        horizon: int = HORIZON,
        step_interval_sec: int = STEP_INTERVAL_SEC,
        predictions: dict[str, np.ndarray] | None = None,
        measurements: dict[str, float | None] | None = None,
        time_windows: dict[str, tuple[list[int], np.ndarray]] | None = None,
        start: pendulum.DateTime | None = None,
    ) -> None:
        self._horizon = horizon
        self._step_interval_sec = step_interval_sec
        self.predictions: dict[str, np.ndarray] = predictions or {}
        self.measurements: dict[str, float | None] = measurements or {}
        self.time_windows: dict[str, tuple[list[int], np.ndarray]] = time_windows or {}

        if start is None:
            start = pendulum.datetime(2024, 6, 1, 0, 0, 0, tz="UTC")
        self.step_times: tuple[pendulum.DateTime, ...] = tuple(
            start.add(seconds=i * step_interval_sec) for i in range(horizon)
        )
        self.step_interval: pendulum.Duration = pendulum.duration(
            seconds=step_interval_sec
        )
        self.horizon: int = horizon

    # Mimic SimulationContext interface
    def resolve_prediction(self, key: str) -> np.ndarray:
        if key in self.predictions:
            return self.predictions[key]
        return np.zeros(self._horizon, dtype=np.float64)

    def resolve_measurement(self, key: str) -> Optional[float]:
        if not key or not key.strip():
            return None
        return self.measurements.get(key)

    def resolve_config_cycle_time_windows(
        self, config_path: str
    ) -> tuple[list[int], np.ndarray]:
        if config_path in self.time_windows:
            return self.time_windows[config_path]
        raise KeyError(f"No time window configured for path '{config_path}'")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_port() -> tuple[EnergyPort, ...]:
    """One AC sink port on 'bus_ac'."""
    return (
        EnergyPort(
            port_id="p_ac",
            bus_id="bus_ac",
            direction=PortDirection.SINK,
        ),
    )


def _make_param(
    *,
    min_charge_power_w: float = MIN_CHARGE_POWER_W,
    max_charge_power_w: float = MAX_CHARGE_POWER_W,
    high_current_threshold_w: float = HIGH_CURRENT_THRESHOLD_W,
    deep_standby_power_w: float = 1.0,
    standby_power_w: float = 5.0,
    charger_efficiency: float = 0.95,
    ev_charger_efficiency_low: float = 0.77,
    ev_charger_efficiency_high: float = 0.90,
    ev_battery_efficiency: float = 0.96,
    hold_time_efficiency: float = 0.98,
    control_efficiency: float = 0.97,
    ev_battery_capacity_wh: float = BATTERY_CAPACITY_WH,
    ev_min_soc_factor: float = 0.1,
    ev_max_soc_factor: float = 0.9,
    ev_target_soc_factor: float = 0.8,
    ev_initial_soc_factor_key: str = "",
    ev_connected_measurement_key: str = "",
    connection_time_window_key: str | None = None,
    import_price_amt_kwh_key: str | None = None,
    export_price_amt_kwh_key: str | None = None,
    ev_lcos_amt_kwh: float = 0.0,
) -> EVChargerParam:
    return EVChargerParam(
        device_id="ev_charger1",
        ports=_make_port(),
        min_charge_power_w=min_charge_power_w,
        max_charge_power_w=max_charge_power_w,
        high_current_threshold_w=high_current_threshold_w,
        deep_standby_power_w=deep_standby_power_w,
        standby_power_w=standby_power_w,
        charger_efficiency=charger_efficiency,
        ev_charger_efficiency_low=ev_charger_efficiency_low,
        ev_charger_efficiency_high=ev_charger_efficiency_high,
        ev_battery_efficiency=ev_battery_efficiency,
        hold_time_efficiency=hold_time_efficiency,
        control_efficiency=control_efficiency,
        ev_battery_capacity_wh=ev_battery_capacity_wh,
        ev_min_soc_factor=ev_min_soc_factor,
        ev_max_soc_factor=ev_max_soc_factor,
        ev_target_soc_factor=ev_target_soc_factor,
        ev_initial_soc_factor_key=ev_initial_soc_factor_key,
        ev_connected_measurement_key=ev_connected_measurement_key,
        connection_time_window_key=connection_time_window_key,
        import_price_amt_kwh_key=import_price_amt_kwh_key,
        export_price_amt_kwh_key=export_price_amt_kwh_key,
        ev_lcos_amt_kwh=ev_lcos_amt_kwh,
    )


def _make_device(param: EVChargerParam | None = None) -> EVChargerDevice:
    if param is None:
        param = _make_param()
    return EVChargerDevice(param=param, device_index=0, port_index=0)


def _setup_device(
    device: EVChargerDevice,
    ctx: FakeContext | None = None,
) -> FakeContext:
    if ctx is None:
        ctx = FakeContext()
    device.setup_run(cast(SimulationContext, ctx))
    return ctx


# ===========================================================================
# 1. EVChargerParam — construction and validation
# ===========================================================================


class TestEVChargerParam:
    def test_valid_construction(self) -> None:
        p = _make_param()
        assert p.device_id == "ev_charger1"
        assert p.max_charge_power_w == MAX_CHARGE_POWER_W
        assert p.ev_min_soc_wh == pytest.approx(0.1 * BATTERY_CAPACITY_WH)
        assert p.ev_max_soc_wh == pytest.approx(0.9 * BATTERY_CAPACITY_WH)
        assert p.ev_target_soc_wh == pytest.approx(0.8 * BATTERY_CAPACITY_WH)

    def test_min_gt_max_power_raises(self) -> None:
        with pytest.raises(ValueError, match="min_charge_power_w"):
            _make_param(min_charge_power_w=12_000.0, max_charge_power_w=11_000.0)

    def test_max_power_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_charge_power_w"):
            _make_param(max_charge_power_w=0.0)

    def test_efficiency_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="charger_efficiency"):
            _make_param(charger_efficiency=0.0)
        with pytest.raises(ValueError, match="charger_efficiency"):
            _make_param(charger_efficiency=1.1)

    def test_soc_factors_inverted_raises(self) -> None:
        with pytest.raises(ValueError, match="SoC factors"):
            _make_param(ev_min_soc_factor=0.9, ev_max_soc_factor=0.1)

    def test_target_soc_outside_window_raises(self) -> None:
        with pytest.raises(ValueError, match="ev_target_soc_factor"):
            _make_param(
                ev_min_soc_factor=0.1,
                ev_max_soc_factor=0.9,
                ev_target_soc_factor=0.95,
            )

    def test_negative_standby_power_raises(self) -> None:
        with pytest.raises(ValueError, match="standby_power_w"):
            _make_param(standby_power_w=-1.0)

    def test_combined_max_efficiency(self) -> None:
        p = _make_param()
        # combined_max_efficiency is the DC-cable-to-battery efficiency at
        # high current. charger_efficiency is excluded — it is captured only
        # in the AC wall-draw formula (ac_w = charge_w / charger_eff + standby).
        expected = (
            p.ev_charger_efficiency_high
            * p.ev_battery_efficiency
            * p.hold_time_efficiency
            * p.control_efficiency
        )
        assert p.combined_max_efficiency == pytest.approx(expected)

    def test_lcos_default_zero(self) -> None:
        p = _make_param()
        assert p.ev_lcos_amt_kwh == 0.0


# ===========================================================================
# 2. EVChargerCommonSettings — settings and to_genetic2_param conversion
# ===========================================================================


class TestEVChargerCommonSettings:
    def _make_settings(self, **overrides) -> EVChargerCommonSettings:
        defaults = dict(
            device_id="ev_charger1",
            ports=[
                PortConfig(port_id="p_ac", bus_id="bus_ac", direction="sink")
            ],
        )
        defaults.update(overrides)
        return EVChargerCommonSettings(**defaults)

    def test_default_construction(self) -> None:
        s = self._make_settings()
        assert s.max_charge_power_w == 11_000.0
        assert s.ev_battery_capacity_wh == 60_000.0

    def test_soc_validation_min_ge_max_raises(self) -> None:
        with pytest.raises(ValueError, match="ev_min_soc_factor"):
            self._make_settings(ev_min_soc_factor=0.9, ev_max_soc_factor=0.5)

    def test_min_charge_gt_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_charge_power_w"):
            self._make_settings(
                min_charge_power_w=15_000.0, max_charge_power_w=11_000.0
            )

    def test_to_genetic2_param_types(self) -> None:
        s = self._make_settings()
        p = s.to_genetic2_param()
        assert isinstance(p, EVChargerParam)
        assert p.device_id == "ev_charger1"
        assert p.max_charge_power_w == s.max_charge_power_w
        assert p.ev_battery_capacity_wh == s.ev_battery_capacity_wh
        assert p.charger_efficiency == s.charger_efficiency

    def test_to_genetic2_param_price_keys_hardcoded(self) -> None:
        s = self._make_settings()
        p = s.to_genetic2_param()
        assert p.import_price_amt_kwh_key == "elecprice_marketprice_amt_kwh"
        assert p.export_price_amt_kwh_key == "feed_in_tariff_amt_kwh"

    def test_measurement_keys_empty_when_no_keys(self) -> None:
        s = self._make_settings(
            ev_initial_soc_factor_key="", ev_connected_measurement_key=""
        )
        assert s.measurement_keys == []

    def test_measurement_keys_populated(self) -> None:
        s = self._make_settings(
            ev_initial_soc_factor_key="ev1_soc",
            ev_connected_measurement_key="ev1_connected",
        )
        assert "ev1_soc" in s.measurement_keys
        assert "ev1_connected" in s.measurement_keys


# ===========================================================================
# 3. EVChargerDevice.setup_run — connection state resolution
# ===========================================================================


class TestSetupRun:
    def test_fallback_always_connected(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        assert dev._connected is not None
        assert np.all(dev._connected == 1.0)

    def test_measurement_connected(self) -> None:
        dev = _make_device(_make_param(ev_connected_measurement_key="ev_plug"))
        ctx = FakeContext(measurements={"ev_plug": 1.0})
        _setup_device(dev, ctx)
        assert np.all(dev._connected == 1.0)

    def test_measurement_absent(self) -> None:
        dev = _make_device(_make_param(ev_connected_measurement_key="ev_plug"))
        ctx = FakeContext(measurements={"ev_plug": 0.0})
        _setup_device(dev, ctx)
        assert np.all(dev._connected == 0.0)

    def test_measurement_none_fallback_to_always_connected(self) -> None:
        """When measurement returns None the device falls back to always-connected."""
        dev = _make_device(_make_param(ev_connected_measurement_key="ev_plug"))
        ctx = FakeContext(measurements={})  # key absent → None
        _setup_device(dev, ctx)
        assert np.all(dev._connected == 1.0)

    def test_time_window_sets_connected_steps(self) -> None:
        """A time window restricting to first 8 hours should mark steps 0-7 connected."""
        param = _make_param(connection_time_window_key="devices/ev_chargers/ev1/windows")
        dev = _make_device(param)
        # Manually build a connected array: first 8 steps active
        matrix = np.zeros((1, HORIZON), dtype=np.float64)
        matrix[0, :8] = 1.0
        ctx = FakeContext(
            time_windows={"devices/ev_chargers/ev1/windows": ([0], matrix)}
        )
        _setup_device(dev, ctx)
        assert dev._connected is not None
        assert np.all(dev._connected[:8] == 1.0)
        assert np.all(dev._connected[8:] == 0.0)

    def test_measurement_overrides_time_window(self) -> None:
        """When measurement says absent (0.0), it overrides a time-window saying present."""
        param = _make_param(
            ev_connected_measurement_key="ev_plug",
            connection_time_window_key="devices/ev_chargers/ev1/windows",
        )
        dev = _make_device(param)
        matrix = np.ones((1, HORIZON), dtype=np.float64)  # window says always present
        ctx = FakeContext(
            measurements={"ev_plug": 0.0},
            time_windows={"devices/ev_chargers/ev1/windows": ([0], matrix)},
        )
        _setup_device(dev, ctx)
        # Measurement (absent) wins
        assert np.all(dev._connected == 0.0)

    def test_initial_soc_from_measurement(self) -> None:
        param = _make_param(ev_initial_soc_factor_key="ev1_soc")
        dev = _make_device(param)
        ctx = FakeContext(measurements={"ev1_soc": 0.5})
        _setup_device(dev, ctx)
        assert dev._initial_soc_wh == pytest.approx(0.5 * BATTERY_CAPACITY_WH)

    def test_initial_soc_default_when_no_measurement(self) -> None:
        """No measurement key → defaults to ev_min_soc_factor."""
        dev = _make_device(_make_param(ev_initial_soc_factor_key=""))
        _setup_device(dev)
        assert dev._initial_soc_wh == pytest.approx(0.1 * BATTERY_CAPACITY_WH)

    def test_initial_soc_out_of_range_raises(self) -> None:
        param = _make_param(ev_initial_soc_factor_key="ev1_soc")
        dev = _make_device(param)
        ctx = FakeContext(measurements={"ev1_soc": 1.5})
        with pytest.raises(ValueError, match="initial SoC factor"):
            _setup_device(dev, ctx)

    def test_import_price_resolved(self) -> None:
        param = _make_param(import_price_amt_kwh_key="elecprice_marketprice_amt_kwh")
        dev = _make_device(param)
        prices = np.full(HORIZON, 0.30, dtype=np.float64)
        ctx = FakeContext(predictions={"elecprice_marketprice_amt_kwh": prices})
        _setup_device(dev, ctx)
        assert dev._import_price_per_kwh is not None
        np.testing.assert_array_almost_equal(dev._import_price_per_kwh, prices)

    def test_setup_required_before_genome_requirements(self) -> None:
        dev = _make_device()
        with pytest.raises(RuntimeError, match="setup_run"):
            dev.genome_requirements()


# ===========================================================================
# 4. genome_requirements
# ===========================================================================


class TestGenomeRequirements:
    def test_shape_and_bounds(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        slc = dev.genome_requirements()
        assert slc.size == HORIZON
        assert slc.lower_bound is not None
        assert slc.upper_bound is not None
        np.testing.assert_array_equal(slc.lower_bound, np.zeros(HORIZON))
        np.testing.assert_array_equal(slc.upper_bound, np.ones(HORIZON))


# ===========================================================================
# 5. create_batch_state
# ===========================================================================


class TestCreateBatchState:
    def test_initial_soc_broadcast(self) -> None:
        param = _make_param(ev_initial_soc_factor_key="ev1_soc")
        dev = _make_device(param)
        ctx = FakeContext(measurements={"ev1_soc": 0.4})
        _setup_device(dev, ctx)
        pop = 10
        state = dev.create_batch_state(pop, HORIZON)
        expected_soc = 0.4 * BATTERY_CAPACITY_WH
        # All individuals start at the same SoC
        np.testing.assert_array_almost_equal(
            state.soc_wh[:, 0], np.full(pop, expected_soc)
        )

    def test_state_shapes(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        pop, hor = 5, HORIZON
        state = dev.create_batch_state(pop, hor)
        assert state.charge_factors.shape == (pop, hor)
        assert state.soc_wh.shape == (pop, hor)
        assert state.ac_power_w.shape == (pop, hor)
        assert state.connected.shape == (hor,)
        assert state.population_size == pop
        assert state.horizon == hor
        assert len(state.step_times) == hor

    def test_requires_setup_run(self) -> None:
        dev = _make_device()
        with pytest.raises(RuntimeError, match="setup_run"):
            dev.create_batch_state(4, HORIZON)


# ===========================================================================
# 6. apply_genome_batch — repair, physics, SoC advance
# ===========================================================================


class TestApplyGenomeBatch:
    """Tests for the core simulation loop."""

    def _run(
        self,
        genome: np.ndarray,
        connected: np.ndarray | None = None,
        initial_soc_factor: float = 0.0,
        param_overrides: dict | None = None,
    ) -> tuple[EVChargerDevice, EVChargerBatchState, np.ndarray]:
        """Helper: set up device, run apply_genome_batch, return device/state/genome."""
        kw = {} if param_overrides is None else param_overrides
        param = _make_param(**kw)
        dev = _make_device(param)
        soc_key = "ev1_soc" if not kw.get("ev_initial_soc_factor_key") else kw["ev_initial_soc_factor_key"]
        ctx = FakeContext(
            measurements={"ev1_soc": initial_soc_factor},
            horizon=genome.shape[1],
        )
        # Temporarily override _make_param to use the soc key
        param2 = _make_param(
            ev_initial_soc_factor_key="ev1_soc",
            **{k: v for k, v in kw.items() if k != "ev_initial_soc_factor_key"},
        )
        dev = _make_device(param2)
        _setup_device(dev, ctx)
        if connected is not None:
            dev._connected = connected.astype(np.float64)
        state = dev.create_batch_state(genome.shape[0], genome.shape[1])
        repaired = dev.apply_genome_batch(state, genome.copy())
        return dev, state, repaired

    # ---- Absent EV --------------------------------------------------------

    def test_ev_absent_forces_zero_charge(self) -> None:
        """When EV is absent all steps, charge_factor must be 0 after repair."""
        pop, hor = 4, 6
        genome = np.ones((pop, hor), dtype=np.float64)  # request full charge
        connected = np.zeros(hor, dtype=np.float64)  # EV absent
        dev, state, repaired = self._run(genome, connected=connected)
        np.testing.assert_array_equal(repaired, np.zeros((pop, hor)))
        np.testing.assert_array_equal(state.charge_factors, np.zeros((pop, hor)))

    def test_ev_absent_deep_standby_power(self) -> None:
        """AC draw when EV absent == deep_standby_power_w."""
        pop, hor = 2, 4
        genome = np.ones((pop, hor))
        connected = np.zeros(hor)
        dev, state, _ = self._run(
            genome,
            connected=connected,
            param_overrides={"deep_standby_power_w": 2.5},
        )
        np.testing.assert_array_almost_equal(
            state.ac_power_w, np.full((pop, hor), 2.5)
        )

    # ---- Connected EV — idle ----------------------------------------------

    def test_ev_connected_idle_standby_power(self) -> None:
        """Zero genome → standby power when EV is connected."""
        pop, hor = 3, 4
        genome = np.zeros((pop, hor))
        dev, state, _ = self._run(genome, connected=np.ones(hor))
        np.testing.assert_array_almost_equal(
            state.ac_power_w, np.full((pop, hor), 5.0)
        )

    # ---- Deadband repair --------------------------------------------------

    def test_below_deadband_zeroed(self) -> None:
        """charge_factor encoding power < min_charge_power_w must be zeroed."""
        pop, hor = 2, 4
        # Encode a power just below the minimum (1 W below)
        below_factor = (MIN_CHARGE_POWER_W - 1.0) / MAX_CHARGE_POWER_W
        genome = np.full((pop, hor), below_factor)
        dev, state, repaired = self._run(genome, connected=np.ones(hor))
        np.testing.assert_array_equal(repaired, np.zeros((pop, hor)))

    def test_above_deadband_passes(self) -> None:
        """Charge request well above min should not be zeroed by deadband."""
        pop, hor = 2, 4
        factor = (MIN_CHARGE_POWER_W * 2.0) / MAX_CHARGE_POWER_W
        genome = np.full((pop, hor), factor)
        dev, state, repaired = self._run(genome, connected=np.ones(hor))
        assert np.all(repaired > 0)

    # ---- SoC cap ----------------------------------------------------------

    def test_soc_cap_prevents_overcharge(self) -> None:
        """Battery must never exceed ev_max_soc_wh after simulation."""
        pop, hor = 4, 24
        genome = np.ones((pop, hor))  # max charge every step
        dev, state, _ = self._run(genome, connected=np.ones(hor), initial_soc_factor=0.0)
        max_soc_wh = 0.9 * BATTERY_CAPACITY_WH
        assert np.all(state.soc_wh <= max_soc_wh + 1.0)  # 1 Wh tolerance

    def test_soc_increases_when_charging(self) -> None:
        """SoC at step t+1 must be >= SoC at step t when EV is charging."""
        pop, hor = 2, 12
        genome = np.full((pop, hor), 0.5)
        dev, state, _ = self._run(genome, connected=np.ones(hor), initial_soc_factor=0.0)
        # SoC is non-decreasing (charging only device)
        for t in range(1, hor):
            assert np.all(state.soc_wh[:, t] >= state.soc_wh[:, t - 1] - 1e-6)

    def test_soc_unchanged_when_absent(self) -> None:
        """SoC must not change when EV is absent."""
        pop, hor = 2, 4
        genome = np.ones((pop, hor))
        connected = np.zeros(hor)
        dev, state, _ = self._run(
            genome, connected=connected, initial_soc_factor=0.3
        )
        expected_soc = 0.3 * BATTERY_CAPACITY_WH
        np.testing.assert_array_almost_equal(
            state.soc_wh, np.full((pop, hor), expected_soc)
        )

    # ---- Efficiency chain -------------------------------------------------

    def test_ac_power_reflects_charger_efficiency(self) -> None:
        """AC wall draw = charge_w / charger_eff + standby.

        charge_w is the DC cable power (gene × max_charge_power_w).
        The wall-box charger_efficiency is the only AC→DC conversion stage.
        Hold/control losses are DC-side and appear only in stored energy.
        """
        pop, hor = 1, 1
        # Use a charge factor that is well above high_current_threshold
        charge_w = MAX_CHARGE_POWER_W  # 11 000 W
        factor = charge_w / MAX_CHARGE_POWER_W
        genome = np.full((pop, hor), factor)
        dev, state, _ = self._run(genome, connected=np.ones(hor), initial_soc_factor=0.0)
        p = dev.param
        expected_draw = (
            charge_w / p.charger_efficiency
            + p.standby_power_w
        )
        np.testing.assert_allclose(state.ac_power_w[0, 0], expected_draw, rtol=1e-3)

    def test_low_current_efficiency_used_below_threshold(self) -> None:
        """When charge_w < threshold, low-current efficiency must apply.

        charge_w is DC cable power. Stored = charge_w × low_eff × bat_eff
        × hold_eff × control_eff. charger_efficiency is excluded (AC draw only).
        initial_soc_factor=0.0 → initial SoC = 0 Wh.
        """
        pop, hor = 1, 1
        # Request just below high_current_threshold
        charge_w = HIGH_CURRENT_THRESHOLD_W * 0.5  # low current
        factor = charge_w / MAX_CHARGE_POWER_W
        genome = np.full((pop, hor), factor)
        dev, state, _ = self._run(genome, connected=np.ones(hor), initial_soc_factor=0.0)
        p = dev.param
        step_h = STEP_INTERVAL_SEC / 3600.0
        expected_stored = (
            charge_w
            * p.ev_charger_efficiency_low
            * p.ev_battery_efficiency
            * p.hold_time_efficiency
            * p.control_efficiency
            * step_h
        )
        # initial_soc_factor=0.0 → initial SoC = 0 Wh (not 0.1*cap)
        actual_stored = state.soc_wh[0, 0] - 0.0 * BATTERY_CAPACITY_WH
        np.testing.assert_allclose(actual_stored, expected_stored, rtol=1e-3)

    def test_high_current_efficiency_used_above_threshold(self) -> None:
        """When charge_w >= threshold, high-current efficiency must apply.

        charge_w is DC cable power. Stored = charge_w × high_eff × bat_eff
        × hold_eff × control_eff. charger_efficiency is excluded (AC draw only).
        initial_soc_factor=0.0 → initial SoC = 0 Wh.
        """
        pop, hor = 1, 1
        charge_w = MAX_CHARGE_POWER_W  # well above threshold
        factor = charge_w / MAX_CHARGE_POWER_W
        genome = np.full((pop, hor), factor)
        dev, state, _ = self._run(genome, connected=np.ones(hor), initial_soc_factor=0.0)
        p = dev.param
        step_h = STEP_INTERVAL_SEC / 3600.0
        expected_stored = (
            charge_w
            * p.ev_charger_efficiency_high
            * p.ev_battery_efficiency
            * p.hold_time_efficiency
            * p.control_efficiency
            * step_h
        )
        # initial_soc_factor=0.0 → initial SoC = 0 Wh (not 0.1*cap)
        actual_stored = state.soc_wh[0, 0] - 0.0 * BATTERY_CAPACITY_WH
        np.testing.assert_allclose(actual_stored, expected_stored, rtol=1e-3)

    # ---- Lamarckian write-back --------------------------------------------

    def test_lamarckian_writeback_zeroes_below_deadband(self) -> None:
        below_factor = (MIN_CHARGE_POWER_W - 1.0) / MAX_CHARGE_POWER_W
        pop, hor = 3, 5
        genome_in = np.full((pop, hor), below_factor)
        dev, state, repaired = self._run(genome_in, connected=np.ones(hor))
        # Original object must be modified in-place
        np.testing.assert_array_equal(repaired, np.zeros((pop, hor)))

    def test_lamarckian_writeback_is_inplace(self) -> None:
        """apply_genome_batch must return the same array object it received."""
        pop, hor = 2, 4
        param = _make_param(ev_initial_soc_factor_key="ev1_soc")
        dev = _make_device(param)
        ctx = FakeContext(measurements={"ev1_soc": 0.0})
        _setup_device(dev, ctx)
        state = dev.create_batch_state(pop, hor)
        genome = np.full((pop, hor), 0.5)
        genome_id = id(genome)
        returned = dev.apply_genome_batch(state, genome)
        assert id(returned) == genome_id


# ===========================================================================
# 7. build_device_request
# ===========================================================================


class TestBuildDeviceRequest:
    def test_request_energy_shape(self) -> None:
        pop, hor = 5, HORIZON
        dev = _make_device()
        _setup_device(dev)
        state = dev.create_batch_state(pop, hor)
        genome = np.full((pop, hor), 0.5)
        dev.apply_genome_batch(state, genome.copy())
        req = dev.build_device_request(state)
        assert req.port_requests[0].energy_wh.shape == (pop, hor)

    def test_request_energy_nonnegative(self) -> None:
        """EV charger is a pure sink — energy must always be ≥ 0."""
        pop, hor = 3, HORIZON
        dev = _make_device()
        _setup_device(dev)
        state = dev.create_batch_state(pop, hor)
        genome = np.random.default_rng(42).uniform(0, 1, (pop, hor))
        dev.apply_genome_batch(state, genome.copy())
        req = dev.build_device_request(state)
        assert np.all(req.port_requests[0].energy_wh >= 0.0)

    def test_min_energy_is_zero(self) -> None:
        """Minimum accepted energy is 0 (charger can be throttled to zero)."""
        pop, hor = 2, 4
        dev = _make_device()
        _setup_device(dev)
        state = dev.create_batch_state(pop, hor)
        genome = np.ones((pop, hor))
        dev.apply_genome_batch(state, genome.copy())
        req = dev.build_device_request(state)
        np.testing.assert_array_equal(req.port_requests[0].min_energy_wh, 0.0)

    def test_is_slack_false(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        state = dev.create_batch_state(2, 4)
        dev.apply_genome_batch(state, np.ones((2, 4)))
        req = dev.build_device_request(state)
        assert req.port_requests[0].is_slack is False


# ===========================================================================
# 8. apply_device_grant
# ===========================================================================


class TestApplyDeviceGrant:
    def test_grant_updates_ac_power(self) -> None:
        from akkudoktoreos.simulation.genetic2.arbitrator import DeviceGrant, PortGrant

        pop, hor = 3, HORIZON
        dev = _make_device()
        _setup_device(dev)
        state = dev.create_batch_state(pop, hor)
        dev.apply_genome_batch(state, np.full((pop, hor), 0.5))

        granted_wh = np.full((pop, hor), 5000.0 * (STEP_INTERVAL_SEC / 3600.0))
        grant = DeviceGrant(
            device_index=0,
            port_grants=(PortGrant(port_index=0, granted_wh=granted_wh),),
        )
        dev.apply_device_grant(state, grant)
        expected_power = 5000.0
        np.testing.assert_array_almost_equal(state.ac_power_w, expected_power)


# ===========================================================================
# 9. compute_cost
# ===========================================================================


class TestComputeCost:
    def _charged_state(
        self,
        *,
        initial_soc_factor: float = 0.0,
        charge_factor: float = 1.0,
        import_price: float = 0.30,
        lcos: float = 0.0,
        target_soc_factor: float = 0.8,
    ) -> tuple[EVChargerDevice, EVChargerBatchState]:
        param = _make_param(
            ev_initial_soc_factor_key="ev1_soc",
            import_price_amt_kwh_key="elecprice_marketprice_amt_kwh",
            ev_lcos_amt_kwh=lcos,
            ev_target_soc_factor=target_soc_factor,
        )
        dev = _make_device(param)
        prices = np.full(HORIZON, import_price)
        ctx = FakeContext(
            measurements={"ev1_soc": initial_soc_factor},
            predictions={"elecprice_marketprice_amt_kwh": prices},
        )
        _setup_device(dev, ctx)
        pop = 4
        state = dev.create_batch_state(pop, HORIZON)
        genome = np.full((pop, HORIZON), charge_factor)
        dev.apply_genome_batch(state, genome.copy())
        return dev, state

    def test_cost_shape(self) -> None:
        pop = 4
        dev, state = self._charged_state()
        cost = dev.compute_cost(state)
        assert cost.shape == (pop, 1)

    def test_zero_shortfall_zero_cost(self) -> None:
        """When battery fully charged above target, SoC shortfall cost = 0."""
        # Start at high SoC, keep charging → will hit max cap, no shortfall
        dev, state = self._charged_state(
            initial_soc_factor=0.85,
            charge_factor=1.0,
            target_soc_factor=0.8,
        )
        cost = dev.compute_cost(state)
        # Terminal SoC should be at or above target (0.8 × 60 000 = 48 000 Wh)
        # No shortfall → cost should be 0
        assert np.all(cost >= 0.0)
        np.testing.assert_array_almost_equal(cost, 0.0, decimal=2)

    def test_shortfall_positive_cost(self) -> None:
        """When EV is absent the whole horizon, SoC stays at initial (below target)."""
        param = _make_param(
            ev_initial_soc_factor_key="ev1_soc",
            import_price_amt_kwh_key="elecprice_marketprice_amt_kwh",
            ev_target_soc_factor=0.8,
        )
        dev = _make_device(param)
        prices = np.full(HORIZON, 0.30)
        ctx = FakeContext(
            measurements={"ev1_soc": 0.0},
            predictions={"elecprice_marketprice_amt_kwh": prices},
        )
        _setup_device(dev, ctx)
        # Force EV absent for all steps
        dev._connected = np.zeros(HORIZON)
        pop = 2
        state = dev.create_batch_state(pop, HORIZON)
        genome = np.ones((pop, HORIZON))  # genome irrelevant when absent
        dev.apply_genome_batch(state, genome.copy())
        cost = dev.compute_cost(state)
        # SoC = 0, target = 0.8 × 60000 = 48000 Wh shortfall
        # cost = 48000 / 1000 * 0.30 = 14.4 currency
        assert np.all(cost > 0.0)
        np.testing.assert_allclose(cost.flatten(), 14.4, rtol=0.01)

    def test_shortfall_cost_uses_import_price(self) -> None:
        """Doubling import price should double the shortfall penalty."""
        _, state_low = self._charged_state(
            initial_soc_factor=0.0,
            charge_factor=0.0,
            import_price=0.15,
            target_soc_factor=0.8,
        )
        _, state_high = self._charged_state(
            initial_soc_factor=0.0,
            charge_factor=0.0,
            import_price=0.30,
            target_soc_factor=0.8,
        )
        # Get devices separately
        param = _make_param(
            ev_initial_soc_factor_key="ev1_soc",
            import_price_amt_kwh_key="elecprice_marketprice_amt_kwh",
        )
        dev_low = _make_device(param)
        ctx_low = FakeContext(
            measurements={"ev1_soc": 0.0},
            predictions={"elecprice_marketprice_amt_kwh": np.full(HORIZON, 0.15)},
        )
        _setup_device(dev_low, ctx_low)
        dev_low._connected = np.zeros(HORIZON)
        pop = 2
        state_l = dev_low.create_batch_state(pop, HORIZON)
        dev_low.apply_genome_batch(state_l, np.zeros((pop, HORIZON)))
        cost_low = dev_low.compute_cost(state_l)

        dev_high = _make_device(param)
        ctx_high = FakeContext(
            measurements={"ev1_soc": 0.0},
            predictions={"elecprice_marketprice_amt_kwh": np.full(HORIZON, 0.30)},
        )
        _setup_device(dev_high, ctx_high)
        dev_high._connected = np.zeros(HORIZON)
        state_h = dev_high.create_batch_state(pop, HORIZON)
        dev_high.apply_genome_batch(state_h, np.zeros((pop, HORIZON)))
        cost_high = dev_high.compute_cost(state_h)

        np.testing.assert_allclose(cost_high, cost_low * 2.0, rtol=1e-6)

    def test_lcos_adds_to_cost(self) -> None:
        """With LCOS > 0 the total cost must be higher than without."""
        _, state_no_lcos = self._charged_state(
            initial_soc_factor=0.0, charge_factor=0.5, lcos=0.0
        )
        dev_no_lcos, _ = self._charged_state(
            initial_soc_factor=0.0, charge_factor=0.5, lcos=0.0
        )
        cost_no_lcos = dev_no_lcos.compute_cost(state_no_lcos)

        dev_lcos, state_lcos = self._charged_state(
            initial_soc_factor=0.0, charge_factor=0.5, lcos=0.05
        )
        cost_lcos = dev_lcos.compute_cost(state_lcos)

        assert np.all(cost_lcos >= cost_no_lcos)

    def test_cost_nonnegative(self) -> None:
        dev, state = self._charged_state()
        cost = dev.compute_cost(state)
        assert np.all(cost >= 0.0)


# ===========================================================================
# 10. extract_instructions
# ===========================================================================


class TestExtractInstructions:
    def test_instruction_count(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        pop, hor = 2, HORIZON
        state = dev.create_batch_state(pop, hor)
        genome = np.full((pop, hor), 0.5)
        dev.apply_genome_batch(state, genome.copy())
        instrs = dev.extract_instructions(state, individual_index=0)
        assert len(instrs) == hor

    def test_charge_mode_when_charging(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        pop, hor = 1, 4
        state = dev.create_batch_state(pop, hor)
        genome = np.full((pop, hor), 0.5)
        dev.apply_genome_batch(state, genome.copy())
        instrs = dev.extract_instructions(state, individual_index=0)
        # All steps connected (fallback) and charging → mode must be CHARGE
        for instr in instrs:
            assert instr.operation_mode_id == "CHARGE"

    def test_standby_mode_when_idle_connected(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        pop, hor = 1, 4
        state = dev.create_batch_state(pop, hor)
        genome = np.zeros((pop, hor))  # no charging
        dev.apply_genome_batch(state, genome.copy())
        instrs = dev.extract_instructions(state, individual_index=0)
        for instr in instrs:
            assert instr.operation_mode_id == "STANDBY"

    def test_deep_standby_when_absent(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        dev._connected = np.zeros(HORIZON)
        pop, hor = 1, HORIZON
        state = dev.create_batch_state(pop, hor)
        genome = np.ones((pop, hor))
        dev.apply_genome_batch(state, genome.copy())
        instrs = dev.extract_instructions(state, individual_index=0)
        for instr in instrs:
            assert instr.operation_mode_id == "DEEP_STANDBY"

    def test_factor_reflects_charge_factor(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        pop, hor = 1, 4
        state = dev.create_batch_state(pop, hor)
        charge_factor = 0.6
        genome = np.full((pop, hor), charge_factor)
        dev.apply_genome_batch(state, genome.copy())
        instrs = dev.extract_instructions(state, individual_index=0)
        for instr in instrs:
            # factor should be close to charge_factor (deadband may zero some)
            assert 0.0 <= instr.operation_mode_factor <= 1.0

    def test_execution_time_matches_step_times(self) -> None:
        dev = _make_device()
        ctx = _setup_device(dev)
        pop, hor = 1, HORIZON
        state = dev.create_batch_state(pop, hor)
        dev.apply_genome_batch(state, np.zeros((pop, hor)))
        instrs = dev.extract_instructions(state, individual_index=0)
        for i, instr in enumerate(instrs):
            assert instr.execution_time == ctx.step_times[i]

    def test_resource_id_matches_device_id(self) -> None:
        dev = _make_device()
        _setup_device(dev)
        state = dev.create_batch_state(1, HORIZON)
        dev.apply_genome_batch(state, np.zeros((1, HORIZON)))
        instrs = dev.extract_instructions(state, individual_index=0)
        for instr in instrs:
            assert instr.resource_id == "ev_charger1"


# ===========================================================================
# 11. Mixed connection: partial horizon connected
# ===========================================================================


class TestPartialConnection:
    def test_partial_connection_respected(self) -> None:
        """Steps 0-11: connected (charging); steps 12-23: absent (deep standby)."""
        dev = _make_device()
        _setup_device(dev)
        connected = np.array(
            [1.0] * 12 + [0.0] * 12, dtype=np.float64
        )
        dev._connected = connected
        pop, hor = 2, HORIZON
        state = dev.create_batch_state(pop, hor)
        genome = np.full((pop, hor), 1.0)
        dev.apply_genome_batch(state, genome.copy())

        # Steps 12-23: charge factor must be 0
        np.testing.assert_array_equal(state.charge_factors[:, 12:], 0.0)

        # Steps 12-23: AC draw must equal deep_standby_power_w
        np.testing.assert_array_almost_equal(
            state.ac_power_w[:, 12:], dev.param.deep_standby_power_w
        )

        # Instructions: first 12 steps CHARGE (if above deadband), last 12 DEEP_STANDBY
        instrs = dev.extract_instructions(state, individual_index=0)
        for t in range(12):
            assert instrs[t].operation_mode_id in ("CHARGE", "STANDBY")
        for t in range(12, 24):
            assert instrs[t].operation_mode_id == "DEEP_STANDBY"


# ===========================================================================
# 12. Objective names
# ===========================================================================


class TestObjectiveNames:
    def test_objective_names(self) -> None:
        dev = _make_device()
        assert dev.objective_names == ["energy_cost_amt"]


# ===========================================================================
# 13. EVChargerParam — combined_efficiency helper
# ===========================================================================


class TestCombinedEfficiency:
    def test_low_current_path(self) -> None:
        p = _make_param()
        charge_w = np.array([1000.0])  # below 3680 W threshold
        eff = EVChargerDevice._combined_efficiency(charge_w, p)
        expected = (
            p.ev_charger_efficiency_low
            * p.ev_battery_efficiency
            * p.hold_time_efficiency
            * p.control_efficiency
        )
        np.testing.assert_allclose(eff, expected, rtol=1e-9)

    def test_high_current_path(self) -> None:
        p = _make_param()
        charge_w = np.array([11_000.0])  # above threshold
        eff = EVChargerDevice._combined_efficiency(charge_w, p)
        expected = (
            p.ev_charger_efficiency_high
            * p.ev_battery_efficiency
            * p.hold_time_efficiency
            * p.control_efficiency
        )
        np.testing.assert_allclose(eff, expected, rtol=1e-9)

    def test_at_threshold_uses_high(self) -> None:
        """Power exactly at threshold → high-current efficiency."""
        p = _make_param()
        charge_w = np.array([HIGH_CURRENT_THRESHOLD_W])
        eff = EVChargerDevice._combined_efficiency(charge_w, p)
        expected = (
            p.ev_charger_efficiency_high
            * p.ev_battery_efficiency
            * p.hold_time_efficiency
            * p.control_efficiency
        )
        np.testing.assert_allclose(eff, expected, rtol=1e-9)

    def test_vectorised_mixed(self) -> None:
        """Array with mixed values returns element-wise correct efficiencies."""
        p = _make_param()
        charge_w = np.array([500.0, 11_000.0, 2_000.0, 5_000.0])
        eff = EVChargerDevice._combined_efficiency(charge_w, p)
        low = (
            p.ev_charger_efficiency_low
            * p.ev_battery_efficiency
            * p.hold_time_efficiency
            * p.control_efficiency
        )
        high = (
            p.ev_charger_efficiency_high
            * p.ev_battery_efficiency
            * p.hold_time_efficiency
            * p.control_efficiency
        )
        expected = np.array([low, high, low, high])
        np.testing.assert_allclose(eff, expected, rtol=1e-9)


# ===========================================================================
# 14. Integration: EVChargerDevice + GridConnectionDevice arbitration
# ===========================================================================


class TestIntegrationWithGridConnection:
    """Smoke test: EV charger draws from the AC bus via the arbitrator."""

    def test_ev_charger_draws_from_grid(self) -> None:
        """After arbitration the grid should supply the EV charger's AC demand."""
        from akkudoktoreos.devices.genetic2.gridconnection import (
            GridConnectionDevice,
            GridConnectionParam,
        )
        from akkudoktoreos.devices.devicesabc import EnergyBus, EnergyCarrier
        from akkudoktoreos.simulation.genetic2.arbitrator import (
            BusTopology,
            VectorizedBusArbitrator,
        )

        pop, hor = 2, 6

        # --- Devices ------------------------------------------------------
        ev_port = EnergyPort(port_id="p_ac", bus_id="bus_ac", direction=PortDirection.SINK)
        ev_param = EVChargerParam(
            device_id="ev1",
            ports=(ev_port,),
            min_charge_power_w=0.0,  # disable deadband for simplicity
            max_charge_power_w=7_400.0,
            high_current_threshold_w=3_680.0,
            deep_standby_power_w=1.0,
            standby_power_w=5.0,
            charger_efficiency=1.0,
            ev_charger_efficiency_low=1.0,
            ev_charger_efficiency_high=1.0,
            ev_battery_efficiency=1.0,
            hold_time_efficiency=1.0,
            control_efficiency=1.0,
            ev_battery_capacity_wh=60_000.0,
            ev_min_soc_factor=0.0,
            ev_max_soc_factor=1.0,
            ev_target_soc_factor=0.8,
            ev_initial_soc_factor_key="",
            ev_connected_measurement_key="",
            connection_time_window_key=None,
            import_price_amt_kwh_key=None,
            export_price_amt_kwh_key=None,
        )
        ev_dev = EVChargerDevice(param=ev_param, device_index=0, port_index=0)

        grid_port = EnergyPort(
            port_id="p_ac", bus_id="bus_ac", direction=PortDirection.BIDIRECTIONAL
        )
        grid_param = GridConnectionParam(
            device_id="grid1",
            ports=(grid_port,),
            max_import_power_w=25_000.0,
            # max_export_power_w controls the grid's injection capacity onto the
            # local bus (= household import capacity in the arbitrator's terms).
            # Must be > 0 for the slack to cover the EV charger's deficit.
            max_export_power_w=25_000.0,
            import_cost_per_kwh=0.30,
            export_revenue_per_kwh=0.08,
        )
        grid_dev = GridConnectionDevice(param=grid_param, device_index=1, port_index=1)

        # --- Context ------------------------------------------------------
        ctx = FakeContext(horizon=hor)
        ev_dev.setup_run(cast(SimulationContext, ctx))
        grid_dev.setup_run(cast(SimulationContext, ctx))

        # --- Batch states ------------------------------------------------
        ev_state = ev_dev.create_batch_state(pop, hor)
        grid_state = grid_dev.create_batch_state(pop, hor)

        # --- Genomes (charge at half power) ------------------------------
        ev_genome = np.full((pop, hor), 0.5)
        ev_dev.apply_genome_batch(ev_state, ev_genome.copy())

        # --- Build requests ----------------------------------------------
        ev_req = ev_dev.build_device_request(ev_state)
        grid_req = grid_dev.build_device_request(grid_state)

        # --- Arbitrate ---------------------------------------------------
        topo = BusTopology(
            port_to_bus=np.array([0, 0], dtype=int),  # both on bus_ac
            num_buses=1,
        )
        arb = VectorizedBusArbitrator(topo, horizon=hor)
        grants = arb.arbitrate([ev_req, grid_req])

        ev_grant = next(g for g in grants if g.device_index == 0)
        grid_grant = next(g for g in grants if g.device_index == 1)

        ev_dev.apply_device_grant(ev_state, ev_grant)
        grid_dev.apply_device_grant(grid_state, grid_grant)

        # Grid import = slack injecting onto the bus → granted_wh is NEGATIVE
        # (sign convention: positive = consuming from bus, negative = injecting).
        assert np.all(grid_state.granted_wh < 0), (
            "Grid import shows as negative granted_wh (injecting onto bus). "
            f"Got: {grid_state.granted_wh}"
        )

        # The EV charger's AC draw = charge_factor * max_charge_w / charger_eff + standby.
        # All efficiencies set to 1.0 in this test, so ac_draw = charge_w + standby.
        charge_w_expected = ev_dev.param.max_charge_power_w * 0.5
        expected_ac_w = (
            charge_w_expected / ev_dev.param.charger_efficiency
            + ev_dev.param.standby_power_w
        )
        np.testing.assert_allclose(
            ev_state.ac_power_w,
            expected_ac_w,
            rtol=1e-3,
        )

        # Compute cost for both devices
        ev_cost = ev_dev.compute_cost(ev_state)
        grid_cost = grid_dev.compute_cost(grid_state)
        assert ev_cost.shape == (pop, 1)
        assert grid_cost.shape[0] == pop
