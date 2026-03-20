"""Tests for the grid connection slack device.

Module under test
-----------------
``akkudoktoreos.devices.genetic2.gridconnection``

Design under test
-----------------
The grid connection is the slack device on the AC bus.  It has no genome —
the GA makes no decisions about it.  Instead it offers the full import/export
window to the bus arbitrator every step and records whatever the arbitrator
grants.  It is the *only* device that contributes an electricity-cost fitness
objective.

Pricing modes
~~~~~~~~~~~~~
* **Flat rate**: ``import_cost_per_kwh`` / ``export_revenue_per_kwh`` from
  ``GridConnectionParam``, applied uniformly across all steps.
* **Time-of-use**: optional ``import_price_amt_kwh_key`` / ``export_price_amt_kwh_key``
  resolved from the ``SimulationContext`` during ``setup_run``.  When set they
  replace the flat rate *for that direction only*.

Sign convention (consistent with the rest of the framework)::

    positive granted_wh  → consuming from the AC bus (grid import)
    negative granted_wh  → injecting into the AC bus (grid export)

``_step_interval_sec`` is stored as a plain ``float`` (seconds), consistent
with every other device.  Division by 3600 in ``compute_cost`` and
``build_device_request`` converts it to hours.

Param construction
------------------
``GridConnectionParam`` inherits ``device_id`` and ``ports`` from
``DeviceParam``.  There are no separate ``port_id`` / ``bus_id`` kwargs on
the param itself — the port identity lives inside an ``EnergyPort`` object
passed as ``ports=(EnergyPort(...),)``.

``FakeContext`` exposes ``resolve_prediction(key)`` matching the real
``SimulationContext`` interface that ``GridConnectionDevice.setup_run`` calls.

Test strategy
-------------
All tests use a ``FakeContext`` stand-in.  No real simulation context,
arbitrator, or engine is required.  Cost arithmetic is verified by hand
against the exact formula in the source:

    cost = Σ_t ( import_wh[t] / 1000 * import_price[t]
               - export_wh[t] / 1000 * export_price[t] )

    peak_kw = max_t( max(granted_wh[t], 0) / step_h / 1000 )

Tolerances: rtol=1e-9 throughout; zero checks use atol=1e-9.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pendulum import Duration

from akkudoktoreos.devices.devicesabc import EnergyPort, PortDirection
from akkudoktoreos.devices.genetic2.gridconnection import (
    GridConnectionBatchState,
    GridConnectionDevice,
    GridConnectionParam,
)
from akkudoktoreos.simulation.genetic2.arbitrator import DeviceGrant, PortGrant
from akkudoktoreos.simulation.genetic2.simulation import SimulationContext
from akkudoktoreos.utils.datetimeutil import to_datetime

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STEP_INTERVAL = 3600.0           # 1-hour steps [s]
STEP_H = STEP_INTERVAL / 3600.0  # = 1.0 h
HORIZON = 6
POP = 4

IMPORT_KEY = "import_price"
EXPORT_KEY = "export_price"

MAX_IMPORT_W = 10_000.0   # 10 kW
MAX_EXPORT_W =  5_000.0   #  5 kW
IMPORT_COST  =      0.30  # EUR / kWh
EXPORT_REV   =      0.08  # EUR / kWh


# ---------------------------------------------------------------------------
# FakeContext
# ---------------------------------------------------------------------------

class FakeContext:
    """Minimal SimulationContext stand-in.

    ``context.step_interval`` is a ``pendulum.Duration`` — matching the real
    ``SimulationContext`` interface that ``GridConnectionDevice.setup_run``
    expects.  ``context.resolve_prediction(key)`` returns the pre-loaded price
    array for the given key, matching the method name called by ``setup_run``.

    Call sites pass this to setup_run() via cast(SimulationContext, ctx) so
    mypy is satisfied while the duck-typed runtime behaviour is preserved.
    """

    def __init__(
        self,
        horizon: int = HORIZON,
        step_interval_sec: float = STEP_INTERVAL,
        import_prices: np.ndarray | None = None,
        export_prices: np.ndarray | None = None,
    ) -> None:
        self.step_times = tuple(to_datetime(i * step_interval_sec) for i in range(horizon))
        self.step_interval = Duration(seconds=int(step_interval_sec))
        self.horizon = horizon
        self._prices: dict[str, np.ndarray] = {}
        if import_prices is not None:
            self._prices[IMPORT_KEY] = import_prices
        if export_prices is not None:
            self._prices[EXPORT_KEY] = export_prices

    def resolve_prediction(self, key: str) -> np.ndarray:
        """Return the price array for ``key``, matching the real SimulationContext API."""
        try:
            return self._prices[key].copy()
        except KeyError:
            raise KeyError(key)


def make_context(
    horizon: int = HORIZON,
    import_prices: np.ndarray | None = None,
    export_prices: np.ndarray | None = None,
) -> FakeContext:
    return FakeContext(horizon, STEP_INTERVAL, import_prices, export_prices)


# ---------------------------------------------------------------------------
# Param factory
# ---------------------------------------------------------------------------

def make_param(
    device_id: str = "grid",
    port_id: str = "p_ac",
    bus_id: str = "bus_ac",
    max_import_w: float = MAX_IMPORT_W,
    max_export_w: float = MAX_EXPORT_W,
    import_cost: float = IMPORT_COST,
    export_rev: float = EXPORT_REV,
    import_price_amt_kwh_key: str | None = None,
    export_price_amt_kwh_key: str | None = None,
    include_peak: bool = False,
) -> GridConnectionParam:
    port = EnergyPort(port_id=port_id, bus_id=bus_id, direction=PortDirection.BIDIRECTIONAL)
    return GridConnectionParam(
        device_id=device_id,
        ports=(port,),
        max_import_power_w=max_import_w,
        max_export_power_w=max_export_w,
        import_cost_per_kwh=import_cost,
        export_revenue_per_kwh=export_rev,
        import_price_amt_kwh_key=import_price_amt_kwh_key,
        export_price_amt_kwh_key=export_price_amt_kwh_key,
        include_peak_power_objective=include_peak,
    )


# ---------------------------------------------------------------------------
# Device factory
# ---------------------------------------------------------------------------

def make_device(
    param: GridConnectionParam | None = None,
    horizon: int = HORIZON,
    device_index: int = 0,
    port_index: int = 0,
    import_prices: np.ndarray | None = None,
    export_prices: np.ndarray | None = None,
) -> GridConnectionDevice:
    if param is None:
        param = make_param()
    ctx = make_context(horizon, import_prices, export_prices)
    device = GridConnectionDevice(param, device_index, port_index)
    device.setup_run(cast(SimulationContext, ctx))
    return device


def make_grant(granted: np.ndarray, device_index: int = 0) -> DeviceGrant:
    return DeviceGrant(
        device_index=device_index,
        port_grants=(PortGrant(port_index=0, granted_wh=granted),),
    )


# ============================================================
# TestGridConnectionParamValidation
# ============================================================

class TestGridConnectionParamValidation:
    def test_zero_import_power_raises(self) -> None:
        with pytest.raises(ValueError, match="max_import_power_w"):
            make_param(max_import_w=0.0)

    def test_negative_import_power_raises(self) -> None:
        with pytest.raises(ValueError, match="max_import_power_w"):
            make_param(max_import_w=-1.0)

    def test_negative_export_power_raises(self) -> None:
        with pytest.raises(ValueError, match="max_export_power_w"):
            make_param(max_export_w=-1.0)

    def test_negative_import_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="import_cost_per_kwh"):
            make_param(import_cost=-0.01)

    def test_negative_export_revenue_raises(self) -> None:
        with pytest.raises(ValueError, match="export_revenue_per_kwh"):
            make_param(export_rev=-0.01)

    def test_zero_export_power_allowed(self) -> None:
        p = make_param(max_export_w=0.0)
        assert p.max_export_power_w == 0.0

    def test_zero_import_cost_allowed(self) -> None:
        p = make_param(import_cost=0.0)
        assert p.import_cost_per_kwh == 0.0

    def test_valid_param_constructs(self) -> None:
        p = make_param()
        assert p.max_import_power_w == MAX_IMPORT_W
        assert p.max_export_power_w == MAX_EXPORT_W


# ============================================================
# TestGridConnectionParamHashability
# ============================================================

class TestGridConnectionParamHashability:
    def test_hashable(self) -> None:
        p = make_param()
        assert isinstance(hash(p), int)

    def test_equal_params_same_hash(self) -> None:
        p1 = make_param(max_import_w=8_000.0)
        p2 = make_param(max_import_w=8_000.0)
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_different_params_not_equal(self) -> None:
        p1 = make_param(import_cost=0.30)
        p2 = make_param(import_cost=0.35)
        assert p1 != p2


# ============================================================
# TestGridConnectionDeviceTopology
# ============================================================

class TestGridConnectionDeviceTopology:
    def test_device_id_matches_param(self) -> None:
        param = make_param(device_id="grid_main")
        device = GridConnectionDevice(param, 0, 0)
        assert device.device_id == "grid_main"

    def test_exactly_one_port(self) -> None:
        device = GridConnectionDevice(make_param(), 0, 0)
        assert len(device.ports) == 1

    def test_port_is_bidirectional(self) -> None:
        device = GridConnectionDevice(make_param(), 0, 0)
        assert device.ports[0].direction == PortDirection.BIDIRECTIONAL

    def test_port_id_matches_param(self) -> None:
        param = make_param(port_id="ac_main")
        device = GridConnectionDevice(param, 0, 0)
        assert device.ports[0].port_id == param.ports[0].port_id

    def test_bus_id_matches_param(self) -> None:
        param = make_param(bus_id="bus_230v")
        device = GridConnectionDevice(param, 0, 0)
        assert device.ports[0].bus_id == param.ports[0].bus_id

    def test_objective_names_without_peak(self) -> None:
        device = GridConnectionDevice(make_param(include_peak=False), 0, 0)
        assert device.objective_names == ["energy_cost_eur"]

    def test_objective_names_with_peak(self) -> None:
        device = GridConnectionDevice(make_param(include_peak=True), 0, 0)
        assert device.objective_names == ["energy_cost_eur", "peak_import_kw"]

    def test_device_index_stored(self) -> None:
        device = GridConnectionDevice(make_param(), device_index=3, port_index=0)
        assert device._device_index == 3

    def test_port_index_stored(self) -> None:
        device = GridConnectionDevice(make_param(), device_index=0, port_index=5)
        assert device._port_index == 5


# ============================================================
# TestSetupRun
# ============================================================

class TestSetupRun:
    def test_step_times_stored(self) -> None:
        ctx = make_context(horizon=4)
        device = GridConnectionDevice(make_param(), 0, 0)
        device.setup_run(cast(SimulationContext, ctx))
        assert device._step_times == ctx.step_times

    def test_num_steps_stored(self) -> None:
        device = make_device(horizon=5)
        assert device._num_steps == 5

    def test_step_interval_stored(self) -> None:
        device = make_device()
        assert device._step_interval_sec == STEP_INTERVAL

    def test_import_price_resolved_when_key_set(self) -> None:
        prices = np.linspace(0.20, 0.40, HORIZON)
        param = make_param(import_price_amt_kwh_key=IMPORT_KEY)
        device = make_device(param=param, import_prices=prices)
        assert device._import_price_per_kwh is not None
        np.testing.assert_allclose(device._import_price_per_kwh, prices)

    def test_export_price_resolved_when_key_set(self) -> None:
        prices = np.linspace(0.05, 0.12, HORIZON)
        param = make_param(export_price_amt_kwh_key=EXPORT_KEY)
        device = make_device(param=param, export_prices=prices)
        assert device._export_price_per_kwh is not None
        np.testing.assert_allclose(device._export_price_per_kwh, prices)

    def test_import_price_none_when_no_key(self) -> None:
        device = make_device(param=make_param(import_price_amt_kwh_key=None))
        assert device._import_price_per_kwh is None

    def test_export_price_none_when_no_key(self) -> None:
        device = make_device(param=make_param(export_price_amt_kwh_key=None))
        assert device._export_price_per_kwh is None

    def test_wrong_shape_import_price_raises(self) -> None:
        wrong = np.ones(HORIZON + 2)
        ctx = FakeContext(horizon=HORIZON, import_prices=wrong)
        param = make_param(import_price_amt_kwh_key=IMPORT_KEY)
        device = GridConnectionDevice(param, 0, 0)
        with pytest.raises(ValueError, match="import price series"):
            device.setup_run(cast(SimulationContext, ctx))

    def test_wrong_shape_export_price_raises(self) -> None:
        wrong = np.ones(HORIZON - 1)
        ctx = FakeContext(horizon=HORIZON, export_prices=wrong)
        param = make_param(export_price_amt_kwh_key=EXPORT_KEY)
        device = GridConnectionDevice(param, 0, 0)
        with pytest.raises(ValueError, match="export price series"):
            device.setup_run(cast(SimulationContext, ctx))


# ============================================================
# TestGenomeRequirements
# ============================================================

class TestGenomeRequirements:
    def test_returns_none_after_setup(self) -> None:
        device = make_device()
        device.genome_requirements()  # must not raise; always returns None

    def test_returns_none_before_setup(self) -> None:
        """genome_requirements must be safe to call without setup_run."""
        device = GridConnectionDevice(make_param(), 0, 0)
        device.genome_requirements()  # must not raise; always returns None


# ============================================================
# TestCreateBatchState
# ============================================================

class TestCreateBatchState:
    def test_granted_wh_shape(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        assert state.granted_wh.shape == (POP, HORIZON)

    def test_granted_wh_zero_initialised(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        assert (state.granted_wh == 0.0).all()

    def test_step_times_forwarded(self) -> None:
        ctx = make_context(horizon=4)
        device = GridConnectionDevice(make_param(), 0, 0)
        device.setup_run(cast(SimulationContext, ctx))
        state = device.create_batch_state(POP, 4)
        assert state.step_times == ctx.step_times

    def test_population_size_stored(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        assert state.population_size == POP

    def test_horizon_stored(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        assert state.horizon == HORIZON


# ============================================================
# TestApplyGenomeBatch
# ============================================================

class TestApplyGenomeBatch:
    def test_returns_genome_unchanged(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        genome = np.random.default_rng(0).random((POP, 3))
        original = genome.copy()
        result = device.apply_genome_batch(state, genome)
        np.testing.assert_array_equal(result, original)

    def test_noop_on_empty_genome(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        genome = np.zeros((POP, 0))
        result = device.apply_genome_batch(state, genome)
        assert result.shape == (POP, 0)


# ============================================================
# TestBuildDeviceRequest
# ============================================================

class TestBuildDeviceRequest:
    def test_device_index_correct(self) -> None:
        device = make_device(device_index=7)
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.device_index == 7

    def test_port_index_correct(self) -> None:
        device = make_device(port_index=2)
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].port_index == 2

    def test_is_slack_true(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].is_slack is True

    def test_energy_wh_equals_max_import_times_step_h(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        expected = MAX_IMPORT_W * STEP_H
        np.testing.assert_allclose(req.port_requests[0].energy_wh, expected)

    def test_min_energy_wh_equals_neg_max_export_times_step_h(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        expected = -(MAX_EXPORT_W * STEP_H)
        np.testing.assert_allclose(req.port_requests[0].min_energy_wh, expected)

    def test_zero_export_min_energy_wh_all_zero(self) -> None:
        param = make_param(max_export_w=0.0)
        device = make_device(param=param)
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        np.testing.assert_allclose(req.port_requests[0].min_energy_wh, 0.0)

    def test_request_shapes_pop_horizon(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].energy_wh.shape == (POP, HORIZON)
        assert req.port_requests[0].min_energy_wh.shape == (POP, HORIZON)

    def test_all_rows_identical(self) -> None:
        """Grid capacity is genome-independent: every population row is equal."""
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        energy = req.port_requests[0].energy_wh
        for i in range(1, POP):
            np.testing.assert_array_equal(energy[i], energy[0])


# ============================================================
# TestApplyDeviceGrant
# ============================================================

class TestApplyDeviceGrant:
    def test_granted_wh_updated(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        awarded = np.ones((POP, HORIZON)) * 500.0
        device.apply_device_grant(state, make_grant(awarded))
        np.testing.assert_allclose(state.granted_wh, awarded)

    def test_negative_grant_stored(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        awarded = -np.ones((POP, HORIZON)) * 300.0
        device.apply_device_grant(state, make_grant(awarded))
        np.testing.assert_allclose(state.granted_wh, awarded)

    def test_shape_preserved_after_grant(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        device.apply_device_grant(state, make_grant(np.zeros((POP, HORIZON))))
        assert state.granted_wh.shape == (POP, HORIZON)


# ============================================================
# TestComputeCost — flat-rate import
# ============================================================

class TestComputeCostFlatImport:
    def test_cost_shape_without_peak(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        assert cost.shape == (POP, 1)

    def test_zero_grant_zero_cost(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        np.testing.assert_allclose(cost, 0.0, atol=1e-9)

    def test_pure_import_cost_formula(self) -> None:
        device = make_device()
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, :] = 1_000.0
        cost = device.compute_cost(state)
        expected = HORIZON * 1.0 * IMPORT_COST
        assert cost[0, 0] == pytest.approx(expected, rel=1e-9)

    def test_cost_scales_linearly_with_import(self) -> None:
        device = make_device()
        state_a = device.create_batch_state(1, HORIZON)
        state_b = device.create_batch_state(1, HORIZON)
        state_a.granted_wh[0, :] = 500.0
        state_b.granted_wh[0, :] = 1_000.0
        cost_a = device.compute_cost(state_a)[0, 0]
        cost_b = device.compute_cost(state_b)[0, 0]
        assert cost_b == pytest.approx(2.0 * cost_a, rel=1e-9)

    def test_population_rows_computed_independently(self) -> None:
        device = make_device()
        state = device.create_batch_state(2, HORIZON)
        state.granted_wh[0, :] = 1_000.0
        state.granted_wh[1, :] = 2_000.0
        cost = device.compute_cost(state)
        assert cost[1, 0] == pytest.approx(2.0 * cost[0, 0], rel=1e-9)


# ============================================================
# TestComputeCost — flat-rate export
# ============================================================

class TestComputeCostFlatExport:
    def test_pure_export_revenue_formula(self) -> None:
        device = make_device()
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, :] = -1_000.0
        cost = device.compute_cost(state)
        expected = -HORIZON * 1.0 * EXPORT_REV
        assert cost[0, 0] == pytest.approx(expected, rel=1e-9)

    def test_net_cost_negative_on_pure_export(self) -> None:
        device = make_device()
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, :] = -500.0
        cost = device.compute_cost(state)
        assert cost[0, 0] < 0.0


# ============================================================
# TestComputeCost — flat-rate mixed
# ============================================================

class TestComputeCostFlatMixed:
    def test_mixed_import_export(self) -> None:
        device = make_device(horizon=4)
        state = device.create_batch_state(1, 4)
        state.granted_wh[0, 0] =  2_000.0
        state.granted_wh[0, 1] = -1_000.0
        cost = device.compute_cost(state)
        expected = 2.0 * IMPORT_COST - 1.0 * EXPORT_REV
        assert cost[0, 0] == pytest.approx(expected, rel=1e-9)

    def test_equal_import_export_at_same_rate_cancels(self) -> None:
        device = make_device(param=make_param(import_cost=0.20, export_rev=0.20), horizon=2)
        state = device.create_batch_state(1, 2)
        state.granted_wh[0, 0] =  1_000.0
        state.granted_wh[0, 1] = -1_000.0
        cost = device.compute_cost(state)
        assert cost[0, 0] == pytest.approx(0.0, abs=1e-9)


# ============================================================
# TestComputeCost — time-of-use prices
# ============================================================

class TestComputeCostTOU:
    def test_tou_import_overrides_flat_rate(self) -> None:
        tou = np.full(HORIZON, 0.50)
        param = make_param(import_price_amt_kwh_key=IMPORT_KEY, import_cost=0.30)
        device = make_device(param=param, import_prices=tou)
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, :] = 1_000.0
        cost = device.compute_cost(state)
        assert cost[0, 0] == pytest.approx(HORIZON * 1.0 * 0.50, rel=1e-9)

    def test_tou_export_overrides_flat_rate(self) -> None:
        tou = np.full(HORIZON, 0.15)
        param = make_param(export_price_amt_kwh_key=EXPORT_KEY, export_rev=0.08)
        device = make_device(param=param, export_prices=tou)
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, :] = -1_000.0
        cost = device.compute_cost(state)
        assert cost[0, 0] == pytest.approx(-HORIZON * 1.0 * 0.15, rel=1e-9)

    def test_tou_prices_applied_per_step(self) -> None:
        prices = np.array([0.10, 0.60])
        param = make_param(import_price_amt_kwh_key=IMPORT_KEY)
        device = make_device(param=param, horizon=2, import_prices=prices)
        state = device.create_batch_state(1, 2)
        state.granted_wh[0, 0] = 1_000.0
        state.granted_wh[0, 1] = 1_000.0
        cost = device.compute_cost(state)
        assert cost[0, 0] == pytest.approx(1.0 * 0.10 + 1.0 * 0.60, rel=1e-9)

    def test_tou_import_key_does_not_affect_export_direction(self) -> None:
        tou_import = np.full(HORIZON, 0.50)
        param = make_param(import_price_amt_kwh_key=IMPORT_KEY, export_rev=EXPORT_REV)
        device = make_device(param=param, import_prices=tou_import)
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, :] = -1_000.0
        cost = device.compute_cost(state)
        assert cost[0, 0] == pytest.approx(-HORIZON * 1.0 * EXPORT_REV, rel=1e-9)

    def test_tou_export_key_does_not_affect_import_direction(self) -> None:
        tou_export = np.full(HORIZON, 0.15)
        param = make_param(export_price_amt_kwh_key=EXPORT_KEY, import_cost=IMPORT_COST)
        device = make_device(param=param, export_prices=tou_export)
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, :] = 1_000.0
        cost = device.compute_cost(state)
        assert cost[0, 0] == pytest.approx(HORIZON * 1.0 * IMPORT_COST, rel=1e-9)


# ============================================================
# TestComputeCost — peak import objective
# ============================================================

class TestComputeCostPeak:
    def test_cost_shape_with_peak(self) -> None:
        device = make_device(param=make_param(include_peak=True))
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        assert cost.shape == (POP, 2)

    def test_peak_column_is_max_import_power_kw(self) -> None:
        device = make_device(param=make_param(include_peak=True))
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, 0] = 2_000.0
        state.granted_wh[0, 2] = 5_000.0
        cost = device.compute_cost(state)
        assert cost[0, 1] == pytest.approx(5.0, rel=1e-9)

    def test_export_steps_do_not_inflate_peak(self) -> None:
        device = make_device(param=make_param(include_peak=True))
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, 0] = -3_000.0
        cost = device.compute_cost(state)
        assert cost[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_pure_export_peak_is_zero(self) -> None:
        device = make_device(param=make_param(include_peak=True))
        state = device.create_batch_state(1, HORIZON)
        state.granted_wh[0, :] = -1_000.0
        cost = device.compute_cost(state)
        assert cost[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_peak_computed_independently_per_individual(self) -> None:
        device = make_device(param=make_param(include_peak=True))
        state = device.create_batch_state(2, HORIZON)
        state.granted_wh[0, 0] = 3_000.0
        state.granted_wh[1, 0] = 7_000.0
        cost = device.compute_cost(state)
        assert cost[0, 1] == pytest.approx(3.0, rel=1e-9)
        assert cost[1, 1] == pytest.approx(7.0, rel=1e-9)

    def test_energy_cost_column_unaffected_by_peak_flag(self) -> None:
        def cost_col0(include_peak: bool) -> float:
            device = make_device(param=make_param(include_peak=include_peak))
            state = device.create_batch_state(1, HORIZON)
            state.granted_wh[0, :] = 1_000.0
            return float(device.compute_cost(state)[0, 0])

        assert cost_col0(False) == pytest.approx(cost_col0(True), rel=1e-9)


# ============================================================
# TestExtractInstructions
# ============================================================

class TestExtractInstructions:
    def test_returns_empty_list(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        assert device.extract_instructions(state, 0) == []

    def test_returns_empty_list_for_all_individuals(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        for i in range(POP):
            assert device.extract_instructions(state, i) == []
