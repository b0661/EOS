"""Tests for the fixed (non-controllable) household load device.

Module under test
-----------------
``akkudoktoreos.devices.genetic2.fixedload``

Design under test
-----------------
``FixedLoadDevice`` models a load whose power consumption is driven entirely
by an external forecast resolved from the ``SimulationContext`` during
``setup_run``.  The optimizer has no control over it:

* No genome — ``genome_requirements`` always returns ``None``.
* No cost objective — ``compute_cost`` returns shape ``(population_size, 0)``.
* No S2 instructions — ``extract_instructions`` always returns ``[]``.

The device submits the forecast as a pure sink request to the bus arbitrator
every generation.  The same forecast applies identically to every individual
in the population.

Forecast handling
~~~~~~~~~~~~~~~~~
* Negative forecast values are clamped to zero in ``setup_run`` — a fixed
  load only consumes, never injects.
* The forecast key is resolved via ``context.resolve_prediction(key)``.
* A wrong-shape forecast raises ``ValueError`` with a descriptive message.

Sign convention (consistent with the rest of the framework)::

    positive energy_wh  → consuming from the AC bus (load)
    negative energy_wh  → injecting (not applicable; clamped to 0)

``_step_interval_sec`` is stored as a plain ``float`` (seconds).  Division
by 3600 in ``build_device_request`` converts it to hours for energy [Wh].

Param construction
------------------
``FixedLoadParam`` inherits ``device_id`` and ``ports`` from ``DeviceParam``
and adds ``load_power_w_key``.  An empty ``load_power_w_key`` or an empty
``ports`` tuple raises ``ValueError`` from ``__post_init__``.

``FakeContext`` exposes ``resolve_prediction(key)`` matching the real
``SimulationContext`` interface that ``FixedLoadDevice.setup_run`` calls.

Test strategy
-------------
All tests use a ``FakeContext`` stand-in.  No real simulation context,
arbitrator, or engine is required.  Request arithmetic is verified by hand:

    energy_wh[t] = max(0, forecast_w[t]) * step_h

Tolerances: rtol=1e-9 throughout; zero checks use atol=1e-9.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pendulum import Duration

from akkudoktoreos.devices.devicesabc import EnergyPort, PortDirection
from akkudoktoreos.devices.genetic2.fixedload import (
    FixedLoadBatchState,
    FixedLoadDevice,
    FixedLoadParam,
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

LOAD_KEY = "loadforecast_power_w"
ALT_KEY  = "my_custom_load_w"

FLAT_LOAD_W = 500.0   # W — constant forecast used in most tests


# ---------------------------------------------------------------------------
# FakeContext
# ---------------------------------------------------------------------------

class FakeContext:
    """Minimal SimulationContext stand-in.

    ``context.step_interval`` is a ``pendulum.Duration`` — matching the real
    ``SimulationContext`` interface that ``FixedLoadDevice.setup_run`` expects.
    ``context.resolve_prediction(key)`` returns the pre-loaded forecast array
    for the given key.

    Call sites pass this to ``setup_run()`` via ``cast(SimulationContext, ctx)``
    so mypy is satisfied while the duck-typed runtime behaviour is preserved.
    """

    def __init__(
        self,
        horizon: int = HORIZON,
        step_interval_sec: float = STEP_INTERVAL,
        forecasts: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.step_times = tuple(to_datetime(i * step_interval_sec) for i in range(horizon))
        self.step_interval = Duration(seconds=int(step_interval_sec))
        self.horizon = horizon
        self._forecasts: dict[str, np.ndarray] = forecasts or {}

    def resolve_prediction(self, key: str) -> np.ndarray:
        """Return the forecast array for ``key``, matching the real SimulationContext API."""
        try:
            return self._forecasts[key].copy()
        except KeyError:
            raise KeyError(key)


def make_context(
    horizon: int = HORIZON,
    load_w: np.ndarray | None = None,
    key: str = LOAD_KEY,
    step_interval_sec: float = STEP_INTERVAL,
) -> FakeContext:
    """Build a FakeContext with a flat or custom load forecast under ``key``."""
    if load_w is None:
        load_w = np.full(horizon, FLAT_LOAD_W)
    return FakeContext(
        horizon=horizon,
        step_interval_sec=step_interval_sec,
        forecasts={key: load_w},
    )


# ---------------------------------------------------------------------------
# Param factory
# ---------------------------------------------------------------------------

def make_param(
    device_id: str = "base_load",
    port_id: str = "p_ac",
    bus_id: str = "bus_ac",
    load_power_w_key: str = LOAD_KEY,
) -> FixedLoadParam:
    port = EnergyPort(port_id=port_id, bus_id=bus_id, direction=PortDirection.SINK)
    return FixedLoadParam(
        device_id=device_id,
        ports=(port,),
        load_power_w_key=load_power_w_key,
    )


# ---------------------------------------------------------------------------
# Device factory
# ---------------------------------------------------------------------------

def make_device(
    param: FixedLoadParam | None = None,
    horizon: int = HORIZON,
    device_index: int = 0,
    port_index: int = 0,
    load_w: np.ndarray | None = None,
    key: str = LOAD_KEY,
    step_interval_sec: float = STEP_INTERVAL,
) -> FixedLoadDevice:
    if param is None:
        param = make_param(load_power_w_key=key)
    ctx = make_context(horizon=horizon, load_w=load_w, key=key,
                       step_interval_sec=step_interval_sec)
    device = FixedLoadDevice(param, device_index, port_index)
    device.setup_run(cast(SimulationContext, ctx))
    return device


def make_grant(granted: np.ndarray, device_index: int = 0) -> DeviceGrant:
    return DeviceGrant(
        device_index=device_index,
        port_grants=(PortGrant(port_index=0, granted_wh=granted),),
    )


# ============================================================
# TestFixedLoadParamValidation
# ============================================================

class TestFixedLoadParamValidation:
    def test_empty_ports_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one port"):
            FixedLoadParam(device_id="x", ports=(), load_power_w_key=LOAD_KEY)

    def test_empty_key_raises(self) -> None:
        port = EnergyPort(port_id="p", bus_id="b", direction=PortDirection.SINK)
        with pytest.raises(ValueError, match="load_power_w_key"):
            FixedLoadParam(device_id="x", ports=(port,), load_power_w_key="")

    def test_valid_param_constructs(self) -> None:
        p = make_param()
        assert p.device_id == "base_load"
        assert p.load_power_w_key == LOAD_KEY

    def test_custom_key_stored(self) -> None:
        p = make_param(load_power_w_key=ALT_KEY)
        assert p.load_power_w_key == ALT_KEY


# ============================================================
# TestFixedLoadParamHashability
# ============================================================

class TestFixedLoadParamHashability:
    def test_hashable(self) -> None:
        p = make_param()
        assert isinstance(hash(p), int)

    def test_equal_params_same_hash(self) -> None:
        p1 = make_param(device_id="load_a")
        p2 = make_param(device_id="load_a")
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_different_params_not_equal(self) -> None:
        p1 = make_param(load_power_w_key=LOAD_KEY)
        p2 = make_param(load_power_w_key=ALT_KEY)
        assert p1 != p2

    def test_usable_as_dict_key(self) -> None:
        p = make_param()
        d = {p: "value"}
        assert d[p] == "value"


# ============================================================
# TestFixedLoadDeviceTopology
# ============================================================

class TestFixedLoadDeviceTopology:
    def test_device_id_matches_param(self) -> None:
        param = make_param(device_id="fridge")
        device = FixedLoadDevice(param, 0, 0)
        assert device.device_id == "fridge"

    def test_exactly_one_port(self) -> None:
        device = FixedLoadDevice(make_param(), 0, 0)
        assert len(device.ports) == 1

    def test_port_is_sink(self) -> None:
        device = FixedLoadDevice(make_param(), 0, 0)
        assert device.ports[0].direction == PortDirection.SINK

    def test_port_id_matches_param(self) -> None:
        param = make_param(port_id="ac_load")
        device = FixedLoadDevice(param, 0, 0)
        assert device.ports[0].port_id == param.ports[0].port_id

    def test_bus_id_matches_param(self) -> None:
        param = make_param(bus_id="bus_230v")
        device = FixedLoadDevice(param, 0, 0)
        assert device.ports[0].bus_id == param.ports[0].bus_id

    def test_objective_names_empty(self) -> None:
        device = FixedLoadDevice(make_param(), 0, 0)
        assert device.objective_names == []

    def test_device_index_stored(self) -> None:
        device = FixedLoadDevice(make_param(), device_index=5, port_index=0)
        assert device._device_index == 5

    def test_port_index_stored(self) -> None:
        device = FixedLoadDevice(make_param(), device_index=0, port_index=3)
        assert device._port_index == 3


# ============================================================
# TestSetupRun
# ============================================================

class TestSetupRun:
    def test_step_times_stored(self) -> None:
        ctx = make_context(horizon=4)
        device = FixedLoadDevice(make_param(), 0, 0)
        device.setup_run(cast(SimulationContext, ctx))
        assert device._step_times == ctx.step_times

    def test_step_interval_stored(self) -> None:
        device = make_device()
        assert device._step_interval_sec == STEP_INTERVAL

    def test_forecast_resolved_and_stored(self) -> None:
        load = np.arange(HORIZON, dtype=float) * 100.0
        device = make_device(load_w=load)
        np.testing.assert_allclose(device._load_power_w, load)

    def test_negative_forecast_clamped_to_zero(self) -> None:
        load = np.array([-200.0, 300.0, -100.0, 400.0, 0.0, 150.0])
        device = make_device(load_w=load)
        expected = np.maximum(0.0, load)
        np.testing.assert_allclose(device._load_power_w, expected)

    def test_all_negative_forecast_gives_all_zero(self) -> None:
        load = np.full(HORIZON, -500.0)
        device = make_device(load_w=load)
        np.testing.assert_allclose(device._load_power_w, 0.0, atol=1e-9)

    def test_zero_forecast_stored_as_zero(self) -> None:
        load = np.zeros(HORIZON)
        device = make_device(load_w=load)
        np.testing.assert_allclose(device._load_power_w, 0.0, atol=1e-9)

    def test_custom_key_resolved(self) -> None:
        load = np.full(HORIZON, 999.0)
        param = make_param(load_power_w_key=ALT_KEY)
        device = make_device(param=param, load_w=load, key=ALT_KEY)
        np.testing.assert_allclose(device._load_power_w, 999.0)

    def test_wrong_shape_raises(self) -> None:
        wrong = np.ones(HORIZON + 3)
        ctx = FakeContext(horizon=HORIZON, forecasts={LOAD_KEY: wrong})
        device = FixedLoadDevice(make_param(), 0, 0)
        with pytest.raises(ValueError, match="load forecast must have shape"):
            device.setup_run(cast(SimulationContext, ctx))

    def test_missing_key_raises(self) -> None:
        ctx = FakeContext(horizon=HORIZON, forecasts={})
        device = FixedLoadDevice(make_param(), 0, 0)
        with pytest.raises(KeyError):
            device.setup_run(cast(SimulationContext, ctx))

    def test_step_interval_15min(self) -> None:
        device = make_device(step_interval_sec=900.0)
        assert device._step_interval_sec == 900.0


# ============================================================
# TestGenomeRequirements
# ============================================================

class TestGenomeRequirements:
    def test_returns_none_after_setup(self) -> None:
        device = make_device()
        assert device.genome_requirements() is None

    def test_returns_none_before_setup(self) -> None:
        device = FixedLoadDevice(make_param(), 0, 0)
        assert device.genome_requirements() is None


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
        np.testing.assert_allclose(state.granted_wh, 0.0, atol=1e-9)

    def test_step_times_forwarded(self) -> None:
        ctx = make_context(horizon=4)
        device = FixedLoadDevice(make_param(), 0, 0)
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

    def test_raises_before_setup(self) -> None:
        device = FixedLoadDevice(make_param(), 0, 0)
        with pytest.raises(RuntimeError, match="setup_run"):
            device.create_batch_state(POP, HORIZON)


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

    def test_state_unchanged_after_call(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        before = state.granted_wh.copy()
        device.apply_genome_batch(state, np.ones((POP, 5)))
        np.testing.assert_array_equal(state.granted_wh, before)


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

    def test_is_slack_false(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].is_slack is False

    def test_energy_wh_equals_forecast_times_step_h(self) -> None:
        load = np.arange(1, HORIZON + 1, dtype=float) * 100.0
        device = make_device(load_w=load)
        state = device.create_batch_state(1, HORIZON)
        req = device.build_device_request(state)
        expected = load * STEP_H
        np.testing.assert_allclose(req.port_requests[0].energy_wh[0], expected, rtol=1e-9)

    def test_min_energy_wh_all_zero(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        np.testing.assert_allclose(req.port_requests[0].min_energy_wh, 0.0, atol=1e-9)

    def test_request_shape_pop_horizon(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        assert req.port_requests[0].energy_wh.shape == (POP, HORIZON)
        assert req.port_requests[0].min_energy_wh.shape == (POP, HORIZON)

    def test_all_population_rows_identical(self) -> None:
        """Forecast is genome-independent: every population row must be equal."""
        load = np.arange(1, HORIZON + 1, dtype=float) * 200.0
        device = make_device(load_w=load)
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        energy = req.port_requests[0].energy_wh
        for i in range(1, POP):
            np.testing.assert_array_equal(energy[i], energy[0])

    def test_zero_forecast_gives_zero_request(self) -> None:
        device = make_device(load_w=np.zeros(HORIZON))
        state = device.create_batch_state(POP, HORIZON)
        req = device.build_device_request(state)
        np.testing.assert_allclose(req.port_requests[0].energy_wh, 0.0, atol=1e-9)

    def test_negative_forecast_clamped_in_request(self) -> None:
        load = np.array([-300.0, 400.0, -100.0, 600.0, 0.0, 200.0])
        device = make_device(load_w=load)
        state = device.create_batch_state(1, HORIZON)
        req = device.build_device_request(state)
        expected = np.maximum(0.0, load) * STEP_H
        np.testing.assert_allclose(req.port_requests[0].energy_wh[0], expected, rtol=1e-9)

    def test_step_interval_15min_scales_energy(self) -> None:
        step_sec = 900.0
        step_h = step_sec / 3600.0
        load = np.full(HORIZON, 1000.0)
        device = make_device(load_w=load, step_interval_sec=step_sec)
        state = device.create_batch_state(1, HORIZON)
        req = device.build_device_request(state)
        np.testing.assert_allclose(
            req.port_requests[0].energy_wh[0],
            1000.0 * step_h,
            rtol=1e-9,
        )

    def test_raises_before_setup(self) -> None:
        device = FixedLoadDevice(make_param(), 0, 0)
        # Manually create a minimal state to reach build_device_request
        state = FixedLoadBatchState(
            granted_wh=np.zeros((POP, HORIZON)),
            population_size=POP,
            horizon=HORIZON,
            step_times=tuple(to_datetime(i * STEP_INTERVAL) for i in range(HORIZON)),
        )
        with pytest.raises(RuntimeError, match="setup_run"):
            device.build_device_request(state)


# ============================================================
# TestApplyDeviceGrant
# ============================================================

class TestApplyDeviceGrant:
    def test_granted_wh_updated(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        awarded = np.ones((POP, HORIZON)) * 450.0
        device.apply_device_grant(state, make_grant(awarded))
        np.testing.assert_allclose(state.granted_wh, awarded)

    def test_partial_grant_stored(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        partial = np.random.default_rng(1).uniform(0, 500, (POP, HORIZON))
        device.apply_device_grant(state, make_grant(partial))
        np.testing.assert_allclose(state.granted_wh, partial)

    def test_zero_grant_stored(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        device.apply_device_grant(state, make_grant(np.zeros((POP, HORIZON))))
        np.testing.assert_allclose(state.granted_wh, 0.0, atol=1e-9)

    def test_shape_preserved_after_grant(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        device.apply_device_grant(state, make_grant(np.zeros((POP, HORIZON))))
        assert state.granted_wh.shape == (POP, HORIZON)

    def test_grant_overwrites_previous_value(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        device.apply_device_grant(state, make_grant(np.full((POP, HORIZON), 100.0)))
        device.apply_device_grant(state, make_grant(np.full((POP, HORIZON), 200.0)))
        np.testing.assert_allclose(state.granted_wh, 200.0)


# ============================================================
# TestComputeCost
# ============================================================

class TestComputeCost:
    def test_cost_shape_is_pop_zero(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        assert cost.shape == (POP, 0)

    def test_cost_shape_after_grant(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        device.apply_device_grant(state, make_grant(np.ones((POP, HORIZON)) * 500.0))
        cost = device.compute_cost(state)
        assert cost.shape == (POP, 0)

    def test_no_objective_names(self) -> None:
        device = make_device()
        assert device.objective_names == []

    def test_objective_names_count_matches_cost_columns(self) -> None:
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        cost = device.compute_cost(state)
        assert cost.shape[1] == len(device.objective_names)


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

    def test_returns_empty_list_with_instruction_context(self) -> None:
        from akkudoktoreos.devices.devicesabc import InstructionContext
        device = make_device()
        state = device.create_batch_state(POP, HORIZON)
        ctx = InstructionContext(grid_granted_wh=None, step_interval_sec=STEP_INTERVAL)
        assert device.extract_instructions(state, 0, instruction_context=ctx) == []
