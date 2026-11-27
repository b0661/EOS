"""Golden inputs of OHMRAKEL42 as it runs on the live installation (snapshot e9f39ad5).

Each case is a self-contained input file ``tests/testdata/ohmrakel42/golden_<case>_input.json``
(configuration, runtime request per run, measured histories, outdoor temperature and presence)
and an expected file ``golden_<case>_expected.json`` (native solution, optimization solution,
energy management plan and thermal plans per run, or the error a run raised). The port of
OHMRAKEL42 to the optimizer architecture of main replays the same input files and has to give the
same expected output (characterization, see ADR 0008 on main).

A missing input file is built from the fixtures of the snapshot's tests. A missing or differing
expected file is written next to it as ``new_golden_<case>_expected.json``; the failure message
gives the command to accept it.
"""

import builtins
import json
import math
import re
import sys
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import pytest
import test_searchscenarios as sc
from test_ohmrakel42ems import ems_world  # noqa: F401  (registers the fixture)
from test_planningrequestdevices import MANY_DEVICES
from test_requestfixtures import (  # noqa: F401  (registers the fixture)
    BATTERY,
    DISHWASHER,
    INVERTER,
    VEHICLE,
    fake_measurement_fixture,
    forecast_series,
)

from akkudoktoreos.config.config import ConfigEOS
from akkudoktoreos.core.ems import EnergyManagement
from akkudoktoreos.core.emsettings import EnergyManagementMode
from akkudoktoreos.measurement.quality import MeasurementSample
from akkudoktoreos.optimization.request import PlanningRequest
from akkudoktoreos.optimization.request.devices import DeviceResolver
from akkudoktoreos.optimization.request.thermal import ThermalResolver
from akkudoktoreos.optimization.solutionmodel import Ohmrakel42Solution
from akkudoktoreos.prediction.presence import PresenceEstimate
from akkudoktoreos.utils.datetimeutil import DateTime, to_datetime

DIR_PROJECT_ROOT = Path(__file__).parent.parent
if str(DIR_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DIR_PROJECT_ROOT))

from scripts.capture_ohmrakel42_golden import Recorder, call_key  # noqa: E402

DIR_TESTDATA = Path(__file__).parent / "testdata"
DIR_GOLDEN = DIR_TESTDATA / "ohmrakel42"

TIMEZONE = "Europe/Berlin"
START = "2026-09-16T10:00:00+02:00"
# The day the clocks go back in Europe/Berlin (25 hours).
DST_START = "2026-10-25T10:00:00+01:00"
HISTORY_MINUTES = 14 * 60
MASK = "<masked>"
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# The wall-clock time an ID is made at, e.g. ``plan-ohmrakel42@2026-11-27T14:43:57.263909+00:00``.
ID_TIME_PATTERN = re.compile(r"@\d{4}-\d{2}-\d{2}T[0-9:.]+(?:[+-]\d{2}:\d{2}|Z)?")

# Values that depend on the wall clock or the machine, not on the input.
VOLATILE_KEYS = {"generated_at", "seconds", "issued_at", "valid_until"}


# ---------------------------------------------------------------------------------------
# Building the input files from the fixtures of the snapshot's tests


def _series(
    start: DateTime, value: Callable[[int], float], minutes: int = HISTORY_MINUTES
) -> list[list[Any]]:
    """A value every minute up to ``start``; ``value`` gets the minute index (0 the oldest)."""
    return [
        [(start - timedelta(minutes=minutes - index)).isoformat(), value(index)]
        for index in range(minutes + 1)
    ]


def _thermal_measurements(start: DateTime) -> dict[str, list[list[Any]]]:
    """The histories of ``thermal_world`` in ``test_planningrequestthermal`` (running 100 min)."""
    running = 100
    return {
        "living-temp": _series(start, lambda _: 20.8),
        "outdoor-temp": _series(start, lambda _: 4.0),
        "hp-power": _series(
            start, lambda i: 1200.0 if i > HISTORY_MINUTES - running else 5.0
        ),
        "hp-heat": _series(start, lambda i: 3200.0 if i > HISTORY_MINUTES - running else 0.0),
    }


THERMAL: dict[str, Any] = {
    "thermal": {
        "measurement_max_age_seconds": 300,
        "zones": {
            "living": {
                "enabled": True,
                "room_temperature_measurement_key": "living-temp",
                "outdoor_temperature_measurement_key": "outdoor-temp",
                "model_reference": "calibrated",
                "model_parameters": sc.HEAT_BUILDING.model_dump(mode="json"),
                "comfort": {
                    "min_temperature_c": 20.0,
                    "target_temperature_c": 21.0,
                    "max_temperature_c": 23.0,
                },
                "unoccupied_comfort": {
                    "min_temperature_c": 17.0,
                    "target_temperature_c": 17.5,
                    "max_temperature_c": 23.0,
                },
                "presence_area_id": "living",
            }
        },
    },
    "presence": {"provider": "PresenceHistory", "zones": {"living": {"measurement_key": "away"}}},
    "measurement": {"channels": {"away": {"quantity": "presence", "unit": "1"}}},
}
HEAT_SOURCE: dict[str, Any] = {
    "kind": "heat_pump",
    "thermal_power_measurement_key": "hp-heat",
    "enabled": True,
    "max_electric_power_w": 3000,
    "min_electric_power_w": 400,
    "zone_ids": ["living"],
    "electric_power_measurement_key": "hp-power",
    "heat_pump": {
        "custom_profile": sc.HEAT_PROFILE.model_dump(mode="json"),
        "delivery_temperature_c": 20.0,
        "electrical": sc.HEAT_ELECTRICAL.model_dump(mode="json"),
    },
}
SIMPLE_DEVICES: dict[str, Any] = {
    "batteries": {"bat1": BATTERY},
    "electric_vehicles": {"ev1": VEHICLE},
    "inverters": {"inv1": INVERTER},
    "home_appliances": {"dishwasher": DISHWASHER},
}
SIMPLE_SOC = {"bat1": 55, "ev1": 10}
HEAT_DEVICES: dict[str, Any] = {
    "batteries": {"bat1": BATTERY},
    "inverters": {"inv1": INVERTER},
    "heat_sources": {"hp1": HEAT_SOURCE},
}
MANY_HEAT_DEVICES: dict[str, Any] = {
    **MANY_DEVICES,
    "batteries": {
        key: {**value, "pv_charge_switchable": False}
        for key, value in MANY_DEVICES["batteries"].items()
    },
    "heat_sources": {"hp1": HEAT_SOURCE},
}
MANY_SOC = {"bat1": 40, "bat2": 70, "ev1": 20, "ev2": 30}
# The presence of ``FakePresence`` in ``test_planningrequestthermal``: away 12:00 to 17:00.
PRESENCE = {
    "away_from_hour": 12,
    "away_to_hour": 17,
    "probability_away": 0.02,
    "probability_home": 0.95,
}


def _settings(
    devices: Optional[dict[str, Any]],
    *,
    interval: int,
    horizon: int = 24,
    tail_hours: int = 0,
    terminal_value_mode: str = "FIXED",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """The configuration of a case, in the snapshot's layout."""
    settings: dict[str, Any] = {
        "general": {"timezone": TIMEZONE},
        "prediction": {"hours": 48},
        "optimization": {
            "horizon_hours": horizon,
            "tail_horizon_hours": tail_hours,
            "terminal_value_mode": terminal_value_mode,
            "interval_sec": interval,
        },
    }
    if devices is not None:
        settings["devices"] = devices
    settings.update(extra or {})
    return settings


def _run(
    start: str,
    interval: int,
    soc: Mapping[str, float],
    *,
    changed_devices: Optional[list[str]] = None,
    **request: Any,
) -> dict[str, Any]:
    """One run of a case: its start and its runtime request."""
    body: dict[str, Any] = {
        "soc": dict(soc),
        "forecasts": forecast_series("optimize_input_1.json", interval),
        **request,
    }
    return {"start": start, "request": body, "changed_devices": changed_devices or []}


def _config_file(name: str) -> dict[str, Any]:
    """A stored configuration file of the test data, without its server settings."""
    with (DIR_TESTDATA / name).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.pop("server", None)
    return data


def _thermal_case(
    devices: dict[str, Any], interval: int, soc: Mapping[str, float], **request: Any
) -> dict[str, Any]:
    """A case with a heat pump in the thermal world of the snapshot's tests."""
    return {
        "settings": _settings(devices, interval=interval, extra=THERMAL),
        "runs": [_run(START, interval, soc, **request)],
        "measurements": _thermal_measurements(to_datetime(START, in_timezone=TIMEZONE)),
        "outdoor_temperature_c": 4.0,
        "presence": PRESENCE,
    }


def _electrical_case(settings: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """A case without heat pump: no histories, no outdoor temperature, no presence."""
    return {
        "settings": settings,
        "runs": runs,
        "measurements": {},
        "outdoor_temperature_c": None,
        "presence": None,
    }


def _start_plus(start: str, minutes: int) -> str:
    return to_datetime(start, in_timezone=TIMEZONE).add(minutes=minutes).isoformat()


CASE_BUILDERS: dict[str, Callable[[], dict[str, Any]]] = {
    "simple_3600": lambda: _electrical_case(
        _settings(SIMPLE_DEVICES, interval=3600), [_run(START, 3600, SIMPLE_SOC)]
    ),
    "simple_900": lambda: _electrical_case(
        _settings(SIMPLE_DEVICES, interval=900), [_run(START, 900, SIMPLE_SOC)]
    ),
    "simple_horizon36_3600": lambda: _electrical_case(
        _settings(SIMPLE_DEVICES, interval=3600, horizon=36), [_run(START, 3600, SIMPLE_SOC)]
    ),
    "simple_dst_3600": lambda: _electrical_case(
        _settings(SIMPLE_DEVICES, interval=3600), [_run(DST_START, 3600, SIMPLE_SOC)]
    ),
    "simple_tail_auto_3600": lambda: _electrical_case(
        _settings(SIMPLE_DEVICES, interval=3600, tail_hours=24, terminal_value_mode="AUTO"),
        [_run(START, 3600, SIMPLE_SOC)],
    ),
    "simple_vehicle_available_3600": lambda: _electrical_case(
        _settings(SIMPLE_DEVICES, interval=3600),
        [
            _run(
                START,
                3600,
                SIMPLE_SOC,
                vehicle_available={"ev1": [False] * 8 + [True] * 16},
            )
        ],
    ),
    "simple_warm_start_900": lambda: _electrical_case(
        _settings(SIMPLE_DEVICES, interval=900),
        [
            _run(START, 900, SIMPLE_SOC),
            _run(_start_plus(START, 15), 900, SIMPLE_SOC),
            _run(_start_plus(START, 30), 900, {"bat1": 50, "ev1": 20}, changed_devices=["ev1"]),
        ],
    ),
    "many_devices_3600": lambda: _electrical_case(
        _settings(MANY_DEVICES, interval=3600), [_run(START, 3600, MANY_SOC)]
    ),
    "heat_pump_3600": lambda: _thermal_case(HEAT_DEVICES, 3600, {"bat1": 50}),
    "heat_pump_900": lambda: _thermal_case(HEAT_DEVICES, 900, {"bat1": 50}),
    "heat_pump_blocks_absence_3600": lambda: _thermal_case(
        HEAT_DEVICES,
        3600,
        {"bat1": 50},
        heat_source_blocks={
            "hp1": [
                {
                    "start": _start_plus(START, 3 * 60),
                    "end": _start_plus(START, 5 * 60),
                }
            ]
        },
        absences=[{"start": _start_plus(START, 60), "end": _start_plus(START, 9 * 60)}],
    ),
    "many_devices_heat_pump_3600": lambda: _thermal_case(MANY_HEAT_DEVICES, 3600, MANY_SOC),
    # Stored configuration files (Q14 a). The snapshot rejects them (no inverter for the
    # battery, no devices); the port follows "optimization parameters are always valid" and
    # declares the difference.
    "config_minimal_now": lambda: _electrical_case(
        {**_config_file("eos_config_minimal_now.json"), "general": {"timezone": TIMEZONE}},
        [_run(START, 3600, {})],
    ),
    "config_andreas_now": lambda: _electrical_case(
        {**_config_file("eos_config_andreas_now.json"), "general": {"timezone": TIMEZONE}},
        [_run(START, 3600, {"pv_akku": 50, "ev0": 30})],
    ),
}


# ---------------------------------------------------------------------------------------
# Running a case


class GoldenMeasurement:
    """The measured histories of a case; an unknown key is an error, as in the snapshot."""

    def __init__(self, series: dict[str, list[list[Any]]]) -> None:
        self.series = {
            key: [(to_datetime(stamp), float(value)) for stamp, value in rows]
            for key, rows in series.items()
        }

    async def get_samples(self, key: str, start: Any, end: Any, **_: Any) -> list[MeasurementSample]:
        """Return the stored samples of a thermal channel within the range."""
        if key not in self.series:
            raise ValueError(f"No measurement channel definition for {key!r}.")
        return [
            MeasurementSample(date_time=stamp, key=key, value=value)
            for stamp, value in self.series[key]
            if start <= stamp < end
        ]

    async def key_to_lists(self, key: str, *_: Any, **__: Any) -> tuple[list[Any], list[Any]]:
        """The electrical devices have no measurements: every key is unknown."""
        raise KeyError(key)


class GoldenPresence:
    """Presence by hour of day, as ``FakePresence`` of the snapshot's thermal tests."""

    def __init__(self, rule: dict[str, Any], timezone: Any) -> None:
        self.rule = rule
        self.timezone = timezone

    async def forecast_zone(
        self, zone: str, *, now: Any, starts: list[Any], **_: Any
    ) -> list[PresenceEstimate]:
        """Return one estimate per start."""
        if any(start < now for start in starts):  # like the real provider
            raise ValueError("Presence forecast requires prospective intervals")
        estimates = []
        for start in starts:
            hour = start.astimezone(self.timezone).hour
            away = self.rule["away_from_hour"] <= hour < self.rule["away_to_hour"]
            estimates.append(
                PresenceEstimate(
                    date_time=start,
                    probability=self.rule["probability_away" if away else "probability_home"],
                    require_comfort=not away,
                    source="history",
                )
            )
        return estimates


class GoldenPrediction:
    """A constant outdoor temperature for three days from local midnight of the first run."""

    def __init__(
        self, outdoor_c: Optional[float], presence: Optional[GoldenPresence], origin: DateTime
    ) -> None:
        self.outdoor_c = outdoor_c
        self.presence = presence
        self.origin = origin

    async def update_data(self) -> None:
        """Nothing to update."""

    async def key_to_raw_series(self, key: str, dropna: bool = False) -> pd.Series:
        """Return the hourly outdoor temperature; every other key is unknown."""
        if key != "weather_temp_air" or self.outdoor_c is None:
            raise KeyError(key)
        index = [self.origin + timedelta(hours=h) for h in range(72)]
        return pd.Series([self.outdoor_c] * 72, index=pd.DatetimeIndex(index))

    def provider_by_id(self, provider_id: str) -> Any:
        """Return the presence provider."""
        if self.presence is None:
            raise KeyError(provider_id)
        return self.presence


def mask(value: Any) -> Any:
    """Replace wall-clock and machine dependent values; keep everything the input decides."""
    if isinstance(value, dict):
        return {
            key: (MASK if key in VOLATILE_KEYS else mask(item)) for key, item in value.items()
        }
    if isinstance(value, list):
        return [mask(item) for item in value]
    if isinstance(value, str):
        return ID_TIME_PATTERN.sub("@<now>", UUID_PATTERN.sub("<uuid>", value))
    return value


class ReplayProxy:
    """Answer the calls recorded by ``scripts/capture_ohmrakel42_golden.py``.

    A call that was not recorded fails the test: the run asked for something else than the
    captured run did.
    """

    def __init__(self, prefix: str, calls: dict[str, Any]) -> None:
        self._prefix = prefix
        self._calls = calls

    def __getattr__(self, name: str) -> Any:
        qualified = f"{self._prefix}.{name}"
        if name == "provider_by_id":
            return lambda provider_id: ReplayProxy(f"{qualified}:{provider_id}", self._calls)
        kinds = {
            entry["async"]
            for key, entry in self._calls.items()
            if json.loads(key)["call"] == qualified
        }
        if not kinds:
            raise AttributeError(f"No call of {qualified} was recorded.")

        def answer(*args: Any, **kwargs: Any) -> Any:
            key = call_key(qualified, args, kwargs)
            if key not in self._calls:
                pytest.fail(f"Call not recorded in the capture: {key}")
            return self._answer(name, self._calls[key])

        if kinds == {True}:

            async def answer_async(*args: Any, **kwargs: Any) -> Any:
                return answer(*args, **kwargs)

            return answer_async
        return answer

    @staticmethod
    def _answer(name: str, entry: dict[str, Any]) -> Any:
        """Rebuild the recorded answer (or exception) of a call."""
        if "raises" in entry:
            error = getattr(builtins, entry["raises"], None)
            if not (isinstance(error, type) and issubclass(error, Exception)):
                error = RuntimeError
            raise error(entry["message"])
        value = entry["returns"]
        if entry["type"] == "Series":
            index = pd.to_datetime(value["index"], utc=True)
            values = [math.nan if item is None else item for item in value["values"]]
            return pd.Series(values, index=pd.DatetimeIndex(index), dtype=float)
        if name == "key_to_lists":
            dates, values = value
            return [to_datetime(stamp) for stamp in dates], values
        if name == "get_samples":
            return [MeasurementSample.model_validate(item) for item in value]
        if name == "forecast_zone":
            return [PresenceEstimate.model_validate(item) for item in value]
        return value


def case_names() -> list[str]:
    """The built cases and the captured ones (input files without a builder)."""
    stored = {
        path.name[len("golden_") : -len("_input.json")]
        for path in DIR_GOLDEN.glob("golden_*_input.json")
    }
    return sorted(set(CASE_BUILDERS) | stored)


def load_case(name: str) -> dict[str, Any]:
    """Return the input of a case, building and writing it first if it does not exist."""
    path = DIR_GOLDEN / f"golden_{name}_input.json"
    if not path.exists():
        DIR_GOLDEN.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(CASE_BUILDERS[name](), handle, indent=1)
            handle.write("\n")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


async def run_case(
    case: dict[str, Any],
    ems: EnergyManagement,
    config_eos: ConfigEOS,
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Run every run of a case through the energy management; return the outputs per run."""
    config_eos.merge_settings_from_dict(case["settings"])
    measurement: Any
    prediction: Any
    if "recorded_calls" in case:
        measurement = ReplayProxy("measurement", case["recorded_calls"])
        prediction = ReplayProxy("prediction", case["recorded_calls"])
    else:
        first_start = to_datetime(case["runs"][0]["start"], in_timezone=TIMEZONE)
        measurement = GoldenMeasurement(case["measurements"])
        presence = (
            GoldenPresence(case["presence"], first_start.timezone) if case["presence"] else None
        )
        prediction = GoldenPrediction(
            case["outdoor_temperature_c"], presence, first_start.start_of("day")
        )
    monkeypatch.setattr(ThermalResolver, "measurement", measurement)
    monkeypatch.setattr(ThermalResolver, "prediction", prediction)
    monkeypatch.setattr(DeviceResolver, "measurement", measurement)
    monkeypatch.setattr(PlanningRequest, "prediction", prediction)
    return await run_case_with_patched_world(case, ems)


async def run_case_with_patched_world(
    case: dict[str, Any], ems: EnergyManagement
) -> list[dict[str, Any]]:
    """Run every run of a case against the measurement and prediction patched in already."""
    outputs: list[dict[str, Any]] = []
    for run in case["runs"]:
        request = PlanningRequest.model_validate(run["request"])
        try:
            solution = await ems.run(
                start_datetime=to_datetime(run["start"], in_timezone=TIMEZONE),
                mode=EnergyManagementMode.OPTIMIZATION,
                ohmrakel42_request=request,
                changed_devices=run["changed_devices"],
            )
        except Exception as exc:  # the snapshot raises on invalid input; record the cause
            outputs.append({"error": f"{type(exc).__name__}: {exc}"})
            continue
        optimization_solution = ems.optimization_solution()
        plan = ems.plan()
        outputs.append(
            {
                "error": None,
                "solution": (
                    solution.model_dump(mode="json")
                    if isinstance(solution, Ohmrakel42Solution)
                    else None
                ),
                "optimization_solution": (
                    optimization_solution.model_dump(mode="json")
                    if optimization_solution
                    else None
                ),
                "plan": plan.model_dump(mode="json") if plan else None,
                "thermal_plans": [p.model_dump(mode="json") for p in ems.thermal_plans()],
            }
        )
    return mask(json.loads(json.dumps(outputs)))


@pytest.mark.asyncio
@pytest.mark.parametrize("name", case_names())
async def test_golden(
    name: str,
    ems_world: EnergyManagement,  # noqa: F811
    config_eos: ConfigEOS,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot gives the stored output for every golden input."""
    case = load_case(name)
    outputs = await run_case(case, ems_world, config_eos, monkeypatch)
    if "captured_outputs" in case:
        # The replay of a capture gives what the live run gave.
        assert outputs == mask(case["captured_outputs"])

    expected_path = DIR_GOLDEN / f"golden_{name}_expected.json"
    new_path = DIR_GOLDEN / f"new_golden_{name}_expected.json"
    expected: Any = None
    if expected_path.exists():
        with expected_path.open("r", encoding="utf-8") as handle:
            expected = json.load(handle)
    if outputs == expected:
        new_path.unlink(missing_ok=True)
        return
    with new_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(outputs, handle, indent=1, sort_keys=True)
        handle.write("\n")
    pytest.fail(
        f"Output of golden case {name!r} differs from {expected_path.name}.\n"
        f"Accept it with: mv {new_path} {expected_path}"
    )


@pytest.mark.asyncio
async def test_a_capture_replays_to_the_outputs_of_the_captured_run(
    ems_world: EnergyManagement,  # noqa: F811
    config_eos: ConfigEOS,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proxies of the capture script record a heat pump run that the replay repeats."""
    case = CASE_BUILDERS["heat_pump_3600"]()
    calls: dict[str, Any] = {}
    config_eos.merge_settings_from_dict(case["settings"])
    first_start = to_datetime(case["runs"][0]["start"], in_timezone=TIMEZONE)
    presence = GoldenPresence(case["presence"], first_start.timezone)
    measurement = Recorder(GoldenMeasurement(case["measurements"]), "measurement", calls)
    prediction = Recorder(
        GoldenPrediction(case["outdoor_temperature_c"], presence, first_start.start_of("day")),
        "prediction",
        calls,
    )
    with monkeypatch.context() as patch:
        patch.setattr(ThermalResolver, "measurement", measurement)
        patch.setattr(ThermalResolver, "prediction", prediction)
        patch.setattr(DeviceResolver, "measurement", measurement)
        patch.setattr(PlanningRequest, "prediction", prediction)
        captured = await run_case_with_patched_world(case, ems_world)
    assert any("forecast_zone" in key for key in calls)
    assert any("get_samples" in key for key in calls)

    replayed_case = {**case, "recorded_calls": json.loads(json.dumps(calls))}
    EnergyManagement._ohmrakel42_warm_start = None
    replayed = await run_case(replayed_case, ems_world, config_eos, monkeypatch)
    assert replayed == captured
    assert replayed[0]["error"] is None and replayed[0]["thermal_plans"]
