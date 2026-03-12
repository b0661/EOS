"""Controllable home appliance device settings."""

from typing import TYPE_CHECKING, Optional

from pydantic import Field, computed_field, model_validator

from akkudoktoreos.config.configabc import ConfigScope, CycleTimeWindowSequence
from akkudoktoreos.devices.settings.devicebasesettings import (
    DevicesBaseSettings,
    PortsMixin,
)

if TYPE_CHECKING:
    from akkudoktoreos.devices.genetic2.homeapplianc import HomeApplianceParam
    from akkudoktoreos.devices.genetic.homeappliance import HomeApplianceParameters


class HomeApplianceCommonSettings(PortsMixin, DevicesBaseSettings):
    """Controllable home appliance device settings.

    Represents a shiftable load whose start time — and optionally the
    start times for multiple sequential runs — can be deferred by the
    optimiser within per-cycle allowed time windows.

    The number of remaining cycles to plan is determined at runtime by
    reading ``cycles_completed_measurement_key`` from the measurement
    store inside ``HomeApplianceDevice.setup_run``.

    ``num_cycles`` is required when ``cycle_time_windows`` is ``None``
    (unconstrained); when windows are provided it is derived from
    ``cycle_time_windows.num_cycles()``.

    Single-cycle, unconstrained (start any time)
    ---------------------------------------------

    ::

        device_id: dishwasher
        consumption_wh: 1500
        duration_h: 2
        num_cycles: 1
        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: sink

    Single-cycle, constrained to one window
    ----------------------------------------

    Each window's ``value`` field carries the **cycle index** (0-based).
    Windows without a ``value`` are ignored by the optimizer::

        device_id: dishwasher
        consumption_wh: 1500
        duration_h: 2
        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: sink
        cycle_time_windows:
          windows:
            - start_time: "10:00"
              duration: "12 hours"
              value: 0

    Multi-cycle, per-cycle windows
    --------------------------------

    Two cycles, each with its own window.  Cycle 0 runs in the morning,
    cycle 1 in the evening::

        device_id: washing_machine
        consumption_wh: 2000
        duration_h: 2
        min_cycle_gap_h: 1
        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: sink
        cycle_time_windows:
          windows:
            - start_time: "07:00"
              duration: "5 hours"
              value: 0
            - start_time: "17:00"
              duration: "5 hours"
              value: 1

    Multi-cycle, shared window (both cycles may run any time 10:00-20:00)
    -----------------------------------------------------------------------

    Assign the same-shaped windows to distinct cycle indices so the
    optimizer can place them independently::

        cycle_time_windows:
          windows:
            - start_time: "10:00"
              duration: "10 hours"
              value: 0
            - start_time: "10:00"
              duration: "10 hours"
              value: 1

    Port wiring guidance
    --------------------

    Standard AC appliance::

        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: sink
    """

    consumption_wh: int = Field(
        ...,
        gt=0,
        json_schema_extra={
            "description": "Energy consumption per run cycle [Wh].",
            "examples": [2000],
        },
    )
    duration_h: int = Field(
        ...,
        gt=0,
        le=24,
        json_schema_extra={
            "description": "Run duration per cycle [h] (1-24).",
            "examples": [2],
        },
    )
    num_cycles: int = Field(
        default=1,
        ge=1,
        json_schema_extra={
            "description": (
                "Number of times the appliance must run within the horizon. "
                "Required when cycle_time_windows is null (unconstrained). "
                "Ignored when cycle_time_windows is provided -- the number "
                "of distinct cycle indices in the windows defines num_cycles. "
                "Defaults to 1."
            ),
            "examples": [1, 2],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    cycle_time_windows: Optional[CycleTimeWindowSequence] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Per-cycle allowed scheduling time windows. "
                "Each window's value field specifies the cycle index (0-based). "
                "When null, the appliance may start at any step and num_cycles "
                "must be set explicitly."
            ),
            "examples": [
                None,
                {
                    "windows": [
                        {"start_time": "07:00", "duration": "5 hours", "value": 0},
                        {"start_time": "17:00", "duration": "5 hours", "value": 1},
                    ]
                },
            ],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    min_cycle_gap_h: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": (
                "Minimum idle time between the end of one cycle and the start "
                "of the next [h]. Applies uniformly between all consecutive cycles. "
                "0 means back-to-back runs are permitted."
            ),
            "examples": [0, 1, 4],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )
    cycles_completed_measurement_key: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Measurement store key holding the number of cycles already "
                "completed in the current planning day. Read by "
                "HomeApplianceDevice.setup_run via context.resolve_measurement. "
                "Defaults to '{device_id}.cycles_completed' when null."
            ),
            "examples": ["dishwasher.cycles_completed", None],
            "x-scope": [str(ConfigScope.GENETIC2)],
        },
    )

    @model_validator(mode="after")
    def _validate_num_cycles_specified(self) -> "HomeApplianceCommonSettings":
        """Require num_cycles when windows are not provided."""
        if self.cycle_time_windows is None and self.num_cycles is None:
            raise ValueError(
                "num_cycles must be set when cycle_time_windows is null. "
                "Provide either cycle_time_windows (windows define cycle count) "
                "or set num_cycles explicitly."
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_num_cycles(self) -> int:
        """Number of cycles as seen by the optimizer.

        Derived from cycle_time_windows.num_cycles() when windows are
        provided; falls back to the explicit num_cycles field otherwise.
        """
        if self.cycle_time_windows is not None:
            return self.cycle_time_windows.num_cycles()
        return self.num_cycles

    # ------------------------------------------------------------------
    # GENETIC domain conversion
    # ------------------------------------------------------------------

    def to_genetic_param(self) -> "HomeApplianceParameters":
        """Return HomeApplianceParameters for the GENETIC optimizer."""
        from akkudoktoreos.devices.genetic.homeappliance import HomeApplianceParameters

        return HomeApplianceParameters(
            device_id=self.device_id,
            consumption_wh=float(self.consumption_wh),
            duration_h=self.duration_h,
            time_windows=self.cycle_time_windows,
        )

    # ------------------------------------------------------------------
    # GENETIC2 domain conversion
    # ------------------------------------------------------------------

    def to_genetic2_param(self) -> "HomeApplianceParam":
        """Return an immutable HomeApplianceParam for the GENETIC2 optimizer.

        CycleTimeWindowSequence objects are not stored directly in the param
        (Pydantic models are not hashable and cannot live in a frozen dataclass).
        Instead, a time_window_key config path is built from config_index
        so that HomeApplianceDevice.setup_run can retrieve the sequence via
        context.resolve_config_cycle_time_windows::

            param  = settings.to_genetic2_param(index)
            device = HomeApplianceDevice(param, device_index, port_index)

        When cycle_time_windows is None (unconstrained), time_window_key
        is set to None and no config lookup is performed at runtime.

        Returns:
            HomeApplianceParam
        """
        from akkudoktoreos.devices.genetic2.homeapplianc import HomeApplianceParam

        time_window_key = (
            f"devices/home_appliances/{self.device_id}/cycle_time_windows"
            if self.cycle_time_windows is not None
            else None
        )

        return HomeApplianceParam(
            device_id=self.device_id,
            ports=self.ports_to_genetic2_param(),
            consumption_wh=float(self.consumption_wh),
            duration_h=self.duration_h,
            num_cycles=self.effective_num_cycles,
            min_cycle_gap_h=self.min_cycle_gap_h,
            cycles_completed_measurement_key=(
                self.cycles_completed_measurement_key or f"{self.device_id}.cycles_completed"
            ),
            time_window_key=time_window_key,
        )

    # ------------------------------------------------------------------
    # Measurement keys
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measurement_keys(self) -> list[str]:
        """Measurement keys this appliance reads.

        Exposes the completed-cycles key so the broader system knows to
        populate it after each run completes.
        """
        key = self.cycles_completed_measurement_key or f"{self.device_id}.cycles_completed"
        return [key]
