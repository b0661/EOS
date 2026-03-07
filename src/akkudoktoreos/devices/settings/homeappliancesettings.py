"""Controllable home appliance device settings."""

from typing import Optional

from pydantic import Field, computed_field, model_validator

from akkudoktoreos.utils.datetimeutil import TimeWindowSequence

from akkudoktoreos.devices.settings.devicebasesettings import DevicesBaseSettings, PortsMixin


class HomeApplianceCommonSettings(PortsMixin, DevicesBaseSettings):
    """Controllable home appliance device settings.

    Represents a shiftable load whose start time — and optionally the
    start times for multiple sequential runs — can be deferred by the
    optimiser within per-cycle allowed time windows.

    The number of remaining cycles to plan is determined at runtime by
    reading ``cycles_completed_measurement_key`` from the measurement
    store inside ``HomeApplianceDevice.setup_run``.

    Single-cycle example (backward-compatible)
    ------------------------------------------
    ::

        device_id: dishwasher
        consumption_wh: 1500
        duration_h: 2
        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: sink
        cycle_time_windows:
          - windows:
              - start_time: "10:00"
                duration: "12 hours"

    Multi-cycle example (two runs, per-cycle windows, gap enforced)
    ---------------------------------------------------------------
    ::

        device_id: washing_machine
        consumption_wh: 2000
        duration_h: 2
        min_cycle_gap_h: 1
        ports:
          - port_id: p_ac
            bus_id: bus_ac
            direction: sink
        cycle_time_windows:
          - windows:                        # first run: morning
              - start_time: "07:00"
                duration: "5 hours"
          - windows:                        # second run: evening
              - start_time: "17:00"
                duration: "5 hours"

    Backward-compatible single-window example
    ------------------------------------------
    The old ``time_windows`` field is still accepted and automatically
    promoted to a single-entry ``cycle_time_windows``::

        device_id: dishwasher
        consumption_wh: 1500
        duration_h: 2
        time_windows:
          windows:
            - start_time: "10:00"
              duration: "12 hours"

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
            "description": "Run duration per cycle [h] (1–24).",
            "examples": [2],
        },
    )
    cycle_time_windows: list[Optional[TimeWindowSequence]] = Field(
        default_factory=lambda: [None],
        json_schema_extra={
            "description": (
                "Per-cycle allowed scheduling time windows. "
                "One entry per required run; the list length defines num_cycles. "
                "A null entry means any step within the horizon is valid for that cycle."
            ),
            "examples": [
                [None],
                [
                    {"windows": [{"start_time": "07:00", "duration": "5 hours"}]},
                    {"windows": [{"start_time": "17:00", "duration": "5 hours"}]},
                ],
            ],
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
        },
    )

    # ------------------------------------------------------------------
    # Backward-compatibility shim for old single time_windows field
    # ------------------------------------------------------------------

    time_windows: Optional[TimeWindowSequence] = Field(
        default=None,
        exclude=True,   # never serialised; consumed on input only
        json_schema_extra={
            "description": (
                "Deprecated. Single allowed time window for a one-cycle appliance. "
                "Automatically promoted to cycle_time_windows. "
                "Use cycle_time_windows for new configurations."
            ),
        },
    )

    @model_validator(mode="after")
    def _promote_legacy_time_windows(self) -> "HomeApplianceCommonSettings":
        """Fold the old ``time_windows`` into ``cycle_time_windows``."""
        if self.time_windows is not None:
            if self.cycle_time_windows == [None]:
                self.cycle_time_windows = [self.time_windows]
            self.time_windows = None
        return self

    @model_validator(mode="after")
    def _validate_cycle_windows_not_empty(self) -> "HomeApplianceCommonSettings":
        if not self.cycle_time_windows:
            raise ValueError(
                "cycle_time_windows must contain at least one entry. "
                "Use [null] to allow any start time for a single-cycle appliance."
            )
        return self

    # ------------------------------------------------------------------
    # GENETIC2 domain conversion
    # ------------------------------------------------------------------

    def to_genetic2_param(self):
        """Return an immutable ``HomeApplianceParam`` for the GENETIC2 optimizer.

        ``TimeWindowSequence`` objects in ``cycle_time_windows`` are converted
        to tuples of ``CycleTimeWindow`` — a hashable, comparable value type —
        so that ``HomeApplianceParam`` is a proper value object suitable for
        use as a dictionary or cache key.

        Returns:
            HomeApplianceParam
        """
        # Import here to avoid a top-level dependency on the genetic2 layer.
        from akkudoktoreos.simulation.genetic2.homeappliance import (
            HomeApplianceParam,
            cycle_windows_from_sequence,
        )

        return HomeApplianceParam(
            device_id=self.device_id,
            ports=self._domain_ports(),
            consumption_wh=float(self.consumption_wh),
            duration_h=self.duration_h,
            num_cycles=len(self.cycle_time_windows),
            min_cycle_gap_h=self.min_cycle_gap_h,
            cycles_completed_measurement_key=(
                self.cycles_completed_measurement_key
                or f"{self.device_id}.cycles_completed"
            ),
            cycle_time_windows=tuple(
                cycle_windows_from_sequence(w) for w in self.cycle_time_windows
            ),
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
        key = (
            self.cycles_completed_measurement_key
            or f"{self.device_id}.cycles_completed"
        )
        return [key]
