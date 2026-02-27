"""Tests for immutable device parameter dataclasses in devicesabc.py.

Covers:
    - Valid construction of every parameter type
    - All __post_init__ validation paths (one test per violated constraint)
    - Immutability enforcement (FrozenInstanceError on assignment)
    - Hashability and value equality contracts
    - Optional field defaults
    - Topology types: EnergyPort, EnergyBus, EnergyBusConstraint
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from akkudoktoreos.devices.devicesabc import (
    ApplianceOperationMode,
    BatteryOperationMode,
    BatteryParam,
    EnergyBus,
    EnergyBusConstraint,
    EnergyCarrier,
    EnergyPort,
    HeatPumpParam,
    InverterParam,
    PVParam,
)

# ============================================================
# Shared fixtures
# ============================================================


@pytest.fixture()
def ac_port() -> EnergyPort:
    return EnergyPort(port_id="p_ac", bus_id="bus_ac")


@pytest.fixture()
def dc_port() -> EnergyPort:
    return EnergyPort(port_id="p_dc", bus_id="bus_dc")


@pytest.fixture()
def heat_port() -> EnergyPort:
    return EnergyPort(port_id="p_heat", bus_id="bus_heat")


@pytest.fixture()
def minimal_battery(ac_port) -> BatteryParam:
    """A fully valid BatteryParam with sensible defaults."""
    return BatteryParam(
        device_id="bat_0",
        ports=(ac_port,),
        operation_modes=(BatteryOperationMode.SELF_CONSUMPTION,),
        capacity_wh=10_000.0,
        charging_efficiency=0.95,
        discharging_efficiency=0.95,
        levelized_cost_of_storage_kwh=0.05,
        max_charge_power_w=5_000.0,
        min_charge_power_w=100.0,
        charge_rates=None,
        min_soc_factor=0.1,
        max_soc_factor=0.9,
    )


@pytest.fixture()
def minimal_inverter(ac_port, dc_port) -> InverterParam:
    return InverterParam(
        device_id="inv_0",
        ports=(dc_port, ac_port),
        max_power_w=6_000.0,
        efficiency=0.97,
    )


@pytest.fixture()
def minimal_pv(dc_port) -> PVParam:
    return PVParam(
        device_id="pv_0",
        ports=(dc_port,),
        peak_power_w=8_000.0,
        tilt_deg=30.0,
        azimuth_deg=180.0,
    )


@pytest.fixture()
def minimal_heat_pump(heat_port) -> HeatPumpParam:
    return HeatPumpParam(
        device_id="hp_0",
        ports=(heat_port,),
        operation_modes=(ApplianceOperationMode.RUN,),
        thermal_power_w=9_000.0,
        cop=3.5,
    )


# ============================================================
# EnergyPort
# ============================================================


class TestEnergyPort:
    def test_valid_construction_minimal(self):
        port = EnergyPort(port_id="p1", bus_id="bus1")
        assert port.port_id == "p1"
        assert port.bus_id == "bus1"
        assert port.max_power_w is None

    def test_valid_construction_with_power_limit(self):
        port = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=3_000.0)
        assert port.max_power_w == 3_000.0

    def test_immutable(self):
        port = EnergyPort(port_id="p1", bus_id="bus1")
        with pytest.raises(FrozenInstanceError):
            port.port_id = "other"

    def test_equal_instances_are_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=1_000.0)
        b = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=1_000.0)
        assert a == b

    def test_equal_instances_hash_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=1_000.0)
        b = EnergyPort(port_id="p1", bus_id="bus1", max_power_w=1_000.0)
        assert hash(a) == hash(b)

    def test_different_instances_are_not_equal(self):
        a = EnergyPort(port_id="p1", bus_id="bus1")
        b = EnergyPort(port_id="p2", bus_id="bus1")
        assert a != b

    def test_usable_as_dict_key(self):
        port = EnergyPort(port_id="p1", bus_id="bus1")
        d = {port: "value"}
        assert d[port] == "value"

    def test_usable_as_set_member(self):
        a = EnergyPort(port_id="p1", bus_id="bus1")
        b = EnergyPort(port_id="p1", bus_id="bus1")
        assert len({a, b}) == 1


# ============================================================
# EnergyBusConstraint
# ============================================================


class TestEnergyBusConstraint:
    def test_valid_construction_all_none(self):
        c = EnergyBusConstraint()
        assert c.max_sinks is None
        assert c.max_sources is None

    def test_valid_construction_with_values(self):
        c = EnergyBusConstraint(max_sinks=3, max_sources=1)
        assert c.max_sinks == 3
        assert c.max_sources == 1

    def test_immutable(self):
        c = EnergyBusConstraint(max_sinks=2)
        with pytest.raises(FrozenInstanceError):
            c.max_sinks = 5

    def test_equality(self):
        assert EnergyBusConstraint(max_sinks=2) == EnergyBusConstraint(max_sinks=2)
        assert EnergyBusConstraint(max_sinks=2) != EnergyBusConstraint(max_sinks=3)

    def test_hashable(self):
        a = EnergyBusConstraint(max_sinks=1, max_sources=1)
        b = EnergyBusConstraint(max_sinks=1, max_sources=1)
        assert hash(a) == hash(b)


# ============================================================
# EnergyBus
# ============================================================


class TestEnergyBus:
    def test_valid_construction_minimal(self):
        bus = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        assert bus.bus_id == "bus_ac"
        assert bus.carrier == EnergyCarrier.AC
        assert bus.constraint is None

    def test_valid_construction_with_constraint(self):
        constraint = EnergyBusConstraint(max_sinks=4)
        bus = EnergyBus(bus_id="bus_dc", carrier=EnergyCarrier.DC, constraint=constraint)
        assert bus.constraint.max_sinks == 4

    def test_all_carriers_accepted(self):
        for carrier in EnergyCarrier:
            bus = EnergyBus(bus_id=f"bus_{carrier.value}", carrier=carrier)
            assert bus.carrier == carrier

    def test_immutable(self):
        bus = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        with pytest.raises(FrozenInstanceError):
            bus.bus_id = "other"

    def test_equality(self):
        a = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        b = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        assert a == b

    def test_hashable(self):
        a = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        b = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self):
        bus = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)
        d = {bus: 42}
        assert d[bus] == 42


# ============================================================
# BatteryParam
# ============================================================


class TestBatteryParamConstruction:
    def test_valid_minimal(self, minimal_battery):
        assert minimal_battery.device_id == "bat_0"
        assert minimal_battery.capacity_wh == 10_000.0
        assert minimal_battery.min_soc_factor == 0.1
        assert minimal_battery.max_soc_factor == 0.9

    def test_valid_with_charge_rates(self, ac_port):
        bat = BatteryParam(
            device_id="bat_1",
            ports=(ac_port,),
            operation_modes=(BatteryOperationMode.SELF_CONSUMPTION,),
            capacity_wh=5_000.0,
            charging_efficiency=0.9,
            discharging_efficiency=0.9,
            levelized_cost_of_storage_kwh=0.1,
            max_charge_power_w=3_000.0,
            min_charge_power_w=0.0,
            charge_rates=(0.25, 0.5, 0.75, 1.0),
            min_soc_factor=0.0,
            max_soc_factor=1.0,
        )
        assert bat.charge_rates == (0.25, 0.5, 0.75, 1.0)

    def test_multiple_operation_modes_accepted(self, ac_port):
        bat = BatteryParam(
            device_id="bat_2",
            ports=(ac_port,),
            operation_modes=(
                BatteryOperationMode.SELF_CONSUMPTION,
                BatteryOperationMode.PEAK_SHAVING,
            ),
            capacity_wh=5_000.0,
            charging_efficiency=0.9,
            discharging_efficiency=0.9,
            levelized_cost_of_storage_kwh=0.1,
            max_charge_power_w=3_000.0,
            min_charge_power_w=0.0,
            charge_rates=None,
            min_soc_factor=0.0,
            max_soc_factor=1.0,
        )
        assert len(bat.operation_modes) == 2

    def test_immutable(self, minimal_battery):
        with pytest.raises(FrozenInstanceError):
            minimal_battery.capacity_wh = 999.0

    def test_equality_and_hash(self, ac_port):
        kwargs = dict(
            device_id="bat_x",
            ports=(ac_port,),
            operation_modes=(BatteryOperationMode.IDLE,),
            capacity_wh=1_000.0,
            charging_efficiency=0.9,
            discharging_efficiency=0.9,
            levelized_cost_of_storage_kwh=0.05,
            max_charge_power_w=500.0,
            min_charge_power_w=0.0,
            charge_rates=None,
            min_soc_factor=0.0,
            max_soc_factor=1.0,
        )
        a = BatteryParam(**kwargs)
        b = BatteryParam(**kwargs)
        assert a == b
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self, minimal_battery):
        d = {minimal_battery: "cached_result"}
        assert d[minimal_battery] == "cached_result"


class TestBatteryParamValidation:
    """One test per violated constraint in BatteryParam.__post_init__."""

    def _make(self, ac_port, **overrides):
        """Build a valid BatteryParam with selected fields overridden."""
        defaults = dict(
            device_id="bat_val",
            ports=(ac_port,),
            operation_modes=(BatteryOperationMode.SELF_CONSUMPTION,),
            capacity_wh=10_000.0,
            charging_efficiency=0.95,
            discharging_efficiency=0.95,
            levelized_cost_of_storage_kwh=0.05,
            max_charge_power_w=5_000.0,
            min_charge_power_w=100.0,
            charge_rates=None,
            min_soc_factor=0.1,
            max_soc_factor=0.9,
        )
        defaults.update(overrides)
        return BatteryParam(**defaults)

    def test_capacity_zero_raises(self, ac_port):
        with pytest.raises(ValueError, match="capacity_wh"):
            self._make(ac_port, capacity_wh=0.0)

    def test_capacity_negative_raises(self, ac_port):
        with pytest.raises(ValueError, match="capacity_wh"):
            self._make(ac_port, capacity_wh=-1.0)

    def test_max_charge_power_zero_raises(self, ac_port):
        with pytest.raises(ValueError, match="max_charge_power_w"):
            self._make(ac_port, max_charge_power_w=0.0)

    def test_max_charge_power_negative_raises(self, ac_port):
        with pytest.raises(ValueError, match="max_charge_power_w"):
            self._make(ac_port, max_charge_power_w=-100.0)

    def test_min_charge_power_negative_raises(self, ac_port):
        with pytest.raises(ValueError, match="min_charge_power_w"):
            self._make(ac_port, min_charge_power_w=-1.0)

    def test_min_charge_power_exceeds_max_raises(self, ac_port):
        with pytest.raises(ValueError, match="min_charge_power_w"):
            self._make(ac_port, min_charge_power_w=6_000.0, max_charge_power_w=5_000.0)

    def test_charging_efficiency_zero_raises(self, ac_port):
        with pytest.raises(ValueError, match="charging_efficiency"):
            self._make(ac_port, charging_efficiency=0.0)

    def test_charging_efficiency_above_one_raises(self, ac_port):
        with pytest.raises(ValueError, match="charging_efficiency"):
            self._make(ac_port, charging_efficiency=1.01)

    def test_discharging_efficiency_zero_raises(self, ac_port):
        with pytest.raises(ValueError, match="discharging_efficiency"):
            self._make(ac_port, discharging_efficiency=0.0)

    def test_discharging_efficiency_above_one_raises(self, ac_port):
        with pytest.raises(ValueError, match="discharging_efficiency"):
            self._make(ac_port, discharging_efficiency=1.5)

    def test_min_soc_equals_max_soc_raises(self, ac_port):
        with pytest.raises(ValueError, match="SoC factors"):
            self._make(ac_port, min_soc_factor=0.5, max_soc_factor=0.5)

    def test_min_soc_exceeds_max_soc_raises(self, ac_port):
        with pytest.raises(ValueError, match="SoC factors"):
            self._make(ac_port, min_soc_factor=0.8, max_soc_factor=0.2)

    def test_min_soc_below_zero_raises(self, ac_port):
        with pytest.raises(ValueError, match="SoC factors"):
            self._make(ac_port, min_soc_factor=-0.1, max_soc_factor=0.9)

    def test_max_soc_above_one_raises(self, ac_port):
        with pytest.raises(ValueError, match="SoC factors"):
            self._make(ac_port, min_soc_factor=0.1, max_soc_factor=1.1)

    def test_empty_operation_modes_raises(self, ac_port):
        with pytest.raises(ValueError, match="operation_modes"):
            self._make(ac_port, operation_modes=())

    def test_boundary_efficiency_one_accepted(self, ac_port):
        """Efficiency of exactly 1.0 should be valid."""
        bat = self._make(ac_port, charging_efficiency=1.0, discharging_efficiency=1.0)
        assert bat.charging_efficiency == 1.0

    def test_boundary_soc_zero_and_one_accepted(self, ac_port):
        """Full SoC range [0.0, 1.0] should be valid."""
        bat = self._make(ac_port, min_soc_factor=0.0, max_soc_factor=1.0)
        assert bat.min_soc_factor == 0.0
        assert bat.max_soc_factor == 1.0

    def test_min_charge_power_zero_accepted(self, ac_port):
        """min_charge_power_w of 0.0 should be valid."""
        bat = self._make(ac_port, min_charge_power_w=0.0)
        assert bat.min_charge_power_w == 0.0


# ============================================================
# InverterParam
# ============================================================


class TestInverterParamConstruction:
    def test_valid_construction(self, minimal_inverter):
        assert minimal_inverter.device_id == "inv_0"
        assert minimal_inverter.max_power_w == 6_000.0
        assert minimal_inverter.efficiency == 0.97

    def test_immutable(self, minimal_inverter):
        with pytest.raises(FrozenInstanceError):
            minimal_inverter.max_power_w = 1.0

    def test_equality_and_hash(self, ac_port, dc_port):
        kwargs = dict(
            device_id="inv_x",
            ports=(dc_port, ac_port),
            max_power_w=5_000.0,
            efficiency=0.96,
        )
        a = InverterParam(**kwargs)
        b = InverterParam(**kwargs)
        assert a == b
        assert hash(a) == hash(b)


class TestInverterParamValidation:
    def _make(self, ac_port, dc_port, **overrides):
        defaults = dict(
            device_id="inv_val",
            ports=(dc_port, ac_port),
            max_power_w=5_000.0,
            efficiency=0.97,
        )
        defaults.update(overrides)
        return InverterParam(**defaults)

    def test_max_power_zero_raises(self, ac_port, dc_port):
        with pytest.raises(ValueError, match="max_power_w"):
            self._make(ac_port, dc_port, max_power_w=0.0)

    def test_max_power_negative_raises(self, ac_port, dc_port):
        with pytest.raises(ValueError, match="max_power_w"):
            self._make(ac_port, dc_port, max_power_w=-1.0)

    def test_efficiency_zero_raises(self, ac_port, dc_port):
        with pytest.raises(ValueError, match="efficiency"):
            self._make(ac_port, dc_port, efficiency=0.0)

    def test_efficiency_above_one_raises(self, ac_port, dc_port):
        with pytest.raises(ValueError, match="efficiency"):
            self._make(ac_port, dc_port, efficiency=1.001)

    def test_boundary_efficiency_one_accepted(self, ac_port, dc_port):
        inv = self._make(ac_port, dc_port, efficiency=1.0)
        assert inv.efficiency == 1.0


# ============================================================
# PVParam
# ============================================================


class TestPVParamConstruction:
    def test_valid_construction(self, minimal_pv):
        assert minimal_pv.device_id == "pv_0"
        assert minimal_pv.peak_power_w == 8_000.0
        assert minimal_pv.tilt_deg == 30.0
        assert minimal_pv.azimuth_deg == 180.0

    def test_immutable(self, minimal_pv):
        with pytest.raises(FrozenInstanceError):
            minimal_pv.peak_power_w = 1.0

    def test_equality_and_hash(self, dc_port):
        kwargs = dict(
            device_id="pv_x",
            ports=(dc_port,),
            peak_power_w=4_000.0,
            tilt_deg=15.0,
            azimuth_deg=90.0,
        )
        a = PVParam(**kwargs)
        b = PVParam(**kwargs)
        assert a == b
        assert hash(a) == hash(b)

    def test_azimuth_full_range_accepted(self, dc_port):
        """Azimuth values across the full compass range should be valid."""
        for azimuth in (0.0, 90.0, 180.0, 270.0, 359.9):
            pv = PVParam(
                device_id="pv_az",
                ports=(dc_port,),
                peak_power_w=1_000.0,
                tilt_deg=30.0,
                azimuth_deg=azimuth,
            )
            assert pv.azimuth_deg == azimuth


class TestPVParamValidation:
    def _make(self, dc_port, **overrides):
        defaults = dict(
            device_id="pv_val",
            ports=(dc_port,),
            peak_power_w=8_000.0,
            tilt_deg=30.0,
            azimuth_deg=180.0,
        )
        defaults.update(overrides)
        return PVParam(**defaults)

    def test_peak_power_zero_raises(self, dc_port):
        with pytest.raises(ValueError, match="peak_power_w"):
            self._make(dc_port, peak_power_w=0.0)

    def test_peak_power_negative_raises(self, dc_port):
        with pytest.raises(ValueError, match="peak_power_w"):
            self._make(dc_port, peak_power_w=-500.0)


# ============================================================
# HeatPumpParam
# ============================================================


class TestHeatPumpParamConstruction:
    def test_valid_construction(self, minimal_heat_pump):
        assert minimal_heat_pump.device_id == "hp_0"
        assert minimal_heat_pump.thermal_power_w == 9_000.0
        assert minimal_heat_pump.cop == 3.5

    def test_immutable(self, minimal_heat_pump):
        with pytest.raises(FrozenInstanceError):
            minimal_heat_pump.cop = 1.0

    def test_equality_and_hash(self, heat_port):
        kwargs = dict(
            device_id="hp_x",
            ports=(heat_port,),
            operation_modes=(ApplianceOperationMode.RUN,),
            thermal_power_w=6_000.0,
            cop=4.0,
        )
        a = HeatPumpParam(**kwargs)
        b = HeatPumpParam(**kwargs)
        assert a == b
        assert hash(a) == hash(b)


class TestHeatPumpParamValidation:
    def _make(self, heat_port, **overrides):
        defaults = dict(
            device_id="hp_val",
            ports=(heat_port,),
            operation_modes=(ApplianceOperationMode.RUN,),
            thermal_power_w=9_000.0,
            cop=3.5,
        )
        defaults.update(overrides)
        return HeatPumpParam(**defaults)

    def test_thermal_power_zero_raises(self, heat_port):
        with pytest.raises(ValueError, match="thermal_power_w"):
            self._make(heat_port, thermal_power_w=0.0)

    def test_thermal_power_negative_raises(self, heat_port):
        with pytest.raises(ValueError, match="thermal_power_w"):
            self._make(heat_port, thermal_power_w=-1.0)

    def test_cop_zero_raises(self, heat_port):
        with pytest.raises(ValueError, match="cop"):
            self._make(heat_port, cop=0.0)

    def test_cop_negative_raises(self, heat_port):
        with pytest.raises(ValueError, match="cop"):
            self._make(heat_port, cop=-1.0)

    def test_empty_operation_modes_raises(self, heat_port):
        with pytest.raises(ValueError, match="operation_modes"):
            self._make(heat_port, operation_modes=())

    def test_multiple_operation_modes_accepted(self, heat_port):
        hp = self._make(
            heat_port,
            operation_modes=(ApplianceOperationMode.RUN, ApplianceOperationMode.LIMIT_POWER),
        )
        assert len(hp.operation_modes) == 2
