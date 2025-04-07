import pytest

from akkudoktoreos.simulation.simulationabc import (
    ID,
    Commodity,
    CommodityQuantity,
    ControlType,
    DeviceControlInstruction,
    EnergyManagementPlan,
    InstructionStatus,
    InstructionStatusUpdate,
    NumberRange,
    PowerForecastValue,
    PowerMeasurement,
    PowerRange,
    PowerValue,
    ReceptionStatus,
    Role,
    RoleType,
    SimulationDetails,
    Timer,
    Transition,
)
from akkudoktoreos.utils.datetimeutil import compare_datetimes, to_datetime, to_duration


# ---------- Test PowerValue ----------
class TestPowerValue:
    def test_valid_power_value(self):
        pv = PowerValue(commodity_quantity=CommodityQuantity.ELECTRIC_POWER_L1, value=123.45)
        assert pv.value == 123.45
        assert pv.commodity_quantity == CommodityQuantity.ELECTRIC_POWER_L1


# ---------- Test PowerForecastValue ----------
class TestPowerForecastValue:
    def test_valid_forecast_value(self):
        pf = PowerForecastValue(
            value_expected=200.0, commodity_quantity=CommodityQuantity.ELECTRIC_POWER_L1
        )
        assert pf.value_expected == 200.0
        assert pf.commodity_quantity == CommodityQuantity.ELECTRIC_POWER_L1


# ---------- Test PowerRange ----------
class TestPowerRange:
    def test_valid_power_range(self):
        pr = PowerRange(
            start_of_range=50.0,
            end_of_range=150.0,
            commodity_quantity=CommodityQuantity.ELECTRIC_POWER_L1,
        )
        assert pr.start_of_range == 50.0
        assert pr.end_of_range == 150.0


# ---------- Test NumberRange ----------
class TestNumberRange:
    def test_valid_number_range(self):
        nr = NumberRange(start_of_range=10.0, end_of_range=20.0)
        assert nr.start_of_range == 10.0
        assert nr.end_of_range == 20.0


# ---------- Test PowerMeasurement ----------
class TestPowerMeasurement:
    def test_valid_power_measurement(self):
        ts = to_datetime()
        pv = PowerValue(commodity_quantity=CommodityQuantity.ELECTRIC_POWER_L1, value=100.0)
        pm = PowerMeasurement(measurement_timestamp=ts, values=[pv])
        assert compare_datetimes(pm.measurement_timestamp, ts).equal
        assert len(pm.values) == 1


# ---------- Test Role ----------
class TestRole:
    def test_valid_role(self):
        r = Role(role=RoleType.ENERGY_PRODUCER, commodity=Commodity.ELECTRICITY)
        assert r.role == RoleType.ENERGY_PRODUCER
        assert r.commodity == Commodity.ELECTRICITY


# ---------- Test ReceptionStatus ----------
class TestReceptionStatus:
    def test_valid_reception_status(self):
        rs = ReceptionStatus(
            status="SUCCEEDED",  # Assuming the enum converts from string correctly
            diagnostic_label="All good",
        )
        # Depending on how ReceptionStatus is defined, you might compare with an enum member
        assert rs.status == "SUCCEEDED"
        assert rs.diagnostic_label == "All good"


# ---------- Test Transition ----------
class TestTransition:
    def test_valid_transition(self):
        trans = Transition(
            id="T1",
            from_="OM1",
            to="OM2",
            start_timers=["timer1"],
            blocking_timers=["timer2"],
            transition_costs=50.0,
            transition_duration=to_duration("5 minutes"),
            abnormal_condition_only=False,
        )
        assert trans.id == "T1"
        assert trans.from_ == "OM1"
        assert trans.to == "OM2"
        assert trans.transition_costs == 50.0


# ---------- Test Timer ----------
class TestTimer:
    def test_valid_timer(self):
        now = to_datetime()
        timer = Timer(
            id="Timer1",
            diagnostic_label="Timer for test",
            duration=to_duration("10 minutes"),
            finished_at=now,
        )
        assert timer.id == "Timer1"
        assert timer.diagnostic_label == "Timer for test"
        assert timer.duration == to_duration("10 minutes")
        assert timer.finished_at == now


# ---------- Test InstructionStatusUpdate ----------
class TestInstructionStatusUpdate:
    def test_valid_instruction_status_update(self):
        ts = to_datetime()
        isu = InstructionStatusUpdate(
            instruction_id="I1", status_type=InstructionStatus.NEW, timestamp=ts
        )
        assert isu.instruction_id == "I1"
        assert isu.status_type == InstructionStatus.NEW
        assert isu.timestamp == ts


# ---------- Test SimulationDetails ----------
class TestSimulationDetails:
    def test_valid_simulation_details(self):
        # Create a SimulationDetails instance with required and some optional fields
        sim_details = SimulationDetails(
            simulation_id="SIM001",
            roles=[
                Role(role=RoleType.ENERGY_PRODUCER, commodity=Commodity.ELECTRICITY),
                Role(role=RoleType.ENERGY_CONSUMER, commodity=Commodity.ELECTRICITY),
            ],
            instruction_processing_delay=to_duration("2 seconds"),
            available_control_types=[
                ControlType.OPERATION_MODE_BASED_CONTROL,
                ControlType.FILL_RATE_BASED_CONTROL,
            ],
            provides_forecast=True,
            provides_power_measurement_types=[
                CommodityQuantity.ELECTRIC_POWER_L1,
                CommodityQuantity.HEAT_TEMPERATURE,
            ],
            # Optional fields
            name="Home Battery System",
            manufacturer="EnergyTech",
            model="PowerWall X1",
            serial_number="ET-PWX1-12345",
            currency="EUR",  # Assuming Currency is a string enum
        )

        # Assert required fields are set correctly
        assert sim_details.simulation_id == "SIM001"
        assert len(sim_details.roles) == 2

        # Check first role
        assert sim_details.roles[0].role == RoleType.ENERGY_PRODUCER
        assert sim_details.roles[0].commodity == Commodity.ELECTRICITY

        # Check second role
        assert sim_details.roles[1].role == RoleType.ENERGY_CONSUMER
        assert sim_details.roles[1].commodity == Commodity.ELECTRICITY

        assert sim_details.instruction_processing_delay.seconds == 2
        assert ControlType.OPERATION_MODE_BASED_CONTROL in sim_details.available_control_types
        assert ControlType.FILL_RATE_BASED_CONTROL in sim_details.available_control_types
        assert sim_details.provides_forecast is True
        assert CommodityQuantity.ELECTRIC_POWER_L1 in sim_details.provides_power_measurement_types
        assert CommodityQuantity.HEAT_TEMPERATURE in sim_details.provides_power_measurement_types

        # Assert optional fields are set correctly
        assert sim_details.name == "Home Battery System"
        assert sim_details.manufacturer == "EnergyTech"
        assert sim_details.model == "PowerWall X1"
        assert sim_details.serial_number == "ET-PWX1-12345"
        assert sim_details.currency == "EUR"

        # Assert unset optional fields are None
        assert sim_details.firmware_version is None

    def test_minimal_simulation_details(self):
        # Test with only required fields
        sim_details = SimulationDetails(
            simulation_id="SIM002",
            roles=[Role(role=RoleType.ENERGY_CONSUMER, commodity=Commodity.HEAT)],
            instruction_processing_delay=to_duration("5 seconds"),
            available_control_types=[ControlType.OPERATION_MODE_BASED_CONTROL],
            provides_forecast=False,
            provides_power_measurement_types=[CommodityQuantity.HEAT_TEMPERATURE],
        )

        # Assert required fields
        assert sim_details.simulation_id == "SIM002"
        assert len(sim_details.roles) == 1
        assert sim_details.roles[0].role == RoleType.ENERGY_CONSUMER
        assert sim_details.roles[0].commodity == Commodity.HEAT
        assert sim_details.instruction_processing_delay.seconds == 5
        assert len(sim_details.available_control_types) == 1
        assert sim_details.available_control_types[0] == ControlType.OPERATION_MODE_BASED_CONTROL
        assert sim_details.provides_forecast is False
        assert len(sim_details.provides_power_measurement_types) == 1
        assert sim_details.provides_power_measurement_types[0] == CommodityQuantity.HEAT_TEMPERATURE

        # Assert all optional fields are None
        assert sim_details.name is None
        assert sim_details.manufacturer is None
        assert sim_details.model is None
        assert sim_details.serial_number is None
        assert sim_details.firmware_version is None
        assert sim_details.currency is None

    def test_energy_storage_role(self):
        # Test with energy storage role
        sim_details = SimulationDetails(
            simulation_id="SIM003",
            roles=[Role(role=RoleType.ENERGY_STORAGE, commodity=Commodity.ELECTRICITY)],
            instruction_processing_delay=to_duration("1 second"),
            available_control_types=[
                ControlType.OPERATION_MODE_BASED_CONTROL,
                ControlType.FILL_RATE_BASED_CONTROL,
            ],
            provides_forecast=True,
            provides_power_measurement_types=[CommodityQuantity.ELECTRIC_POWER_3_PHASE_SYM],
            name="Battery Storage Unit",
        )

        # Assert the energy storage role is set correctly
        assert len(sim_details.roles) == 1
        assert sim_details.roles[0].role == RoleType.ENERGY_STORAGE
        assert sim_details.roles[0].commodity == Commodity.ELECTRICITY
        assert sim_details.name == "Battery Storage Unit"
        assert sim_details.firmware_version is None
        assert sim_details.currency is None


# ---------- Test EnergyManagementPlan ----------
@pytest.fixture
def energy_management_plan():
    """Fixture to create a fresh EnergyManagementPlan for each test."""
    return EnergyManagementPlan(
        plan_id=ID("plan1"), generated_at=to_datetime(), instructions=[], comment="Test plan"
    )


@pytest.fixture
def device_control_instruction():
    """Fixture to create a sample DeviceControlInstruction for testing."""
    return DeviceControlInstruction(
        control_id=ID("control1"),
        target_device=ID("device1"),
        commodity_quantity=CommodityQuantity.ELECTRIC_POWER_3_PHASE_SYM,
        start_time=to_datetime(),
        duration=to_duration("2 hours"),
        power=1000.0,
    )


class TestEnergyManagementPlan:
    """Test suite for the EnergyManagementPlan class."""

    def test_add_instruction(self, energy_management_plan, device_control_instruction):
        """Test adding a new instruction to the energy management plan."""
        energy_management_plan.add_instruction(device_control_instruction)

        assert len(energy_management_plan.instructions) == 1
        assert energy_management_plan.instructions[0] == device_control_instruction
        assert energy_management_plan.valid_from == device_control_instruction.start_time
        assert (
            energy_management_plan.valid_until
            == device_control_instruction.start_time + device_control_instruction.duration
        )

    def test_clear_instructions(self, energy_management_plan, device_control_instruction):
        """Test clearing all instructions from the plan."""
        energy_management_plan.add_instruction(device_control_instruction)
        energy_management_plan.clear()

        assert len(energy_management_plan.instructions) == 0
        assert energy_management_plan.get_active_instructions() is None
        assert energy_management_plan.get_next_instruction() is None

    def test_get_active_instructions(self, energy_management_plan, device_control_instruction):
        """Test retrieving active instructions based on the current time."""
        future_instruction = DeviceControlInstruction(
            control_id=ID("control2"),
            target_device=ID("device2"),
            commodity_quantity=CommodityQuantity.ELECTRIC_POWER_3_PHASE_SYM,
            start_time=to_datetime() + to_duration("1 hour"),
            duration=to_duration("2 hours"),
            power=0.0,
        )

        energy_management_plan.add_instruction(device_control_instruction)
        assert len(energy_management_plan.instructions) == 1
        energy_management_plan.add_instruction(future_instruction)
        assert len(energy_management_plan.instructions) == 2

        # Assuming the current time is before the future instruction starts
        active_instructions = energy_management_plan.get_active_instructions()

        assert active_instructions is not None
        assert len(active_instructions) == 1
        assert active_instructions[0] == device_control_instruction

    def test_get_next_instruction(self, energy_management_plan, device_control_instruction):
        """Test retrieving the next upcoming instruction."""
        future_instruction = DeviceControlInstruction(
            control_id=ID("control2"),
            target_device=ID("device2"),
            commodity_quantity=CommodityQuantity.ELECTRIC_POWER_3_PHASE_SYM,
            start_time=to_datetime() + to_duration("1 hour"),
            duration=to_duration("2 hours"),
            power=0.0,
        )

        energy_management_plan.add_instruction(device_control_instruction)
        energy_management_plan.add_instruction(future_instruction)

        # Assuming the current time is before any instruction starts
        next_instruction = energy_management_plan.get_next_instruction()

        assert next_instruction.control_id == future_instruction.control_id

    def test_get_instructions_for_device(self, energy_management_plan, device_control_instruction):
        """Test filtering instructions for a specific device."""
        energy_management_plan.add_instruction(device_control_instruction)

        # Fetch instructions for the target device
        instructions_for_device = energy_management_plan.get_instructions_for_device(ID("device1"))

        assert len(instructions_for_device) == 1
        assert instructions_for_device[0] == device_control_instruction

    def test_add_multiple_instructions(self, energy_management_plan):
        """Test adding multiple instructions and ensure sorting by start_time."""
        instruction1 = DeviceControlInstruction(
            control_id=ID("control1"),
            target_device=ID("device1"),
            commodity_quantity=CommodityQuantity.ELECTRIC_POWER_3_PHASE_SYM,
            start_time=to_datetime() + to_duration("2 hours"),
            duration=to_duration("2 hours"),
            power=1000.0,
        )

        instruction2 = DeviceControlInstruction(
            control_id=ID("control2"),
            target_device=ID("device2"),
            commodity_quantity=CommodityQuantity.ELECTRIC_POWER_3_PHASE_SYM,
            start_time=to_datetime() + to_duration("1 hour"),
            duration=to_duration("2 hours"),
            power=0.0,
        )

        energy_management_plan.add_instruction(instruction1)
        energy_management_plan.add_instruction(instruction2)

        assert energy_management_plan.instructions[0].control_id == instruction2.control_id
        assert energy_management_plan.instructions[1].control_id == instruction1.control_id
        assert compare_datetimes(energy_management_plan.valid_from, instruction2.start_time).equal
        assert compare_datetimes(
            energy_management_plan.valid_until, instruction1.start_time + instruction1.duration
        ).equal
