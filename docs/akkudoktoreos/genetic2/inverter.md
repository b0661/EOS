

# Hybrid inverter device

The hybrid inverter device simulation can simulate:

- battery inverters
- solar inverters
- hybrid inverters

The hybrid inverter device has only one port where it is connectet to the ac bus.

## Operation modes and mode factors

A battery inverter can operate in these operation modes:

- OFF:
    No or minimal energy consumption.
    Energy is taken from the port (ac bus)
- CHARGE:
    Charge battery with energy given by operation mode factor (related to the capacity).
    Energy is taken from the port (ac bus)
- DISCHARGE:
    Discharge Batter with energy given by the operation mode factor
    Energy is provided to the port (ac bus)

A solar inverter can operate in these operation modes:

- OFF:
    No or minimal energy consumption.
    Energy is taken from the port (ac bus)
- PV:
    Provide energy from solar panels.
    Energy is provided to the port (ac bus)

A hybrid inverter can operate in these operation modes:

- OFF:
    No or minimal energy consumption.
    Energy is taken from the port (ac bus)
- PV:
    Provide energy from solar panels only. Battery is idle
    Energy is provided to the port (ac bus)
- CHARGE:
    Charge battery with energy given by operation mode factor (related to the capacity).
    Provide energy from solar panels if not consumed by the battery charging.
    Energy is taken from or delivered to the port (ac bus).
- DISCHARGE:
    Discharge Batter with energy given by the operation mode factor.
    Provide energy from solar panels.
    Energy is provided to the port (ac bus)

## Device parameter dataclass

The parameter dataclass is frozen, slotted, and hashable — safe as dictionary or cache keys.
I carries no mutable state or simulation logic.

The parameters are:

- inverter_type
    The type of the inverter: BATTERY, SOLAR, HYBRID
- off_state_power_consumption_w
- on_state_power_consumption_w
- pv_to_ac_efficiency
- pv_to battery_efficiency
- pv_max_power_w
- pv_min_power_w
- pv_prediction_power_w
    PV power prediction for all simulation steps.
- ac_to_battery_efficiency
- battery_to_ac_efficiency
- battery_capacity_wh
- battery_charge_rates
    Charge rates may be explicitly defined if the battery can only be set to specific charge rates.
- battery_min_charge_rate
- battery_max_charge_rate
- battery_min_discharge_rate
- battery_max_discharge_rate
- battery_min_soc_factor
- battery_max_soc_factor
- battery_initial_soc_factor
    The SoC factor of the battery at the start of the simulation.

## Device genome

The device genome is build up from operation_mode and optionally operation_mode_factor:

Battery inverter genome:

- operation_mode: OFF, CHARGE, DISCHARGE
- operation_mode_factor: 0.0..1.0 (battery charge or discharge power related to battery capacity)

Solar inverter genome:

- operation_mode: OFF, PV

Hybrid inverter genome:

- operation_mode: OFF, PV, CHARGE, DISCHARGE
- operation_mode_factor: 0.0..1.0 (battery charge or discharge power related to battery capacity)

## Repair

If the genome requests an operation mode factor that is infeasable the device simulation corrects it
to the next feasable value.

Battery inverter (type: BATTERY) genome:

- operation mode OFF:
    The operation_mode_factor is set to 0.0
- operation mode CHARGE:
    The operation mode factor clipped to battery_min_charge_rate .. battery_max_charge_rate adapted
    by the ac_to_battery_efficiency.
    If battery_charge_rates are given the operation mode factor is clipped to the next given rate
    (except 0).
- operation mode DISCHARGE:
    The operation mode factor clipped to battery_min_discharge_rate .. battery_max_discharge_rate
    adapted by the battery_to_ac_efficiency.

Solar inverter (type: SOLAR) genome:

- No operation mode factor.

Hybrid Inverter (type: HYBRID) genome:

- operation mode OFF:
    The operation_mode_factor is set to 0.0.
- opration mode PV:
    The operation mode factor is set to 0.0.
- operation mode CHARGE:
    The operation mode factor clipped to battery_min_charge_rate .. battery_max_charge_rate adapted
    by the ac_to_battery_efficiency.
    If battery_charge_rates are given the operation mode factor is clipped to the next given rate
    (except 0).
- operation mode DISCHARGE:
    The operation mode factor clipped to battery_min_discharge_rate .. battery_max_discharge_rate
    adapted by the battery_to_ac_efficiency.

## Arbitration

The hybrid inverter either consumes energy from or injects energy into the port. It requests the
energy that results from PV and battery requests.

The minimum energy requested is defined by the battery_min_charge_rate or the
battery_min_discharge_rate or the PV power prediction, depending on the operation mode. In any case
the requested energy is corrected by the efficiency of the inverter (see Repair).

## Lifecycle

### setup_run()

Called once per optimisation run.

### genome_requirements()

Called once to freeze genome structure.

### create_batch_state()

Repeated once per generation inside evaluate_population().

### apply_genome_batch()

Repeated once per generation inside evaluate_population().

### build_device_request()

Repeated once per generation inside evaluate_population().

### apply_device_grant()

Repeated once per generation inside evaluate_population().

### compute_cost()

Repeated once per generation inside evaluate_population().






