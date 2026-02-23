# Akkudoktor-EOS Simulation Engine

## Overview

The simulation engine models energy flows between physical devices — batteries,
PV arrays, electric vehicles, heat pumps, and grid connections — over a
configurable time horizon. It is designed to support the genetic optimizer
but is fully decoupled from it: any optimizer, or no optimizer at all, can
drive a simulation run.

The design follows three principles:

- **Devices own their own physics.** The engine never inspects device internals.
- **The optimizer owns the schedule.** Devices declare what genome slots they need; the optimizer provides values.
- **The grid balances everything.** After all devices have acted, any surplus feeds in and any deficit is imported.

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                    Optimizer                        │
│  (genetic, linear, rule-based, …)                   │
│                                                     │
│  GenomeAssembler.dispatch(genome, registry)         │
└───────────────────────┬─────────────────────────────┘
                        │ apply_genome(slice) per device
                        ▼
┌─────────────────────────────────────────────────────┐
│                 DeviceRegistry                      │
│                                                     │
│   Battery "main" │ Battery "garage" │ PV │ EV │ …  │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│             EnergySimulationEngine                  │
│                                                     │
│  for each hour:                                     │
│    1. request()      ← all devices declare needs    │
│    2. arbitrate()    ← ResourceArbitrator allocates │
│    3. simulate_hour()← devices act on their grant   │
│    4. aggregate()    ← grid balances remainder      │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
              SimulationResult
         (per-hour + aggregate metrics)
```

---

## Module Layout

```text
akkudoktoreos/
├── devices/genetic2/
│   └── base.py              # EnergyDevice, StorageDevice, GenomeSlice
│
├── simulation/genetic2/
│   ├── flows.py             # EnergyFlows, ResourceRequest, ResourceGrant, Priority
│   ├── registry.py          # DeviceRegistry
│   ├── arbitrator.py        # ArbitratorBase, PriorityArbitrator, ProportionalArbitrator
│   ├── timeseries.py        # SimulationInput, DeviceSchedule
│   ├── result.py            # SimulationResult, HourlyResult, DeviceHourlyState
│   └── engine.py            # EnergySimulationEngine
│
└── optimization/genetic2
    └── genome.py            # GenomeAssembler
```

---

## Core Concepts

### Devices

Every physical device implements `EnergyDevice` and provides four methods:

| Method | When called | Purpose |
|--------|------------|---------|
| `genome_requirements()` | Once at optimizer setup | Declare genome slice size and bounds |
| `apply_genome(slice)` | Once per simulation run | Decode genome into internal schedule |
| `request(hour)` | Each hour, Phase 1 | Declare resource needs for this hour |
| `simulate_hour(hour, grant)` | Each hour, Phase 2 | Act on allocated resources, update state |

Devices that have no optimizable parameters (e.g. a PV array with a fixed
forecast) return `None` from `genome_requirements()` and are skipped by the
genome assembler.

### Sign Convention

All power values are in **Wh per simulation hour**.

| Sign | Meaning |
|------|---------|
| Positive `ac_power_wh` | Device is **providing** AC to the system |
| Negative `ac_power_wh` | Device is **consuming** AC from the system |
| Positive `ac_power_wh` in `ResourceRequest` | Device **wants** to consume |
| Negative `ac_power_wh` in `ResourceRequest` | Device **offers** surplus |

### The Grid as Balancer

The grid is not modelled as a device. After all devices have reported their
`EnergyFlows`, the engine sums net AC power:

```text
net_ac = Σ(device ac flows) − base_load

feedin_wh      = max(net_ac, 0)   # surplus → grid
grid_import_wh = max(-net_ac, 0)  # deficit ← grid
```

---

## Resource Arbitration

Each hour, every device submits a `ResourceRequest` before any device simulates.
The `ResourceArbitrator` sees all requests simultaneously and produces a
`ResourceGrant` for each device.

### Why Arbitration?

Without arbitration, the simulation order determines who gets resources — a
fragile implicit dependency. With arbitration, allocation policy is explicit
and swappable.

### Priority Arbitration (default)

1. **Producers** (negative request values) are always dispatched in full.
   Their output forms the available supply pool.
2. **Consumers** are served in `Priority` order — `CRITICAL` first, `LOW` last.
3. A consumer that would receive less than its declared **minimum** is
   **curtailed entirely** (`grant.curtailed = True`). This prevents devices
   from running in inefficient partial states (e.g. a heat pump below its
   minimum viable power level).
4. Any remaining surplus after all consumers are served is handled by the
   grid balancer.

### Proportional Arbitration

When no device has strict priority over others, `ProportionalArbitrator`
distributes supply proportionally by request size. Devices below their
minimum are curtailed iteratively until the allocation is stable.

### Custom Arbitration

Implement `ArbitratorBase.arbitrate()` to inject any policy — market-based
pricing, phase balancing, contractual obligations — without touching the engine
or any device:

```python
class MyArbitrator(ArbitratorBase):
    def arbitrate(self, requests):
        ...
        return grants  # dict[device_id, ResourceGrant]

engine = EnergySimulationEngine(registry, arbitrator=MyArbitrator())
```

---

## Genome Assembly

The `GenomeAssembler` builds the flat optimizer genome from device declarations
and dispatches slices back to devices before each simulation run.

```python
assembler = GenomeAssembler(registry)

# Inspect the layout
print(assembler.describe())
# GenomeAssembler: total_size=72
#   [   0:  24] battery_main  (size=24, dtype=int, range=[0.0, 2.0])
#              0=idle, 1=discharge, 2=charge
#   [  24:  48] battery_garage (size=24, dtype=int, range=[0.0, 2.0])
#              0=idle, 1=discharge, 2=charge
#   [  48:  72] ev_car        (size=24, dtype=int, range=[0.0, 10.0])
#              charge rate index per hour

# Get bounds for the optimizer
lows, highs = assembler.bounds()

# Dispatch a genome to all devices before simulate()
assembler.dispatch(genome, registry)
```

**Adding a new device automatically extends the genome.** No optimizer or
engine code changes are needed.

Genome slices are validated against each device's declared bounds at dispatch
time. Violations raise a `ValueError` immediately rather than silently
corrupting device schedules.

---

## Running a Simulation

```python
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine
from akkudoktoreos.simulation.genetic2.timeseries import SimulationInput
from akkudoktoreos.optimization.genetic2.genome import GenomeAssembler

# 1. Build registry
registry = DeviceRegistry()
registry.register(PVArray("pv_main", forecast_wh=pv_forecast))
registry.register(Battery("battery_main", capacity_wh=10_000, ...))
registry.register(Battery("battery_garage", capacity_wh=5_000, ...))
registry.register(ElectricVehicle("ev_car", ...))

# 2. Build assembler and generate / dispatch genome
assembler = GenomeAssembler(registry)
genome = assembler.random_genome()          # or from optimizer
assembler.dispatch(genome, registry)

# 3. Build inputs
inputs = SimulationInput(
    start_hour=0,
    end_hour=24,
    load_wh=household_load,
    electricity_price=price_array,
    feed_in_tariff=tariff_array,
)

# 4. Simulate
engine = EnergySimulationEngine(registry)
result = engine.simulate(inputs)

# 5. Use results
print(f"Net balance: {result.net_balance_eur:.2f} EUR")
print(f"battery_main SoC: {result.soc_per_hour('battery_main')}")

# Backwards-compatible dict export
legacy_dict = result.to_dict()
```

---

## Multiple Devices of the Same Type

The registry supports any number of devices of each type. Each device must
have a unique `device_id`. The genome assembler assigns non-overlapping slices
automatically.

```python
registry.register(Battery("battery_house",  capacity_wh=10_000, ...))
registry.register(Battery("battery_garage", capacity_wh=5_000, ...))
registry.register(Battery("battery_shed",   capacity_wh=2_500, ...))
```

Per-device results are accessible by ID:

```python
for device_id in ["battery_house", "battery_garage", "battery_shed"]:
    soc = result.soc_per_hour(device_id)
    flows = result.flows_per_hour(device_id)
```

---

## Extending with New Device Types

Implementing a new device type requires no changes to the engine, arbitrator,
or genome assembler. The following example adds a heat pump that consumes AC
and provides heat:

```python
class HeatPump(EnergyDevice):

    def genome_requirements(self) -> GenomeSlice:
        return GenomeSlice(
            device_id=self.device_id,
            size=self.prediction_hours,
            dtype=int,
            low=0, high=3,
            description="0=off, 1=low, 2=medium, 3=high",
        )

    def apply_genome(self, genome_slice: np.ndarray) -> None:
        self._levels = genome_slice

    def request(self, hour: int) -> ResourceRequest:
        level = self._levels[hour]
        ac_needed = self.power_levels_wh[level]
        return ResourceRequest(
            device_id=self.device_id,
            hour=hour,
            priority=Priority.HIGH,
            ac_power_wh=ac_needed,              # wants AC
            heat_wh=-ac_needed * self.cop,      # offers heat
            min_ac_power_wh=self.power_levels_wh[1],
        )

    def simulate_hour(self, hour: int, grant: ResourceGrant) -> EnergyFlows:
        if grant.curtailed:
            return EnergyFlows()
        heat = grant.ac_power_wh * self.cop
        return EnergyFlows(
            ac_power_wh=-grant.ac_power_wh,
            heat_provided_wh=heat,
            losses_wh=grant.ac_power_wh - heat,
        )

    def reset(self) -> None:
        self._levels = np.zeros(self.prediction_hours, dtype=int)
```

Register it and the simulation picks it up:

```python
registry.register(HeatPump("heatpump_1", cop=3.5, prediction_hours=24))
```

---

## Simulation Result

`SimulationResult` provides typed access to all outputs.

### Aggregate Properties

| Property | Type | Description |
|----------|------|-------------|
| `net_balance_eur` | `float` | Cost minus revenue — optimizer fitness signal |
| `total_cost_eur` | `float` | Total grid import cost in EUR |
| `total_revenue_eur` | `float` | Total feed-in revenue in EUR |
| `total_losses_wh` | `float` | Total conversion losses in Wh |
| `total_feedin_wh` | `float` | Total energy fed to grid in Wh |
| `total_grid_import_wh` | `float` | Total energy drawn from grid in Wh |

### Per-Device Queries

```python
result.soc_per_hour("battery_main")    # list[float | None]
result.flows_per_hour("battery_main")  # list[EnergyFlows | None]
result.all_device_ids()                # set[str]
```

### Backwards Compatibility

`result.to_dict()` exports the legacy key names used by the existing API and
visualization layer, so existing consumers require no changes during migration.

---

## Design Decisions and Trade-offs

**Devices decode their own genome.** This means each device controls its own
encoding (binary states, charge rate indices, continuous factors) without the
optimizer or engine needing to know. The trade-off is that genome structure
is less transparent — `GenomeAssembler.describe()` exists specifically to
compensate for this.

**The arbitrator sees all requests before any device simulates.** This is
more expensive than a sequential pass but necessary for correct priority
handling — a low-priority device must not consume resources that a
high-priority device will later need.

**The grid is implicit, not a device.** This simplifies the model for the
common residential case. If multi-node grid modelling is needed in future,
the grid connection can be promoted to a `GridDevice` that submits requests
and grants like any other device.

**`SimulationResult.to_dict()` provides backwards compatibility.** New code
should use the typed properties. The dict export will be deprecated once all
consumers are migrated.
