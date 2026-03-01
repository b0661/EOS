# genetic2 Framework Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Reference](#module-reference)
   - [devicesabc — Device Abstractions](#devicesabc--device-abstractions)
   - [topology — Topology Validation](#topology--topology-validation)
   - [registry — Device Registry](#registry--device-registry)
   - [arbitrator — Bus Arbitration](#arbitrator--bus-arbitration)
   - [state — Batch Simulation State](#state--batch-simulation-state)
   - [genome — Genome Assembly](#genome--genome-assembly)
   - [engine — Simulation Engine](#engine--simulation-engine)
   - [optimizer — Genetic Optimiser](#optimizer--genetic-optimiser)
     - [extract_best_instructions](#extract_best_instructions)
4. [Data Flow](#data-flow)
5. [Key Design Decisions](#key-design-decisions)
6. [Implementing a New Device](#implementing-a-new-device)
7. [Running an Optimisation](#running-an-optimisation)
8. [Array Shape Reference](#array-shape-reference)
9. [Module Dependency Graph](#module-dependency-graph)

---

## Overview

`genetic2` is a vectorized multi-objective genetic algorithm framework for optimising energy system schedules. It is designed for home energy management problems where multiple devices — batteries, heat pumps, PV arrays, grid connections — share energy buses and compete for limited power over a rolling time horizon.

The framework has three core properties:

**Immutable structure, mutable state.** Devices are structural elements: they hold physical parameters but never runtime data. All per-evaluation data lives in state objects that are created fresh each generation and discarded after cost computation. This makes repeated evaluation safe without re-instantiation.

**Population-vectorized evaluation.** An entire population of candidate schedules is evaluated in a single call. No Python loop over individuals exists anywhere in the hot path. All inner loops are NumPy operations over a `(population_size, horizon)` array axis.

**Separation of concerns.** Physical parameters, topology, arbitration, genome encoding, and optimisation strategy are all in separate modules with well-defined interfaces. A new device type requires implementing one abstract class. A new optimisation strategy requires only a different `scalarize` function or a different outer loop around `GeneticOptimizer`.

---

## Architecture

The framework is split into two package trees that reflect the separation between optimisation-time concerns and simulation-time concerns:

```
akkudoktoreos/
├── devices/
│   └── genetic2/
│       └── devicesabc.py        Device parameters, port types, abstract base classes
│
├── simulation/
│   └── genetic2/
│       ├── topology.py          Structural topology validation
│       ├── registry.py          Device registry
│       ├── arbitrator.py        Bus energy arbitration
│       ├── state.py             Batch simulation state container
│       └── engine.py            Simulation engine and evaluation loop
│
└── optimization/
    └── genetic2/
        ├── genome.py            Genome slice definitions and assembly
        └── optimizer.py         GeneticOptimizer and RollingHorizonOptimizer
```

The dependency direction is strictly one-way: `optimizer` → `engine` → `arbitrator`, `registry`, `topology`, `state` → `devicesabc`. The `genome` module is used by both `engine` and `optimizer` but depends only on `devicesabc`.

---

## Module Reference

### devicesabc — Device Abstractions

**Location:** `akkudoktoreos.devices.genetic2.devicesabc`

This module has two responsibilities that are co-located because they are tightly coupled by the device identity contract.

#### Enumerations

**`EnergyCarrier`** — The carrier type of an energy bus.

| Value | Meaning |
|---|---|
| `AC` | Alternating current |
| `DC` | Direct current |
| `HEAT` | Thermal energy |

**`PortDirection`** — Energy flow direction from the *device's* perspective.

| Value | Meaning |
|---|---|
| `SOURCE` | Device injects energy onto the bus (PV, discharging battery) |
| `SINK` | Device consumes energy from the bus (load, charging battery) |
| `BIDIRECTIONAL` | Device can both inject and consume (grid connection, battery) |

A `BIDIRECTIONAL` port counts as both a source and a sink for topology validation purposes.

**`BatteryOperationMode`**, **`ApplianceOperationMode`** — Symbolic operating modes for batteries and appliances respectively. These express scheduling intent and do not directly drive physics — they are available for use in `DeviceParam` subclasses.

#### Topology Dataclasses

**`EnergyBus`** — An immutable energy bus descriptor.

```python
@dataclass(frozen=True, slots=True)
class EnergyBus:
    bus_id: str
    carrier: EnergyCarrier
    constraint: EnergyBusConstraint | None = None
```

Buses own the carrier type. Ports do not redeclare carrier — they connect to a bus by `bus_id` and inherit its carrier implicitly.

**`EnergyPort`** — An immutable port connecting a device to a bus.

```python
@dataclass(frozen=True, slots=True)
class EnergyPort:
    port_id: str           # Unique within the owning device
    bus_id: str            # References an EnergyBus.bus_id
    direction: PortDirection
    max_power_w: float | None = None
```

**`EnergyBusConstraint`** — Optional structural limits on port counts for a bus.

```python
@dataclass(frozen=True, slots=True)
class EnergyBusConstraint:
    max_sinks: int | None = None
    max_sources: int | None = None
```

#### Device Parameter Dataclasses

All parameter dataclasses are frozen, slotted, and hashable — safe as dictionary or cache keys. They carry no mutable state or simulation logic.

| Class | Key Fields |
|---|---|
| `DeviceParam` | `device_id`, `ports` |
| `BatteryParam` | `capacity_wh`, `charging_efficiency`, `discharging_efficiency`, `max_charge_power_w`, `min_soc_factor`, `max_soc_factor`, `charge_rates` |
| `InverterParam` | `max_power_w`, `efficiency` |
| `PVParam` | `peak_power_w`, `tilt_deg`, `azimuth_deg` |
| `HeatPumpParam` | `thermal_power_w`, `cop`, `operation_modes` |

All parameter classes run validation in `__post_init__` and raise `ValueError` for physically impossible configurations (negative capacity, efficiency > 1, etc.).

#### Abstract Device Base Classes

**`EnergyDevice`** — The full abstract contract for a vectorized batch simulation device.

```
setup_run(step_times, step_interval)                    ← called once per optimisation run
genome_requirements() → GenomeSlice | None              ← called once after setup_run
ports → tuple[EnergyPort, ...]                          ← static property
objective_names → list[str]                             ← static property

create_batch_state(population_size, horizon) → state
apply_genome_batch(state, genome_batch) → repaired_genome
build_device_request(state) → DeviceRequest | None
apply_device_grant(state, grant) → None
compute_cost(state) → np.ndarray                        ← shape (population_size, num_local_objectives)
extract_instructions(state, individual_index) → list[EnergyManagementInstruction]
```

Devices hold configuration but never mutable runtime data. All runtime data lives in the state object returned by `create_batch_state`.

`setup_run` receives `step_times` as `tuple[DateTime, ...]` — not raw floats. `DateTime` objects are passed directly into `SingleStateBatchState.step_times` so every lifecycle method, including `compute_cost` and `extract_instructions`, can use them as S2 `execution_time` values without any conversion.

**`SingleStateEnergyDevice`** — A convenience base class for devices with a single scalar internal state (SoC, temperature, etc.). It implements the full batch simulation loop, leaving only the physics to concrete subclasses.

Subclasses must implement:

| Method | Description |
|---|---|
| `initial_state() → float` | Scalar initial state at step 0 |
| `state_transition_batch(state, power, step_interval) → np.ndarray` | Vectorized one-step state update |
| `power_bounds() → tuple[float, float]` | `(min_power_w, max_power_w)` — used for genome bounds and default repair |
| `ports` | Port tuple |
| `objective_names` | Objective name list |
| `build_device_request` | Arbitration request |
| `apply_device_grant` | Grant application |
| `compute_cost` | Local cost matrix |
| `extract_instructions` | S2 instruction list for one individual (see below) |

Two hooks can be overridden rather than required from scratch. `repair_batch(step, requested_power, current_state)` provides state-dependent repair (e.g. clamping charge power when the battery is near full); the default clamps to `power_bounds()`. `extract_instructions(state, individual_index)` converts the simulated schedule into a list of `EnergyManagementInstruction` objects; the base class raises `NotImplementedError`, which `GeneticOptimizer.extract_best_instructions` silently skips.

**Sign convention:** positive power = consumption (sink), negative power = generation (source), following the load convention.

**`SingleStateBatchState`** — The mutable state container for `SingleStateEnergyDevice`.

```python
@dataclass
class SingleStateBatchState:
    schedule: np.ndarray              # (population_size, horizon) — repaired power schedule [W]
    state: np.ndarray                 # (population_size,) — current internal state
    population_size: int
    horizon: int
    step_times: tuple[DateTime, ...]  # length == horizon — timestamps for each step
```

`step_times` is populated by `SingleStateEnergyDevice.create_batch_state` from the `_step_times` stored during `setup_run`. It is available to every lifecycle method that receives a `SingleStateBatchState`:

- **`compute_cost`** — use for time-of-use tariff pricing, peak-window penalties, etc.
- **`extract_instructions`** — use directly as S2 `execution_time` values; no float-to-DateTime conversion needed.

---

### topology — Topology Validation

**Location:** `akkudoktoreos.simulation.genetic2.topology`

`TopologyValidator.validate(devices, buses)` runs at engine construction time and raises `TopologyValidationError` on the first structural violation found.

**What is validated:**

- All `port_id` values are unique within each device.
- Every port's `bus_id` references a bus in the provided bus list.
- Every bus that has at least one connected port has at least one source (or bidirectional) port and at least one sink (or bidirectional) port.
- If a bus declares an `EnergyBusConstraint`, the actual sink and source counts must not exceed `max_sinks` and `max_sources`.

**What is not validated:**

- Carrier matching — ports inherit carrier from the bus they connect to.
- Power balance or flow feasibility — that is the arbitrator's responsibility at runtime.
- Unused buses — buses with no connected ports are silently ignored.

---

### registry — Device Registry

**Location:** `akkudoktoreos.simulation.genetic2.registry`

`DeviceRegistry` is an ordered container for `EnergyDevice` instances. Device registration order is preserved and determines gene ordering in the assembled genome.

```python
registry = DeviceRegistry()
registry.register(battery)
registry.register(heat_pump)

for device in registry.all_devices():   # yields in registration order
    ...
```

Registering a device with a duplicate `device_id` raises `ValueError`. Devices are looked up by `device_id` via `registry.get(device_id)`.

---

### arbitrator — Bus Arbitration

**Location:** `akkudoktoreos.simulation.genetic2.arbitrator`

The `VectorizedBusArbitrator` resolves competing energy requests across all buses in a single call, processing the entire population at once.

#### Data Structures

**`PortRequest`** — A device's energy request for one port over the full population and horizon.

```python
@dataclass(frozen=True, slots=True)
class PortRequest:
    port_index: int
    energy_wh: np.ndarray       # (population_size, horizon) — positive=consume, negative=inject
    min_energy_wh: np.ndarray   # (population_size, horizon) — minimum acceptable grant
```

**`DeviceRequest`** — All port requests for one device.

```python
@dataclass(frozen=True, slots=True)
class DeviceRequest:
    device_index: int
    port_requests: tuple[PortRequest, ...]
```

**`PortGrant`** / **`DeviceGrant`** — Mirror the request structure; `granted_wh` carries the arbitrated result.

**`BusTopology`** — Static mapping from port indices to bus indices.

```python
@dataclass(frozen=True, slots=True)
class BusTopology:
    port_to_bus: np.ndarray   # (num_ports,) — port_to_bus[i] = bus index for port i
    num_buses: int
```

#### Arbitration Algorithm

For each bus, per individual, per timestep:

1. Sum all consumer requests → `total_demand (population_size, horizon)`.
2. Sum all producer requests → `total_supply (population_size, horizon)`.
3. Compute `supply_ratio = clip(total_supply / total_demand, 0, 1)`.
4. Scale each consumer's grant proportionally: `grant = request * supply_ratio`.
5. Producers always receive their full requested injection.
6. Apply `min_energy_wh` floor to consumer grants after proportional scaling.

The key vectorization detail: port requests are stacked into a `(num_ports, population_size, horizon)` tensor. Bus-level sums reduce over `axis=0` (the port axis), leaving `(population_size, horizon)` results. The `supply_ratio` broadcast inserts a `newaxis` to expand from `(population_size, horizon)` to `(1, population_size, horizon)` before multiplying against the `(num_ports, population_size, horizon)` tensor.

**`ArbitrationPriority`** is available as an enum for future priority-based allocation extensions. The current implementation uses proportional allocation only.

#### Device Index Contract

`DeviceRequest.device_index` must be set by the device's own `build_device_request()`. The engine does not inject or override device indices. The conventional approach is for a device to store its index at construction time or to derive it from the registry.

---

### state — Batch Simulation State

**Location:** `akkudoktoreos.simulation.genetic2.state`

`BatchSimulationState` is the top-level mutable container for one generation's evaluation. It is created fresh at the start of each `evaluate_population()` call.

```python
@dataclass
class BatchSimulationState:
    device_states: dict[str, object]   # device_id → device-specific state object
    population_size: int
```

Device-specific state types (e.g. `SingleStateBatchState`) are defined alongside their device classes. The engine retrieves them by `device_id` and passes them to each device lifecycle method.

---

### genome — Genome Assembly

**Location:** `akkudoktoreos.optimization.genetic2.genome`

#### `GenomeSlice`

An immutable descriptor for one device's segment of the global genome vector. Contains only structural metadata — no gene values.

```python
@dataclass(frozen=True, slots=True)
class GenomeSlice:
    start: int                         # Inclusive start index in the global genome
    size: int                          # Number of genes
    lower_bound: np.ndarray | None     # Per-gene lower bounds, shape (size,)
    upper_bound: np.ndarray | None     # Per-gene upper bounds, shape (size,)

    @property
    def end(self) -> int: ...          # start + size

    def extract(self, genome: np.ndarray) -> np.ndarray:
        # 1-D input (total_genome_size,)           → returns (size,)
        # 2-D input (population_size, total_size)  → returns (population_size, size)
```

`extract()` returns a **view** of the genome array, not a copy. In-place writes to the view affect the original genome. This is intentional — the engine writes repaired genomes back in-place.

#### `AssembledGenome`

The immutable result of genome assembly. Produced once at `genome_requirements()` time and reused for the entire run.

```python
@dataclass(frozen=True, slots=True)
class AssembledGenome:
    total_size: int
    slices: dict[str, GenomeSlice]     # device_id → GenomeSlice
    lower_bounds: np.ndarray           # (total_size,) — -inf where no bound declared
    upper_bounds: np.ndarray           # (total_size,) — +inf where no bound declared
```

Slices are keyed by `device_id` (string), not by device instance, consistent with `DeviceRegistry` and `BatchSimulationState`.

#### `GenomeAssembler`

Iterates over devices in registration order, calls `genome_requirements()` on each, and assigns non-overlapping index ranges. Devices returning `None` or size-zero slices are skipped.

```python
assembler = GenomeAssembler()
assembled = assembler.assemble(devices)   # devices: Iterable[EnergyDevice]

# Extract one device's genes from a flat or batch genome
genes = GenomeAssembler.extract_device_genome(genome, device_id, assembled)
```

Device registration order determines gene ordering in the global genome — this ordering is stable across all generations of the same run.

---

### engine — Simulation Engine

**Location:** `akkudoktoreos.simulation.genetic2.engine`

`EnergySimulationEngine` orchestrates the full simulation pipeline over an entire population in a single call.

#### Lifecycle States

```
CREATED  →(setup_run)→  RUN_CONFIGURED  →(genome_requirements)→  STRUCTURE_FROZEN
                                ↑                                         |
                                └─────────────(setup_run)────────────────┘
```

`setup_run()` can be called again from `STRUCTURE_FROZEN` to start a new run with different inputs (e.g. a different time horizon or price forecast) without re-instantiating the engine.

#### `EnergySimulationInput`

```python
@dataclass(frozen=True)
class EnergySimulationInput:
    step_times: tuple[DateTime, ...]   # Ordered timestamps, length = horizon
    step_interval: float               # Seconds between steps
```

`step_times` accepts `DateTime` objects (not raw floats). The engine passes them verbatim to `device.setup_run()`, which stores them and forwards them into every `SingleStateBatchState` created during that run.

#### `EvaluationResult`

```python
@dataclass
class EvaluationResult:
    fitness: np.ndarray                     # (population_size, num_objectives)
    objective_names: list[str]              # Column names for fitness
    repaired_genomes: dict[str, np.ndarray] # device_id → (population_size, horizon)
```

`repaired_genomes` contains only entries for devices that actually modified genome values during simulation. The optimiser writes these back into the population before the next generation (Lamarckian repair).

#### `evaluate_population()` Pipeline

```python
result = engine.evaluate_population(genome_population)
# genome_population: dict[device_id, np.ndarray (population_size, horizon)]
```

Internally:

1. **Create batch state** — `device.create_batch_state(pop_size, horizon)` for each device. State is fresh every call.
2. **Apply genomes** — `device.apply_genome_batch(state, genome_slice)` for each genome-controlled device. Devices repair infeasible values in-place. Changed values are recorded in `repaired_genomes`.
3. **Build requests** — `device.build_device_request(state)` for each device.
4. **Arbitrate** — `arbitrator.arbitrate(device_requests)` resolves competing energy claims across all buses.
5. **Apply grants** — `device.apply_device_grant(state, grant)` distributes arbitrated energy back to devices.
6. **Accumulate costs** — `device.compute_cost(state)` for each device. Local cost columns are mapped to global columns by objective name and summed. Multiple devices sharing an objective name contribute to the same fitness column.

#### `last_batch_state`

```python
@property
def last_batch_state(self) -> BatchSimulationState: ...
```

Holds the `BatchSimulationState` produced by the most recent `evaluate_population` call. Used by `GeneticOptimizer.extract_best_instructions` to retrieve per-device states after a single-individual re-evaluation of the best genome. Raises `RuntimeError` if accessed before any `evaluate_population` call.

#### Objective Naming

Objective names are arbitrary strings declared by each device via `objective_names`. The engine assigns global column indices in stable insertion order across all devices. Sharing a name across devices is intentional — e.g. two devices both contributing to `"energy_cost_eur"` are summed into the same fitness column. The optimiser receives the raw `(population_size, num_objectives)` matrix and is responsible for aggregation.

---

### optimizer — Genetic Optimiser

**Location:** `akkudoktoreos.optimization.genetic2.optimizer`

#### `GeneticOptimizer`

A real-valued genetic algorithm that evaluates the entire population in one vectorized engine call per generation.

```python
optimizer = GeneticOptimizer(
    engine=engine,
    population_size=50,
    generations=100,
    crossover_rate=0.9,
    mutation_rate=0.05,
    mutation_sigma=0.1,         # Fraction of bound range per gene
    tournament_size=3,
    scalarize=default_scalarize,  # (pop, num_obj) → (pop,)
    random_seed=42,
)

result = optimizer.optimize(inputs)   # inputs: EnergySimulationInput
```

`optimize()` calls `engine.setup_run()` and `engine.genome_requirements()` internally. The engine is left in `STRUCTURE_FROZEN` state after the call.

**GA operators:**

- **Initialisation** — Uniform random sampling within `[lower_bounds, upper_bounds]`. Genes with infinite bounds are initialised to 0.0.
- **Tournament selection** — `tournament_size` individuals drawn at random; lowest scalar fitness wins.
- **BLX-α crossover** — Two parents produce two children as convex combinations: `child = α * p1 + (1-α) * p2` where α is drawn uniformly from [0, 1]. Both children are within the parents' convex hull. Applied with probability `crossover_rate`.
- **Gaussian mutation** — Per-gene perturbation with probability `mutation_rate`. Noise sigma = `mutation_sigma * bound_range` per gene, making mutation scale-invariant. Genes are clipped to bounds after mutation.
- **Lamarckian repair** — After each `evaluate_population()` call, `EvaluationResult.repaired_genomes` is spliced back into the flat population before selection. Infeasible individuals are corrected rather than penalised.

**Multi-objective scalarization:**

The `scalarize` argument accepts any `Callable[[np.ndarray], np.ndarray]` that maps `(population_size, num_objectives)` to `(population_size,)`. The default `default_scalarize` applies equal-weight sum. Custom strategies (weighted sum, Chebyshev, ε-constraint) can be substituted without any other changes.

#### `OptimizationResult`

```python
@dataclass
class OptimizationResult:
    best_genome: np.ndarray             # (total_genome_size,) — flat best genome
    best_fitness_vector: np.ndarray     # (num_objectives,) — raw multi-objective costs
    best_scalar_fitness: float          # Scalarized fitness of best individual
    objective_names: list[str]          # Column names matching best_fitness_vector
    generations_run: int
    history: list[GenerationStats]      # One entry per generation
    assembled: AssembledGenome          # Use assembled.slices[device_id] to extract schedules
```

To extract a per-device schedule from `best_genome`:

```python
slc = result.assembled.slices["battery_0"]
battery_schedule = result.best_genome[slc.start:slc.end]   # (horizon,)
```

#### `extract_best_instructions`

```python
def extract_best_instructions(
    self,
    result: OptimizationResult,
    inputs: EnergySimulationInput,
) -> dict[str, list[EnergyManagementInstruction]]:
```

Re-runs the engine for a population of size 1 containing `result.best_genome`, then calls `device.extract_instructions(state, 0)` on every registered device. Returns a `dict[device_id, list[EnergyManagementInstruction]]`.

Devices that have not implemented `extract_instructions` (i.e. the base class raises `NotImplementedError`) are silently omitted from the result. An empty dict is returned if no device implements the method.

The engine is left in `STRUCTURE_FROZEN` state after the call. This method must be called after `optimize()`, not during it.

```python
# After optimisation:
result = optimizer.optimize(inputs)
instructions = optimizer.extract_best_instructions(result, inputs)

# One list of instructions per device that implemented extract_instructions:
battery_instructions = instructions["bat_0"]   # list[EnergyManagementInstruction]
for instr in battery_instructions:
    print(instr.execution_time, instr.type)
```

#### `GenerationStats`

```python
@dataclass
class GenerationStats:
    generation: int              # Zero-based index
    best_scalar_fitness: float
    mean_scalar_fitness: float
    num_repaired: int            # Devices that had genomes repaired this generation
```

#### `RollingHorizonOptimizer`

Solves long horizons by optimising one time window at a time and committing only the first `roll_steps` of each optimised window to the final schedule.

```python
rho = RollingHorizonOptimizer(
    engine=engine,
    all_step_times=tuple(range(96)),   # 96-step horizon
    step_interval=3600.0,
    window_size=24,                    # Optimise 24 steps at a time
    roll_steps=8,                      # Commit 8 steps, re-optimise the rest
    # All remaining kwargs forwarded to GeneticOptimizer:
    population_size=40,
    generations=50,
    random_seed=0,
)

schedule = rho.optimize()
# schedule: dict[device_id, np.ndarray(total_steps,)]
```

**Window mechanics:** With `total_steps=96`, `window_size=24`, `roll_steps=8`, the windows are `[0,24)`, `[8,32)`, `[16,40)`, … The last window is truncated automatically when it extends past the end of the horizon. A fresh `GeneticOptimizer` is created for each window, calling `engine.setup_run()` with the window's `step_times` slice.

`roll_steps > window_size` raises `ValueError` at construction time.

---

## Data Flow

```
EnergySimulationInput
        │
        ▼
GeneticOptimizer.optimize()
        │
        ├─► engine.setup_run()
        ├─► engine.genome_requirements()
        │       └─► device.setup_run()        (all devices)
        │       └─► device.genome_requirements()
        │
        ├─► _build_assembled_genome()         → AssembledGenome
        ├─► _init_population()                → (pop_size, total_genome_size)
        │
        └─► for each generation:
                │
                ├─► _split_population()       → dict[device_id, (pop, horizon)]
                │
                ├─► engine.evaluate_population()
                │       │
                │       ├─► device.create_batch_state()     (all devices)
                │       ├─► device.apply_genome_batch()     (genome devices only)
                │       │       └─► _simulate_batch()
                │       │               └─► repair_batch()  (each step)
                │       │               └─► state_transition_batch()
                │       │
                │       ├─► device.build_device_request()   (all devices)
                │       ├─► arbitrator.arbitrate()
                │       ├─► device.apply_device_grant()     (all devices)
                │       └─► device.compute_cost()           (all devices)
                │               └─► accumulate into (pop_size, num_objectives)
                │
                ├─► _apply_repairs()          ← Lamarckian write-back
                ├─► scalarize()               → (pop_size,)
                ├─► track best individual
                └─► _breed()                  → next generation
                        ├─► _tournament_select()
                        ├─► _blx_crossover()
                        └─► _mutate()

        └─► OptimizationResult

# After optimize() — called separately, not inside the GA loop:
optimizer.extract_best_instructions(result, inputs)
        │
        ├─► engine.setup_run()                  (re-configure for best genome)
        ├─► engine.genome_requirements()
        ├─► engine.evaluate_population({best_genome reshaped to pop=1})
        │       └─► full pipeline as above, population_size=1
        │
        ├─► engine.last_batch_state             (state from the re-evaluation)
        │
        └─► for each device:
                device.extract_instructions(device_state, individual_index=0)
                    └─► uses state.step_times for execution_time fields
                    └─► returns list[EnergyManagementInstruction]
        │
        └─► dict[device_id, list[EnergyManagementInstruction]]
```

---

## Key Design Decisions

### Why is the genome a flat float64 array?

Genes represent physical power values in watts — continuous quantities with known upper and lower bounds. A flat `(population_size, total_genome_size)` array lets all GA operators (crossover, mutation, selection, initialisation) work with standard NumPy operations on contiguous memory. The `GenomeAssembler` handles slicing the flat array into per-device sub-arrays before passing them to the engine.

Integer genes would only be appropriate for devices with genuinely discrete control sets (e.g. a device that selects from a fixed set of charge rate fractions). The architecture supports this — `GenomeSlice` and `AssembledGenome` are dtype-agnostic — but no existing device requires it.

### Why are slices keyed by `device_id` (string) rather than device instance?

Using device instances as dictionary keys requires them to be hashable. Hashability based on identity (`id()`) would make lookups fragile across runs. Hashability based on content (frozen dataclass) would make it impossible to have two batteries with different runtime state but identical parameters. `device_id` strings are unambiguous, hashable, and consistent across `DeviceRegistry`, `BatchSimulationState`, `AssembledGenome`, and `EvaluationResult.repaired_genomes`.

### Why does carrier validation not happen at the port level?

A port's carrier type is determined entirely by the bus it connects to — not by the device or the port itself. Putting carrier on the port would create a redundant field that could silently disagree with the bus. The topology validator checks bus references and port uniqueness; carrier compatibility is structurally guaranteed by the bus owning the carrier type.

### Why is topology validated at engine construction, not at registration time?

Individual device registrations cannot validate bus references because the full bus list is only known at engine construction time. Validating at registration time would require pre-registering buses, adding ordering constraints with no benefit. Validating at construction gives a single, complete validation pass with access to all devices and all buses simultaneously.

### Why is `BatchSimulationState` mutable?

The engine updates device states in-place during `_simulate_batch()`. Creating new state objects at every step would allocate `horizon × population_size` intermediate arrays per generation, applying significant GC pressure in the inner loop. Mutability is the correct design for this lifecycle. `BatchSimulationState` is created fresh each generation, so there is no inter-generation state leakage.

### Why is Lamarckian repair used instead of penalty functions?

Penalty functions allow infeasible individuals to persist in the population, wasting evaluation budget on schedules that violate physical constraints. Lamarckian repair writes the corrected genotype back into the population immediately, so infeasible individuals never survive to the next generation. This is appropriate here because repair is always possible (clamping is always a valid correction) and cheap relative to evaluation cost.

### Why do `step_times` live in `SingleStateBatchState` rather than on the device?

Two alternatives were considered. Storing `step_times` on the device (via `setup_run`) would have made them available to `extract_instructions` but not to `compute_cost` — which receives only `state`, not `self`. Storing them in the state object makes them available to every lifecycle method that receives a state: `compute_cost` (useful for time-of-use tariff pricing), `apply_device_grant`, and `extract_instructions`. The cost is one extra tuple reference per state allocation — negligible.

The related decision to change `step_times` from `tuple[float, ...]` to `tuple[DateTime, ...]` throughout the API eliminates all float-to-`DateTime` conversions from device code. Every device that builds an S2 instruction can write `execution_time=state.step_times[step]` directly.

### Why no threading in the evaluator?

NumPy releases the Python GIL during array operations, so `ThreadPoolExecutor` over NumPy calls adds synchronisation overhead without parallelism benefit. The vectorization over the population axis is the correct performance strategy — the entire population is processed in a single `evaluate_population()` call with no Python loop over individuals.

---

## Implementing a New Device

The minimum implementation for a device using `SingleStateEnergyDevice`:

```python
from akkudoktoreos.devices.devicesabc import (
    EnergyPort, PortDirection, SingleStateEnergyDevice, SingleStateBatchState
)
from akkudoktoreos.simulation.genetic2.arbitrator import DeviceGrant, DeviceRequest, PortRequest
from akkudoktoreos.core.s2 import FRBCInstruction, EnergyManagementInstruction
import numpy as np

class MyBattery(SingleStateEnergyDevice):

    def __init__(self, device_id: str, device_index: int,
                 capacity_wh: float, max_power_w: float) -> None:
        super().__init__()
        self.device_id = device_id
        self._device_index = device_index
        self._capacity = capacity_wh
        self._max_power = max_power_w

    # ---- Topology ----

    @property
    def ports(self):
        return (EnergyPort(
            port_id="p0",
            bus_id="bus_dc",
            direction=PortDirection.BIDIRECTIONAL,
        ),)

    @property
    def objective_names(self):
        return ["energy_cost_eur"]

    # ---- Physics ----

    def initial_state(self) -> float:
        return self._capacity * 0.5    # Start at 50% SoC

    def state_transition_batch(self, state, power, step_interval):
        # state = SoC [Wh]; power > 0 means charging
        return state + power * step_interval / 3600.0

    def power_bounds(self):
        return (-self._max_power, self._max_power)

    def repair_batch(self, step, requested_power, current_state):
        # Default clamp first
        power = np.clip(requested_power, -self._max_power, self._max_power)
        # Prevent overcharge: cap charge so SoC doesn't exceed capacity
        max_charge = (self._capacity - current_state) * 3600.0 / self._step_interval
        power = np.minimum(power, max_charge)
        # Prevent overdischarge: cap discharge so SoC doesn't go below 0
        max_discharge = current_state * 3600.0 / self._step_interval
        power = np.maximum(power, -max_discharge)
        return power

    # ---- Arbitration ----

    def build_device_request(self, state: SingleStateBatchState):
        # Positive schedule = charging = consuming from bus (positive energy_wh)
        # Negative schedule = discharging = injecting into bus (negative energy_wh)
        pr = PortRequest(
            port_index=0,
            energy_wh=state.schedule,
            min_energy_wh=np.zeros_like(state.schedule),
        )
        return DeviceRequest(device_index=self._device_index, port_requests=(pr,))

    def apply_device_grant(self, state, grant: DeviceGrant) -> None:
        # Overwrite schedule with what was actually granted
        state.schedule[:] = grant.port_grants[0].granted_wh

    # ---- Cost ----

    def compute_cost(self, state: SingleStateBatchState) -> np.ndarray:
        # state.step_times[i] is the DateTime for step i — use for time-of-use pricing.
        # Example: flat cost proportional to total energy cycled.
        cycled_wh = np.sum(np.abs(state.schedule), axis=1, keepdims=True)
        return cycled_wh * 0.05 / 1000.0   # 0.05 EUR/kWh levelized cost

    # ---- S2 Instructions ----

    def extract_instructions(
        self,
        state: SingleStateBatchState,
        individual_index: int,
    ) -> list[EnergyManagementInstruction]:
        # state.step_times is tuple[DateTime, ...] aligned with state.schedule columns.
        # Use it directly for execution_time — no float conversion needed.
        schedule_row = state.schedule[individual_index]   # (horizon,) [W]
        instructions = []
        for power_w, dt in zip(schedule_row, state.step_times):
            # Normalise power to fill_rate: map [-max, +max] W → [0, 1]
            fill_rate = float(np.clip(
                (power_w + self._max_power) / (2 * self._max_power), 0.0, 1.0
            ))
            instructions.append(FRBCInstruction(
                resource_id=self.device_id,
                execution_time=dt,
                actuator_id="charge_actuator",
                operation_mode_id="charge_mode",
                operation_mode_factor=fill_rate,
            ))
        return instructions
```

---

## Running an Optimisation

```python
from akkudoktoreos.simulation.genetic2.registry import DeviceRegistry
from akkudoktoreos.simulation.genetic2.arbitrator import BusTopology, VectorizedBusArbitrator
from akkudoktoreos.simulation.genetic2.engine import EnergySimulationEngine, EnergySimulationInput
from akkudoktoreos.optimization.genetic2.optimizer import GeneticOptimizer
from akkudoktoreos.devices.devicesabc import EnergyBus, EnergyCarrier
import numpy as np

# 1. Define buses
dc_bus = EnergyBus(bus_id="bus_dc", carrier=EnergyCarrier.DC)
ac_bus = EnergyBus(bus_id="bus_ac", carrier=EnergyCarrier.AC)

# 2. Instantiate devices
battery = MyBattery("bat_0", device_index=0, capacity_wh=10000, max_power_w=5000)
grid    = GridConnection("grid_0", device_index=1)   # your implementation

# 3. Register devices
registry = DeviceRegistry()
registry.register(battery)
registry.register(grid)

# 4. Define bus topology (port_to_bus maps port_index → bus_index)
# Port 0 = battery.p0 on dc_bus (index 0), Port 1 = grid.p0 on ac_bus (index 1)
topology = BusTopology(
    port_to_bus=np.array([0, 1]),
    num_buses=2,
)
horizon = 24
arbitrator = VectorizedBusArbitrator(topology, horizon=horizon)

# 5. Build engine (validates topology immediately)
engine = EnergySimulationEngine(registry, [dc_bus, ac_bus], arbitrator)

# 6. Configure and run optimiser
from akkudoktoreos.utils.datetimeutil import to_datetime
import datetime

start = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
inputs = EnergySimulationInput(
    step_times=tuple(to_datetime(start + datetime.timedelta(hours=i)) for i in range(horizon)),
    step_interval=3600.0,
)

optimizer = GeneticOptimizer(
    engine=engine,
    population_size=50,
    generations=200,
    mutation_rate=0.05,
    mutation_sigma=0.1,
    tournament_size=3,
    random_seed=42,
)

result = optimizer.optimize(inputs)

# 7. Extract results
print(f"Best fitness: {result.best_scalar_fitness:.2f}")
print(f"Objectives: {result.objective_names}")

battery_schedule = result.best_genome[
    result.assembled.slices["bat_0"].start :
    result.assembled.slices["bat_0"].end
]
print(f"Battery schedule [W]: {battery_schedule}")

# 8. Extract S2 instructions for the best individual
instructions = optimizer.extract_best_instructions(result, inputs)
# instructions["bat_0"]: list[EnergyManagementInstruction], one per step
for instr in instructions["bat_0"]:
    print(f"{instr.execution_time}  factor={instr.operation_mode_factor:.2f}")

# 9. Optional: rolling-horizon for longer forecasts
from akkudoktoreos.optimization.genetic2.optimizer import RollingHorizonOptimizer

all_times = tuple(
    to_datetime(start + datetime.timedelta(hours=i)) for i in range(96)
)
rho = RollingHorizonOptimizer(
    engine=engine,
    all_step_times=all_times,
    step_interval=3600.0,
    window_size=24,
    roll_steps=8,
    population_size=50,
    generations=100,
    random_seed=0,
)
full_schedule = rho.optimize()
# full_schedule["bat_0"]: np.ndarray of shape (96,)
```

---

## Array Shape Reference

| Array | Shape | Axis 0 | Axis 1 | Axis 2 |
|---|---|---|---|---|
| `PortRequest.energy_wh` | `(pop, horizon)` | individual | time step | — |
| `PortGrant.granted_wh` | `(pop, horizon)` | individual | time step | — |
| `SingleStateBatchState.schedule` | `(pop, horizon)` | individual | time step | — |
| `SingleStateBatchState.state` | `(pop,)` | individual | — | — |
| `SingleStateBatchState.step_times` | `(horizon,)` tuple | time step | — | — |
| Arbitrator internal energy tensor | `(n_ports, pop, horizon)` | port | individual | time step |
| `EvaluationResult.fitness` | `(pop, n_obj)` | individual | objective | — |
| `repaired_genomes[device_id]` | `(pop, horizon)` | individual | time step | — |
| Flat population (optimizer) | `(pop, genome_size)` | individual | gene | — |
| `best_genome` | `(genome_size,)` | gene | — | — |
| `best_fitness_vector` | `(n_obj,)` | objective | — | — |

`pop` = `population_size`, `n_obj` = `num_objectives`, `genome_size` = `total_genome_size`

---

## Module Dependency Graph

```
optimizer.py
    ├── genome.py
    ├── s2.py              (EnergyManagementInstruction — extract_best_instructions return type)
    └── engine.py
            ├── registry.py
            │       └── devicesabc.py
            ├── topology.py
            │       └── devicesabc.py
            ├── arbitrator.py
            ├── state.py
            └── devicesabc.py
                    ├── genome.py  (GenomeSlice only)
                    └── s2.py      (EnergyManagementInstruction — extract_instructions signature)
```

All arrows point in the direction of the import dependency. There are no circular imports. `devicesabc` is the only module imported by both the `simulation` and `optimization` package trees. `s2` is imported by `devicesabc` (for the `extract_instructions` return type annotation) and by `optimizer` (for the `extract_best_instructions` return type annotation).
