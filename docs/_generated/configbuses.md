## Configuration for all energy buses in the system

Validates that all ``bus_id`` values are unique. Each device's
``ports`` field must reference a ``bus_id`` that exists here —
this cross-reference is validated by the engine at construction time,
not here (to keep config loading independent of engine instantiation).

Call ``to_genetic2_param()`` to obtain the ``list[EnergyBus]`` required by
``EnergySimulationEngine``.

<!-- pyml disable line-length -->
:::{table} buses
:widths: 10 20 10 5 5 30
:align: left

| Name | Environment Variable | Type | Read-Only | Default | Description |
| ---- | -------------------- | ---- | --------- | ------- | ----------- |
| buses | `EOS_BUSES__BUSES` | `list[akkudoktoreos.devices.settings.devicebasesettings.BusConfig]` | `rw` | `required` | List of energy buses in the system. |
:::
<!-- pyml enable line-length -->

<!-- pyml disable no-emphasis-as-heading -->
**Example Input/Output**
<!-- pyml enable no-emphasis-as-heading -->

<!-- pyml disable line-length -->
```json
   {
       "buses": {
           "buses": [
               {
                   "bus_id": "bus_dc",
                   "carrier": "dc",
                   "constraint": null
               },
               {
                   "bus_id": "bus_ac",
                   "carrier": "ac",
                   "constraint": null
               }
           ]
       }
   }
```
<!-- pyml enable line-length -->

### Optional structural constraints on port counts for a bus

Attributes:
    max_sinks: Maximum number of sink ports allowed on this bus,
        or ``None`` for no limit.
    max_sources: Maximum number of source ports allowed on this bus,
        or ``None`` for no limit.

<!-- pyml disable line-length -->
:::{table} buses::buses::list::constraint
:widths: 10 10 5 5 30
:align: left

| Name | Type | Read-Only | Default | Description |
| ---- | ---- | --------- | ------- | ----------- |
| max_sinks | `Optional[int]` | `rw` | `None` | Maximum number of sink ports on this bus. |
| max_sources | `Optional[int]` | `rw` | `None` | Maximum number of source ports on this bus. |
:::
<!-- pyml enable line-length -->

<!-- pyml disable no-emphasis-as-heading -->
**Example Input/Output**
<!-- pyml enable no-emphasis-as-heading -->

<!-- pyml disable line-length -->
```json
   {
       "buses": {
           "buses": [
               {
                   "constraint": {
                       "max_sinks": 1,
                       "max_sources": 1
                   }
               }
           ]
       }
   }
```
<!-- pyml enable line-length -->

### Configuration for a single energy bus

A bus is a connection point that devices attach to via ports. All ports
on the same bus exchange energy of the same carrier type.

Attributes:
    bus_id: Unique identifier for this bus. Referenced by
        ``PortConfig.bus_id`` on all devices that connect to it.
    carrier: Energy carrier type flowing through this bus.
        ``"ac"`` = AC electricity, ``"dc"`` = DC electricity,
        ``"heat"`` = thermal energy.
    constraint: Optional structural limits on the number of source or
        sink ports. Used by topology validation at engine construction.

<!-- pyml disable line-length -->
:::{table} buses::buses::list
:widths: 10 10 5 5 30
:align: left

| Name | Type | Read-Only | Default | Description |
| ---- | ---- | --------- | ------- | ----------- |
| bus_id | `str` | `rw` | `required` | Unique bus identifier. Referenced by device port configs. |
| carrier | `<enum 'EnergyCarrier'>` | `rw` | `required` | Energy carrier type: 'ac', 'dc', or 'heat'. |
| constraint | `Optional[akkudoktoreos.devices.settings.devicebasesettings.BusConstraintConfig]` | `rw` | `None` | Optional structural constraints on the number of source or sink ports. Validated at engine construction time. |
:::
<!-- pyml enable line-length -->

<!-- pyml disable no-emphasis-as-heading -->
**Example Input/Output**
<!-- pyml enable no-emphasis-as-heading -->

<!-- pyml disable line-length -->
```json
   {
       "buses": {
           "buses": [
               {
                   "bus_id": "bus_ac",
                   "carrier": "ac",
                   "constraint": null
               }
           ]
       }
   }
```
<!-- pyml enable line-length -->
