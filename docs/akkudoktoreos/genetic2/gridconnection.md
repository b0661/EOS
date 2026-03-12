# Grid connection device

What GridConnectionDevice needs to do:

- No genome
    The grid is the passive residual of all other devices. It has no genes to optimise;
    it just absorbs or injects whatever the AC bus needs after all other devices have been granted.
- build_device_request
    Request to cover the remaining bus imbalance. The simplest approach: request a very large
    bidirectional range so the arbitrator always grants it (it's the slack bus). Or don't participate in
    arbitration at all and measure the residual after grants are applied.
- apply_device_grant
    Record the granted import/export per step.
- compute_cost
    Import_wh *import_cost_per_kwh - export_wh* export_revenue_per_kwh, returning shape
    (population_size, 1) under the "energy_cost_eur" objective name.
- extract_instructions
    No S2 instructions needed (the grid isn't controllable).
