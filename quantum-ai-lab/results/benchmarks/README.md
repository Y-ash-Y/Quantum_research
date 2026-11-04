# Benchmark families

| Folders | Evidence |
|---|---|
| `fixed-policy-v2`, `robustness-v2`, `count-aware-v2`, `nuisance-v2` | Classical/physical comparisons and honest controls |
| `loss-aware-v2`, `window-power-v2` | Gain-aware and finite-window screening |
| `calibration-protocol-v1`, `allocation-frontier-v1` | Reference-data accounting and allocation choices |
| `exact-comparison-v1`, `best-response-stability-v2`, `off-grid-geometry-v1`, `detector-power-v1` | Restricted search, geometry and power studies |
| `drift-pilot-v1`, `direct-drift-pilot-v1`, `classical-drift-audit-v1` | Drift rules and selected paired classical replay |

`*.json` files identify the declared settings and aggregate results; published
`*.csv.gz` files retain the selected synthetic observations or decisions needed
for replay. Some campaigns additionally use large local arrays that are not
part of the public package.

Read the associated [protocol and interpretation](../../docs/README.md) before
combining numbers across folders: the endpoints, exposure budgets and allowed
honest variation differ. Version labels are experiment identifiers, not a
promise that every later detector is uniformly better.
