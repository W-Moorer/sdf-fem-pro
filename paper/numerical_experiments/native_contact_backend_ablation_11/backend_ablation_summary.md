# Native Contact Backend Ablation

This ablation separates SFC backend engineering choices from the independent CalculiX accuracy comparison.
CalculiX is intentionally skipped here; the reference for endpoint differences is the non-vectorized surface-to-surface quadrature backend.

## Backends

- `node_to_surface`: preserved centroid/sample backend.
- `surface_to_surface_reference`: scalar reference surface quadrature backend.
- `surface_to_surface_vectorized`: vectorized quadrature and force assembly.
- `surface_to_surface_vectorized_grouped_projection`: vectorized quadrature with forced spatial-hash grouped SDF field build.

## Outputs

- `backend_ablation.csv`
- `figures/backend_ablation_runtime.png`
- `figures/backend_ablation_runtime.pdf`
- `figures/backend_ablation_endpoint_metrics.png`
- `figures/backend_ablation_endpoint_metrics.pdf`
- `figures/backend_ablation_overview.png`
- `figures/backend_ablation_overview.pdf`

## Timing Rows

| Backend | SFC total s | Field update s | Query s | Linear solve s | Endpoint disp diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| `node_to_surface` | 2.900228e+00 | 8.526329e-01 | 1.834925e+00 | 4.530120e-02 | 8.679244e-04 |
| `surface_to_surface_reference` | 1.730898e+01 | 1.300163e+00 | 1.458134e+01 | 4.267650e-02 | 0.000000e+00 |
| `surface_to_surface_vectorized` | 1.342239e+00 | 1.202841e+00 | 6.078520e-02 | 5.237890e-02 | 4.348113e-16 |
| `surface_to_surface_vectorized_grouped_projection` | 2.351295e+01 | 2.308684e+01 | 6.823160e-02 | 4.622450e-02 | 4.348113e-16 |

## Scope

- Supports the paper statement that the backend contains both node-to-surface and surface-to-surface paths.
- Supports an engineering-cost ablation among preserved/reference/vectorized/batch field-build implementations.
- Does not replace the CalculiX native-contact comparison or SDF field accuracy/Jacobian claim gates.
