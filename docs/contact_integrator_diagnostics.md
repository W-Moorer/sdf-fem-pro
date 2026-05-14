# Contact Integrator Diagnostics

## Purpose

This note records the block/plane contact diagnostics after changing the SFC
validation contact integration from one face centroid to three in-face triangle
quadrature points. It also separates HHT-alpha dissipation from contact/time
step effects by running `alpha=0` Newmark checks and reduced time increments.

These diagnostics are validation-only. They do not modify the core `src/sfc`
solver.

## Commands

Three-point quadrature, HHT alpha `-0.05`, `dt=0.002`:

```bash
python validation/run_calculix_drop_impact_comparison.py --quick --case block_drop --duration 1.0 --dt 0.002 --output-frequency 1 --initial-velocity-z 0.0 --gravity 9.81 --contact-stiffness 5000 --out-dir results/calculix_gentle_contact_quadrature
```

Three-point quadrature, Newmark alpha `0`, `dt=0.002`:

```bash
python validation/run_calculix_drop_impact_comparison.py --quick --case block_drop --duration 1.0 --dt 0.002 --output-frequency 1 --initial-velocity-z 0.0 --gravity 9.81 --contact-stiffness 5000 --hht-alpha 0 --out-dir results/calculix_gentle_contact_newmark_alpha0
```

Newmark time-step checks:

```bash
python validation/run_calculix_drop_impact_comparison.py --quick --case block_drop --duration 1.0 --dt 0.001 --output-frequency 1 --initial-velocity-z 0.0 --gravity 9.81 --contact-stiffness 5000 --hht-alpha 0 --out-dir results/calculix_gentle_contact_newmark_dt001
python validation/run_calculix_drop_impact_comparison.py --quick --case block_drop --duration 1.0 --dt 0.0005 --output-frequency 1 --initial-velocity-z 0.0 --gravity 9.81 --contact-stiffness 5000 --hht-alpha 0 --out-dir results/calculix_gentle_contact_newmark_dt0005
```

HHT time-step checks:

```bash
python validation/run_calculix_drop_impact_comparison.py --quick --case block_drop --duration 1.0 --dt 0.001 --output-frequency 1 --initial-velocity-z 0.0 --gravity 9.81 --contact-stiffness 5000 --out-dir results/calculix_gentle_contact_hht_dt001
python validation/run_calculix_drop_impact_comparison.py --quick --case block_drop --duration 1.0 --dt 0.0005 --output-frequency 1 --initial-velocity-z 0.0 --gravity 9.81 --contact-stiffness 5000 --out-dir results/calculix_gentle_contact_hht_dt0005
```

## Summary Metrics

| Run | CalculiX completed | first contact C/SFC | z rel. error | gap Linf error | C max force | SFC max force | C active | SFC active | SFC final energy drift | SFC final z |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HHT dt=0.002 | true | 0.096 / 0.096 | 1.406341e-02 | 1.797834e-02 | 35.1383 | 32.5677 | 14 | 6 | -2.427503e-04 | 0.430159 |
| HHT dt=0.001 | false | 0.096 / 0.096 | 2.332804e-04 | 2.531019e-03 | 24.1120 | 32.7795 | 14 | 6 | 3.112712e-04 | 0.429701 |
| HHT dt=0.0005 | false | 0.096 / 0.096 | 2.092275e-04 | 2.556400e-03 | 23.8802 | 32.8253 | 14 | 6 | 3.474994e-04 | 0.429570 |
| Newmark dt=0.002 | false | 0.096 / 0.096 | 2.571739e-04 | 2.495084e-03 | 24.5637 | 32.6222 | 14 | 6 | 4.593720e-05 | 0.430101 |
| Newmark dt=0.001 | false | 0.096 / 0.096 | 2.336644e-04 | 2.533671e-03 | 24.0991 | 32.7861 | 14 | 6 | 3.428155e-04 | 0.429675 |
| Newmark dt=0.0005 | false | 0.096 / 0.096 | 2.093221e-04 | 2.556940e-03 | 23.8777 | 32.8272 | 14 | 6 | 3.529187e-04 | 0.429564 |

For the incomplete rows, the `z rel. error` and `gap Linf error` are computed
only over the partial CalculiX output interval. Those rows should not be used
as supported external validation claims.

## Interpretation

The three-point triangle quadrature increases the SFC active contact samples
from the earlier two bottom-face centroid points to six bottom-face quadrature
points in the `block_drop r1` model. This moves the SFC contact integration
closer to CalculiX surface-to-surface contact, but the active counts are still
not equivalent: CalculiX reports 14 contact spring elements at peak contact,
while SFC reports 6 active quadrature points.

The HHT `dt=0.002` row is the only completed CalculiX row in this diagnostic
set. The first contact time remains aligned at `0.096 s`, and the physical
rebound-height gate remains supported. However, the center-of-mass curve still
does not match as a full time-history reference. The SFC force peak is closer
to CalculiX than the earlier centroid formulation, but the rebound peaks still
decay differently.

The `alpha=0` Newmark runs show that SFC can complete the 1-second history
without HHT damping, but CalculiX direct integration with `ALPHA=0` exits early
with return code `201` due to persistent-contact energy residual divergence.
Therefore `alpha=0` is useful here as an SFC internal diagnostic, not as a full
external CalculiX reference.

The SFC Newmark time-step results are close across `dt=0.002`, `0.001`, and
`0.0005`: final mass-center height changes from `0.430101` to `0.429564`, and
the energy drift remains below `4e-4` relative. This suggests that the remaining
SFC/CalculiX curve mismatch is dominated by contact formulation and external
solver contact behavior rather than by SFC time-step error alone.

## Current Conclusion

The updated SFC validation contact is more physically appropriate than the
single-centroid approximation because it integrates pressure over multiple
in-face points. It still does not reproduce CalculiX time histories exactly.
The remaining mismatch is mainly attributable to different contact
discretization, active contact generation, and CalculiX convergence behavior in
persistent contact. The evidence supports contact timing and qualitative
response checks, not a high-precision dynamic contact time-history claim.
