# CalculiX-Aligned Drop Curve Rerun

## Purpose

The old `results/calculix_drop_impact_1s` curves were generated with the older
drop-impact runner and should not be used as final evidence.  The reruns below
use the current CalculiX-aligned path:

- CalculiX C3D4 one-point mass;
- persistent face-to-face contact springs;
- increment-start spring areas;
- active CDIS gap metrics;
- rank-aware C3D4 mass diagnostics;
- curve-only mode to avoid expensive per-step audit diagnostics.

## Successful 1s Block Drop Rerun

Command:

```bash
python validation/run_geometric_nonlinear_contact_validation.py \
  --curve-only \
  --duration 1 \
  --dt 0.001 \
  --case block_drop \
  --resolution 1 \
  --initial-velocity-z -2 \
  --contact-mode persistent_calculix_c3d4_f2f \
  --cutback-policy calculix_direct \
  --calculix-timeout-seconds 900 \
  --out-dir results/calculix_aligned_block_drop_1s_dt001
```

Output directory:

```text
results/calculix_aligned_block_drop_1s_dt001/
```

Key curve metrics:

| Metric | Value |
| --- | ---: |
| CalculiX completed | `true` |
| contact activation error | `1.39e-17` |
| z_cm L2 relative error | `2.60e-4` |
| max penetration relative error | `2.32e-2` |
| peak normal force relative error | `9.82e-3` |
| peak CELS/contact energy relative error | `2.02e-2` |
| max CNUM error | `0` |
| contact-zone von Mises relative error | `3.94e-3` |

Generated curve plots:

- `calculix_drop_z_cm.png`
- `calculix_drop_min_gap.png`
- `calculix_drop_force_proxy.png`
- `calculix_drop_contact_energy.png`
- `calculix_drop_total_energy.png`

Conclusion: the updated block-drop curve is usable as a CalculiX-aligned
longer-time curve comparison for the current scoped model.

## Failed Default 1s Sphere-Like Drop Attempt

Command attempted:

```bash
python validation/run_geometric_nonlinear_contact_validation.py \
  --curve-only \
  --duration 1 \
  --dt 0.005 \
  --case sphere_like_drop \
  --resolution 1 \
  --initial-velocity-z -2 \
  --contact-mode persistent_calculix_c3d4_f2f \
  --cutback-policy calculix_direct \
  --out-dir results/calculix_aligned_drop_impact_1s
```

Result: not usable.  CalculiX reports an impact-rule reduction of the maximum
allowed increment and exits with return code `201`.  The default sphere contact
stiffness (`2e5`) with direct fixed `dt=0.005` does not produce a completed
external reference.

## Failed 1s Sphere-Like Drop With Softer Contact

Command:

```bash
python validation/run_geometric_nonlinear_contact_validation.py \
  --curve-only \
  --duration 1 \
  --dt 0.005 \
  --case sphere_like_drop \
  --resolution 1 \
  --initial-velocity-z -2 \
  --contact-stiffness 20000 \
  --contact-mode persistent_calculix_c3d4_f2f \
  --cutback-policy calculix_direct \
  --calculix-timeout-seconds 900 \
  --out-dir results/calculix_aligned_sphere_drop_1s_k2e4
```

Result: not usable as a final curve comparison.  The short probe completed, but
the 1s run still ended with CalculiX return code `201` and only partial
external contact output.  SFC then continued beyond the partial CalculiX
reference and diverged, so the plotted 1s sphere curves must not be interpreted
as a completed external comparison.

## Current Acceptance

- Use `results/calculix_aligned_block_drop_1s_dt001` for the updated 1s block
  drop curve.
- Do not use the old `results/calculix_drop_impact_1s` curves.
- Do not use the current 1s sphere-like drop curves as paper evidence.
- To obtain a valid sphere-like drop comparison, the next step is either:
  - run a much smaller fixed time increment, likely `dt <= 0.001`, with a longer
    wall-time budget; or
  - use a gentler sphere contact setup that CalculiX completes and label it as
    a separate physical case.

