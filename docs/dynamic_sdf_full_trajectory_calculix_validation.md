# Dynamic-SDF Full Trajectory CalculiX Validation

## Purpose

This is the paper-facing dynamic contact validation requested for full
trajectory comparison. Unlike the deformed-state replay check, this run compares
the native SFC trajectory against an external CalculiX trajectory over time.

The SFC side uses:

- geometric nonlinear StVK TET4 mechanics in the validation backend;
- C3D4 one-point mass alignment;
- persistent face-to-face contact spring lifecycle;
- hard linear pressure-overclosure;
- fixed direct increments matching the generated CalculiX `*DYNAMIC, DIRECT`
  path;
- `persistent_dynamic_sdf_calculix_f2f`, where the contact query is driven by
  the current-surface dynamic SDF adapter.

## Paper Command

```bash
python validation/run_geometric_nonlinear_contact_validation.py \
  --curve-only \
  --duration 1 \
  --dt 0.001 \
  --case block_drop \
  --resolution 1 \
  --initial-velocity-z -2 \
  --contact-mode persistent_dynamic_sdf_calculix_f2f \
  --cutback-policy calculix_direct \
  --calculix-timeout-seconds 900 \
  --out-dir results/dynamic_sdf_full_trajectory_block_drop_1s_dt001
```

## Outputs

- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/geometric_contact_history.csv`
- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/geometric_contact_calculix_comparison.csv`
- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/geometric_contact_alignment_diagnostics.csv`
- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/geometric_contact_claims.csv`
- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/geometric_contact_validation_summary.md`
- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/calculix_drop_z_cm.png`
- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/calculix_drop_min_gap.png`
- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/calculix_drop_force_proxy.png`
- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/calculix_drop_contact_energy.png`
- `results/dynamic_sdf_full_trajectory_block_drop_1s_dt001/calculix_drop_total_energy.png`

## Latest 1s Result

| Quantity | Value |
| --- | ---: |
| CalculiX completed | `true` |
| SFC contact mode | `persistent_dynamic_sdf_calculix_f2f` |
| duration | `1.0 s` |
| dt | `0.001 s` |
| contact activation time error | `1.39e-17 s` |
| z_cm L2 relative error | `2.60e-4` |
| max penetration relative error | `2.32e-2` |
| peak normal force relative error | `9.82e-3` |
| peak contact energy relative error | `2.02e-2` |
| max CNUM absolute error | `0` |
| rebound height CalculiX | `6.15529975e-1` |
| rebound height SFC | `6.15669095e-1` |
| contact-zone von Mises relative error | `3.94e-3` |

The scoped acceptance gate reports `passed_scoped_gate`.

## Quick Regression Command

```bash
python validation/run_geometric_nonlinear_contact_validation.py \
  --quick \
  --curve-only \
  --case block_drop \
  --resolution 1 \
  --contact-mode persistent_dynamic_sdf_calculix_f2f \
  --cutback-policy calculix_direct \
  --out-dir results/dynamic_sdf_full_trajectory_quick
```

The quick run currently reports:

- `z_cm_l2_rel_error = 3.98e-06`
- `max_penetration_rel_error = 2.07e-04`
- `peak_normal_force_rel_error = 6.75e-04`
- `max_cnum_abs_error = 0`

## Supported Paper Statement

For the scoped block-plane impact benchmark, using the CalculiX-aligned
mechanics/contact lifecycle and replacing the contact query with the
current-surface dynamic SDF backend preserves the full dynamic contact
trajectory against CalculiX to small relative errors over the tested 1 second
run.

## Limitations

- This is a scoped block-plane benchmark, not arbitrary contact equivalence.
- It validates the current `persistent_dynamic_sdf_calculix_f2f` path, not every
  legacy contact mode.
- It does not claim full source-level CalculiX equivalence or private branch
  equivalence.
- Sphere-like drop remains a separate case and should not be claimed until the
  external CalculiX reference completes robustly.
- The result is still limited to the implemented geometric nonlinear StVK TET4
  validation backend and frictionless normal contact.
