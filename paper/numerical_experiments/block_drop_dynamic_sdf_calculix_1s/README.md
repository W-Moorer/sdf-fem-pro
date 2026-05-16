# Locked Numerical Experiment: Dynamic-SDF Block Drop vs CalculiX

## Status

This folder freezes the paper-bound numerical experiment for the full dynamic
trajectory comparison between SFC dynamic-SDF contact and CalculiX.

The locked case is:

- case: `block_drop`
- duration: `1.0 s`
- time step: `0.001 s`
- initial vertical velocity: `-2`
- SFC contact mode: `persistent_dynamic_sdf_calculix_f2f`
- cutback policy: `calculix_direct`
- external solver: CalculiX

This is the binding paper case for the current dynamic-SDF full-trajectory
claim. Future code changes must not weaken this case without explicitly
updating the evidence, thresholds, and paper wording.

## Reproduce

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

## Primary Figure

- `figures/calculix_drop_z_cm.png`

Secondary figures:

- `figures/calculix_drop_min_gap.png`
- `figures/calculix_drop_force_proxy.png`
- `figures/calculix_drop_contact_energy.png`
- `figures/calculix_drop_total_energy.png`

## Locked Metrics

The locked comparison row is in `data/trajectory_comparison.csv`.

| Metric | Locked value |
| --- | ---: |
| contact activation time error | `1.3877787807814457e-17` |
| z_cm L2 relative error | `2.6014929027548756e-04` |
| max penetration relative error | `2.3167937699955544e-02` |
| peak normal force relative error | `9.816750590013102e-03` |
| peak contact energy relative error | `2.0223011325782192e-02` |
| max CNUM absolute error | `0.0` |
| contact-zone von Mises relative error | `3.937602844759979e-03` |

The acceptance thresholds are versioned in `locked_thresholds.csv` and enforced
by `tests/test_locked_paper_experiments.py`.

## Supported Claim

For this scoped block-plane impact benchmark, replacing the contact query with
the current-surface dynamic SDF backend preserves the native SFC dynamic
trajectory against CalculiX to small relative errors over the tested 1 second
run.

## Boundaries

This locked case does not claim arbitrary contact equivalence. It does not cover
friction, self-contact, sphere drop, production BVH, private CalculiX branch
equivalence, or general nonlinear contact beyond this scoped benchmark.
