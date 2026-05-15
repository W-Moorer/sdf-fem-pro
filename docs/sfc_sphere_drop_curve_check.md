# SFC Sphere Drop Curve Check

Date: 2026-05-16

This note records the latest SFC-only sphere drop curve runs. The generated
CSV and plot files are under `results/`, which is ignored by git.

## Commands

```bash
python validation/run_geometric_nonlinear_contact_validation.py --curve-only --skip-calculix --duration 1 --dt 0.005 --case sphere_like_drop --resolution 1 --initial-velocity-z -2 --contact-mode persistent_calculix_c3d4_f2f --cutback-policy calculix_direct --out-dir results/sfc_sphere_drop_1s_dt005_curve
```

```bash
python validation/run_geometric_nonlinear_contact_validation.py --curve-only --skip-calculix --duration 3 --dt 0.005 --case sphere_like_drop --resolution 1 --initial-velocity-z -2 --contact-mode persistent_calculix_c3d4_f2f --cutback-policy calculix_direct --out-dir results/sfc_sphere_drop_3s_dt005_curve
```

An attempted `dt=0.001` run did not finish in the available time budget:
the 1 s job exceeded 30 minutes and the 3 s job exceeded 60 minutes before
producing output files.

## Outputs

1 s run:

- `results/sfc_sphere_drop_1s_dt005_curve/geometric_contact_history.csv`
- `results/sfc_sphere_drop_1s_dt005_curve/calculix_drop_z_cm.png`
- `results/sfc_sphere_drop_1s_dt005_curve/calculix_drop_min_gap.png`
- `results/sfc_sphere_drop_1s_dt005_curve/calculix_drop_force_proxy.png`
- `results/sfc_sphere_drop_1s_dt005_curve/calculix_drop_contact_energy.png`
- `results/sfc_sphere_drop_1s_dt005_curve/calculix_drop_total_energy.png`

3 s run:

- `results/sfc_sphere_drop_3s_dt005_curve/geometric_contact_history.csv`
- `results/sfc_sphere_drop_3s_dt005_curve/calculix_drop_z_cm.png`
- `results/sfc_sphere_drop_3s_dt005_curve/calculix_drop_min_gap.png`
- `results/sfc_sphere_drop_3s_dt005_curve/calculix_drop_force_proxy.png`
- `results/sfc_sphere_drop_3s_dt005_curve/calculix_drop_contact_energy.png`
- `results/sfc_sphere_drop_3s_dt005_curve/calculix_drop_total_energy.png`

## Metrics

| Run | Rows | End time | Contact activation | z_cm initial | z_cm final | z_cm max | Max penetration | Peak normal force | Peak contact energy | Final relative total-energy drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 s, dt=0.005 | 201 | 1.0 | 0.025 | 0.542085 | 36.079990 | 36.079990 | 2.107352 | 4407.510538 | 771.249111 | 9817.307402 |
| 3 s, dt=0.005 | 601 | 3.0 | 0.025 | 0.542085 | 85.905402 | 85.905402 | 2.107352 | 4407.510538 | 771.249111 | 9890.942053 |

## Interpretation

These SFC-only curves are not physically acceptable validation evidence. The
center of mass is numerically launched far above the initial height after
contact, and the total mechanical energy increases by several orders of
magnitude. The result indicates that the current sphere-drop configuration
still has a serious contact/nonlinear dynamics stability issue.

The runs are useful as a failure reproduction case, not as a paper claim.
The already-aligned block-drop curve remains the valid current external
comparison evidence; sphere drop still needs stabilization or a validated
external reference setup before it can be used.
