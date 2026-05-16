# CalculiX Scheibe/Ball SDF Validation

This package extends the official CalculiX evidence line beyond small contact-law examples.

## Cases

| Case | Type | Status | Main metrics |
|---|---|---|---|
| scheibe2f2f_c3d8_nonlinear_static_sdf_replay | nonlinear_static_replay | supported | max gap error `1.025350993882543e-09`, max pressure rel. `4.471130716758415e-05` |
| ball_c3d8_dynamic_drop_native_trajectory | native_dynamic | native_external_comparison_failed | z_cm rel. `4.1345077189948836e-10`, gap rel. ``, force rel. `0.0` |

## Claim Gates

| Case | Claim | Allowed | Reason |
|---|---|---|---|
| scheibe2f2f_c3d8_nonlinear_static_sdf_replay | deforming_sdf_gap_matches_calculix_copen | true | requires FRD DISP/CONTACT replay gap and pressure metrics below thresholds |
| scheibe2f2f_c3d8_nonlinear_static_sdf_replay | native_sfc_trajectory_equivalence | false | scheibe2f2f is deformed-state replay, not a native SFC solve |
| scheibe2f2f_c3d8_nonlinear_static_sdf_replay | efficiency | false | this replay isolates correctness, not timing acceleration |
| ball_c3d8_dynamic_drop_native_trajectory | native_sfc_trajectory_equivalence | false | requires native SFC C3D8 trajectory, CalculiX dynamic trajectory, active contact, and error metrics below thresholds |
| ball_c3d8_dynamic_drop_native_trajectory | external_correctness | false | requires the native trajectory equivalence gate for this dynamic contact case |
| ball_c3d8_dynamic_drop_native_trajectory | efficiency | false | ball runner is a correctness trajectory check, not a repeated timing benchmark |

## Commands

- `wsl --exec bash -lc "cd /mnt/e/workspace/sdf-fem-pro/paper/numerical_experiments/calculix_scheibe_ball_sdf_validation/calculix_runs/scheibe2f2f_c3d8_nonlinear_static_sdf_replay && ccx scheibe2f2f"`
- `wsl --exec bash -lc "cd /mnt/e/workspace/sdf-fem-pro/paper/numerical_experiments/calculix_scheibe_ball_sdf_validation/calculix_runs/ball_c3d8_dynamic_drop && ccx ball"`

## Figures

- `figures\scheibe2f2f_sdf_replay_scatter.png`: Scheibe2f2f FRD COPEN/CPRESS versus SFC deformed-SDF replay
- `figures\ball_dynamic_drop_native_trajectory.png`: Official ball dynamic drop SFC native trajectory versus CalculiX

## Interpretation

`scheibe2f2f` is a deformed-state replay validation: SFC reads the final CalculiX-deformed C3D8 surface and reproduces COPEN/CPRESS through the dynamic SDF and the same pressure-overclosure law. It does not claim a native SFC nonlinear static trajectory.

`ball` is a native C3D8 dynamic trajectory comparison against the official CalculiX drop example. The claim gate remains false unless the native SFC trajectory, contact activation, and error metrics pass the generated thresholds.
