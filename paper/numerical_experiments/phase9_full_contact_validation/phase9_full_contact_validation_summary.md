# Phase-9 Full Contact Validation

This package organizes the evidence needed for static/dynamic, linear/nonlinear, multi-element contact claims.

The current short-term multi-element scope is C3D4/TET4 plus C3D8. C3D10 is explicitly out of scope because no backend is registered.

## Commands

- `python validation/run_phase9_full_contact_validation.py --quick --out-dir paper\numerical_experiments\phase9_full_contact_validation`
- `wsl --exec bash -lc "cd /mnt/e/workspace/sdf-fem-pro/paper/numerical_experiments/phase9_full_contact_validation/c3d8_linear_dynamic_calculix/linear_dynamic_block_plane_c3d8_r1 && ccx linear_dynamic_block_plane_c3d8_r1"`
- `wsl --exec bash -lc "cd /mnt/e/workspace/sdf-fem-pro/paper/numerical_experiments/phase9_full_contact_validation/c3d8_linear_dynamic_calculix/linear_dynamic_block_block_c3d8_r1 && ccx linear_dynamic_block_block_c3d8_r1"`
- `wsl --exec bash -lc "cd /mnt/e/workspace/sdf-fem-pro/paper/numerical_experiments/phase9_full_contact_validation/curved_nonplanar_contact/calculix_runs/curved_nonplanar_c3d8_r2 && ccx curved_nonplanar_c3d8_r2"`

## Evidence Matrix

| Case | Analysis | Linearity | Native SFC | CalculiX | External correctness | Trajectory equivalence | Efficiency | Status |
|---|---|---|---|---|---|---|---|---|
| c3d8_linear_static_contact | static | linear | true | true | true | false | false | supported |
| c3d8_nonlinear_static_contact | static | geometric_nonlinear | true | true | true | false | false | supported |
| c3d8_linear_dynamic_block_plane_contact | dynamic | linear | true | true | true | true | true | supported |
| c3d8_linear_dynamic_block_block_contact | dynamic | linear | true | true | true | true | true | supported |
| c3d8_nonlinear_dynamic_block_plane_contact | dynamic | geometric_nonlinear | true | true | true | true | false | supported |
| c3d8_nonlinear_dynamic_block_block_contact | dynamic | geometric_nonlinear | true | true | true | true | false | supported |
| c3d8_curved_nonplanar_contact_replay | dynamic_replay | geometric_nonlinear | true | true | false | false | false | external_sdf_replay_metric_failed |

## Claim Gates

| Case | Claim | Allowed | Reason |
|---|---|---|---|
| c3d8_linear_static_contact | external_correctness | true | requires native SFC result and CalculiX comparison |
| c3d8_linear_static_contact | trajectory_equivalence | false | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_linear_static_contact | efficiency | false | requires acceleration/timing evidence for this case |
| c3d8_nonlinear_static_contact | external_correctness | true | requires native SFC result and CalculiX comparison |
| c3d8_nonlinear_static_contact | trajectory_equivalence | false | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_nonlinear_static_contact | efficiency | false | requires acceleration/timing evidence for this case |
| c3d8_linear_dynamic_block_plane_contact | external_correctness | true | requires native SFC result and CalculiX comparison |
| c3d8_linear_dynamic_block_plane_contact | trajectory_equivalence | true | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_linear_dynamic_block_plane_contact | efficiency | true | requires acceleration/timing evidence for this case |
| c3d8_linear_dynamic_block_block_contact | external_correctness | true | requires native SFC result and CalculiX comparison |
| c3d8_linear_dynamic_block_block_contact | trajectory_equivalence | true | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_linear_dynamic_block_block_contact | efficiency | true | requires acceleration/timing evidence for this case |
| c3d8_nonlinear_dynamic_block_plane_contact | external_correctness | true | requires native SFC result and CalculiX comparison |
| c3d8_nonlinear_dynamic_block_plane_contact | trajectory_equivalence | true | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_nonlinear_dynamic_block_plane_contact | efficiency | false | requires acceleration/timing evidence for this case |
| c3d8_nonlinear_dynamic_block_block_contact | external_correctness | true | requires native SFC result and CalculiX comparison |
| c3d8_nonlinear_dynamic_block_block_contact | trajectory_equivalence | true | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_nonlinear_dynamic_block_block_contact | efficiency | false | requires acceleration/timing evidence for this case |
| c3d8_curved_nonplanar_contact_replay | external_correctness | false | requires native SFC result and CalculiX comparison |
| c3d8_curved_nonplanar_contact_replay | trajectory_equivalence | false | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_curved_nonplanar_contact_replay | efficiency | false | requires acceleration/timing evidence for this case |

## Figures

- `figures\phase9_claim_gate_matrix.png`: Phase-9 claim support matrix
- `figures\phase9_contact_error_metrics.png`: Phase-9 contact error metrics

## Additional Tables

- `phase9_c3d4_c3d8_side_by_side.csv`: C3D4 and C3D8 contact cases in one comparison table.
- `phase9_solver_timing.csv`: native SFC and CalculiX wall-clock timing where both commands are run by this evidence package.
- `phase9_curved_nonplanar_contact_external.csv`: warped/non-planar C3D8 master-surface external replay check.

## Current Conclusion

The current evidence supports C3D8 linear static contact correctness, native C3D8 linear dynamic block-plane/block-block trajectory comparison, native C3D8 geometric-nonlinear static contact comparison on the contactenergy reference, native C3D8 nonlinear block-plane/block-block dynamics when the CSV error gates pass, and a warped/non-planar C3D8 external current-surface replay check. Efficiency is only allowed per case when the timing CSV has both SFC and CalculiX wall times and speedup is greater than one. For generated CalculiX dynamic steps, CDIS rows may be empty; in that case the gap metric is computed by replaying the CalculiX displacement geometry with the same current-surface dynamic-SDF query rather than by inventing unavailable CDIS output.
