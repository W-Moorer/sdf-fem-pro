# Phase-9 Full Contact Validation

This package organizes the evidence needed for static/dynamic, linear/nonlinear, multi-element contact claims.

The current short-term multi-element scope is C3D4/TET4 plus C3D8. C3D10 is explicitly out of scope because no backend is registered.

## Commands

- `python validation/run_phase9_full_contact_validation.py --quick --out-dir results\phase9_full_contact_validation`

## Evidence Matrix

| Case | Analysis | Linearity | Native SFC | CalculiX | External correctness | Trajectory equivalence | Efficiency | Status |
|---|---|---|---|---|---|---|---|---|
| c3d8_linear_static_contact | static | linear | true | true | true | false | false | supported |
| c3d8_nonlinear_static_contact | static | geometric_nonlinear | true | true | true | false | false | supported |
| c3d8_linear_dynamic_contact | dynamic | linear | false | true | false | false | false | external_replay_only |
| c3d8_nonlinear_dynamic_contact | dynamic | geometric_nonlinear | true | true | true | true | false | supported |

## Claim Gates

| Case | Claim | Allowed | Reason |
|---|---|---|---|
| c3d8_linear_static_contact | external_correctness | true | requires native SFC result and CalculiX comparison |
| c3d8_linear_static_contact | trajectory_equivalence | false | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_linear_static_contact | efficiency | false | requires acceleration/timing evidence for this case |
| c3d8_nonlinear_static_contact | external_correctness | true | requires native SFC result and CalculiX comparison |
| c3d8_nonlinear_static_contact | trajectory_equivalence | false | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_nonlinear_static_contact | efficiency | false | requires acceleration/timing evidence for this case |
| c3d8_linear_dynamic_contact | external_correctness | false | requires native SFC result and CalculiX comparison |
| c3d8_linear_dynamic_contact | trajectory_equivalence | false | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_linear_dynamic_contact | efficiency | false | requires acceleration/timing evidence for this case |
| c3d8_nonlinear_dynamic_contact | external_correctness | true | requires native SFC result and CalculiX comparison |
| c3d8_nonlinear_dynamic_contact | trajectory_equivalence | true | requires native SFC trajectory and CalculiX trajectory comparison |
| c3d8_nonlinear_dynamic_contact | efficiency | false | requires acceleration/timing evidence for this case |

## Figures

- `figures\phase9_claim_gate_matrix.png`: Phase-9 claim support matrix
- `figures\phase9_contact_error_metrics.png`: Phase-9 contact error metrics

## Current Conclusion

The current evidence supports C3D8 linear static contact correctness and native C3D8 geometric-nonlinear static contact comparison on the contactenergy reference. Native C3D8 nonlinear dynamics now runs for the block-plane case and is compared against CalculiX, but the trajectory-equivalence claim is allowed only when the CSV error gates pass. C3D8 linear dynamic remains external replay evidence, and per-case efficiency claims remain blocked without matching timing evidence.
