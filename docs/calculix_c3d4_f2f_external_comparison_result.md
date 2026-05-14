# CalculiX C3D4 F2F External Comparison Result

## Command

```bash
python validation/run_geometric_nonlinear_contact_validation.py --contact-mode calculix_c3d4_f2f --out-dir results/geometric_nonlinear_contact_validation_c3d4_f2f
```

The run completed successfully and wrote:

- `results/geometric_nonlinear_contact_validation_c3d4_f2f/geometric_contact_history.csv`
- `results/geometric_nonlinear_contact_validation_c3d4_f2f/geometric_contact_calculix_comparison.csv`
- `results/geometric_nonlinear_contact_validation_c3d4_f2f/geometric_contact_mesh_convergence.csv`
- `results/geometric_nonlinear_contact_validation_c3d4_f2f/geometric_contact_timestep_convergence.csv`
- `results/geometric_nonlinear_contact_validation_c3d4_f2f/geometric_contact_stress_clouds.csv`
- `results/geometric_nonlinear_contact_validation_c3d4_f2f/external_solver_commands.csv`
- `results/geometric_nonlinear_contact_validation_c3d4_f2f/geometric_contact_claims.csv`
- `results/geometric_nonlinear_contact_validation_c3d4_f2f/geometric_contact_validation_summary.md`

## Compared Against Previous Result

Previous reference directory:

```text
results/geometric_nonlinear_contact_validation_full
```

That previous result used the older SFC three-point triangle quadrature contact path.
The new result uses `calculix_c3d4_f2f`, i.e. one C3D4 slave-face centroid sample
per linear triangular boundary face and hard linear pressure-overclosure.

## Relative Error Comparison

| Resolution | Metric | Old three-point | New C3D4 F2F | Change |
| ---: | --- | ---: | ---: | ---: |
| r1 | max penetration rel. error | `7.727284e-02` | `4.817436e-02` | `-2.909849e-02` |
| r1 | peak RF/normal force rel. error | `3.333271e-02` | `5.722847e-02` | `+2.389576e-02` |
| r1 | CELS/contact energy rel. error | `6.486660e-02` | `1.112276e-01` | `+4.636101e-02` |
| r2 | max penetration rel. error | `1.865451e-01` | `1.600986e-01` | `-2.644655e-02` |
| r2 | peak RF/normal force rel. error | `1.364248e-01` | `1.238871e-01` | `-1.253770e-02` |
| r2 | CELS/contact energy rel. error | `3.018645e-01` | `2.883060e-01` | `-1.355845e-02` |
| r3 | max penetration rel. error | `1.229794e-01` | `1.505151e-01` | `+2.753576e-02` |
| r3 | peak RF/normal force rel. error | `1.811129e-01` | `1.880409e-01` | `+6.928053e-03` |
| r3 | CELS/contact energy rel. error | `3.820071e-01` | `3.998027e-01` | `+1.779559e-02` |

Negative change means the new strict C3D4 F2F mode moved closer to CalculiX.
Positive change means it moved farther away.

## New Absolute Metrics

| Resolution | max pen. C | max pen. SFC | peak RF C | peak force SFC | peak CELS C | peak energy SFC | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r1 | `3.144397e-03` | `3.295876e-03` | `4.118874e+01` | `4.354591e+01` | `6.457809e-02` | `7.176096e-02` | `passed_scoped_gate` |
| r2 | `2.743927e-03` | `3.183225e-03` | `3.077931e+01` | `3.459247e+01` | `3.775696e-02` | `4.864252e-02` | `passed_scoped_gate` |
| r3 | `2.690467e-03` | `3.095423e-03` | `2.922631e+01` | `3.472205e+01` | `3.346637e-02` | `4.684631e-02` | `passed_scoped_gate` |

## CNUM / Active Count Check

| Resolution | CalculiX CNUM old | Old SFC active samples | CalculiX CNUM new | New SFC active springs | Old abs. diff. | New abs. diff. |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | `14` | `6` | `14` | `2` | `8` | `12` |
| r2 | `56` | `24` | `56` | `8` | `32` | `48` |
| r3 | `126` | `54` | `126` | `18` | `72` | `108` |

The CNUM comparison became worse numerically because the new strict SFC mode
reports one active spring per active slave triangle, while CalculiX `CNUM` is
reported from its generated contact-element bookkeeping. The source-level read
already showed that CalculiX `CNUM` is not the same quantity as SFC active sample
count. The older three-point quadrature count was closer only by coincidence; it
was not more CalculiX-equivalent.

## Interpretation

The strict C3D4 F2F mode partially improves the external comparison:

- max penetration improves for r1 and r2;
- peak RF/normal force and CELS/contact energy improve for r2;
- r3 becomes slightly worse for penetration, force, and energy;
- CNUM does not become closer because the output definitions remain different.

This means the previous discrepancy was not only the slave-face quadrature rule
or smooth/hard overclosure law. The remaining dominant differences are still:

1. CalculiX persistent contact-element generation and active-set bookkeeping;
2. CalculiX cutback/contact convergence heuristics;
3. CalculiX `CNUM` and `CELS` output definitions;
4. full contact spring lifecycle across increments and Newton iterations.

## Current Conclusion

`calculix_c3d4_f2f` is a better scoped validation mode because it removes one
known discretization mismatch: three-point SFC quadrature vs. one-point C3D4
CalculiX face contact. However, it does not yet make the contact histories
uniformly closer to CalculiX.

The next necessary implementation step is persistent CalculiX-style contact
spring bookkeeping: store contact spring elements across increment/iteration
states, reuse/drop them with a CalculiX-like positive-clearance rule, and report
CNUM/CELS from that same spring-element state.
