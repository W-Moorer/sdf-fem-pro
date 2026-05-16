# Deforming Master Dynamic-SDF Reference Handoff

## Purpose

This validation checks the SFC contact-query method on prescribed current master
surfaces that undergo stretch, shear, and bending-style deformations. It
compares:

- spatial hash + dynamic FEM-induced SDF on the current master surface;
- brute-force all-face closest-projection reference on the same current master
  surface;
- frozen reference-surface SDF baseline, which intentionally does not follow
  the current master deformation.

The brute-force path is validation-only evidence. It is not the production
contact pipeline.

## Commands

Quick smoke:

```bash
python validation/run_deforming_master_sdf_reference.py --quick --out-dir results/deforming_master_sdf_reference
```

Full r1/r2/r3-style surface run:

```bash
python validation/run_deforming_master_sdf_reference.py --out-dir results/deforming_master_sdf_reference_full
```

## Output Files

- `deforming_master_sdf_queries.csv`
- `deforming_master_sdf_forces.csv`
- `deforming_master_sdf_frozen_baseline.csv`
- `deforming_master_sdf_claims.csv`
- `deforming_master_sdf_summary.md`

## Latest Quick Result

| Metric | Value |
| --- | ---: |
| query rows | `48` |
| force rows | `2` |
| frozen baseline rows | `48` |
| max gap absolute error | `0.000000e+00` |
| max normal angle error | `2.580957e-08` |
| max force relative error | `0.000000e+00` |
| frozen-reference max gap difference | `1.468324e-02` |
| frozen-reference max normal-angle difference | `3.603990e-02` |

All claim gates are supported:

- `broad_phase_dynamic_sdf_matches_bruteforce_on_deforming_master`
- `dynamic_sdf_contact_force_matches_bruteforce_reference`
- `frozen_reference_sdf_differs_from_current_deforming_master_sdf`

## Interpretation

For the tested prescribed deforming master surfaces, broad-phase candidate
filtering does not change the dynamic-SDF gap, normal, or assembled penalty
force relative to all-face projection. This supports the SDF/contact-query side
of the method under current-surface deformation.

The frozen baseline rows show that if the SDF is not updated with the current
master coordinates, the same slave samples acquire measurable gap and normal
errors. This is the direct evidence that the FEM-induced dynamic SDF follows the
deforming master surface rather than using a static reference surface.

## Limitations

- The master surface deformation is prescribed, not solved as a coupled second
  deformable body in this runner.
- The comparison is against brute-force closest projection, not CalculiX active
  contact generation.
- This validates gap/normal/contact-force query equivalence, not nonlinear
  material behavior or external solver equivalence.
