# Deforming Master Dynamic-SDF Reference Handoff

## Purpose

This validation checks the SFC contact-query method on prescribed current master
surfaces that undergo stretch, shear, and bending-style deformations. It
compares:

- spatial hash + dynamic FEM-induced SDF on the current master surface;
- brute-force all-face closest-projection reference on the same current master
  surface.

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
- `deforming_master_sdf_claims.csv`
- `deforming_master_sdf_summary.md`

## Latest Full Result

| Metric | Value |
| --- | ---: |
| query rows | `1008` |
| force rows | `9` |
| max gap absolute error | `0.000000e+00` |
| max normal angle error | `2.980232e-08` |
| max force relative error | `0.000000e+00` |

Both claim gates are supported:

- `broad_phase_dynamic_sdf_matches_bruteforce_on_deforming_master`
- `dynamic_sdf_contact_force_matches_bruteforce_reference`

## Interpretation

For the tested prescribed deforming master surfaces, broad-phase candidate
filtering does not change the dynamic-SDF gap, normal, or assembled penalty
force relative to all-face projection. This supports the SDF/contact-query side
of the method under current-surface deformation.

## Limitations

- The master surface deformation is prescribed, not solved as a coupled second
  deformable body in this runner.
- The comparison is against brute-force closest projection, not CalculiX active
  contact generation.
- This validates gap/normal/contact-force query equivalence, not nonlinear
  material behavior or external solver equivalence.
