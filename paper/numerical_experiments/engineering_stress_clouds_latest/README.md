# Locked Numerical Experiment: Latest Engineering Stress Clouds

## Status

This folder freezes the latest paper-facing engineering stress-cloud exports
for:

- 3D cantilever beam external FEM comparison;
- rigid indenter pressing an elastic block;
- deformable-deformable block contact.

These figures are included to make the deformation and stress fields visually
inspectable in the manuscript and supplementary material.

## Reproduce

```bash
python validation/run_phase8_engineering_cases.py --out-dir results/phase8
```

## Primary Data and Figures

- `data/phase8_cantilever_external.csv`
- `data/phase8_rigid_indenter_history.csv`
- `data/phase8_deformable_deformable_history.csv`
- `figures/phase8_cantilever_external_stress_3d.png`
- `figures/phase8_rigid_indenter_stress_3d.png`
- `figures/phase8_deformable_deformable_stress_3d.png`

## Stress Cloud Scheme

This folder intentionally uses the current engineering visualization scheme:

1. element stress is averaged to nodes;
2. boundary triangle colors are interpolated from nodal stress;
3. boundary triangles are subdivided for smoother surface rendering.

This is not the previous incorrect visual scheme where the object could appear
as a nearly uniform block or where element-constant values obscured the body
surface and stress variation.

## Supported Claims

- The cantilever stress field is tied to the external scikit-fem comparison.
- The rigid indenter case shows a prescribed normal-contact response and
  contact-zone stress visualization.
- The deformable-deformable case shows action-reaction balance and stress-cloud
  visualization for the locked penalty contact setup.

## Boundaries

These stress clouds are engineering visualization evidence. Quantitative
external FEM agreement is locked in
`../scikit_fem_cantilever_external/`, and dynamic full-trajectory contact
agreement is locked in `../block_drop_dynamic_sdf_calculix_1s/`.

