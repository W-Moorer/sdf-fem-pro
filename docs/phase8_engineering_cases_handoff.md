# Phase-8 Engineering Cases Handoff

## Scope

Phase 8 adds three paper-facing engineering validation cases without changing
core solver physics:

1. 3D cantilever beam external FEM comparison against scikit-fem.
2. Rigid flat indenter pressing an elastic block.
3. Deformable-deformable block contact with action-reaction and stress-cloud
   evidence.

These are still linear TET4, oriented current-surface distance, spatial-hash
candidate search, local projection, and frictionless normal penalty contact
cases. They do not implement friction, self-contact, nonlinear FEM, barrier
contact, GPU acceleration, or nonlinear contact equilibrium iteration.

## Commands

Full output:

```bash
python validation/run_phase8_engineering_cases.py --out-dir results/phase8
```

Quick/CI output:

```bash
python validation/run_phase8_engineering_cases.py --quick --out-dir results/phase8
```

Test coverage:

```bash
pytest -q tests/test_phase8_engineering_cases.py
```

## Output Files

- `results/phase8/phase8_cantilever_external.csv`
- `results/phase8/phase8_rigid_indenter_history.csv`
- `results/phase8/phase8_rigid_indenter_stress_cloud.csv`
- `results/phase8/phase8_deformable_deformable_history.csv`
- `results/phase8/phase8_deformable_deformable_stress_cloud.csv`
- `results/phase8/phase8_claims.csv`
- `results/phase8/phase8_plots.csv`
- `results/phase8/phase8_summary.md`
- `results/phase8/phase8_cantilever_external_stress_3d.png`
- `results/phase8/phase8_cantilever_external_stress_3d.pdf`
- `results/phase8/phase8_rigid_indenter_force_history.png`
- `results/phase8/phase8_rigid_indenter_force_history.pdf`
- `results/phase8/phase8_rigid_indenter_stress_3d.png`
- `results/phase8/phase8_rigid_indenter_stress_3d.pdf`
- `results/phase8/phase8_deformable_deformable_history.png`
- `results/phase8/phase8_deformable_deformable_history.pdf`
- `results/phase8/phase8_deformable_deformable_stress_3d.png`
- `results/phase8/phase8_deformable_deformable_stress_3d.pdf`

## Metrics

| Claim | Evidence | Gate value | Status |
| --- | --- | ---: | --- |
| 3D beam external FEM agreement | `phase8_cantilever_external.csv::stress_l2_rel_error` | `8.069196e-13` | supported |
| Rigid flat indenter response | `phase8_rigid_indenter_history.csv::normal_force_z` | `-2.880000e+04` | supported |
| Two-block action-reaction balance | `phase8_deformable_deformable_history.csv::action_reaction_imbalance` | `0.000000e+00` | supported |

The non-quick 3D cantilever beam comparison uses resolutions 1, 2, and 3,
corresponding to 15, 120, and 405 TET4 elements. The maximum displacement,
stress, von Mises, and stiffness relative errors against scikit-fem are below
`1e-8` for all reported rows.

The rigid flat indenter case uses 16 top-surface contact samples in the
non-quick run. Positive prescribed indentation produces active contact,
negative block `normal_force_z`, finite block stress, and monotone normal-force
growth.

The deformable-deformable block case reports equal-and-opposite upper/lower
normal force resultants for the tested prescribed approaches. The maximum
reported action-reaction imbalance is zero to the CSV precision.

## Interpretation

- The external FEM case strengthens the conventional FEM comparison beyond the
  earlier internal refinement trend.
- The rigid-indenter case shows that the current contact pipeline can produce a
  physically interpretable contact force and stress field under a prescribed
  engineering-style indentation.
- The deformable-deformable case shows that the slave/master penalty force
  assembly remains action-reaction balanced in a two-body setting.

## Limitations

- The rigid indenter is a flat rigid underside surface, not a curved Hertzian
  punch.
- The contact cases use prescribed approaches and one-pass penalty force
  evaluation; they are not nonlinear contact equilibrium solves.
- Stress plots are visualization outputs using nodal averaging and subdivided
  boundary triangles. Quantitative checks remain in the CSV files.
- These cases do not broaden the solver beyond linear TET4 elasticity and
  frictionless penalty normal contact.
