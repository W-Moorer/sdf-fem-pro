# Locked Numerical Experiment: CalculiX Contactenergy C3D8 Replay

## Status

This folder freezes the paper-adjacent external contact-law evidence based on
the official CalculiX `contactenergy.inp` C3D8 test. CalculiX performs the
static C3D8 surface-to-surface contact solve. SFC then replays the final
deformed geometry with the current-surface dynamic SDF query on triangulated
C3D8 boundary faces and evaluates the same linear pressure-overclosure law.

This validates that the dynamic SDF contact query is not topologically bound to
TET4 surfaces: once the current finite-element boundary is represented as
oriented triangles, the same query can evaluate a C3D8/HEX8 surface.

## Reproduce

```bash
python validation/run_calculix_contactenergy_replay.py --out-dir results/calculix_contactenergy_replay
```

For CI or quick local replay without running `ccx`:

```bash
python validation/run_calculix_contactenergy_replay.py --quick --skip-calculix --out-dir results/calculix_contactenergy_replay_quick
```

## Primary Data and Figures

- `data/calculix_contactenergy_replay.csv`
- `data/calculix_contactenergy_claims.csv`
- `data/calculix_contactenergy_commands.csv`
- `data/calculix_contactenergy_raw_cels.csv`
- `data/calculix_contactenergy_plots.csv`
- `data/calculix_contactenergy_stress_strain_cloud.csv`
- `data/contactenergy.inp`
- `data/contactenergy.dat`
- `figures/calculix_contactenergy_stress_strain_3d.png`
- `figures/calculix_contactenergy_stress_strain_3d.pdf`
- `figures/calculix_contactenergy_contact_pressure_3d.png`
- `figures/calculix_contactenergy_contact_pressure_3d.pdf`

## Visualization Scheme

The paper-facing visualization uses the same current engineering cloud scheme
as the scikit-fem cantilever evidence:

1. C3D8 element-center engineering strain and linear elastic stress are
   post-processed from the final CalculiX displacement field;
2. element von Mises stress and engineering strain norm are averaged to nodes;
3. triangulated C3D8 boundary faces interpolate the nodal values;
4. boundary triangles are subdivided for a smoother Matplotlib surface
   rendering.

The deformed geometry is shown with a labeled `1000x` displacement scale because
the official static contactenergy displacement is only \(O(10^{-4})\) in model
units. This makes the deformation visible without changing the quantitative CSV
values.

## Locked Metrics

The locked run reports:

- contact force relative error: `3.797932679461269e-08`
- contact energy relative error: `3.798346405165345e-08`
- dynamic SDF backend: `current_surface_triangulated_c3d8_faces`
- element type: `c3d8`

The acceptance thresholds are versioned in `locked_thresholds.csv` and enforced
by `tests/test_locked_paper_experiments.py`.

## Supported Claim

For the official CalculiX `contactenergy.inp` C3D8 contact-energy reference,
SFC dynamic SDF replay on triangulated current C3D8 boundary faces reproduces
the total contact spring energy and normal force to the locked tolerance.

## Boundaries

This is not a TET4 trajectory-equivalence claim. It is a C3D8 static
contact-law/energy replay. SFC does not solve C3D8 mechanics in this case; it
replays CalculiX final displacements with the current-surface dynamic SDF
contact query.
