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

## Primary Data

- `data/calculix_contactenergy_replay.csv`
- `data/calculix_contactenergy_claims.csv`
- `data/calculix_contactenergy_commands.csv`
- `data/calculix_contactenergy_raw_cels.csv`
- `data/contactenergy.inp`
- `data/contactenergy.dat`

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

