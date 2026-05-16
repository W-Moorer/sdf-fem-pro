# RecurDyn Gear CalculiX Conversion Handoff

## Scope

Source model: `assets/jiandanjiaolian.rmd`.

Goal: convert the RecurDyn flexible gear contact model into a CalculiX reference deck and produce stress/strain VTK clouds for review in ParaView.

This is an external validation runner only. It does not modify `src/sfc` and does not make RecurDyn or CalculiX a core solver dependency.

## Implemented Runner

Command:

```bash
python validation/run_recurdyn_gear_calculix.py --rmd assets/jiandanjiaolian.rmd --out-dir results/recurdyn_gear_calculix
```

Important options:

- `--duration 1.0`
- `--dt 0.001`
- `--drive-mode rigid-body|prescribed-surface|fixed`
- `--slave-surface-mode node|element-face`
- `--automatic-increment`
- `--explicit`
- `--contact-stiffness-scale`

## RMD Mapping

Parsed model content:

| Quantity | Value |
|---|---:|
| Flexible nodes | 13,630 |
| Flexible elements | 64,644 C3D4 / TET4 |
| Flexible contact surface triangles | 10,738 |
| Flexible contact nodes | 5,371 |
| C3D4 surface face refs matched | 10,738 |
| Rigid gear surface nodes | 1,641 |
| Rigid gear surface triangles | 3,278 |
| FRBE hub fixed nodes | 607 |
| Material E | 200000 N/mm^2 |
| Material nu | 0.285 |
| Density for CalculiX | 7.85e-9 tonne/mm^3 |

Contact mapping:

- RecurDyn contact: `IGGEOMID=3`, `JGGEOMID=2`, `INODECONTACT=1`, `K=100000`, `KORDER=2`, `C=10`.
- CalculiX node mode: `TYPE=NODE TO SURFACE`.
- CalculiX face mode: RecurDyn surface triangles mapped to C3D4 element faces, then `TYPE=SURFACE TO SURFACE`.
- RecurDyn `KORDER=2` is not exactly reproduced by CalculiX linear pressure-overclosure; the runner records this as an approximation.

Rigid gear motion:

- Default faithful mode uses CalculiX `*RIGID BODY` with a rotational node.
- Engineering cloud mode can use `--drive-mode prescribed-surface`, which applies the final rigid rotation displacement directly to rigid surface nodes. This avoids explicit-dynamics rigid-body/contact MPC incompatibility but is an approximation of the RecurDyn rigid-body joint.

## Commands Actually Run

Parser/deck smoke:

```bash
python validation/run_recurdyn_gear_calculix.py --quick --skip-calculix --out-dir results/recurdyn_gear_calculix_smoke
```

Rigid-body automatic-increment smoke:

```bash
python validation/run_recurdyn_gear_calculix.py --quick --output-every 1 --out-dir results/recurdyn_gear_calculix_ccx_smoke --timeout 300
```

Prescribed-surface explicit smoke:

```bash
python validation/run_recurdyn_gear_calculix.py --quick --explicit --drive-mode prescribed-surface --output-every 1 --out-dir results/recurdyn_gear_calculix_prescribed_explicit_smoke --timeout 300
```

Prescribed-surface C3D4 face-to-face automatic smoke:

```bash
python validation/run_recurdyn_gear_calculix.py --quick --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --output-every 1 --out-dir results/recurdyn_gear_calculix_f2f_smoke --timeout 300 --max-vtk-frames 5
```

Requested 1 s attempt:

```bash
python validation/run_recurdyn_gear_calculix.py --duration 1.0 --dt 0.001 --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --output-every 1 --out-dir results/recurdyn_gear_calculix_1s --timeout 300 --max-vtk-frames 5
```

## Observed Results

| Run | Result |
|---|---|
| Parser/deck smoke | Passed. Generated CalculiX deck, metadata, preview figure, and placeholder VTK. |
| Rigid-body automatic | Entered CalculiX solve and printed displacement/stress data, but 0.002 s did not complete within 300 s because of repeated contact cutbacks. |
| Prescribed-surface explicit | Failed in CalculiX explicit contact path due invalid NaN energy behavior; not acceptable as a validation run. |
| Prescribed-surface face-to-face automatic | Generated 5 real CalculiX `.dat`-derived VTK frames before timeout; reached about 0.000494 s of the requested 0.002 s smoke. |
| Requested 1 s run | Timed out at 300 s. It generated one real `.dat`-derived VTK frame, but did not complete the requested 1 s interval. |

Generated partial VTK evidence:

```text
results/recurdyn_gear_calculix_f2f_smoke/vtk/recurdyn_gear_calculix_frame_0000.vtk
results/recurdyn_gear_calculix_f2f_smoke/vtk/recurdyn_gear_calculix_frame_0001.vtk
results/recurdyn_gear_calculix_f2f_smoke/vtk/recurdyn_gear_calculix_frame_0002.vtk
results/recurdyn_gear_calculix_f2f_smoke/vtk/recurdyn_gear_calculix_frame_0003.vtk
results/recurdyn_gear_calculix_f2f_smoke/vtk/recurdyn_gear_calculix_frame_0004.vtk
```

Requested 1 s partial VTK:

```text
results/recurdyn_gear_calculix_1s/vtk/recurdyn_gear_calculix_frame_0000.vtk
```

## Acceptance

Conversion infrastructure: **PASS**.

Evidence:

- The RMD mesh, material, contact surfaces, hub node list, and rigid surface are parsed.
- CalculiX `.inp` generation supports C3D4 solid mesh, S3 rigid surface, node-to-surface contact, face-to-face contact, stress/strain print requests, and VTK export.
- Tests cover parser counts and generated deck structure.

Full 1 s CalculiX stress/strain cloud at `dt=0.001`: **NOT PASSED YET**.

Reason:

- The faithful rigid-body mapping is dominated by CalculiX contact cutbacks.
- The explicit path is incompatible with nonlinear rigid-body/contact MPCs, and the prescribed-surface explicit attempt produced invalid energy output.
- The face-to-face automatic mapping generated real stress/strain VTK frames but timed out before completing even the 0.002 s smoke interval.

## Recommended Next Step

Before claiming this RecurDyn gear case as external paper evidence, reduce it to a stable CalculiX benchmark:

1. Verify initial clearance/penetration between the two gear surfaces.
2. Run a static contact preload solve before transient motion.
3. Use C3D4 face-to-face contact as the primary CalculiX contact mode.
4. Tune pressure-overclosure law from RecurDyn `KORDER=2` to a CalculiX tabular or exponential law, rather than using a one-value linear approximation.
5. Only after the 0.002 s smoke completes should the full `T=1s`, `dt=0.001s` run be attempted.

This case should not yet be used as a final paper result.
