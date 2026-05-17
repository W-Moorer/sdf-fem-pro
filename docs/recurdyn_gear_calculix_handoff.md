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
- `--dt 0.0005` by default; use smaller values for expensive dynamic checks
- `--output-every 20` by default; lower output frequency keeps `.dat` and VTK generation manageable
- `--drive-mode rigid-body|prescribed-surface|fixed`
- `--slave-surface-mode node|element-face`
- `--automatic-increment`
- `--explicit`
- `--contact-stiffness-scale`
- `--analysis dynamic|static-preload|preload-dynamic|preload-restart-dynamic`
- `--contact-law linear|exponential|tabular`
- `--contact-adjust VALUE`
- `--contact-init-distance VALUE`

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
- Rigid `GGEOM` nodes are now transformed with the parsed RMD hierarchy: `x_global = T_part * T_marker * x_local`.
- For this model, `GGEOM / 2` uses `RM=12`; marker 12 belongs to part 3, has `QP=(17.46, -1.366543853054, -0.478799197013)`, and part 3 has `REULER=(0, 0.00349065850398866, 0)`.

## Commands Actually Run

Initial distance/contact-law diagnostic after applying PART/MARKER transforms:

```bash
python validation/run_recurdyn_gear_calculix.py --quick --skip-calculix --drive-mode prescribed-surface --slave-surface-mode element-face --analysis preload-dynamic --contact-adjust 0.0 --gap-candidate-count 16 --out-dir results/recurdyn_gear_contact_initialization_quick
```

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

Static preload with contact-law proxy slope:

```bash
python validation/run_recurdyn_gear_calculix.py --analysis static-preload --preload-rotation 0.0 --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --contact-stiffness-scale 0.04522388059701491 --gap-candidate-count 16 --output-every 1 --out-dir results/recurdyn_gear_static_preload_fixed --timeout 300 --max-vtk-frames 5
```

Two-step preload-to-dynamic smoke:

```bash
python validation/run_recurdyn_gear_calculix.py --analysis preload-dynamic --duration 0.002 --dt 0.001 --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --contact-stiffness-scale 0.04522388059701491 --gap-candidate-count 16 --output-every 1 --out-dir results/recurdyn_gear_preload_dynamic_smoke --timeout 450 --max-vtk-frames 8
```

Restart preload-to-dynamic deck check with reduced output frequency and tabular contact-law proxy:

```bash
python validation/run_recurdyn_gear_calculix.py --quick --skip-calculix --analysis preload-restart-dynamic --duration 0.0005 --dt 0.0005 --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --contact-stiffness-scale 0.04522388059701491 --contact-law tabular --gap-candidate-count 16 --output-every 20 --out-dir results/recurdyn_gear_restart_tabular_deck_check --max-vtk-frames 4
```

Small-step same-job linear preload-to-dynamic smoke:

```bash
python validation/run_recurdyn_gear_calculix.py --analysis preload-dynamic --duration 0.001 --dt 0.0005 --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --contact-stiffness-scale 0.04522388059701491 --contact-law linear --gap-candidate-count 16 --output-every 20 --out-dir results/recurdyn_gear_dynamic_strategy_linear_dt0005_smoke --timeout 300 --max-vtk-frames 4
```

Small-step exponential-law smoke:

```bash
python validation/run_recurdyn_gear_calculix.py --analysis preload-dynamic --duration 0.0005 --dt 0.0005 --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --contact-stiffness-scale 0.04522388059701491 --contact-law exponential --gap-candidate-count 16 --output-every 20 --out-dir results/recurdyn_gear_dynamic_strategy_exponential_dt0005_smoke --timeout 300 --max-vtk-frames 4
```

Rigid-body two-step preload-to-dynamic smoke:

```bash
python validation/run_recurdyn_gear_calculix.py --analysis preload-dynamic --duration 0.002 --dt 0.001 --automatic-increment --drive-mode rigid-body --slave-surface-mode element-face --contact-stiffness-scale 0.04522388059701491 --gap-candidate-count 16 --output-every 1 --out-dir results/recurdyn_gear_preload_dynamic_rigidbody_smoke --timeout 450 --max-vtk-frames 8
```

Dynamic smoke with CalculiX contact adjustment:

```bash
python validation/run_recurdyn_gear_calculix.py --duration 0.002 --dt 0.001 --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --contact-stiffness-scale 0.04522388059701491 --contact-adjust 0.0 --gap-candidate-count 16 --output-every 1 --out-dir results/recurdyn_gear_adjust_dynamic_smoke --timeout 450 --max-vtk-frames 8
```

Dynamic 1 s attempt with contact-law proxy slope:

```bash
python validation/run_recurdyn_gear_calculix.py --duration 1.0 --dt 0.001 --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --contact-stiffness-scale 0.04522388059701491 --gap-candidate-count 16 --output-every 10 --out-dir results/recurdyn_gear_dynamic_1s_law_aligned --timeout 300 --max-vtk-frames 5
```

Requested 1 s attempt:

```bash
python validation/run_recurdyn_gear_calculix.py --duration 1.0 --dt 0.001 --automatic-increment --drive-mode prescribed-surface --slave-surface-mode element-face --output-every 1 --out-dir results/recurdyn_gear_calculix_1s --timeout 300 --max-vtk-frames 5
```

## Observed Results

Initial geometry diagnostic after applying the RMD PART/MARKER transform:

| Quantity | Value |
|---|---:|
| Contact-distance samples | 16,109 |
| Minimum unsigned closest distance | 4.1899e-6 mm |
| Samples with unsigned distance < 1e-3 mm | 36 |
| Minimum local signed gap | -0.271447 mm |
| Negative local signed-gap samples | 1,009 |

The unsigned closest distance is the geometric proximity metric. The local signed gap only indicates which side of the nearest oriented open rigid-surface triangle the sample lies on; it is not a robust penetration classifier for this open gear surface. Therefore the old wording "initial penetration" was too strong. The updated evidence supports "near-contact / local overclosure-side samples" rather than a confirmed large geometric interpenetration.

Contact initialization from unsigned distance and the parsed RecurDyn law:

| Quantity | Value |
|---|---:|
| Activation band | `BPEN = 0.01 mm` |
| Active unsigned-distance candidates | 153 / 16,109 |
| Local-negative active candidates | 68 |
| Maximum initialization overclosure | 9.9958e-3 mm |
| Maximum RecurDyn `KORDER=2` force proxy | 9.9916 |
| Sum RecurDyn `KORDER=2` force proxy | 639.095 |
| Recommended initialization | `static_preload_then_dynamic_restart` |
| CONTACT ADJUST recommended | false |

Interpretation: the initialization uses unsigned closest distance to decide which samples enter the RecurDyn `BPEN` activation band, then applies the parsed `K=100000`, `KORDER=2` law as a sample-level force proxy. This is not an integrated physical resultant. It is a start-up strategy diagnostic. Because the evidence indicates near-contact candidates but not robust closed-surface penetration, the recommended next run is static preload followed by dynamic restart/continuation, not geometry adjustment with `CONTACT ADJUST`.

Contact-law proxy diagnostic:

| Fit | Linear slope | Relative RMS error |
|---|---:|---:|
| Parsed K | 100000 | 20.443 |
| Endpoint fit | 6000 | 0.403 |
| Least-squares fit | 4522.388 | 0.250 |

The least-squares proxy slope corresponds to `--contact-stiffness-scale 0.04522388059701491`. This is only a proxy for RecurDyn `KORDER=2`, not a proven exact RecurDyn law.

| Run | Result |
|---|---|
| Parser/deck smoke | Passed. Generated CalculiX deck, metadata, preview figure, and placeholder VTK. |
| Rigid-body automatic | Entered CalculiX solve and printed displacement/stress data, but 0.002 s did not complete within 300 s because of repeated contact cutbacks. |
| Prescribed-surface explicit | Failed in CalculiX explicit contact path due invalid NaN energy behavior; not acceptable as a validation run. |
| Prescribed-surface face-to-face automatic | Generated 5 real CalculiX `.dat`-derived VTK frames before timeout; reached about 0.000494 s of the requested 0.002 s smoke. |
| Static preload with fitted law | Passed. Completed 10 static increments, generated stress/strain VTK frames, and contact totals. |
| Unsigned-distance + RecurDyn-law initialization | Passed as a diagnostic. It found 153 candidates inside `BPEN=0.01 mm`, recommends `static_preload_then_dynamic_restart`, and does not recommend `CONTACT ADJUST`. |
| Two-step preload-to-dynamic smoke | Passed for `T=0.002s`. It completed the static preload step and two dynamic increments, and generated 8 `.dat`-derived VTK frames. The dynamic step is still an engineering prescribed-surface mapping, and the energy output must be treated as a diagnostic rather than final validation evidence. |
| Rigid-body two-step preload-to-dynamic smoke | Failed in CalculiX with `*ERROR in add_sm_st: coefficient should be 0`; the faithful rigid-body contact/MPC mapping is blocked for this model. |
| Same-job `dt=0.0005` linear preload-to-dynamic smoke | Static preload completed, but the first dynamic increment did not complete within 300 s. The `.cvg` log showed roughly 19k contact elements during dynamic convergence. |
| `dt=0.0005` exponential contact-law smoke | The deck was syntactically generated, but the actual CalculiX run stalled during static preload within 300 s. This law is therefore not currently preferred for this model. |
| `preload-restart-dynamic` deck path | Implemented as two CalculiX jobs: a static preload deck with `*RESTART, WRITE, FREQUENCY=1`, followed by a restart dynamic deck with `*RESTART, READ, STEP=1`. The runner copies the preload `.rout` to the restart job `.rin` before launching the dynamic job. |
| `preload-restart-dynamic`, `dt=0.0005`, linear law smoke | The preload job completed and the `.rout` file was copied to the restart `.rin`. The restart dynamic job started and advanced to about `3.1875e-4 s` of the requested `5e-4 s`, then hit the 300 s restart timeout. With `--output-every 20`, no dynamic `.dat` frame was produced before timeout. |
| Dynamic smoke with `CONTACT PAIR ADJUST=0.0` | Timed out at 450 s. It produced partial VTK/contact rows but only advanced to about `6.37e-4 s` of a `0.002 s` target, so CalculiX-side adjustment of local overclosure-side samples alone does not make the transient acceptable. |
| Dynamic 1 s with fitted law | Still timed out at 300 s; impact rules forced maximum increment to 1e-5 near the first contact increment. |
| Requested 1 s run | Timed out at 300 s. It generated one real `.dat`-derived VTK frame, but did not complete the requested 1 s interval. |

Current dynamic strategy decision:

- Use `--dt 0.0005` or smaller for follow-up dynamic trials.
- Use `--output-every 20` or larger unless dense field output is explicitly required.
- Keep `--slave-surface-mode element-face` so the validation remains face-to-face contact.
- Prefer the linear fitted proxy first because the exponential proxy stalled during preload.
- The tabular pressure-overclosure law is available as a RecurDyn `KORDER` proxy deck-generation option; it must be accepted by CalculiX on this model before it is used as validation evidence.
- Prefer `--analysis preload-restart-dynamic` for long dynamic attempts because it separates accepted preload state generation from the transient contact solve.

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

Static preload VTK:

```text
results/recurdyn_gear_static_preload_fixed/vtk/recurdyn_gear_calculix_frame_0000.vtk
results/recurdyn_gear_static_preload_fixed/vtk/recurdyn_gear_calculix_frame_0001.vtk
results/recurdyn_gear_static_preload_fixed/vtk/recurdyn_gear_calculix_frame_0002.vtk
results/recurdyn_gear_static_preload_fixed/vtk/recurdyn_gear_calculix_frame_0003.vtk
results/recurdyn_gear_static_preload_fixed/vtk/recurdyn_gear_calculix_frame_0004.vtk
```

Two-step preload-to-dynamic VTK:

```text
results/recurdyn_gear_preload_dynamic_smoke/vtk/recurdyn_gear_calculix_frame_0000.vtk
results/recurdyn_gear_preload_dynamic_smoke/vtk/recurdyn_gear_calculix_frame_0001.vtk
results/recurdyn_gear_preload_dynamic_smoke/vtk/recurdyn_gear_calculix_frame_0002.vtk
results/recurdyn_gear_preload_dynamic_smoke/vtk/recurdyn_gear_calculix_frame_0003.vtk
results/recurdyn_gear_preload_dynamic_smoke/vtk/recurdyn_gear_calculix_frame_0004.vtk
results/recurdyn_gear_preload_dynamic_smoke/vtk/recurdyn_gear_calculix_frame_0005.vtk
results/recurdyn_gear_preload_dynamic_smoke/vtk/recurdyn_gear_calculix_frame_0006.vtk
results/recurdyn_gear_preload_dynamic_smoke/vtk/recurdyn_gear_calculix_frame_0007.vtk
```

## Acceptance

Conversion infrastructure: **PASS**.

Evidence:

- The RMD mesh, material, contact surfaces, hub node list, and rigid surface are parsed.
- CalculiX `.inp` generation supports C3D4 solid mesh, S3 rigid surface, node-to-surface contact, face-to-face contact, stress/strain print requests, and VTK export.
- Tests cover parser counts and generated deck structure.

Static preload stress/strain cloud after contact-law proxy alignment: **PASS**.

Unsigned-distance contact initialization: **PASS**.

Reason:

- The contact initialization now uses unsigned closest distance as the geometric metric.
- The activation band defaults to the parsed RecurDyn `BPEN=0.01 mm`.
- The initialization force proxy uses the parsed RecurDyn `K=100000`, `KORDER=2` law rather than the linear CalculiX proxy.
- The diagnostic recommends static preload before dynamic continuation and does not recommend `CONTACT ADJUST` from the current evidence.

Two-step preload-to-dynamic smoke: **PARTIAL PASS**.

Reason:

- The generated deck now supports a static preload step followed by dynamic continuation in the same CalculiX job.
- The prescribed-surface version completed a short `T=0.002s`, `dt=0.001s` smoke and exported stress/strain VTK frames.
- This does not yet validate the full `T=1s` transient, and the prescribed-surface drive remains an approximation of the original RecurDyn rigid-body joint.

Initial local-overclosure cleanup with `ADJUST=0.0`: **NOT SUFFICIENT**.

Reason:

- CalculiX accepted the generated `CONTACT PAIR, ADJUST=0.0` deck, but the short dynamic run still timed out.
- It advanced only to about `6.37e-4s` of `0.002s` after 450 s wall time.
- Therefore the blocking issue is not just initial slave-node adjustment; it is the combination of dense gear contact, prescribed-surface drive, contact law mismatch, and severe cutback behavior.

Full 1 s CalculiX transient stress/strain cloud at `dt=0.001`: **NOT PASSED YET**.

Reason:

- The faithful rigid-body mapping is dominated by CalculiX contact cutbacks.
- The explicit path is incompatible with nonlinear rigid-body/contact MPCs, and the prescribed-surface explicit attempt produced invalid energy output.
- The face-to-face automatic mapping generated real stress/strain VTK frames but timed out before completing even the 0.002 s smoke interval.
- The initial distance diagnostic shows near-contact samples and many local signed-gap samples on the negative side of open rigid-surface triangles, but it no longer proves broad initial geometric penetration.
- A static preload with the contact-law proxy slope is stable.
- A short preload-to-dynamic continuation completes, but full `1s` dynamic evidence has not been produced.
- `CONTACT PAIR ADJUST=0.0` alone does not remove the dynamic bottleneck.

## Recommended Next Step

Before claiming this RecurDyn gear case as external paper evidence, reduce it to a stable CalculiX benchmark:

1. Continue using unsigned closest distance, not local signed side alone, to verify initial geometric clearance.
2. Use the completed static preload result as the initial state for a longer transient only after the short preload-to-dynamic run has physically acceptable energy and contact histories.
3. Use C3D4 face-to-face contact as the primary CalculiX contact mode.
4. Tune pressure-overclosure law from RecurDyn `KORDER=2` to a CalculiX tabular or exponential law, rather than using a one-value linear approximation.
5. Prefer an actual restart workflow if the two-step deck becomes too expensive, but treat it as equivalent only after displacement, reaction, contact energy, and stress histories are checked.
6. Only after a physically acceptable preload-to-transient smoke completes should the full `T=1s`, `dt=0.001s` run be accepted.

This case should not yet be used as a final paper result.
