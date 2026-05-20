# Flexible Gear Explicit Lagrangian-SDF Comparison

This validation case uses the supplied commercial flexible-gear Abaqus model as an external reference.  The runner in `validation/run_flexible_gear_explicit_sdf_comparison.py` parses the original Abaqus input deck, emits a compact Abaqus/Explicit input deck with frictionless linear penalty contact, exports Abaqus VTK frames, and compares them with an SFC Lagrangian-SDF gear replay.

The core SFC package does not import Abaqus. Abaqus is used only as an external reference solver and VTK exporter for this validation script.

## Reproduced Run

Command:

```powershell
python validation\run_flexible_gear_explicit_sdf_comparison.py `
  --out-dir results\flexible_gear_explicit_sdf `
  --duration 0.00005 `
  --fixed-dt 5e-9 `
  --output-interval 0.00001 `
  --contact-stiffness 5e9 `
  --max-contact-samples 1000 `
  --abaqus-command C:\SIMULIA\Commands\abaqus.bat
```

The Abaqus/Explicit stable time increment estimate for the parsed model was `9.42063e-9`, so the reproduced run uses `dt = 5e-9`.

## Outputs

- Abaqus VTK manifest: `results/flexible_gear_explicit_sdf/vtk/frame_manifest.csv`
- SFC VTK manifest: `results/flexible_gear_explicit_sdf/sfc_vtk/sfc_frame_manifest.csv`
- History: `results/flexible_gear_explicit_sdf/flexible_gear_history.csv`
- Metrics: `results/flexible_gear_explicit_sdf/flexible_gear_metrics.csv`
- Timing: `results/flexible_gear_explicit_sdf/flexible_gear_timing.csv`
- Curves: `results/flexible_gear_explicit_sdf/flexible_gear_displacement_curves.png`
- SDF contact history: `results/flexible_gear_explicit_sdf/flexible_gear_sdf_contact_history.png`

## Current Metrics

| Metric | L2 relative error | Max absolute error |
|---|---:|---:|
| Gear 1 mean displacement norm | 9.36% | 4.39e-6 m |
| Gear 1 max displacement norm | 11.05% | 6.46e-6 m |
| Gear 2 mean displacement norm | 0.63% | 3.04e-7 m |
| Gear 2 max displacement norm | 0.48% | 3.50e-7 m |

## Timing

| Solver path | Wall time |
|---|---:|
| Abaqus/Explicit reported analysis wall time | 37.0 s |
| SFC Lagrangian-SDF replay core time | 17.39 s |

This comparison supports the scoped validation claim that the Lagrangian-SDF contact geometry path can replay the same gear kinematics with lower core runtime than the Abaqus/Explicit external reference for this short flexible-gear case. It is not a claim of full source-level Abaqus equivalence.
