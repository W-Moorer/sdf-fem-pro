# Phase-9 External Visual Validation and Backend Scope Handoff

## Commands

```bash
python validation/run_external_visual_validation.py --quick --out-dir results/external_visual
pytest -q
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Purpose

Phase-9 adds visual audit artifacts and backend-distribution documentation without changing the core dynamic narrow-band SDF method.

`pytest -q` output:

```text
259 passed in 387.51s (0:06:27)
```

`paper/main.pdf` was regenerated from `paper/main.tex`. The final LaTeX log has no unresolved-reference or overfull-box warnings.

The visual validation supports physical plausibility for the tested generated reference case. It does not prove the SDF acceleration claim, and it does not establish CalculiX/source-level external solver equivalence.

## Generated Outputs

`results/external_visual`:

- `external_visual_metrics.csv`
- `external_visual_summary.md`
- `vtk/sfc_visual_fields.vtu`
- `vtk/reference_visual_fields.vtu`
- `vtk/absolute_error_visual_fields.vtu`
- `vtk/sdf_phi_slice.vtu`
- `vtk/sdf_contact_samples.vtu`
- `figures/external_visual_fields.png`
- `figures/external_visual_fields.pdf`
- `figures/dynamic_sdf_field_visualization.png`
- `figures/dynamic_sdf_field_visualization.pdf`
- `sdf_phi_slice.csv`
- `sdf_contact_samples.csv`

## Metrics

From `external_visual_metrics.csv`:

- Displacement L2 relative error: `1.404040e-02`.
- von Mises L2 relative error: `2.699310e-02`.
- Strain-norm L2 relative error: `2.104498e-02`.
- Gap Linf absolute error: `4.000000e-04`.
- Pressure L2 relative error: `9.654852e-03`.
- Contact-force relative error: `8.775744e-03`.
- Deformation scale used in figures: `4.0`.

## Visual Artifacts

`external_visual_fields.png` renders deformed 3D boundary-surface clouds and compares:

- SFC displacement magnitude / reference displacement magnitude / absolute error.
- SFC von Mises stress / reference von Mises stress / absolute error.
- SFC strain norm / reference strain norm / absolute error.
- SFC contact pressure / reference contact pressure / absolute error.

SFC and reference columns use matched color scales; error columns use separate error color scales. The rows are surface-rendered fields, not projected 2D point clouds.

`dynamic_sdf_field_visualization.png` renders 3D surface/slice views:

- Current FEM boundary colored by interpolated gap.
- Narrow-band SDF phi slice.
- Narrow-band validity slice.
- Contact-pressure surface with scalar-gradient normals.

## Backend Documentation

Added:

- `docs/backend_api.md`
- `docs/distribution_scope.md`

The package is documented as:

```text
research backend for dynamic SDF field contact queries
```

not:

```text
general-purpose production FEM contact solver
```

## Scope and Constraints

No friction, self-contact, nonlinear FEM main method, GPU path, barrier contact, POD, neural SDF, Abaqus dependency, or core CalculiX dependency was introduced.

The generated external reference is used for visual validation only. External solver importers may remain optional validation tools, but they are not required by core SDF field construction, field query, contact Jacobian assembly, or penalty force assembly.

## Manuscript Status

Updated:

- `paper/main.tex`
- `paper/manuscript.md`
- `paper/main.pdf`

Added subsection:

- `External Visual FEM Validation`

The subsection explicitly states that visual validation supports physical auditability, not SDF acceleration or external-solver equivalence.

All current paper-facing validation figures are regenerated with Times-style English fonts through the shared Matplotlib configuration in the validation scripts.
