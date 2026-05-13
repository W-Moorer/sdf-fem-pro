# Paper Draft Package

This directory contains the manuscript draft for the Phase-6 paper-result
package plus the Phase-7 physical-comparison supplement.

Primary TeX draft:

- `main.tex`
- `references.bib`

Markdown source draft:

- `manuscript.md`

Build from this directory with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Evidence source:

- `../results/paper_final/`
- `../results/phase7/`
- `../results/external_fem/`
- `../docs/phase6_final_experiment_handoff.md`
- `../docs/phase7_physical_validation_handoff.md`
- `../docs/external_fem_comparison_handoff.md`
- `../docs/validation_plan.md`

Scope for the submission draft:

- Linear TET4 finite elements.
- Oriented local current-surface distance.
- Spatial-hash broad phase.
- Local closest-point projection.
- Frictionless penalty normal contact.
- Speedup claims only against brute-force all-triangle projection.
- Stress/strain correctness claims only for affine small-strain linear
  elasticity.
- Standard FEM displacement/stress trends use the finest internal TET4 run as
  reference, not an external commercial solver.
- The primary standard-FEM visual comparison is the Phase-7 3D boundary-surface
  von Mises stress plot, not only scalar trend lines.
- External open-source FEM comparison uses scikit-fem for the same linear TET4
  cantilever model.

Unsupported claims that must not be added without new evidence:

- Frictional contact.
- Self-contact.
- Nonlinear FEM or large-deformation material models.
- GPU acceleration.
- Barrier contact.
- Production BVH or IPC superiority.
- Robust global SDF sign for arbitrary non-manifold geometry.
