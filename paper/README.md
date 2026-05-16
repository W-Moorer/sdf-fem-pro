# Paper Draft Package

This directory contains the manuscript draft for the Phase-6 paper-result
package plus the Phase-7 physical-comparison and external FEM/contact
supplements.

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
- `../results/external_contact_solver/`
- `../results/calculix_contactenergy_replay/`
- `../results/c3d8_contact_trajectory_quick/`
- `../docs/phase6_final_experiment_handoff.md`
- `../docs/phase7_physical_validation_handoff.md`
- `../docs/external_fem_comparison_handoff.md`
- `../docs/external_contact_solver_handoff.md`
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
- External contact solver evidence uses SfePy to solve a two-body penalty
  contact problem and replays the final deformed geometry with SFC
  current-surface gap queries.
- CalculiX `contactenergy.inp` evidence is a C3D8 contact-law/energy replay
  showing that the dynamic SDF query consumes triangulated current boundary
  surfaces and is not inherently TET4-bound.
- Generated CalculiX C3D8 block-plane and block-block trajectory evidence is
  an external trajectory replay: CalculiX supplies the deformed states, and SFC
  replays current-surface dynamic SDF gap, force, energy, and stress-cloud
  diagnostics on triangulated C3D8 boundaries.

Unsupported claims that must not be added without new evidence:

- Frictional contact.
- Self-contact.
- Nonlinear FEM or large-deformation material models.
- Native nonlinear C3D8 dynamic trajectory equivalence to CalculiX.
- GPU acceleration.
- Barrier contact.
- Production BVH or IPC superiority.
- Robust global SDF sign for arbitrary non-manifold geometry.
