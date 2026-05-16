# CalculiX Deformable Dynamic-SDF Contact Validation

## Purpose

This validation checks the SFC dynamic surface-SDF geometry layer against an
external CalculiX deformed contact state. CalculiX solves a small
two-deformable-block normal-compression contact problem. SFC then replays the
final CalculiX-deformed coordinates and evaluates the slave surface points
against the current lower-block master surface with `dynamic_surface_sdf`.

This is a geometry replay validation. It is not a claim that the full native
SFC two-body nonlinear contact trajectory is identical to CalculiX.

## Paper Command

```bash
python validation/run_calculix_deformable_sdf_contact_validation.py --timeout 240 --out-dir results/calculix_deformable_sdf_contact_paper
```

## Outputs

- `results/calculix_deformable_sdf_contact_paper/calculix_deformable_sdf_contact_comparison.csv`
- `results/calculix_deformable_sdf_contact_paper/calculix_deformable_sdf_contact_claims.csv`
- `results/calculix_deformable_sdf_contact_paper/calculix_deformable_sdf_contact_commands.csv`
- `results/calculix_deformable_sdf_contact_paper/calculix_deformable_sdf_contact_paper_metrics.csv`
- `results/calculix_deformable_sdf_contact_paper/calculix_deformable_sdf_contact_plots.csv`
- `results/calculix_deformable_sdf_contact_paper/calculix_deformable_sdf_contact_summary.md`
- `results/calculix_deformable_sdf_contact_paper/calculix_deformable_sdf_contact_gap_error.png`
- `results/calculix_deformable_sdf_contact_paper/calculix_deformable_sdf_contact_gap_error.pdf`

## Latest Paper Result

The latest paper run completed all nine external CalculiX cases: three mesh
resolutions `r1/r2/r3` and three prescribed approaches `0.02/0.04/0.06`.

| Resolution | Master faces | Approach | CalculiX min CDIS | SFC min dynamic-SDF gap | Difference | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 0.02 | no contact rows | `1.500000e-02` | n/a | sign agreement |
| 1 | 2 | 0.04 | `-4.086740e-03` | `-4.086740e-03` | `1.0e-10` | gap agreement |
| 1 | 2 | 0.06 | `-2.039199e-02` | `-2.039200e-02` | `7.0e-09` | gap agreement |
| 2 | 8 | 0.02 | no contact rows | `1.500000e-02` | n/a | sign agreement |
| 2 | 8 | 0.04 | `-4.086740e-03` | `-4.086740e-03` | `1.0e-10` | gap agreement |
| 2 | 8 | 0.06 | `-2.039199e-02` | `-2.039200e-02` | `7.0e-09` | gap agreement |
| 3 | 18 | 0.02 | no contact rows | `1.500000e-02` | n/a | sign agreement |
| 3 | 18 | 0.04 | `-4.086740e-03` | `-4.086740e-03` | `1.0e-10` | gap agreement |
| 3 | 18 | 0.06 | `-2.039199e-02` | `-2.039200e-02` | `7.0e-09` | gap agreement |

Supported gates:

- CalculiX completes the scoped two-deformable-block references: `9/9`.
- SFC dynamic-SDF replay agrees with CalculiX contact/no-contact sign.
- In this lateral-constrained normal-compression replay, SFC minimum gap agrees
  with CalculiX minimum `CDIS` to output precision, with maximum difference
  `7.000000024071085e-09`.
- The evidence includes three mesh resolutions and six completed contact cases.

No convergence order is claimed. The evidence supports a paper statement about
external deformed-state SDF replay across the tested mesh resolutions.

## Important Parser Fix

Earlier replay diagnostics showed impossible coordinates such as lateral
displacements of order `1e3`. The cause was not the SDF query. The shared
CalculiX `.dat` displacement parser continued reading numeric rows after the
node displacement table and overwrote node displacements with later
`CDIS/CSTR/CELS` contact-print rows. The parser now records each node once per
displacement block and ignores subsequent contact output rows for that time
block.

## Scope and Limitations

- The CalculiX input fixes all lateral `x/y` DOFs to isolate a controlled
  normal-compression contact state.
- The validation uses C3D4 blocks and frictionless normal contact.
- This validates current-surface SDF gap evaluation on an externally deformed
  contact state.
- It does not validate full SFC/CalculiX trajectory equivalence, CalculiX
  private contact lifecycle branches, friction, self-contact, or nonlinear
  production contact behavior.
