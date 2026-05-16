# CalculiX Deformable Dynamic-SDF Contact Validation

## Purpose

This validation checks the SFC dynamic surface-SDF geometry layer against an
external CalculiX deformed contact state. CalculiX solves a small
two-deformable-block normal-compression contact problem. SFC then replays the
final CalculiX-deformed coordinates and evaluates the slave surface points
against the current lower-block master surface with `dynamic_surface_sdf`.

This is a geometry replay validation. It is not a claim that the full native
SFC two-body nonlinear contact trajectory is identical to CalculiX.

## Command

```bash
python validation/run_calculix_deformable_sdf_contact_validation.py --quick --resolution 1 --timeout 180 --out-dir results/calculix_deformable_sdf_contact_quick
```

## Outputs

- `results/calculix_deformable_sdf_contact_quick/calculix_deformable_sdf_contact_comparison.csv`
- `results/calculix_deformable_sdf_contact_quick/calculix_deformable_sdf_contact_claims.csv`
- `results/calculix_deformable_sdf_contact_quick/calculix_deformable_sdf_contact_commands.csv`
- `results/calculix_deformable_sdf_contact_quick/calculix_deformable_sdf_contact_summary.md`

## Latest Quick Result

The latest quick run completed both external CalculiX cases:

| Approach | CalculiX min CDIS | SFC min dynamic-SDF gap | Difference | Status |
| ---: | ---: | ---: | ---: | --- |
| 0.02 | no contact rows | `1.500000e-02` | n/a | sign agreement |
| 0.06 | `-2.039199e-02` | `-2.039200e-02` | `7.0e-09` | gap agreement |

Supported gates:

- CalculiX completes the scoped two-deformable-block reference.
- SFC dynamic-SDF replay agrees with CalculiX contact/no-contact sign.
- In this lateral-constrained normal-compression replay, SFC minimum gap agrees
  with CalculiX minimum `CDIS` to output precision.

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
