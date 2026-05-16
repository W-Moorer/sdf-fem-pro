# Phase-9 Full Contact Validation

This locked package is generated from `validation/run_phase9_full_contact_validation.py` in quick mode with the local CalculiX reference enabled.

Scope:
- Short-term multi-element scope is C3D4/TET4 plus C3D8. C3D10 is explicitly excluded.
- C3D8 linear static contact uses the existing contactenergy replay evidence.
- C3D8 geometric-nonlinear static contact uses native SFC C3D8 StVK mechanics and CalculiX `contactenergy` output.
- C3D8 geometric-nonlinear dynamic contact uses native SFC C3D8 HHT/Newmark block-plane dynamics and CalculiX trajectory output.
- Efficiency claims remain blocked unless case-specific timing evidence is generated.

Primary files:
- `phase9_full_contact_validation.csv`
- `phase9_claim_gates.csv`
- `phase9_full_contact_validation_summary.md`
- `native_c3d8_nonlinear_static_contactenergy.csv`
- `native_c3d8_nonlinear_dynamic_block_plane.csv`
- `native_c3d8_nonlinear_dynamic_comparison.csv`
- `figures/`
- `vtk/`
