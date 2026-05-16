# Phase-9 Full Contact Validation

This locked package is generated from `validation/run_phase9_full_contact_validation.py` in quick mode with the local CalculiX reference enabled.

Scope:
- Short-term multi-element scope is C3D4/TET4 plus C3D8. C3D10 is explicitly excluded.
- C3D8 linear static contact uses the existing contactenergy replay evidence.
- C3D8 linear dynamic contact uses native SFC C3D8 Newmark dynamics and no-NLGEOM CalculiX dynamic references for both block-plane and block-block.
- C3D8 geometric-nonlinear static contact uses native SFC C3D8 StVK mechanics and CalculiX `contactenergy` output.
- C3D8 geometric-nonlinear dynamic contact uses native SFC C3D8 HHT/Newmark block-plane and block-block dynamics with CalculiX trajectory output.
- `phase9_c3d4_c3d8_side_by_side.csv` places the locked C3D4 block-plane trajectory next to the C3D8 dynamic rows.
- `phase9_solver_timing.csv` records native SFC and CalculiX wall time where both commands are run by this package.
- `phase9_curved_nonplanar_contact_external.csv` is a warped/non-planar C3D8 external diagnostic; it is generated, but its RF/CELS/CDIS metrics do not pass the external-correctness gate.
- Efficiency claims are enabled only for rows whose timing evidence has speedup greater than one and whose correctness gate passes.

Primary files:
- `phase9_full_contact_validation.csv`
- `phase9_claim_gates.csv`
- `phase9_c3d4_c3d8_side_by_side.csv`
- `phase9_solver_timing.csv`
- `phase9_curved_nonplanar_contact_external.csv`
- `phase9_full_contact_validation_summary.md`
- `native_c3d8_linear_dynamic_block_plane.csv`
- `native_c3d8_linear_dynamic_block_block.csv`
- `native_c3d8_linear_dynamic_comparison.csv`
- `native_c3d8_nonlinear_static_contactenergy.csv`
- `native_c3d8_nonlinear_dynamic_block_plane.csv`
- `native_c3d8_nonlinear_dynamic_block_block.csv`
- `native_c3d8_nonlinear_dynamic_comparison.csv`
- `figures/`
- `vtk/`
