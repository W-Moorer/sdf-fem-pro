# Official CalculiX Contact Example Line

This package records which official CalculiX examples are suitable for
the paper-facing SFC validation path. It is intentionally conservative:
an official example is not allowed to support full SFC trajectory or
field equivalence until a native SFC solve and metric-level comparison
exist for the same model.

## Selected Examples

| Case | File | Role | Elements | Contact | Step | Paper use |
| --- | --- | --- | --- | --- | --- | --- |
| contactenergy_c3d8_static_energy | `contactenergy.inp` | official_static_contact_energy_reference | C3D8 | SURFACE TO SURFACE | STATIC | direct_reference_already_locked |
| scheibe2f2f_c3d8_nlgeom_static | `scheibe2f2f.inp.gz` | official_nonlinear_static_surface_to_surface_candidate | C3D8 | SURFACE TO SURFACE | STATIC | candidate_requires_native_sfc_metric_extraction |
| ball_c3d8_dynamic_drop | `ball.inp.gz` | official_dynamic_drop_candidate | C3D8;S8 | NODE TO SURFACE | DYNAMIC | candidate_requires_s8_floor_equivalent_or_filtered_comparison |
| contact1_c3d8_exponential_law | `contact1.inp` | small_contact_law_alignment | C3D8 | NODE TO SURFACE | STATIC | law_alignment_reference |
| contact3_c3d8_linear_law | `contact3.inp` | small_contact_law_alignment | C3D8 | NODE TO SURFACE | STATIC | law_alignment_reference |
| contact6_c3d8_stiff_linear_law | `contact6.inp` | small_contact_law_alignment | C3D8 | NODE TO SURFACE | STATIC | law_alignment_reference |

## Claim Gates

| Case | Claim | Allowed | Reason |
| --- | --- | --- | --- |
| contactenergy_c3d8_static_energy | official_calculix_example_available | true | official input was found and parsed |
| contactenergy_c3d8_static_energy | paper_external_reference_direct | true | direct paper use is currently limited to the locked contactenergy C3D8 law/energy reference |
| contactenergy_c3d8_static_energy | small_law_alignment_reference | false | small C3D8 law examples are allowed for pressure-overclosure and contact-print alignment only |
| contactenergy_c3d8_static_energy | full_trajectory_or_full_field_equivalence | false | requires a native SFC solve plus metric-level comparison before this official example can support full equivalence |
| contactenergy_c3d8_static_energy | recommended_next_adaptation | false | scheibe2f2f and ball are the best official engineering candidates but need dedicated native SFC comparison paths |
| scheibe2f2f_c3d8_nlgeom_static | official_calculix_example_available | true | official input was found and parsed |
| scheibe2f2f_c3d8_nlgeom_static | paper_external_reference_direct | false | direct paper use is currently limited to the locked contactenergy C3D8 law/energy reference |
| scheibe2f2f_c3d8_nlgeom_static | small_law_alignment_reference | false | small C3D8 law examples are allowed for pressure-overclosure and contact-print alignment only |
| scheibe2f2f_c3d8_nlgeom_static | full_trajectory_or_full_field_equivalence | false | requires a native SFC solve plus metric-level comparison before this official example can support full equivalence |
| scheibe2f2f_c3d8_nlgeom_static | recommended_next_adaptation | true | scheibe2f2f and ball are the best official engineering candidates but need dedicated native SFC comparison paths |
| ball_c3d8_dynamic_drop | official_calculix_example_available | true | official input was found and parsed |
| ball_c3d8_dynamic_drop | paper_external_reference_direct | false | direct paper use is currently limited to the locked contactenergy C3D8 law/energy reference |
| ball_c3d8_dynamic_drop | small_law_alignment_reference | false | small C3D8 law examples are allowed for pressure-overclosure and contact-print alignment only |
| ball_c3d8_dynamic_drop | full_trajectory_or_full_field_equivalence | false | requires a native SFC solve plus metric-level comparison before this official example can support full equivalence |
| ball_c3d8_dynamic_drop | recommended_next_adaptation | true | scheibe2f2f and ball are the best official engineering candidates but need dedicated native SFC comparison paths |
| contact1_c3d8_exponential_law | official_calculix_example_available | true | official input was found and parsed |
| contact1_c3d8_exponential_law | paper_external_reference_direct | false | direct paper use is currently limited to the locked contactenergy C3D8 law/energy reference |
| contact1_c3d8_exponential_law | small_law_alignment_reference | true | small C3D8 law examples are allowed for pressure-overclosure and contact-print alignment only |
| contact1_c3d8_exponential_law | full_trajectory_or_full_field_equivalence | false | requires a native SFC solve plus metric-level comparison before this official example can support full equivalence |
| contact1_c3d8_exponential_law | recommended_next_adaptation | false | scheibe2f2f and ball are the best official engineering candidates but need dedicated native SFC comparison paths |
| contact3_c3d8_linear_law | official_calculix_example_available | true | official input was found and parsed |
| contact3_c3d8_linear_law | paper_external_reference_direct | false | direct paper use is currently limited to the locked contactenergy C3D8 law/energy reference |
| contact3_c3d8_linear_law | small_law_alignment_reference | true | small C3D8 law examples are allowed for pressure-overclosure and contact-print alignment only |
| contact3_c3d8_linear_law | full_trajectory_or_full_field_equivalence | false | requires a native SFC solve plus metric-level comparison before this official example can support full equivalence |
| contact3_c3d8_linear_law | recommended_next_adaptation | false | scheibe2f2f and ball are the best official engineering candidates but need dedicated native SFC comparison paths |
| contact6_c3d8_stiff_linear_law | official_calculix_example_available | true | official input was found and parsed |
| contact6_c3d8_stiff_linear_law | paper_external_reference_direct | false | direct paper use is currently limited to the locked contactenergy C3D8 law/energy reference |
| contact6_c3d8_stiff_linear_law | small_law_alignment_reference | true | small C3D8 law examples are allowed for pressure-overclosure and contact-print alignment only |
| contact6_c3d8_stiff_linear_law | full_trajectory_or_full_field_equivalence | false | requires a native SFC solve plus metric-level comparison before this official example can support full equivalence |
| contact6_c3d8_stiff_linear_law | recommended_next_adaptation | false | scheibe2f2f and ball are the best official engineering candidates but need dedicated native SFC comparison paths |

## Run Status

| Case | Status | DAT | FRD | Wall time |
| --- | --- | --- | --- | ---: |
| contactenergy_c3d8_static_energy | ok | true | True | 0.9143053996376693 |
| scheibe2f2f_c3d8_nlgeom_static | ok | false | True | 0.5435123001225293 |
| ball_c3d8_dynamic_drop | ok | true | True | 0.5752305998466909 |
| contact1_c3d8_exponential_law | ok | true | True | 0.5350178997032344 |
| contact3_c3d8_linear_law | ok | true | True | 0.5361699000932276 |
| contact6_c3d8_stiff_linear_law | ok | true | True | 0.5549443997442722 |

## Recommended Use

1. `contactenergy.inp` remains the direct C3D8 static contact-law and contact-energy reference.
2. `contact1/contact3/contact6` should be used for small pressure-overclosure law alignment diagnostics.
3. `scheibe2f2f.inp.gz` is the best official nonlinear static C3D8 surface-to-surface candidate, but it needs FRD field extraction and a native SFC comparison before paper claims.
4. `ball.inp.gz` is the best official dynamic drop candidate, but its shell-floor/node-to-surface formulation must be matched or explicitly filtered before trajectory claims.
