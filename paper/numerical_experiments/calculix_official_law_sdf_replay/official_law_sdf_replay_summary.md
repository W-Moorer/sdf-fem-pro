# Official CalculiX Contact Law + Deforming-SDF Replay

CalculiX solves the selected official C3D8 contact examples. SFC then
replays the final deformed configuration by querying the dynamic SDF on
the current master surface and evaluating the pressure-overclosure law.

## Metrics

| Case | Law | contact points | max gap error | max pressure error | status |
| --- | --- | ---: | ---: | ---: | --- |
| contactenergy_c3d8_surface_linear | LINEAR | 1 | 1.523947797382047e-16 | 1.5239502290524023e-11 | ok |
| contact1_c3d8_node_exponential | EXPONENTIAL | 1 | 1.999848878710096e-14 | 6.484461685631127e-07 | ok |
| contact3_c3d8_node_linear | LINEAR | 1 | 4.109999929243475e-09 | 0.00027218718749966777 | ok |
| contact6_c3d8_node_stiff_linear | LINEAR | 1 | 6.999582021250431e-13 | 3.074321948612393e-05 | ok |

## Claim Gates

| Case | Claim | Allowed | Reason |
| --- | --- | --- | --- |
| contactenergy_c3d8_surface_linear | deforming_sdf_gap_matches_calculix_cdis | true | requires instrumented CalculiX CDIS and SFC current-surface gap agreement |
| contactenergy_c3d8_surface_linear | pressure_law_matches_calculix_cstr | true | requires pressure-overclosure replay and CalculiX CSTR agreement |
| contactenergy_c3d8_surface_linear | linear_pressure_law_matches_calculix_cstr | true | allowed only for linear pressure-overclosure rows with CSTR agreement |
| contactenergy_c3d8_surface_linear | contact_energy_matches_calculix_cels | true | currently only the official contactenergy surface-to-surface case has a locked total CELS match |
| contactenergy_c3d8_surface_linear | native_sfc_trajectory_equivalence | false | this runner replays CalculiX deformed states; it does not run a native SFC trajectory |
| contact1_c3d8_node_exponential | deforming_sdf_gap_matches_calculix_cdis | true | requires instrumented CalculiX CDIS and SFC current-surface gap agreement |
| contact1_c3d8_node_exponential | pressure_law_matches_calculix_cstr | true | requires pressure-overclosure replay and CalculiX CSTR agreement |
| contact1_c3d8_node_exponential | linear_pressure_law_matches_calculix_cstr | false | allowed only for linear pressure-overclosure rows with CSTR agreement |
| contact1_c3d8_node_exponential | contact_energy_matches_calculix_cels | false | currently only the official contactenergy surface-to-surface case has a locked total CELS match |
| contact1_c3d8_node_exponential | native_sfc_trajectory_equivalence | false | this runner replays CalculiX deformed states; it does not run a native SFC trajectory |
| contact3_c3d8_node_linear | deforming_sdf_gap_matches_calculix_cdis | true | requires instrumented CalculiX CDIS and SFC current-surface gap agreement |
| contact3_c3d8_node_linear | pressure_law_matches_calculix_cstr | true | requires pressure-overclosure replay and CalculiX CSTR agreement |
| contact3_c3d8_node_linear | linear_pressure_law_matches_calculix_cstr | true | allowed only for linear pressure-overclosure rows with CSTR agreement |
| contact3_c3d8_node_linear | contact_energy_matches_calculix_cels | false | currently only the official contactenergy surface-to-surface case has a locked total CELS match |
| contact3_c3d8_node_linear | native_sfc_trajectory_equivalence | false | this runner replays CalculiX deformed states; it does not run a native SFC trajectory |
| contact6_c3d8_node_stiff_linear | deforming_sdf_gap_matches_calculix_cdis | true | requires instrumented CalculiX CDIS and SFC current-surface gap agreement |
| contact6_c3d8_node_stiff_linear | pressure_law_matches_calculix_cstr | true | requires pressure-overclosure replay and CalculiX CSTR agreement |
| contact6_c3d8_node_stiff_linear | linear_pressure_law_matches_calculix_cstr | true | allowed only for linear pressure-overclosure rows with CSTR agreement |
| contact6_c3d8_node_stiff_linear | contact_energy_matches_calculix_cels | false | currently only the official contactenergy surface-to-surface case has a locked total CELS match |
| contact6_c3d8_node_stiff_linear | native_sfc_trajectory_equivalence | false | this runner replays CalculiX deformed states; it does not run a native SFC trajectory |

## Boundaries

- This is a deformed-state replay validation, not native SFC trajectory equivalence.
- `contactenergy` supports the total CELS energy replay claim.
- `contact3` and `contact6` support linear pressure-overclosure diagnostics.
- `contact1` is retained as an exponential-law diagnostic; the gap comparison is meaningful, but full exponential energy equivalence is not claimed.
