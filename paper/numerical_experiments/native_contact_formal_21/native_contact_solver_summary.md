# Independent CalculiX/SFC Native Contact Trajectory

This is the strict independent-solver trajectory runner: CalculiX builds and solves a native `*CONTACT PAIR` model; SFC independently builds and solves the corresponding `DynamicNarrowBandSDF + field_contact` model.

The cases use open benchmark-style contact modeling schemes only as geometry/loading inspiration. The comparison is not a claim of matching FuzzyContact published results.

## Outputs

- `native_contact_solver_comparison.csv`
- `native_contact_solver_sfc_history.csv`
- `native_contact_solver_commands.csv`
- `native_contact_solver_timing.csv`
- `figures/native_contact_displacement_curves.png`
- `figures/native_contact_displacement_curves.pdf`
- `figures/native_contact_solver_errors.png`
- `figures/native_contact_solver_errors.pdf`
- `figures/native_contact_solver_timing_breakdown.png`
- `figures/native_contact_solver_timing_breakdown.pdf`

## Native Solver Agreement Gate

| Problem | Step | disp rel | strain rel | stress rel | VM rel | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `problem_1` | 1 | 1.352881e-02 | 1.869638e-02 | 1.726835e-02 | 1.586128e-02 | passed_native_solver_agreement |
| `problem_1` | 2 | 3.941682e-02 | 4.521653e-02 | 4.346932e-02 | 4.415077e-02 | passed_native_solver_agreement |
| `problem_1` | 3 | 1.118729e-02 | 1.591953e-02 | 1.433259e-02 | 1.447111e-02 | passed_native_solver_agreement |
| `problem_1` | 4 | 1.243152e-02 | 1.640379e-02 | 1.464610e-02 | 1.479453e-02 | passed_native_solver_agreement |
| `problem_1` | 5 | 8.327165e-03 | 1.192624e-02 | 9.998782e-03 | 9.893222e-03 | passed_native_solver_agreement |
| `problem_1` | 6 | 5.503863e-03 | 8.204639e-03 | 6.424074e-03 | 5.680983e-03 | passed_native_solver_agreement |
| `problem_1` | 7 | 7.075130e-03 | 1.004951e-02 | 7.620670e-03 | 7.115001e-03 | passed_native_solver_agreement |
| `problem_1` | 8 | 5.327019e-03 | 7.637989e-03 | 5.757688e-03 | 4.457905e-03 | passed_native_solver_agreement |
| `problem_1` | 9 | 5.436974e-03 | 7.651528e-03 | 5.830118e-03 | 4.231906e-03 | passed_native_solver_agreement |
| `problem_1` | 10 | 6.234897e-03 | 8.694439e-03 | 6.312309e-03 | 4.669128e-03 | passed_native_solver_agreement |
| `problem_1` | 11 | 6.035786e-03 | 8.293555e-03 | 6.625141e-03 | 4.531956e-03 | passed_native_solver_agreement |
| `problem_1` | 12 | 6.206424e-03 | 8.359852e-03 | 7.308166e-03 | 4.963922e-03 | passed_native_solver_agreement |
| `problem_1` | 13 | 6.564260e-03 | 8.763482e-03 | 8.059504e-03 | 5.537013e-03 | passed_native_solver_agreement |
| `problem_1` | 14 | 6.961128e-03 | 9.228358e-03 | 8.896763e-03 | 6.210110e-03 | passed_native_solver_agreement |
| `problem_1` | 15 | 7.422038e-03 | 9.799673e-03 | 9.778935e-03 | 6.927780e-03 | passed_native_solver_agreement |
| `problem_1` | 16 | 7.897146e-03 | 1.038703e-02 | 1.072540e-02 | 7.718565e-03 | passed_native_solver_agreement |
| `problem_1` | 17 | 8.319263e-03 | 1.088786e-02 | 1.186774e-02 | 8.700970e-03 | passed_native_solver_agreement |
| `problem_1` | 18 | 8.766669e-03 | 1.138957e-02 | 1.315754e-02 | 9.842608e-03 | passed_native_solver_agreement |
| `problem_1` | 19 | 9.280044e-03 | 1.203707e-02 | 1.425982e-02 | 1.080076e-02 | passed_native_solver_agreement |
| `problem_1` | 20 | 9.919947e-03 | 1.292474e-02 | 1.513637e-02 | 1.147865e-02 | passed_native_solver_agreement |
| `problem_1` | 21 | 1.048845e-02 | 1.366285e-02 | 1.632092e-02 | 1.247513e-02 | passed_native_solver_agreement |
| `problem_3` | 1 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 | 1.000000e+00 | failed_native_solver_agreement |
| `problem_3` | 2 | 5.050754e-02 | 6.728597e-02 | 6.477352e-02 | 6.297838e-02 | failed_native_solver_agreement |
| `problem_3` | 3 | 2.113364e-02 | 3.365492e-02 | 3.131874e-02 | 2.628925e-02 | passed_native_solver_agreement |
| `problem_3` | 4 | 2.173833e-02 | 3.302205e-02 | 3.078849e-02 | 2.556611e-02 | passed_native_solver_agreement |
| `problem_3` | 5 | 2.112886e-02 | 3.386793e-02 | 3.150503e-02 | 2.647831e-02 | passed_native_solver_agreement |
| `problem_3` | 6 | 2.113349e-02 | 3.616819e-02 | 3.362176e-02 | 2.907188e-02 | passed_native_solver_agreement |
| `problem_3` | 7 | 2.097647e-02 | 3.463736e-02 | 3.218931e-02 | 2.729888e-02 | passed_native_solver_agreement |
| `problem_3` | 8 | 2.100317e-02 | 3.549222e-02 | 3.297305e-02 | 2.824969e-02 | passed_native_solver_agreement |
| `problem_3` | 9 | 2.179844e-02 | 3.769156e-02 | 3.505998e-02 | 3.073329e-02 | passed_native_solver_agreement |
| `problem_3` | 10 | 2.112406e-02 | 3.599150e-02 | 3.342778e-02 | 2.876474e-02 | passed_native_solver_agreement |
| `problem_3` | 11 | 2.112567e-02 | 3.424439e-02 | 3.181782e-02 | 2.676123e-02 | passed_native_solver_agreement |
| `problem_3` | 12 | 2.155033e-02 | 3.346570e-02 | 3.114348e-02 | 2.588664e-02 | passed_native_solver_agreement |
| `problem_3` | 13 | 2.168204e-02 | 3.332791e-02 | 3.102703e-02 | 2.571937e-02 | passed_native_solver_agreement |
| `problem_3` | 14 | 2.195830e-02 | 3.311111e-02 | 3.085491e-02 | 2.547021e-02 | passed_native_solver_agreement |
| `problem_3` | 15 | 2.219565e-02 | 3.294988e-02 | 3.073136e-02 | 2.528810e-02 | passed_native_solver_agreement |
| `problem_3` | 16 | 2.226405e-02 | 3.291620e-02 | 3.070517e-02 | 2.523514e-02 | passed_native_solver_agreement |
| `problem_3` | 17 | 2.217604e-02 | 3.298339e-02 | 3.075206e-02 | 2.527981e-02 | passed_native_solver_agreement |
| `problem_3` | 18 | 2.237411e-02 | 3.288204e-02 | 3.068010e-02 | 2.516089e-02 | passed_native_solver_agreement |
| `problem_3` | 19 | 2.222916e-02 | 3.298076e-02 | 3.074811e-02 | 2.523330e-02 | passed_native_solver_agreement |
| `problem_3` | 20 | 2.244358e-02 | 3.288244e-02 | 3.068176e-02 | 2.511844e-02 | passed_native_solver_agreement |
| `problem_3` | 21 | 2.266841e-02 | 3.278566e-02 | 3.061974e-02 | 2.501100e-02 | passed_native_solver_agreement |
| `problem_4` | 1 | 2.960037e-04 | 2.190308e-03 | 1.371491e-03 | 6.415724e-04 | passed_native_solver_agreement |
| `problem_4` | 2 | 2.970766e-04 | 2.171843e-03 | 1.356190e-03 | 6.323497e-04 | passed_native_solver_agreement |
| `problem_4` | 3 | 3.015559e-04 | 2.167010e-03 | 1.356958e-03 | 6.392641e-04 | passed_native_solver_agreement |
| `problem_4` | 4 | 3.084611e-04 | 2.165137e-03 | 1.362337e-03 | 6.530284e-04 | passed_native_solver_agreement |
| `problem_4` | 5 | 3.177498e-04 | 2.164379e-03 | 1.370495e-03 | 6.717571e-04 | passed_native_solver_agreement |
| `problem_4` | 6 | 3.293556e-04 | 2.164179e-03 | 1.380870e-03 | 6.946027e-04 | passed_native_solver_agreement |
| `problem_4` | 7 | 3.430257e-04 | 2.164310e-03 | 1.393224e-03 | 7.210109e-04 | passed_native_solver_agreement |
| `problem_4` | 8 | 3.585846e-04 | 2.164678e-03 | 1.407420e-03 | 7.505303e-04 | passed_native_solver_agreement |
| `problem_4` | 9 | 3.757748e-04 | 2.165219e-03 | 1.423348e-03 | 7.827609e-04 | passed_native_solver_agreement |
| `problem_4` | 10 | 3.943953e-04 | 2.165911e-03 | 1.440927e-03 | 8.173631e-04 | passed_native_solver_agreement |
| `problem_4` | 11 | 4.142477e-04 | 2.166717e-03 | 1.460094e-03 | 8.540578e-04 | passed_native_solver_agreement |
| `problem_4` | 12 | 4.352067e-04 | 2.167646e-03 | 1.480759e-03 | 8.925371e-04 | passed_native_solver_agreement |
| `problem_4` | 13 | 4.570720e-04 | 2.168667e-03 | 1.502866e-03 | 9.326100e-04 | passed_native_solver_agreement |
| `problem_4` | 14 | 4.797535e-04 | 2.169791e-03 | 1.526340e-03 | 9.740663e-04 | passed_native_solver_agreement |
| `problem_4` | 15 | 5.031257e-04 | 2.170999e-03 | 1.551129e-03 | 1.016742e-03 | passed_native_solver_agreement |
| `problem_4` | 16 | 5.271298e-04 | 2.172299e-03 | 1.577158e-03 | 1.060474e-03 | passed_native_solver_agreement |
| `problem_4` | 17 | 5.515882e-04 | 2.173686e-03 | 1.604399e-03 | 1.105188e-03 | passed_native_solver_agreement |
| `problem_4` | 18 | 5.765394e-04 | 2.175152e-03 | 1.632724e-03 | 1.150679e-03 | passed_native_solver_agreement |
| `problem_4` | 19 | 6.019398e-04 | 2.176700e-03 | 1.662127e-03 | 1.196924e-03 | passed_native_solver_agreement |
| `problem_4` | 20 | 6.276510e-04 | 2.178326e-03 | 1.692540e-03 | 1.243821e-03 | passed_native_solver_agreement |
| `problem_4` | 21 | 6.536840e-04 | 2.180031e-03 | 1.723906e-03 | 1.291308e-03 | passed_native_solver_agreement |

- Passed rows: `61/63`.
- Completed CalculiX jobs: `3/3`.

## Timing

| Problem | SFC total s | CalculiX wall s | CalculiX/SFC | SFC field update s | SFC query s | SFC linear solve s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `problem_1` | 1.113690e+01 | 1.363250e+01 | 1.224084e+00 | 1.025799e+01 | 2.847838e-01 | 4.448315e-01 |
| `problem_3` | 3.371117e+00 | 1.410647e+01 | 4.184509e+00 | 2.301379e+00 | 4.167905e-01 | 5.550105e-01 |
| `problem_4` | 6.929062e+01 | 4.810666e+01 | 6.942738e-01 | 6.286295e+01 | 2.918967e-01 | 6.032349e+00 |

## Scope

- Supports independent CalculiX-native-contact versus SFC-SDF-contact agreement only for rows marked `passed_native_solver_agreement`.
- If rows fail, the output is a negative result identifying contact-law/active-set/discretization mismatch, not an SDF field construction failure by itself.
