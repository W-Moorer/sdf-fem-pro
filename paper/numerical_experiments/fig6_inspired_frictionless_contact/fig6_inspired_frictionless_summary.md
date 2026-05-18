# Fig. 6-Inspired Frictionless C3D8 Contact

This case uses the FuzzyContact Fig. 6 loading sketch only as geometry/loading inspiration.
The contact law is frictionless; no stick-slip claim is made.

- lower C3D8 mesh: `24 x 12 x 4` elements
- driver surface triangles: `96`
- dynamic SDF spacing: `0.25` mm
- wall time: `5.685634` s
- normal-stage min gap: `-2.892235e-02` mm
- shifted-stage min gap: `-2.921181e-02` mm
- shifted-stage active samples: `672`

## Outputs

- `fig6_inspired_frictionless_metrics.csv`
- `vtk/fig6_inspired_normal_loading.vtk`
- `vtk/fig6_inspired_shifted_frictionless.vtk`
- `figures/fig6_inspired_normal_loading_clouds.png`
- `figures/fig6_inspired_normal_loading_clouds.pdf`
- `figures/fig6_inspired_shifted_frictionless_clouds.png`
- `figures/fig6_inspired_shifted_frictionless_clouds.pdf`
- `figures/fig6_inspired_sdf_field_visualization.png`
- `figures/fig6_inspired_sdf_field_visualization.pdf`
