# Long-Time C3D8 Dynamic SDF Contact

- scenario: `free_fall_settling_contact`
- case: `block_block_c3d8`
- element type: `C3D8`
- resolution: `4`
- total time: `3.0` s
- dt: `0.02` s
- gravity: `9.81`
- initial z velocity: `0.0`
- mass-proportional damping alpha: `120.0`
- steps: `150`
- linearity: `linear_damped`
- wall time: `67.432387` s
- max active contact samples: `64`
- min gap: `-9.114282e-05`
- max normal force: `1.128852e+00`
- cloud snapshot time: `0.260000` s
- cloud snapshot rule: peak normal-force state
- exported SFC VTK animation frames: `151`
- SFC PVD: `paper\numerical_experiments\long_time_c3d8_dynamic\vtk_frames\sfc\sfc_block_block_c3d8.pvd`

## Outputs

- `long_time_c3d8_dynamic_history.csv`
- `vtk/long_time_c3d8_peak_contact_clouds.vtk`
- `vtk_frames/sfc/sfc_block_block_c3d8.pvd`
- `vtk_frames/sfc/sfc_block_block_c3d8_0000.vtk`
- `figures/long_time_c3d8_dynamic_histories.png`
- `figures/long_time_c3d8_dynamic_histories.pdf`
- `figures/fine_grid_c3d8_contact_clouds.png`
- `figures/fine_grid_c3d8_contact_clouds.pdf`
