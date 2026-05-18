# Dynamic VTK Frame Export

This directory contains continuously numbered legacy VTK frame sequences and PVD time indexes for ParaView animation.

`sfc` and `calculix` rows preserve each solver's raw output sampling. `*_common_time` rows are resampled to the SFC time grid for side-by-side animation, and `sfc_minus_calculix_common_error` stores pointwise displacement-error fields on the same frame count.

| case | solver | frames | status | pvd |
| --- | --- | ---: | --- | --- |
| block_plane_c3d8 | sfc | 41 |  | `paper\numerical_experiments\dynamic_vtk_frames\block_plane_c3d8\sfc\sfc_block_plane_c3d8_r1.pvd` |
| block_plane_c3d8 | calculix | 69 | exported | `paper\numerical_experiments\dynamic_vtk_frames\block_plane_c3d8\calculix\calculix_block_plane_c3d8_r1.pvd` |
| block_plane_c3d8 | sfc_common_time | 41 | exported | `paper\numerical_experiments\dynamic_vtk_frames\block_plane_c3d8\common_time_frames\sfc\common_sfc_block_plane_c3d8_r1.pvd` |
| block_plane_c3d8 | calculix_common_time | 41 | interpolated_to_sfc_times | `paper\numerical_experiments\dynamic_vtk_frames\block_plane_c3d8\common_time_frames\calculix\common_calculix_block_plane_c3d8_r1.pvd` |
| block_plane_c3d8 | sfc_minus_calculix_common_error | 41 | exported | `paper\numerical_experiments\dynamic_vtk_frames\block_plane_c3d8\common_time_frames\error\common_error_block_plane_c3d8_r1.pvd` |
| block_block_c3d8 | sfc | 41 |  | `paper\numerical_experiments\dynamic_vtk_frames\block_block_c3d8\sfc\sfc_block_block_c3d8_r1.pvd` |
| block_block_c3d8 | calculix | 70 | exported | `paper\numerical_experiments\dynamic_vtk_frames\block_block_c3d8\calculix\calculix_block_block_c3d8_r1.pvd` |
| block_block_c3d8 | sfc_common_time | 41 | exported | `paper\numerical_experiments\dynamic_vtk_frames\block_block_c3d8\common_time_frames\sfc\common_sfc_block_block_c3d8_r1.pvd` |
| block_block_c3d8 | calculix_common_time | 41 | interpolated_to_sfc_times | `paper\numerical_experiments\dynamic_vtk_frames\block_block_c3d8\common_time_frames\calculix\common_calculix_block_block_c3d8_r1.pvd` |
| block_block_c3d8 | sfc_minus_calculix_common_error | 41 | exported | `paper\numerical_experiments\dynamic_vtk_frames\block_block_c3d8\common_time_frames\error\common_error_block_block_c3d8_r1.pvd` |
