# 源齿轮 penalty 对齐运行计划

本文档记录 `commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp`
源工况的 SFC/Abaqus 对齐口径和当前已验证的短程结果。该工况只改变接触方式和积分计算方式：
Abaqus 源模型的几何、材料、TET4 网格、RP-MPC、RP1 角速度和 RP2 力矩保持不变。

## 当前推荐运行口径

- 接触：无摩擦线性 penalty normal contact，`pressure_stiffness = 5.0e9`。
- 积分：Abaqus/Standard HHT moderate dissipation 对齐口径。
- 时间步：固定 `dt = 1.0e-5 s`。
- 全时长目标：`duration = 0.05 s`，即 5000 个增量。
- SFC 初始动力学口径：初始加速度为零，初始 HHT history force 为上一已接受步平衡；本源 deck 没有前置 preload step，因此该历史项为零。
- 动画输出：建议隔步输出，`vtk_frame_stride = 2`。
- 云图输出：建议使用 scalar-only VTK，保留位移幅值、von Mises、node-averaged von Mises、strain norm、node-averaged strain norm、equivalent elastic strain 等云图字段，不写完整 tensor blocks。

## 已验证 10-step 结果

命令：

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --duration 0.0001 `
  --dt 0.00001 `
  --pressure-stiffness 5e9 `
  --write-sfc-vtk `
  --vtk-frame-stride 2 `
  --vtk-scalars-only `
  --history-frame-stride 2 `
  --abaqus-vtk-manifest results\flexible_gear_source_penalty_abaqus_stride2_10steps_rpdiag\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\flexible_gear_source_drive_penalty_stride2_10steps_scalars_only
```

结果：

- SFC wall time: `119.325812 s`。
- `timing_source_residual_seconds`: `37.157087 s`。
- `timing_source_linear_seconds`: `62.588299 s`。
- `timing_source_history_seconds`: `7.172796 s`。
- `timing_source_vtk_seconds`: `12.265461 s`。
- SFC VTK frame count: `6`。
- Scalar-only VTK frame size: about `22 MB/frame`，此前 tensor VTK 约 `100 MB/frame`。

与 Abaqus/Standard penalty VTK manifest 对齐后的末帧误差：

- max displacement magnitude relative error: `4.071%`。
- node-averaged von Mises relative error: `2.135%`。
- node-averaged equivalent elastic strain relative error: `2.135%`。
- RP2 rotation relative error: `3.738%`。
- RP2 angular velocity relative error: `3.856%`。

## 全时长命令模板

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --use-source-timing `
  --pressure-stiffness 5e9 `
  --write-sfc-vtk `
  --vtk-frame-stride 2 `
  --vtk-scalars-only `
  --history-frame-stride 2 `
  --out-dir results\flexible_gear_source_drive_penalty_full_scalars_only
```

按 10-step 结果线性外推，全时长仍然很长，并且隔步 scalar-only VTK 仍会产生约 100 GB 量级输出。
因此下一步工程优化应优先处理 `timing_source_linear_seconds`，即 reduced penalty correction / CG 修正路径。
