# Source gear VTK manifest alignment current result

本记录对应商业软件齿轮工况 `commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp`。

## 工况口径

- 原始 Abaqus 步：`*Dynamic`，`1e-05, 0.05, 1e-10, 5e-05`。
- 当前 SFC 建模/驱动：读取同一 `gear_contact.inp` 的网格、材料、RP-MPC hub、G1 绕 z 角速度和 G2 绕 z 扭矩。
- 当前 SFC 接触：`source_penalty` 罚函数接触。
- 当前 SFC 输出：VTK 隔位/隔步保存由 runner 参数控制；动画使用 node-averaged stress/strain 和固定全局色标。

## 当前可核验基线

当前使用已有 100-step SFC 与 Abaqus VTK manifest 做同步长对比：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_100steps_stride10_p95\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_penalty_abaqus_100steps_stride10_p95\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_100steps_stride10_p95_alignment_current
```

输出：

- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/sfc_vs_abaqus_vtk_metric_errors.csv`
- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/sfc_vs_abaqus_vtk_metric_curves.png`
- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/animation_fixed_color_ranges.csv`
- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/paraview_fixed_range_animation.py`
- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/source_gear_vtk_manifest_alignment_summary.md`

末帧误差：

| 指标 | 相对误差 |
| --- | ---: |
| displacement magnitude | 3.927% |
| p95 node-averaged von Mises | 20.832% |
| p95 node-averaged equivalent elastic strain | 20.832% |

## 结论

该结果证明当前输出流水线已经能从 SFC/Abaqus 的 VTK 云图字段生成位移、应力、应变曲线，并给出固定色标 ParaView 脚本，解决动画因逐帧自动缩放造成的应力/应变闪动问题。

但这还不是最终验收所需的 0.05 s 全时长结果。全时长仍需要按同一 `gear_contact.inp` 的 `0.05 s / 1e-5 s` 跑完 SFC，并与 Abaqus 全程 VTK manifest 做同一入口的对比。

## 更新：source-penalty 同步帧首块

为了从零建立全时长可续跑结果，当前新建了目录：

- SFC：`results/source_gear_penalty_full_stride2_match_step`
- Abaqus penalty：`results/source_gear_abaqus_penalty_full_stride2_match_step_0002`

SFC 命令口径：

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --duration 0.0002 `
  --dt 0.00001 `
  --pressure-stiffness 5e9 `
  --write-sfc-vtk `
  --vtk-frame-stride 2 `
  --vtk-scalars-only `
  --history-frame-stride 2 `
  --source-checkpoint results\source_gear_penalty_full_stride2_match_step\source_drive_checkpoint.npz `
  --source-checkpoint-stride 20 `
  --out-dir results\source_gear_penalty_full_stride2_match_step
```

Abaqus penalty 命令口径：

```powershell
python validation\run_flexible_gear_source_penalty_abaqus.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --out-dir results\source_gear_abaqus_penalty_full_stride2_match_step_0002 `
  --pressure-stiffness 5e9 `
  --frame-stride 2 `
  --dt 0.00001 `
  --duration 0.0002 `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --scalars-only
```

该首块结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 2e-4 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 11 |
| Abaqus VTK 帧数 | 11 |
| SFC wall time | 77.042 s |
| Abaqus analysis wall time | 282.670 s |
| 末帧 displacement magnitude rel. error | 1.395% |
| 末帧 p95 node-averaged von Mises rel. error | 39.672% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 39.672% |

PVD 时间戳已经同步为 `0:2e-5:2e-4`。SFC 目录中已写入 `source_drive_checkpoint.npz`，后续可用同一路径和 `--resume-source-checkpoint` 继续推进到 `0.05 s`。

## 更新：续跑到 `4e-4 s`

SFC 已从同一个 checkpoint 续跑到 `4e-4 s`：

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --duration 0.0004 `
  --dt 0.00001 `
  --pressure-stiffness 5e9 `
  --write-sfc-vtk `
  --vtk-frame-stride 2 `
  --vtk-scalars-only `
  --history-frame-stride 2 `
  --source-checkpoint results\source_gear_penalty_full_stride2_match_step\source_drive_checkpoint.npz `
  --resume-source-checkpoint `
  --source-checkpoint-stride 20 `
  --out-dir results\source_gear_penalty_full_stride2_match_step
```

Abaqus penalty 同步对照：

```powershell
python validation\run_flexible_gear_source_penalty_abaqus.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --out-dir results\source_gear_abaqus_penalty_full_stride2_match_step_0004 `
  --pressure-stiffness 5e9 `
  --frame-stride 2 `
  --dt 0.00001 `
  --duration 0.0004 `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --scalars-only
```

独立 manifest 对比：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0004\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0004_alignment
```

`4e-4 s` 阶段结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 4e-4 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 21 |
| Abaqus VTK 帧数 | 21 |
| SFC 续跑 wall time | 66.340 s |
| Abaqus analysis wall time | 573.906 s |
| 末帧 displacement magnitude rel. error | 4.506% |
| 末帧 p95 node-averaged von Mises rel. error | 1.451% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 1.451% |
| 末帧 max node-averaged von Mises rel. error | 0.117% |
| 末帧 mean node-averaged von Mises rel. error | 1.640% |

PVD 时间戳已经同步为 `0:2e-5:4e-4`。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 40，可继续向 `0.05 s` 推进。

## 更新：续跑到 `6e-4 s`

SFC 已继续从同一个 checkpoint 续跑到 `6e-4 s`。Abaqus penalty 对照也使用同一 `gear_contact.inp`、同一 `dt=1e-5 s`、同一 `frame_stride=2` 跑到 `6e-4 s`。

输出目录：

- SFC：`results/source_gear_penalty_full_stride2_match_step`
- Abaqus penalty：`results/source_gear_abaqus_penalty_full_stride2_match_step_0006`
- 独立 manifest 对比：`results/source_gear_penalty_full_stride2_match_step_0006_alignment`

独立 manifest 对比命令：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0006\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0006_alignment
```

`6e-4 s` 阶段结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 6e-4 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 31 |
| Abaqus VTK 帧数 | 31 |
| SFC 续跑 wall time | 81.696 s |
| Abaqus analysis wall time | 881.962 s |
| 末帧 displacement magnitude rel. error | 4.693% |
| 末帧 p95 node-averaged von Mises rel. error | 6.275% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 6.275% |
| 末帧 max node-averaged von Mises rel. error | 5.390% |
| 末帧 mean node-averaged von Mises rel. error | 4.019% |

PVD 时间戳已经同步为 `0:2e-5:6e-4`。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 60，可继续向 `0.05 s` 推进。

## 更新：续跑到 `8e-4 s`

SFC 已继续从同一个 checkpoint 续跑到 `8e-4 s`。Abaqus penalty 对照也使用同一 `gear_contact.inp`、同一 `dt=1e-5 s`、同一 `frame_stride=2` 跑到 `8e-4 s`。

输出目录：

- SFC：`results/source_gear_penalty_full_stride2_match_step`
- Abaqus penalty：`results/source_gear_abaqus_penalty_full_stride2_match_step_0008`
- 独立 manifest 对比：`results/source_gear_penalty_full_stride2_match_step_0008_alignment`

独立 manifest 对比命令：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0008\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0008_alignment
```

`8e-4 s` 阶段结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 8e-4 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 41 |
| Abaqus VTK 帧数 | 41 |
| SFC 续跑 wall time | 82.328 s |
| Abaqus analysis wall time | 1180.277 s |
| 末帧 displacement magnitude rel. error | 4.453% |
| 末帧 p95 node-averaged von Mises rel. error | 8.510% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 8.510% |
| 末帧 max node-averaged von Mises rel. error | 0.563% |
| 末帧 mean node-averaged von Mises rel. error | 9.122% |

PVD 时间戳已经同步为 `0:2e-5:8e-4`。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 80，可继续向 `0.05 s` 推进。
