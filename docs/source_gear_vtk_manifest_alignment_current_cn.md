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

## 更新：续跑到 `1.0e-3 s`

SFC 已继续从同一个 checkpoint 续跑到 `1.0e-3 s`。Abaqus penalty 对照也使用同一 `gear_contact.inp`、同一 `dt=1e-5 s`、同一 `frame_stride=2` 跑到 `1.0e-3 s`。

输出目录：

- SFC：`results/source_gear_penalty_full_stride2_match_step`
- Abaqus penalty：`results/source_gear_abaqus_penalty_full_stride2_match_step_0010`
- 独立 manifest 对比：`results/source_gear_penalty_full_stride2_match_step_0010_alignment`

独立 manifest 对比命令：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0010\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0010_alignment
```

`1.0e-3 s` 阶段结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 1.0e-3 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 51 |
| Abaqus VTK 帧数 | 51 |
| SFC 续跑 wall time | 73.069 s |
| Abaqus analysis wall time | 1542.746 s |
| 末帧 displacement magnitude rel. error | 3.927% |
| 末帧 p95 node-averaged von Mises rel. error | 20.832% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 20.832% |
| 末帧 max node-averaged von Mises rel. error | 23.166% |
| 末帧 mean node-averaged von Mises rel. error | 1.065% |

该时间点的整体位移和 mean stress/strain 仍与 Abaqus penalty 对照接近，但 p95/max stress/strain 误差重新升高，说明局部接触峰值在该瞬态更敏感。当前记录保留原始 p95/max/mean 三种口径，不用单一指标替代全部云图场判断。

PVD 时间戳已经同步为 `0:2e-5:1.0e-3`。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 100，可继续向 `0.05 s` 推进。

## 更新：续跑到 `1.2e-3 s`

SFC 已继续从同一个 checkpoint 续跑到 `1.2e-3 s`。Abaqus penalty 对照也使用同一 `gear_contact.inp`、同一 `dt=1e-5 s`、同一 `frame_stride=2` 跑到 `1.2e-3 s`。两边均使用线性罚函数接触；VTK 均为隔帧保存，不保存每一个增量。

输出目录：

- SFC：`results/source_gear_penalty_full_stride2_match_step`
- Abaqus penalty：`results/source_gear_abaqus_penalty_full_stride2_match_step_0012`
- 独立 manifest 对比：`results/source_gear_penalty_full_stride2_match_step_0012_alignment`

独立 manifest 对比命令：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0012\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0012_alignment
```

`1.2e-3 s` 阶段结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 1.2e-3 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 61 |
| Abaqus VTK 帧数 | 61 |
| SFC 续跑 wall time | 62.424 s |
| Abaqus analysis wall time | 1860.786 s |
| Abaqus VTK export wall time | 914.661 s |
| 末帧 displacement magnitude rel. error | 3.188% |
| 末帧 p95 node-averaged von Mises rel. error | 36.049% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 36.049% |
| 末帧 max node-averaged von Mises rel. error | 10.890% |
| 末帧 mean node-averaged von Mises rel. error | 22.822% |
| 末帧 gear 1 prescribed rotation rel. error | 8.14e-6% |
| 末帧 gear 2 rotation rel. error | 3.910% |
| 末帧 gear 2 angular velocity rel. error | 2.669% |

位移、源 deck 规定的 gear 1 角速度/转角以及 gear 2 的整体动力学响应仍保持接近；p95 stress/strain 误差在该瞬态升高，说明局部接触应力分位数比位移和刚体运动量更敏感。后续若要进一步降低应力/应变误差，应优先继续对齐 Abaqus 的接触压力分布、active patch 覆盖范围、surface-to-surface 约束平均和应力采样位置，而不是改变 SDF 查询或通过阻尼/滤波人为压低峰值。

PVD 时间戳已经同步为 `0:2e-5:1.2e-3`。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 120，可继续向 `0.05 s` 推进。

## 更新：续跑到 `1.4e-3 s`

SFC 已继续从同一个 checkpoint 续跑到 `1.4e-3 s`。Abaqus penalty 对照也使用同一 `gear_contact.inp`、同一 `dt=1e-5 s`、同一 `frame_stride=2` 跑到 `1.4e-3 s`。两边均使用线性罚函数接触；VTK 均为隔帧保存。

输出目录：

- SFC：`results/source_gear_penalty_full_stride2_match_step`
- Abaqus penalty：`results/source_gear_abaqus_penalty_full_stride2_match_step_0014`
- 独立 manifest 对比：`results/source_gear_penalty_full_stride2_match_step_0014_alignment`

独立 manifest 对比命令：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0014\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0014_alignment
```

`1.4e-3 s` 阶段结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 1.4e-3 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 71 |
| Abaqus VTK 帧数 | 71 |
| SFC 续跑 wall time | 60.890 s |
| Abaqus analysis wall time | 2127.960 s |
| Abaqus VTK export wall time | 1071.082 s |
| 末帧 displacement magnitude rel. error | 2.310% |
| 末帧 p95 node-averaged von Mises rel. error | 45.816% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 45.816% |
| 末帧 max node-averaged von Mises rel. error | 20.141% |
| 末帧 mean node-averaged von Mises rel. error | 41.215% |
| 末帧 gear 1 prescribed rotation rel. error | 3.39e-6% |
| 末帧 gear 2 rotation rel. error | 3.566% |
| 末帧 gear 2 angular velocity rel. error | 2.480% |
| 末帧 object 1 max node-averaged von Mises rel. error | 5.895% |
| 末帧 object 2 p95 node-averaged von Mises rel. error | 45.408% |

位移和源 deck 中规定的 gear 1 运动继续保持强一致；gear 2 的整体转动量仍在约 2.5--3.6% 误差范围内。应力/应变误差主要集中在 object 2 的局部接触区域统计，object 1 的 max node-averaged von Mises 已在 6% 内。这个结果进一步说明剩余误差更可能来自 Abaqus 与 SFC 对接触压力分布、active patch 覆盖、surface-to-surface 约束平均和应力采样位置的定义差异，而不是转动单位、时间步长或全局位移曲线错配。

PVD 时间戳已经同步为 `0:2e-5:1.4e-3`。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 140，可继续向 `0.05 s` 推进。
