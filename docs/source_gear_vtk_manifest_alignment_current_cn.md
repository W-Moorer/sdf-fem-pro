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

## 更新：续跑到 `1.6e-3 s`

SFC 已继续从同一个 checkpoint 续跑到 `1.6e-3 s`。Abaqus penalty 对照也使用同一 `gear_contact.inp`、同一 `dt=1e-5 s`、同一 `frame_stride=2` 跑到 `1.6e-3 s`。两边均使用线性罚函数接触；VTK 均为隔帧保存。

输出目录：

- SFC：`results/source_gear_penalty_full_stride2_match_step`
- Abaqus penalty：`results/source_gear_abaqus_penalty_full_stride2_match_step_0016`
- 独立 manifest 对比：`results/source_gear_penalty_full_stride2_match_step_0016_alignment`

独立 manifest 对比命令：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0016\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0016_alignment
```

`1.6e-3 s` 阶段结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 1.6e-3 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 81 |
| Abaqus VTK 帧数 | 81 |
| SFC 续跑 wall time | 56.538 s |
| Abaqus analysis wall time | 2591.466 s |
| Abaqus VTK export wall time | 1226.457 s |
| 末帧 displacement magnitude rel. error | 1.209% |
| 末帧 p95 node-averaged von Mises rel. error | 52.750% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 52.750% |
| 末帧 max node-averaged von Mises rel. error | 51.606% |
| 末帧 mean node-averaged von Mises rel. error | 48.600% |
| 末帧 gear 1 prescribed rotation rel. error | 3.39e-6% |
| 末帧 gear 2 rotation rel. error | 3.283% |
| 末帧 gear 2 angular velocity rel. error | 2.192% |
| 末帧 object 1 max displacement rel. error | 0.020% |
| 末帧 object 2 max displacement rel. error | 1.209% |

位移曲线和源 deck 中规定的 gear 1 运动继续收敛式接近 Abaqus；gear 2 的整体转动误差也保持在约 2--3% 水平。相反，node-averaged stress/strain 的 p95、max、mean 误差都明显升高，说明现阶段不能把 SFC 与 Abaqus 的应力云图差异解释为转动单位、时间步长或动画色标问题。后续优先级应转向：

1. 对齐 Abaqus surface-to-surface penalty 接触压力在约束区域内的分布/平均方式；
2. 检查 active patch 覆盖范围是否漏掉或过早剔除齿面边缘接触；
3. 对齐 Abaqus 与 SFC 的应力/应变采样位置和 nodal averaging 规则；
4. 再比较完整应力/应变云图，而不是仅用单一分位数判断场误差。

PVD 时间戳已经同步为 `0:2e-5:1.6e-3`。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 160，可继续向 `0.05 s` 推进。

## 更新：续跑到 `1.8e-3 s`

SFC 已继续从同一个 checkpoint 续跑到 `1.8e-3 s`。Abaqus penalty 对照也使用同一 `gear_contact.inp`、同一 `dt=1e-5 s`、同一 `frame_stride=2` 跑到 `1.8e-3 s`。两边均使用线性罚函数接触；VTK 均为隔帧保存。

输出目录：

- SFC：`results/source_gear_penalty_full_stride2_match_step`
- Abaqus penalty：`results/source_gear_abaqus_penalty_full_stride2_match_step_0018`
- 独立 manifest 对比：`results/source_gear_penalty_full_stride2_match_step_0018_alignment`

独立 manifest 对比命令：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0018\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0018_alignment
```

`1.8e-3 s` 阶段结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 1.8e-3 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 91 |
| Abaqus VTK 帧数 | 91 |
| SFC 续跑 wall time | 49.230 s |
| Abaqus analysis wall time | 2920.718 s |
| Abaqus VTK export wall time | 1363.377 s |
| 末帧 displacement magnitude rel. error | 0.299% |
| 末帧 p95 node-averaged von Mises rel. error | 62.688% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 62.688% |
| 末帧 max node-averaged von Mises rel. error | 34.996% |
| 末帧 mean node-averaged von Mises rel. error | 60.819% |
| 末帧 gear 1 prescribed rotation rel. error | 3.39e-6% |
| 末帧 gear 2 rotation rel. error | 3.043% |
| 末帧 gear 2 angular velocity rel. error | 2.151% |
| 末帧 object 1 max displacement rel. error | 0.0036% |
| 末帧 object 2 max displacement rel. error | 0.299% |

`1.8e-3 s` 窗口进一步确认：位移曲线已经非常接近 Abaqus penalty 参考，源 deck 规定的 gear 1 转动几乎完全一致，gear 2 的角速度误差仍在约 2% 水平。与此同时，应力/应变的 p95 与 mean 误差继续偏大，且局部 object 1 的 max node-averaged stress 在该瞬态出现异常高的相对误差。这个状态不应再归因于 SFC 应力动画闪动；当前输出已经使用固定色标脚本和 node-averaged 场。后续需要单独做应力口径诊断：

1. 比较 Abaqus 与 SFC 的元素积分点应力、单元中心应力、节点平均应力三种口径；
2. 输出接触区 active samples / contact pressure 云图，确认应力偏差是否来自压力分布而非体响应；
3. 检查 object 1 齿根/齿面局部峰值是否由 nodal averaging 和齿面边缘采样造成；
4. 在完成上述诊断前，论文主曲线应优先展示位移/转角/角速度和完整云图，避免用单一 p95 stress 指标代表全部物理对齐。

PVD 时间戳已经同步为 `0:2e-5:1.8e-3`。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 180，可继续向 `0.05 s` 推进。

## 更新：续跑到 `2.0e-3 s`

SFC 已继续从同一个 checkpoint 续跑到 `2.0e-3 s`。Abaqus penalty 对照也使用同一 `gear_contact.inp`、同一 `dt=1e-5 s`、同一 `frame_stride=2` 跑到 `2.0e-3 s`。两边均使用线性罚函数接触；VTK 均为隔帧保存。

输出目录：

- SFC：`results/source_gear_penalty_full_stride2_match_step`
- Abaqus penalty：`results/source_gear_abaqus_penalty_full_stride2_match_step_0020`
- 独立 manifest 对比：`results/source_gear_penalty_full_stride2_match_step_0020_alignment`

独立 manifest 对比命令：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0020_alignment
```

`2.0e-3 s` 阶段结果：

| 项目 | 数值 |
| --- | ---: |
| 时间窗 | `0 ~ 2.0e-3 s` |
| 固定步长 | `1e-5 s` |
| VTK 保存间隔 | 每 2 步 |
| SFC VTK 帧数 | 101 |
| Abaqus VTK 帧数 | 101 |
| SFC 续跑 wall time | 136.221 s |
| Abaqus analysis wall time | 3320.869 s |
| Abaqus VTK export wall time | 1527.732 s |
| 末帧 displacement magnitude rel. error | 2.716% |
| 末帧 p95 node-averaged von Mises rel. error | 68.646% |
| 末帧 p95 node-averaged equivalent elastic strain rel. error | 68.646% |
| 末帧 max node-averaged von Mises rel. error | 45.675% |
| 末帧 mean node-averaged von Mises rel. error | 69.273% |
| 末帧 gear 1 prescribed rotation rel. error | 8.14e-6% |
| 末帧 gear 2 rotation rel. error | 2.923% |
| 末帧 gear 2 angular velocity rel. error | 2.640% |
| 末帧 object 1 max displacement rel. error | 3.82e-4% |
| 末帧 object 2 max displacement rel. error | 2.716% |
| SFC 末帧 active contact samples | 0 |
| SFC 末帧 min gap | 0 |

`2.0e-3 s` 窗口显示出一个新的诊断信号：SFC 末帧 active contact samples 为 0，且 min gap 为 0，而 Abaqus penalty 参考在相同时间窗的 node-averaged stress/strain 仍明显更高。位移和整体转动仍在低误差范围内，因此当前主要问题不是驱动/积分时间步，而是 contact status、active set 和 stress recovery 的口径差异。后续应优先做：

1. 在 SFC VTK 中输出 active contact samples / contact pressure 的时间序列云图；
2. 从 Abaqus 外部 VTK 或可导出的接触变量中提取同时间窗的接触状态/压力指标；
3. 检查 SFC contact release 条件是否比 Abaqus penalty 接触更早释放；
4. 将应力对比拆成 contact-active 区域、near-contact 区域和全场区域，而不是只报全场 p95/max/mean。

PVD 时间戳已经同步为 `0:2e-5:2.0e-3`。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 200，可继续向 `0.05 s` 推进。

## 更新：SFC 接触状态/压力诊断输出

`2.0e-3 s` 窗口显示 SFC 末帧已释放接触，而 Abaqus penalty 参考仍有较高 stress/strain。为避免继续只看位移和应力分位数，已在 SFC VTK/manifest 输出中加入接触状态诊断字段。该修改不改变接触力、时间积分、罚函数刚度或求解路径，只增加后处理字段。

新增 SFC VTK point-data 字段：

- `contact_pressure_nodeavg`
- `contact_penetration_nodeavg`
- `contact_active_node`
- `contact_gap_min_node`
- `contact_sample_area_weight`

新增 manifest/history 统计字段：

- `active_contact_node_count`
- `max_contact_pressure_nodeavg`
- `p95_contact_pressure_nodeavg`
- `mean_active_contact_pressure_nodeavg`
- `max_contact_penetration_nodeavg`
- `min_contact_gap_node`

验证：

```powershell
pytest -q tests\test_flexible_gear_implicit_lagrangian_sdf.py -k "writes_sfc_vtk_frames or source_drive_path_advances_rp_rotation or checkpoint_resume_matches_continuous_short_run or animation_setup"
```

结果：

```text
4 passed, 20 deselected
```

随后使用当前真实全齿轮 checkpoint 从 step 200 续跑到 step 202，即 `2.02e-3 s`，只生成一个新的 SFC 隔帧 VTK 用于确认诊断字段落盘：

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --duration 0.00202 `
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

`2.02e-3 s` 诊断帧：

| 项目 | 数值 |
| --- | ---: |
| 新增 SFC VTK | `results/source_gear_penalty_full_stride2_match_step/sfc_vtk/sfc_0101.vtk` |
| VTK 字段检查 | `contact_pressure_nodeavg`, `contact_active_node`, `contact_penetration_nodeavg`, `contact_gap_min_node` 已存在 |
| 时间 | `2.02e-3 s` |
| SFC VTK 帧数 | 102 |
| active_contact_node_count | 0 |
| max_contact_pressure_nodeavg | 0 |
| p95_contact_pressure_nodeavg | 0 |
| min_contact_gap_node | 0 |
| max displacement magnitude | 0.041392 |
| p95 node-averaged von Mises | 4.367104e6 |

这说明后续不需要再依赖 aggregate `active_contact_samples` 猜测接触状态，可以直接在 ParaView 中查看 SFC 的接触压力/激活云图。`results/source_gear_penalty_full_stride2_match_step/source_drive_checkpoint.npz` 已更新到 step 202；后续可继续向 `0.05 s` 推进，或者先提取 Abaqus 的相同 contact status/pressure 参考量，再按 contact-active 区域、near-contact 区域和全场区域分别比较 stress/strain。

## 更新：Abaqus penalty 接触场导出诊断

已补充 Abaqus ODB 到 VTK 的接触场导出支持。导出器现在会优先读取 `CPRESS`、`COPEN`、`CSTATUS`，并兼容 Abaqus 可能使用的 `CSTRESS`、`CDISP` 组合输出形式。Abaqus 对未定义接触开闭量会写入接近 `-3.4e38` 的哨兵值；导出器会过滤这类非物理哨兵值，避免把“未定义接触输出”误认为巨大穿透。

新增 Abaqus VTK point-data 字段：

- `contact_pressure_nodeavg`
- `contact_opening_node`
- `contact_penetration_nodeavg`
- `contact_status_node`
- `contact_active_node`

新增 Abaqus manifest 字段：

- `contact_output_available`
- `contact_pressure_source`
- `contact_opening_source`
- `contact_status_source`
- `contact_pressure_value_count`
- `contact_opening_value_count`
- `contact_status_value_count`
- `active_contact_node_count`
- `max_contact_pressure_nodeavg`
- `p95_contact_pressure_nodeavg`
- `mean_active_contact_pressure_nodeavg`
- `max_contact_penetration_nodeavg`
- `min_contact_gap_node`

使用已有 `2.0e-3 s` Abaqus ODB 只导出末帧进行验证，没有重新运行 Abaqus 求解：

```powershell
abaqus python validation\abaqus_odb_to_vtk.py `
  --odb results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_run\gear_contact_source_penalty.odb `
  --out-dir results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_check `
  --stem abaqus_contact_check `
  --frame-stride 1 `
  --time-start 0.001999 `
  --time-end 0.002001 `
  --young 2.05e11 `
  --poisson 0.28 `
  --scalars-only
```

验证输出：

| 项目 | 数值 |
| --- | ---: |
| Abaqus contact-check VTK | `results/source_gear_abaqus_penalty_full_stride2_match_step_0020/abaqus_vtk_contact_check/abaqus_contact_check_0000.vtk` |
| Abaqus contact-check manifest | `results/source_gear_abaqus_penalty_full_stride2_match_step_0020/abaqus_vtk_contact_check/abaqus_contact_check_manifest.csv` |
| time | `0.0020000000949949026` |
| contact_output_available | 1 |
| contact_pressure_source | `CPRESS` |
| contact_opening_source | `COPEN` |
| contact_pressure_value_count | 12716 |
| contact_opening_value_count | 209 |
| active_contact_node_count | 251 |
| max_contact_pressure_nodeavg | `7.7778825e5` |
| mean_active_contact_pressure_nodeavg | `3.445386552846232e5` |
| max_contact_penetration_nodeavg | `1.6066397074609995e-4` |
| min_contact_gap_node | `-1.6066397074609995e-4` |

这一步把 Abaqus 与 SFC 的接触诊断口径补齐到同一类 VTK/manifest 字段。接下来可以将误差拆成三类区域比较：contact-active 区域、near-contact 区域和全场区域；这样能判断当前 stress/strain 差异是来自接触释放/压力分布，还是来自 stress recovery 与节点平均口径。

## 更新：区域化 VTK 场误差诊断

已新增区域化 VTK 后处理脚本：

```powershell
python validation\run_source_gear_vtk_regional_alignment.py `
  --sfc-vtk results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_0100.vtk `
  --abaqus-vtk results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_check\abaqus_contact_check_0000.vtk `
  --sfc-time 0.002000000000 `
  --abaqus-time 0.0020000000949949026 `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0020_regional_alignment
```

该脚本只读取已经导出的 legacy VTK 文件，不读取 Abaqus ODB，也不重新运行 SFC 或 Abaqus。它会按区域输出位移、node-averaged von Mises、strain norm 和 equivalent elastic strain 的 p95/max 误差。目前支持：

- `full`：全部节点；
- `abaqus_active`：Abaqus contact pressure / penetration / active 字段标记的接触节点；
- `sfc_active`：SFC contact 字段标记的接触节点；
- `active_union`：两边 active 节点并集。

同一时间 `2.0e-3 s` 的区域化输出：

| 区域 | 指标 | 节点数 | p95 相对误差 | max 相对误差 |
| --- | --- | ---: | ---: | ---: |
| full | displacement magnitude | 38884 | 2.716% | 2.716% |
| full | von Mises nodeavg | 38884 | 68.646% | 45.675% |
| full | strain norm nodeavg | 38884 | 75.823% | 70.743% |
| full | equivalent elastic strain nodeavg | 38884 | 68.646% | 45.675% |
| Abaqus active | displacement magnitude | 251 | 2.716% | 2.716% |
| Abaqus active | von Mises nodeavg | 251 | 76.635% | 74.845% |
| Abaqus active | strain norm nodeavg | 251 | 85.173% | 84.262% |
| Abaqus active | equivalent elastic strain nodeavg | 251 | 76.635% | 74.845% |

区域化结果进一步说明：当前位移场已经对齐到低误差，但应力/应变差异主要集中在 Abaqus active contact 区域，并且明显高于全场位移误差。这支持后续优先排查 contact pressure distribution、contact release/status 和 stress recovery 口径，而不是再把问题归因到齿轮驱动角速度或时间步长。

补充检查还使用 `2.02e-3 s` 的新 SFC contact VTK 帧与 `2.0e-3 s` Abaqus 末帧做了非最终字段可用性验证：

```powershell
python validation\run_source_gear_vtk_regional_alignment.py `
  --sfc-vtk results\source_gear_penalty_full_stride2_match_step\sfc_vtk\sfc_0101.vtk `
  --abaqus-vtk results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_check\abaqus_contact_check_0000.vtk `
  --sfc-time 0.00202 `
  --abaqus-time 0.0020000000949949026 `
  --out-dir results\source_gear_penalty_full_stride2_match_step_0020_regional_alignment_sfc_contact_check
```

该检查仅用于确认两边 VTK 均可读取 contact 字段；由于时间相差约 `2e-5 s`，不作为最终误差证明。它显示 SFC contact fields present = 1、Abaqus contact fields present = 1，但 SFC active node count = 0、Abaqus active node count = 251。这与前面的诊断一致：当前主要差异是 SFC 在该窗口附近比 Abaqus penalty 更早释放接触或压力分布更弱。

## 更新：同口径短窗口 contact-field 序列

为避免只看单个末帧，已开始把双方导出为带接触字段的同口径序列。直接重导出 Abaqus `0~2.0e-3 s` 的 101 帧 contact VTK 时，桌面任务在约 10 分钟后超时，但已有 40 帧落盘。为避免这种长导出中断后没有索引文件，`validation/abaqus_odb_to_vtk.py` 已改为每写完一帧就增量刷新 `*.pvd` 和 `*_manifest.csv`。这样后续即使分窗或中断，也可以直接使用已完成的帧。

使用已有 Abaqus ODB 导出了一个小窗口 `0~4e-5 s`，不重新运行 Abaqus 求解：

```powershell
abaqus python validation\abaqus_odb_to_vtk.py `
  --odb results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_run\gear_contact_source_penalty.odb `
  --out-dir results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_incremental_check `
  --stem abaqus `
  --frame-stride 1 `
  --time-start 0.0 `
  --time-end 0.00004 `
  --young 2.05e11 `
  --poisson 0.28 `
  --scalars-only
```

Abaqus contact-field 小窗口：

| frame | time | active contact nodes | max contact pressure |
| ---: | ---: | ---: | ---: |
| 0 | `0` | 0 | 0 |
| 1 | `1.999999949e-5` | 138 | `7.3608336e4` |
| 2 | `3.999999899e-5` | 220 | `2.57393297e5` |

随后用相同 `gear_contact.inp`、相同 `dt=1e-5 s`、相同线性罚函数接触，短程重跑 SFC 到 `4e-5 s`：

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --duration 0.00004 `
  --dt 0.00001 `
  --pressure-stiffness 5e9 `
  --write-sfc-vtk `
  --vtk-frame-stride 2 `
  --vtk-scalars-only `
  --history-frame-stride 2 `
  --source-checkpoint results\source_gear_penalty_contact_incremental_check_sfc\source_drive_checkpoint.npz `
  --source-checkpoint-stride 20 `
  --out-dir results\source_gear_penalty_contact_incremental_check_sfc
```

SFC contact-field 小窗口：

| frame | time | active contact nodes | max contact pressure | min gap |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `0` | 0 | 0 | 0 |
| 1 | `2e-5` | 196 | `3.9311277e6` | `-7.8622713e-4` |
| 2 | `4e-5` | 281 | `3.7241691e6` | `-7.4483485e-4` |

同一时间 `4e-5 s` 的区域化误差：

```powershell
python validation\run_source_gear_vtk_regional_alignment.py `
  --sfc-vtk results\source_gear_penalty_contact_incremental_check_sfc\sfc_vtk\sfc_0002.vtk `
  --abaqus-vtk results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_incremental_check\abaqus_0002.vtk `
  --sfc-time 0.00004 `
  --abaqus-time 0.000039999998989515007 `
  --out-dir results\source_gear_penalty_contact_incremental_check_regional_alignment
```

| 区域 | 指标 | 节点数 | p95 相对误差 | max 相对误差 |
| --- | --- | ---: | ---: | ---: |
| full | displacement magnitude | 38884 | 0.040% | 0.055% |
| full | von Mises nodeavg | 38884 | 0.906% | 2.565% |
| full | strain norm nodeavg | 38884 | 34.167% | 48.314% |
| full | equivalent elastic strain nodeavg | 38884 | 0.906% | 2.565% |
| active union | displacement magnitude | 285 | 0.088% | 0.090% |
| active union | von Mises nodeavg | 285 | 1.400% | 1.350% |
| active union | strain norm nodeavg | 285 | 40.167% | 37.932% |
| active union | equivalent elastic strain nodeavg | 285 | 1.400% | 1.350% |

这个短窗口给出两个重要结论：

1. 在接触刚开始阶段，位移、von Mises 和 equivalent elastic strain 已经能与 Abaqus penalty 对齐到低误差；这说明驱动、时间步长、RP 角速度弧度制和整体动力学路径在早期窗口是对的。
2. `strain_norm_nodeavg` 误差显著偏高，但 `equivalent_elastic_strain_nodeavg` 与 von Mises 同步低误差，说明当前 strain-norm 指标不是最可靠的跨软件口径。后续论文曲线应优先报告 displacement、von Mises 和 equivalent elastic strain；若继续报告 strain norm，需要明确 Abaqus `LE` 与 SFC strain recovery 的定义差异。

同时，SFC 早期 contact pressure 峰值明显高于 Abaqus，但 stress/equivalent-strain 在 `4e-5 s` 仍对齐较好。这意味着后续中长窗口应继续追踪 contact pressure/active set 随时间演化，判断后期 stress 差异来自接触释放时序、压力分布积分口径，还是应力恢复口径。

## 更新：区域化误差序列曲线

`validation/run_source_gear_vtk_regional_alignment.py` 已从单帧对比扩展为 manifest 序列对比。新模式会按时间自动配对 SFC/Abaqus VTK 帧，并输出长格式误差 CSV 与 p95 相对误差曲线。该脚本仍然只读取已导出的 VTK/manifest，不读取 Abaqus ODB，也不重新运行求解器。

短窗口序列对比命令：

```powershell
python validation\run_source_gear_vtk_regional_alignment.py `
  --sfc-manifest results\source_gear_penalty_contact_incremental_check_sfc\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_incremental_check\abaqus_manifest.csv `
  --time-tolerance 1e-9 `
  --out-dir results\source_gear_penalty_contact_incremental_check_sequence_alignment
```

输出：

- `results/source_gear_penalty_contact_incremental_check_sequence_alignment/source_gear_regional_vtk_errors.csv`
- `results/source_gear_penalty_contact_incremental_check_sequence_alignment/source_gear_regional_vtk_alignment_summary.md`
- `results/source_gear_penalty_contact_incremental_check_sequence_alignment/source_gear_regional_p95_error_curves.png`

该曲线图使用线图而不是柱状图，并将图例放在图外，避免遮挡曲线。当前短窗口只有 3 个时间点，主要用于验证序列对比链路；后续长窗口分段导出后，同一工具可直接生成更完整的位移、应力、等效应变误差曲线。

短窗口序列最大 p95 误差：

| 区域 | 指标 | 发生时间 | p95 相对误差 |
| --- | --- | ---: | ---: |
| full | displacement magnitude | `2e-5` | 0.103% |
| full | von Mises nodeavg | `4e-5` | 0.906% |
| full | equivalent elastic strain nodeavg | `4e-5` | 0.906% |
| Abaqus active | displacement magnitude | `2e-5` | 0.435% |
| Abaqus active | von Mises nodeavg | `2e-5` | 62.094% |
| Abaqus active | equivalent elastic strain nodeavg | `2e-5` | 62.094% |

由于 `Abaqus active` 区域在 `2e-5 s` 只有 138 个节点，且接触刚建立，局部 active 区域的 p95 对节点集差异非常敏感；主文应优先使用 full-field 曲线与接触压力/active set 曲线共同解释，而不是单独用 active 区域 p95 作为精度结论。

## 更新：同口径序列推进到 `2e-4 s`

已将同口径 contact-field 序列从 `4e-5 s` 推进到 `2e-4 s`。Abaqus 侧仍然只使用已有 ODB 重新导出 VTK，不重新运行 Abaqus 求解；SFC 侧从 `4e-5 s` checkpoint 续跑到 `2e-4 s`，保持同一 `gear_contact.inp`、同一 `dt=1e-5 s`、同一 source RP 驱动和同一线性罚函数接触。两边均为隔帧保存，因此 `0~2e-4 s` 共 11 帧。

Abaqus 导出命令：

```powershell
abaqus python validation\abaqus_odb_to_vtk.py `
  --odb results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_run\gear_contact_source_penalty.odb `
  --out-dir results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_0002 `
  --stem abaqus `
  --frame-stride 1 `
  --time-start 0.0 `
  --time-end 0.0002 `
  --young 2.05e11 `
  --poisson 0.28 `
  --scalars-only
```

SFC 续跑命令：

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
  --source-checkpoint results\source_gear_penalty_contact_incremental_check_sfc\source_drive_checkpoint.npz `
  --resume-source-checkpoint `
  --source-checkpoint-stride 20 `
  --out-dir results\source_gear_penalty_contact_incremental_check_sfc
```

序列对比命令：

```powershell
python validation\run_source_gear_vtk_regional_alignment.py `
  --sfc-manifest results\source_gear_penalty_contact_incremental_check_sfc\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_0002\abaqus_manifest.csv `
  --time-tolerance 1e-9 `
  --out-dir results\source_gear_penalty_contact_0002_sequence_alignment
```

输出：

- `results/source_gear_abaqus_penalty_full_stride2_match_step_0020/abaqus_vtk_contact_0002/abaqus.pvd`
- `results/source_gear_penalty_contact_incremental_check_sfc/sfc_vtk/sfc.pvd`
- `results/source_gear_penalty_contact_0002_sequence_alignment/source_gear_regional_vtk_errors.csv`
- `results/source_gear_penalty_contact_0002_sequence_alignment/source_gear_regional_p95_error_curves.png`
- `results/source_gear_penalty_contact_0002_sequence_alignment/source_gear_regional_vtk_alignment_summary.md`

`2e-4 s` 末帧区域化误差：

| 区域 | 指标 | 节点数 | p95 相对误差 | max 相对误差 |
| --- | --- | ---: | ---: | ---: |
| full | displacement magnitude | 38884 | 1.386% | 1.395% |
| full | von Mises nodeavg | 38884 | 39.672% | 33.284% |
| full | equivalent elastic strain nodeavg | 38884 | 39.672% | 33.284% |
| Abaqus active | displacement magnitude | 466 | 1.448% | 1.448% |
| Abaqus active | von Mises nodeavg | 466 | 66.292% | 65.426% |
| Abaqus active | equivalent elastic strain nodeavg | 466 | 66.292% | 65.426% |

`0~2e-4 s` 过程中，full-field 位移 p95 误差始终保持在约 `1.4%` 以内，说明源 deck 驱动、弧度制角速度、时间步长和整体运动对齐仍然成立。但 von Mises / equivalent elastic strain 从 `1.6e-4 s` 后开始快速偏离，到 `2e-4 s` 达到约 `39.7%`。接触状态也开始出现差异：末帧 SFC active contact nodes 为 `381`，Abaqus active contact nodes 为 `466`；SFC max contact pressure 为 `3.925976e6`，Abaqus max contact pressure 为 `5.8205955e6`。这进一步定位了后续误差来源：需要优先对齐 penalty contact pressure distribution、active/release status 和接触面积积分口径，而不是继续修改驱动或时间步长。

曲线图已改为 ms 横轴并限制刻度数量，避免角标/刻度遮挡曲线；图例保持在图外，适合后续放入论文草图。

## 更新：同口径序列推进到 `5e-4 s`

已继续将同口径 contact-field 序列推进到 `5e-4 s`。Abaqus 侧仍复用已有 ODB 进行 VTK 后处理，不重新运行求解；SFC 侧从 `2e-4 s` checkpoint 续跑到 `5e-4 s`。两边均使用同一源 `gear_contact.inp` 的 RP/MPC、角速度、力矩、材料、网格和边界条件，只将接触统一为线性罚函数并使用当前 SFC 积分/接触实现。

Abaqus 导出命令：

```powershell
abaqus python validation\abaqus_odb_to_vtk.py `
  --odb results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_run\gear_contact_source_penalty.odb `
  --out-dir results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_0005 `
  --stem abaqus `
  --frame-stride 1 `
  --time-start 0.0 `
  --time-end 0.0005 `
  --young 2.05e11 `
  --poisson 0.28 `
  --scalars-only
```

SFC 续跑命令：

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --duration 0.0005 `
  --dt 0.00001 `
  --pressure-stiffness 5e9 `
  --write-sfc-vtk `
  --vtk-frame-stride 2 `
  --vtk-scalars-only `
  --history-frame-stride 2 `
  --source-checkpoint results\source_gear_penalty_contact_incremental_check_sfc\source_drive_checkpoint.npz `
  --resume-source-checkpoint `
  --source-checkpoint-stride 20 `
  --out-dir results\source_gear_penalty_contact_incremental_check_sfc
```

序列对比命令：

```powershell
python validation\run_source_gear_vtk_regional_alignment.py `
  --sfc-manifest results\source_gear_penalty_contact_incremental_check_sfc\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0020\abaqus_vtk_contact_0005\abaqus_manifest.csv `
  --time-tolerance 1e-9 `
  --out-dir results\source_gear_penalty_contact_0005_sequence_alignment
```

输出：

- `results/source_gear_abaqus_penalty_full_stride2_match_step_0020/abaqus_vtk_contact_0005/abaqus.pvd`
- `results/source_gear_penalty_contact_incremental_check_sfc/sfc_vtk/sfc.pvd`
- `results/source_gear_penalty_contact_0005_sequence_alignment/source_gear_regional_vtk_errors.csv`
- `results/source_gear_penalty_contact_0005_sequence_alignment/source_gear_regional_p95_error_curves.png`
- `results/source_gear_penalty_contact_0005_sequence_alignment/source_gear_regional_p95_metric_curves.png`
- `results/source_gear_penalty_contact_0005_sequence_alignment/source_gear_regional_vtk_alignment_summary.md`

注意：Abaqus 当前 ODB 在 `0~5e-4 s` 时间窗内实际输出到 `4.8e-4 s`，因此序列对比自动配到 25 帧；SFC 额外的 `5e-4 s` 帧保留在 VTK/PVD 中，但不强行与缺失的 Abaqus 帧比较。

`4.8e-4 s` 最新配对帧区域化误差：

| 区域 | 指标 | 节点数 | p95 相对误差 | max 相对误差 |
| --- | --- | ---: | ---: | ---: |
| full | displacement magnitude | 38884 | 4.820% | 4.821% |
| full | von Mises nodeavg | 38884 | 0.478% | 2.447% |
| full | equivalent elastic strain nodeavg | 38884 | 0.478% | 2.447% |
| Abaqus active | displacement magnitude | 97 | 4.817% | 4.817% |
| Abaqus active | von Mises nodeavg | 97 | 647.604% | 639.283% |
| Abaqus active | equivalent elastic strain nodeavg | 97 | 647.604% | 639.283% |

`0~4.8e-4 s` 过程中的最大 full-field p95 误差：

| 指标 | 发生时间 | 最大 p95 相对误差 |
| --- | ---: | ---: |
| displacement magnitude | `4.8e-4 s` | 4.820% |
| von Mises nodeavg | `3.0e-4 s` | 50.533% |
| equivalent elastic strain nodeavg | `3.0e-4 s` | 50.533% |

新增实际量值曲线图 `source_gear_regional_p95_metric_curves.png`，直接绘制 SFC 与 Abaqus 的 full-field p95 displacement、p95 von Mises 和 p95 equivalent strain。该图比误差曲线更适合论文展示：位移曲线基本贴合；von Mises / equivalent strain 在 `0.18~0.30 ms` 区间出现明显分叉，随后又重新接近。这说明运动学和整体位移仍然对齐，但中间接触压力/active set 演化导致应力能量分配存在阶段性差异。

`4.8e-4 s` 附近的接触状态：

| 项目 | SFC | Abaqus |
| --- | ---: | ---: |
| latest paired time | `4.8e-4 s` | `4.799999878741801e-4 s` |
| active contact nodes | 445 | 97 |
| max contact pressure | `4.001781e6` | `9.23529375e5` |

这一步把问题进一步定位为：SFC 当前在后段保留了更大的 active 接触区域和更高局部罚压力，而 Abaqus active 区域快速收缩。论文最终若要使用该齿轮长窗口作为强对比，需要继续对齐 Abaqus 的 surface-to-surface penalty pressure distribution、active/release status 和接触面积积分口径；否则该窗口只能作为“当前诊断阶段结果”，不能作为最终精度 claim。
