# 源齿轮 0.003 s 对齐诊断

本文档记录 `commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp`
对应工况在 `0.003 s` 窗口下的 SFC/Abaqus 对齐状态。该诊断只读取已有
VTK manifest 和 regional error CSV，不读取 Abaqus ODB，不运行 Abaqus，也不修改
`src/sfc` 核心求解器。

## 输入数据

- SFC manifest:
  `results/source_gear_penalty_contact_0030_sfc_alpha0/sfc_vtk/sfc_manifest.csv`
- Abaqus manifest:
  `results/source_gear_abaqus_penalty_full_stride2_match_step_0030/abaqus_vtk/abaqus_manifest.csv`
- Regional error CSV:
  `results/source_gear_penalty_contact_0030_alpha0_sequence_alignment/source_gear_regional_vtk_errors.csv`

两边均使用线性罚函数接触；SFC 使用 `hht_alpha=0.0`，避免把无阻尼工况误设为
moderate-dissipation HHT。

## 新增诊断工具

新增：

```powershell
python validation\diagnose_source_gear_alignment.py `
  --sfc-manifest results\source_gear_penalty_contact_0030_sfc_alpha0\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0030\abaqus_vtk\abaqus_manifest.csv `
  --regional-errors results\source_gear_penalty_contact_0030_alpha0_sequence_alignment\source_gear_regional_vtk_errors.csv `
  --out-dir results\source_gear_penalty_contact_0030_alpha0_diagnostics `
  --time-tolerance 1e-8
```

输出：

- `results/source_gear_penalty_contact_0030_alpha0_diagnostics/source_gear_alignment_diagnostics.csv`
- `results/source_gear_penalty_contact_0030_alpha0_diagnostics/source_gear_alignment_diagnostic_summary.csv`
- `results/source_gear_penalty_contact_0030_alpha0_diagnostics/source_gear_alignment_diagnostic_summary.md`

## 末帧结果

在 `t = 0.003 s`：

| 指标 | SFC | Abaqus | 相对误差 |
| --- | ---: | ---: | ---: |
| full displacement p95 | `4.559392e-2` | `4.406546e-2` | `3.469%` |
| full von Mises p95 | `3.204770e7` | `2.706007e7` | `18.432%` |
| full strain norm p95 | `1.688752e-4` | `1.871732e-4` | `9.776%` |
| full equivalent elastic strain p95 | `1.334018e-4` | `1.126403e-4` | `18.432%` |
| active contact nodes | `0` | `88` | active set 不一致 |
| max contact pressure | `0` | `5.660279e5` | `100%` |
| RP1 rotation z | `0.15708 rad` | `0.1570799947 rad` | `3.39e-6%` |
| RP2 rotation z | `9.197807 rad` | `2.575642 rad` | `257.107%` |
| RP2 equivalent-branch rotation z | `2.914622 rad` | `2.575642 rad` | `13.161%` |
| RP2 angular velocity z | `6156.638 rad/s` | `5771.216 rad/s` | `6.678%` |

## 最大误差位置

| 指标 | 最大误差 | 时刻 |
| --- | ---: | ---: |
| full von Mises p95 | `742.299%` | `0.00042 s` |
| active-union von Mises p95 | `2225.597%` | `0.00042 s` |
| RP2 rotation z | `211404.929%` | `0.00252 s` |
| RP2 equivalent-branch rotation z | `0.338980 rad` absolute error | `0.003 s` |
| p95 contact pressure | `100%` | `2.0e-5 s` |

## 结论

位移曲线目前不能作为应力/应变正确性的充分证据。原因是 Gear 1 的转动由
RP 角速度强约束，整体位移 p95 很容易被驱动条件拉到低误差；但应力/应变由
接触压力历史、active/release 时序、RP2 转矩响应和应力恢复口径决定。

当前失败点已经收敛到两类：

1. **RP2 转矩驱动响应仍有差异，但 raw 转角误差不能直接解释。**
   RP1 角速度对齐。RP2 raw 转角在末帧看似差异达到 `257.107%`，但这是因为
   Abaqus `nlgeom=YES` 下 `UR` 是有限转动输出，会在大转角时改变 `2π` 分支；
   SFC manifest 记录的是累计 reduced 角位移。把 SFC 转角折返到最接近 Abaqus
   `UR` 的等效分支后，末帧 RP2 转角误差为 `0.338980 rad`，相对误差
   `13.161%`。因此 RP2 动力学仍需对齐，但不能再把 raw `257%` 当成真实物理误差。
   下一步应检查 Abaqus `*MPC, BEAM` 的有限转动、转动惯量、约束反力和等效虚功口径。

2. **接触状态和压力历史不一致。**
   末帧 SFC active contact nodes 为 `0`，Abaqus 为 `88`；SFC 最大接触压力为
   `0`，Abaqus 为 `5.660279e5`。动态应力是历史量，接触释放时序一旦不同，
   后续应力波和齿根应力趋势就会分叉。

因此，下一步不应继续调接触刚度或阻尼。应优先做通用理论对齐：

1. 对齐 Abaqus `*MPC, BEAM` 的 RP finite-rotation / equivalent virtual work 口径；
2. 对齐 surface-to-surface penalty 的 active/release 和 pressure distribution；
3. 在同一输出口径下比较 von Mises 与应变，避免混用 SFC 小应变范数和 Abaqus
   `LE` 对数应变范数；
4. 只有当 RP2 响应和 contact status 时间线对齐后，再判断 SDF 接触本身的误差。

## 当前验收状态

- displacement latest gate `<=10%`: 通过；
- strain-norm latest gate `<=10%`: 勉强通过；
- von Mises latest gate `<=10%`: 未通过；
- combined latest gate: 未通过。

当前结果还不能满足“应力/应变曲线误差小于等于 10%”的目标。

## 新增 surface-to-surface 约束平均路径

为避免继续调接触刚度，本轮新增了 source-drive penalty 的显式选项：

```powershell
--source-contact-averaging slave_face
```

该选项不改变 SDF closest-feature 查询，也不改变罚函数刚度；它只把同一个 TET4
slave 三角面上的 3 个 `tri3` 采样点按面积合并成一个 slave-face 约束，作为
Abaqus-style surface-to-surface constraint averaging 的通用近似。默认值仍为
`none`，因此不会影响已有路径。

真实源齿轮网格上的 2-step smoke 结果显示该方向是合理的：

| 路径 | t | active contact samples | min gap | normal force | p95 von Mises |
| --- | ---: | ---: | ---: | ---: | ---: |
| no averaging | `2e-5 s` | `276` | `-7.857767e-4` | `22.306268` | `9.213437e6` |
| slave-face averaging | `2e-5 s` | `59` | `-7.469806e-4` | `14.405619` | `9.211585e6` |

这说明当前早期接触过强的问题确实与“逐 Gauss 点独立罚函数”有关；slave-face
constraint averaging 可以显著降低过大的 active 样本数和法向力。下一步需要用
该通用路径重新跑到 `0.003 s`，再与 Abaqus penalty VTK 对比。如果 active/release
时间线仍不能对齐，再继续处理 pressure smoothing 和 contact release 口径。
