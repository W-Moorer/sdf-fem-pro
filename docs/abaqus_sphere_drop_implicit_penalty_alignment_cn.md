# Abaqus 隐式线性罚函数小球跌落对标记录

## 对标对象

- Abaqus 输入文件：`commercial_software_comparison/abaqus_flexible_body_rigid_plane_implicit/sphere_drop/abaqus_run/sphere_drop_implicit.inp`
- Abaqus 结果目录：`commercial_software_comparison/abaqus_flexible_body_rigid_plane_implicit/sphere_drop/vtk`
- 分析类型：Abaqus/Standard 隐式动力学
- 接触律：无摩擦线性罚函数，`pressure-overclosure=LINEAR`
- 法向罚刚度：`5.0e9`
- 总时长：`3.0 s`
- 最大时间增量 / 输出间隔：`0.001 s`
- Abaqus 帧数：`3001`

本次修正了 SFC 验证脚本对 Abaqus 隐式 `*Dynamic` 行的解析。隐式格式为：

```text
initial increment, total time, minimum increment, maximum increment
```

因此总时长应读取第二个数值，而不是最后一个数值。脚本现在也会从 `*Surface Behavior, pressure-overclosure=LINEAR` 下方的数值读取 Abaqus 罚刚度；如果命令行没有显式传入 `--contact-stiffness`，SFC 默认采用该罚刚度。

## SFC 对标设置

- SFC 接触路径：`MaterialSDF + LagrangianSDFContactOracle`
- SFC 积分器：Newmark 隐式步进
- SFC 时间步长：`0.001 s`
- SFC 总时长：`3.0 s`
- SFC 接触罚刚度：`5.0e9`，来自 Abaqus 输入文件
- 接触阻尼：`0.0`
- 刚度比例阻尼：`0.0`
- 质量比例阻尼：`4.0`

质量比例阻尼用于对齐 Abaqus 隐式结果中的总体耗散特征。它不是 Abaqus 输入文件中显式给出的材料参数，因此论文中应表述为验证层的耗散对齐，而不是 Abaqus 源级等价。

## 运行命令

```powershell
python validation\run_abaqus_sphere_drop_full_validation.py `
  --inp commercial_software_comparison\abaqus_flexible_body_rigid_plane_implicit\sphere_drop\abaqus_run\sphere_drop_implicit.inp `
  --vtk-dir commercial_software_comparison\abaqus_flexible_body_rigid_plane_implicit\sphere_drop\vtk `
  --duration 3.0 `
  --dt 0.001 `
  --contact-stiffness 5.0e9 `
  --contact-damping 0.0 `
  --mass-damping 4.0 `
  --stiffness-damping 0.0 `
  --output-stride 1 `
  --out-dir results\abaqus_sphere_drop_implicit_penalty_sfc_aligned
```

## 全程误差结果

| 指标 | 数值 |
| --- | ---: |
| `max_z_cm_abs_error` | `0.0116215886 m` |
| `rms_z_cm_abs_error` | `0.0029100077 m` |
| `max_min_gap_abs_error` | `0.0115649591 m` |
| `full_z_cm_l2_error` | `0.1594142488` |
| `full_min_gap_l2_error` | `0.1592972003` |
| `sfc_first_contact_time` | `0.067 s` |
| `abaqus_first_contact_time` | `0.0649999976 s` |
| `first_contact_time_abs_error` | `0.0020000024 s` |
| `max_von_mises_abs_error` | `6.001501e5` |
| `rms_von_mises_abs_error` | `4.236631e4` |
| `max_strain_norm_abs_error` | `0.0174033072` |
| `rms_strain_norm_abs_error` | `0.0012440639` |
| `newton_iterations_total` | `7020` |
| `newton_failed_steps` | `0` |

直接使用线性罚刚度但不做耗散对齐时，3 s 全程 `rms_z_cm_abs_error` 为 `0.0157027287 m`，`full_z_cm_l2_error` 为 `0.8602172058`。加入质量比例阻尼 `4.0` 后，位移 RMS 误差降到 `0.0029100077 m`，全程位移 L2 误差降到 `0.1594142488`。

## 时间开销

| 项目 | wall time |
| --- | ---: |
| Abaqus/Standard native analysis | `345.819903 s` |
| Abaqus ODB 到 VTK 后处理 | `191.925459 s` |
| Abaqus workflow total | `538.291619 s` |
| SFC solve | `135.108105 s` |

按原生求解 wall time 计，SFC 相对 Abaqus/Standard 为 `2.56x`；按 Abaqus 求解加 ODB-VTK 全流程计为 `3.98x`。论文中应优先报告 `SFC solve time = 135.11 s` 对 `Abaqus native analysis time = 345.82 s`，后处理时间单独列出。

## 输出文件

- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_metrics.csv`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_comparison.csv`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_z_cm.png`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_min_gap.png`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_von_mises.png`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_strain_norm.png`

## 当前结论

这组对标已经把接触律从 Abaqus/Explicit 硬接触调整为 Abaqus/Standard 隐式线性罚函数接触，并让 SFC 自动读取相同罚刚度。SFC 在 3 s 全程中接触触发时间与 Abaqus 相差约 `0.002 s`，位移 RMS 误差约 `2.91 mm`，且完整 SFC solve wall time 小于 Abaqus 原生求解 wall time。

仍需谨慎表述的是：质量比例阻尼是为了对齐隐式动力学耗散响应而加入的验证层参数；它支撑该工况下的结果对齐，但不等价于 Abaqus 内部算法的源级复现。
