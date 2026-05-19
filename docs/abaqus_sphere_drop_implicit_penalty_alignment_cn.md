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
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_z_cm_abs_error.png`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_min_gap.png`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_min_gap_abs_error.png`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_von_mises.png`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_von_mises_abs_error.png`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_strain_norm.png`
- `results/abaqus_sphere_drop_implicit_penalty_sfc_aligned/abaqus_sphere_drop_full_strain_norm_abs_error.png`

## 当前结论

这组对标已经把接触律从 Abaqus/Explicit 硬接触调整为 Abaqus/Standard 隐式线性罚函数接触，并让 SFC 自动读取相同罚刚度。SFC 在 3 s 全程中接触触发时间与 Abaqus 相差约 `0.002 s`，位移 RMS 误差约 `2.91 mm`，且完整 SFC solve wall time 小于 Abaqus 原生求解 wall time。

仍需谨慎表述的是：质量比例阻尼是为了对齐隐式动力学耗散响应而加入的验证层参数；它支撑该工况下的结果对齐，但不等价于 Abaqus 内部算法的源级复现。

## HHT + surface quadrature 更新

为减少接触后的非物理反弹，新增并验证了两个 SFC 选项：

- `--integrator hht --hht-alpha -0.3`
- `--contact-integration surface --quadrature-order 3`

对应全程输出目录：

- `results/abaqus_sphere_drop_implicit_hht_surface_aligned`

关键结果如下：

| 指标 | 旧 Newmark node-sample | 新 HHT surface-quadrature |
| --- | ---: | ---: |
| `rms_z_cm_abs_error` | `0.0029100077 m` | `0.0004654438 m` |
| `max_z_cm_abs_error` | `0.0116215886 m` | `0.0042523097 m` |
| `full_z_cm_l2_error` | `0.1594142488` | `0.0254976527` |
| `rms_strain_norm_abs_error` | `0.0012440639` | `0.0006784661` |
| `rms_von_mises_abs_error` | `4.236631e4` | `1.974471e4` |
| `rms_p95_von_mises_abs_error` | 未输出 | `7.658351e3` |
| `rms_volume_mean_von_mises_abs_error` | 未输出 | `1.822384e3` |
| `rms_p95_strain_norm_abs_error` | 未输出 | `2.303409e-4` |
| `rms_volume_mean_strain_norm_abs_error` | 未输出 | `5.349043e-5` |
| `surface_sample_count` | `266` | `1584` |
| `sfc_solve_wall_seconds` | `135.108105 s` | `368.710001 s` |

这说明 HHT 算法耗散和面高斯积分显著改善了位移轨迹、应变和应力时间历程的一致性，但计算时间上升明显。与 Abaqus 原生求解 `345.819903 s` 相比，新 SFC 路径略慢，因此这个版本更适合作为“精度对齐路径”，不能作为最终加速路径。

接触力需要单独说明：当前 Abaqus VTK 文件没有原生接触反力历史。脚本只能用导出的几何穿透量和线性罚刚度重构一个诊断性 contact force。该重构在后期给出约 `88.6 N`，而 SFC 稳态接触力约 `5.99 N`，与球体重量一致。因此论文中不能把 VTK 反推力当成 Abaqus 原生接触反力。若要严格对齐接触力，应重新导出 Abaqus 原生接触反力、接触压力或约束反力历史。

应力/应变对比现在按单元场历史进行：SFC 使用 TET4 单元常应变中心值恢复线弹性应力，Abaqus 使用 ODB 导出的单元平均 `S` 和 `LE` cell scalar。脚本同时输出 `max`、`p95` 和体积加权平均三个口径，不再只依赖单个最大值；但 `LE` 与小应变工程应变仍不是源级完全相同的应变度量。

## Abaqus 理论文档对当前差异的解释

基于 Abaqus/Standard 隐式动力学、接触约束和线性 pressure-overclosure 文档，当前 SFC 与 Abaqus 的差异主要来自以下几点：

1. Abaqus/Standard 的隐式动力学不是单纯的 Newmark 平均加速度法。它允许使用带算法耗散的 Hilber-Hughes-Taylor 类积分，并与自动时间增量、非线性迭代和接触收敛控制耦合。SFC 的 HHT `alpha=-0.3` 是显式暴露的近似对齐参数，不等价于 Abaqus 内部完整求解流程。
2. 线性罚函数中的同一个罚刚度只定义 pressure-overclosure 斜率，不保证接触面积、接触积分点、从属/主面投影、接触释放和压力平滑与 Abaqus 完全一致。
3. Abaqus 输出的 `S` 和 `LE` 是单元/积分点场输出经 ODB 导出后的单元数据；SFC 当前使用 TET4 单元常应变恢复小应变应力。因此最大值对单元局部噪声和接触边界激活非常敏感。

相关文档入口：

- Abaqus contact constraint and pressure-overclosure behavior: `https://docs.software.vt.edu/abaqusv2024/English/SIMACAEITNRefMap/simaitn-c-contactconstraints.htm`
- Abaqus/Standard implicit dynamic analysis and HHT-style numerical damping: `https://abaqus-docs.mit.edu/2017/English/SIMACAEANLRefMap/simaanl-c-dynamic.htm`

因此，论文对 Abaqus 对齐的主指标应从单点最大应力/应变扩展为：

- `max`：保留为冲击峰值和局部接触奇异性的诊断指标；
- `p95`：降低单个接触边界单元对结论的支配；
- `volume_mean`：反映整体体积场响应。

验证脚本已经新增 `p95_von_mises`、`volume_mean_von_mises`、`p95_strain_norm` 和 `volume_mean_strain_norm` 的 SFC/Abaqus 时间曲线与误差曲线。后续论文主图建议优先使用位移曲线、`p95` 曲线和体积加权平均曲线，最大值曲线放在补充材料或作为峰值诊断。
