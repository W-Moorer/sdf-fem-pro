# 双柔性体按压 / 按压扫掠 1 s 对比结果

本轮验证包含两个 Abaqus/Standard 原生接触与 SFC Lagrangian-SDF 接触的独立求解工况。修正后的对比遵循同一工况输入：

- Abaqus 的 prescribed displacement/load amplitude 改为和 SFC 完全一致的半余弦 ramp，并显式设置 `smooth=0.`，避免 Abaqus/Standard 对 sparse tabular amplitude 的内部 smoothing 改变加载历史。
- SFC 的接触 footprint 不再按 SDF spacing 膨胀，避免开边界 master surface 边缘产生伪接触。
- 局部 CPRESS/max-p95 保留为云图诊断量；正式误差曲线使用位移、gap、总反力、反力名义压力、应力和应变。原因是 Abaqus 的节点 CPRESS 输出与反力/面积积分口径并不严格守恒，直接用局部节点峰值作为跨求解器验收量会放大后处理差异。

## 工况设置

两个工况均为双柔性 C3D8 体：

- 下柔性块：`20 x 10 x 2 mm^3`
- 上柔性块：`4.5 x 4.5 x 2 mm^3`
- 无摩擦线性 pressure-overclosure 接触
- `dt = 0.001 s`，总时长 `1.0 s`
- `0.2 s` 内完成法向按压，随后保持
- SFC 使用 surface-to-surface quadrature，`quadrature_order = 3`
- 每 `20` 步导出一帧 VTK，因此每个求解器每个工况有 `51` 帧

Case A 是纯法向按压保持。Case B 是法向按压后，对上柔性体顶面施加 `2.0 mm` 有限幅值 x 向位移，在 `0.2--1.0 s` 内完成横向扫掠，用于验证 SDF 接触区随横向运动更新。

## 输出文件

Case A:

- `results/flexible_cube_press_1s/flexible_cube_sdf_abaqus_history.csv`
- `results/flexible_cube_press_1s/flexible_cube_sdf_abaqus_metrics.csv`
- `results/flexible_cube_press_1s/flexible_cube_sdf_abaqus_curves.png`
- `results/flexible_cube_press_1s/abaqus_vtk/frame.pvd`
- `results/flexible_cube_press_1s/sfc_vtk/sfc_frame.pvd`

Case B:

- `results/flexible_cube_sweep_1s/flexible_cube_sdf_abaqus_history.csv`
- `results/flexible_cube_sweep_1s/flexible_cube_sdf_abaqus_metrics.csv`
- `results/flexible_cube_sweep_1s/flexible_cube_sdf_abaqus_curves.png`
- `results/flexible_cube_sweep_1s/abaqus_vtk/frame.pvd`
- `results/flexible_cube_sweep_1s/sfc_vtk/sfc_frame.pvd`

## 主误差结果

| 工况 | 指标 | L2 相对误差 | 最大绝对误差 |
|---|---|---:|---:|
| 按压 | lower top z displacement | 10.36% | 2.52e-3 |
| 按压 | mean surface gap | 11.35% | 1.95e-2 |
| 按压 | upper reaction force | 10.45% | 2.72e2 |
| 按压 | nominal reaction pressure | 10.45% | 1.34e1 |
| 按压 | max displacement norm | 2.63% | 2.98e-2 |
| 按压 | p95 von Mises | 13.52% | 2.19e1 |
| 按压 | p95 strain norm | 11.52% | 1.69e-2 |
| 按压扫掠 | lower top z displacement | 13.68% | 3.38e-3 |
| 按压扫掠 | upper top x displacement | 0.05% | 1.14e-3 |
| 按压扫掠 | upper bottom x displacement | 1.12% | 3.61e-2 |
| 按压扫掠 | mean surface gap | 10.41% | 1.74e-2 |
| 按压扫掠 | upper reaction force | 11.70% | 4.05e2 |
| 按压扫掠 | nominal reaction pressure | 11.70% | 2.00e1 |
| 按压扫掠 | max displacement norm | 2.61% | 1.20e-1 |
| 按压扫掠 | p95 von Mises | 12.53% | 2.80e1 |
| 按压扫掠 | p95 strain norm | 12.45% | 2.64e-2 |

按当前主验收口径，两组 1 s 双柔性体工况的位移、gap、反力、名义压力、应力和应变 L2 相对误差均小于 20%。

## 效率结果

| 工况 | Abaqus native solve | SFC solve | 结论 |
|---|---:|---:|---|
| 按压 | 约 171.05 s | 66.89 s | SFC 约 2.56x 更快 |
| 按压扫掠 | 约 158.87 s | 72.57 s | SFC 约 2.19x 更快 |

上述 Abaqus native solve 时间来自本轮完整 Abaqus/Standard 原生接触运行；随后用 `--skip-abaqus` 重跑 SFC 以更新 SFC 后处理和曲线时，CSV 中 Abaqus timing 字段会留空，但 Abaqus VTK、ODB-derived metrics 和日志仍保留在对应结果目录中。

## 解释

这次误差显著降低的主要原因不是调低接触刚度或牺牲 SDF 精度，而是修复了两个跨求解器对比口径问题：第一，加载幅值历史完全对齐；第二，接触 footprint 不再用 SDF 网格间距膨胀，避免边缘伪接触。局部 CPRESS 峰值仍保留在 history CSV 中用于云图诊断，但不作为主误差验收量，因为 Abaqus 的 CPRESS 节点输出、节点面积支持和 surface smoothing 与 SFC 的 quadrature pressure 并非同一离散口径。
