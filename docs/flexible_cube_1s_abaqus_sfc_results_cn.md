# 双柔性体按压 / 按压扫掠 1 s 对比结果

本轮验证包含两个 Abaqus/Standard 原生接触与 SFC Lagrangian-SDF 接触的独立求解工况。

## 工况设置

两个工况均为双柔性 C3D8 体：

- 下柔性块：`20 x 10 x 2 mm^3`；
- 上柔性块：`4.5 x 4.5 x 2 mm^3`；
- 无摩擦线性 pressure-overclosure 接触；
- `dt = 0.001 s`，总时长 `1.0 s`；
- `0.2 s` 内完成法向压入，随后保持；
- SFC 使用 surface-to-surface quadrature，`quadrature_order = 3`；
- 每 `20` 步导出一帧 VTK，因此每个求解器每个工况各 `51` 帧。

Case A 是纯法向按压保持。Case B 是法向按压后，对上柔性体顶面施加 `2.0 mm` 有限幅值 x 向位移，在 `0.2--1.0 s` 内完成横向扫掠。该位移幅值使上柔性体仍留在下柔性块范围内，用于验证 SDF 接触区随横向运动更新，而不是让小块滑出接触区域。

## 输出文件

Case A：

- `results/flexible_cube_press_1s/flexible_cube_sdf_abaqus_history.csv`
- `results/flexible_cube_press_1s/flexible_cube_sdf_abaqus_metrics.csv`
- `results/flexible_cube_press_1s/flexible_cube_sdf_abaqus_curves.png`
- `results/flexible_cube_press_1s/abaqus_vtk/frame.pvd`
- `results/flexible_cube_press_1s/sfc_vtk/sfc_frame.pvd`

Case B：

- `results/flexible_cube_sweep_1s/flexible_cube_sdf_abaqus_history.csv`
- `results/flexible_cube_sweep_1s/flexible_cube_sdf_abaqus_metrics.csv`
- `results/flexible_cube_sweep_1s/flexible_cube_sdf_abaqus_curves.png`
- `results/flexible_cube_sweep_1s/abaqus_vtk/frame.pvd`
- `results/flexible_cube_sweep_1s/sfc_vtk/sfc_frame.pvd`

## 误差结果

| 工况 | 指标 | L2 相对误差 | 最大绝对误差 |
|---|---|---:|---:|
| 按压 | lower top z displacement | 67.70% | 8.60e-3 |
| 按压 | mean surface gap | 27.98% | 5.68e-2 |
| 按压 | max contact pressure | 163.25% | 4.16e2 |
| 按压 | max displacement norm | 25.99% | 2.53e-1 |
| 按压 | p95 von Mises | 25.52% | 4.96e1 |
| 按压 | p95 strain norm | 28.02% | 4.65e-2 |
| 按压扫掠 | upper top x displacement | 12.84% | 2.10e-1 |
| 按压扫掠 | upper bottom x displacement | 18.88% | 4.06e-1 |
| 按压扫掠 | lower top z displacement | 57.17% | 8.60e-3 |
| 按压扫掠 | mean surface gap | 28.23% | 5.68e-2 |
| 按压扫掠 | max contact pressure | 170.10% | 4.16e2 |
| 按压扫掠 | max displacement norm | 14.96% | 3.28e-1 |
| 按压扫掠 | p95 von Mises | 25.01% | 4.96e1 |
| 按压扫掠 | p95 strain norm | 31.81% | 5.45e-2 |

按压工况中的 x 向位移在物理上接近零，`upper_bottom_mean_x_displacement` 的相对误差分母过小，不适合作为主指标；应看最大绝对误差。

## 终点状态

| 工况 | 求解器 | t | top z disp. | upper top x | upper bottom x | mean gap | max pressure | p95 von Mises | p95 strain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 按压 | Abaqus | 1.0 | -8.6066e-3 | 0 | 0 | -8.3919e-2 | 68.14 | 63.89 | 6.07e-2 |
| 按压 | SFC | 1.0 | -8.5390e-3 | 0 | 0 | -6.5754e-2 | 156.44 | 77.70 | 7.50e-2 |
| 按压扫掠 | Abaqus | 1.0 | -8.2188e-3 | 2.0 | 2.0887 | -7.3840e-2 | 61.48 | 77.85 | 8.00e-2 |
| 按压扫掠 | SFC | 1.0 | -7.8543e-3 | 2.0 | 2.1743 | -4.6682e-2 | 126.11 | 118.20 | 1.34e-1 |

## 效率和接触采样

| 工况 | Abaqus native solve | SFC solve | SFC / Abaqus |
|---|---:|---:|---:|
| 按压 | 约 142--147 s | 70.97 s | 约 2.0x 更快 |
| 按压扫掠 | 156.75 s | 71.84 s | 约 2.18x 更快 |

SFC backend breakdown：

| 工况 | 步数 | 平均 active samples | 最大 active samples | field update 总耗时 | field query 总耗时 |
|---|---:|---:|---:|---:|---:|
| 按压 | 1001 | 72.59 | 100 | 1.334 s | 0.125 s |
| 按压扫掠 | 1001 | 67.21 | 100 | 1.324 s | 0.128 s |

## 结论

这两个 1 s 工况已经能支撑以下结论：

- Abaqus 和 SFC 都完成了相同双柔性体无摩擦线性接触工况；
- SFC 与 Abaqus 的位移、应力、应变曲线同趋势，但长时动态误差仍然明显；
- 受控横向扫掠中，上柔性块没有离开下柔性块接触范围，SFC 的 active contact samples 随运动持续更新；
- SFC 完整求解 wall time 低于 Abaqus，且 SDF field update/query 只占总耗时的一小部分；
- 当前最大误差集中在 contact pressure 和 gap，这说明后续要继续对齐的是 Abaqus surface-to-surface pressure smoothing / area integration / contact enforcement 口径，而不是 SDF 查询链路本身。

因此，这组结果适合作为“长时双柔性体 SDF 接触后端可运行且有速度优势”的证据；若要作为最终投稿精度主图，还需要继续压低 pressure/gap 误差。
