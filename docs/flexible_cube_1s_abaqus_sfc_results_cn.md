# 双柔性体按压 / 按压扫掠 1 s 对比结果

本轮验证包含两个 Abaqus/Standard 原生接触与 SFC Lagrangian-SDF 接触的独立求解工况。最新结果已经使用：

- 当前构型四边形面面积权重；
- SFC 高斯点压力到从面节点的面积加权平滑；
- 原始高斯点最大压力作为诊断量，而不再作为主 CPRESS 对比口径。

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
| 按压 | lower top z displacement | 67.66% | 8.60e-3 |
| 按压 | mean surface gap | 27.98% | 5.68e-2 |
| 按压 | max nodal-smoothed contact pressure | 96.27% | 3.71e2 |
| 按压 | p95 nodal-smoothed contact pressure | 87.06% | 3.59e2 |
| 按压 | max displacement norm | 25.99% | 2.53e-1 |
| 按压 | p95 von Mises | 25.55% | 4.96e1 |
| 按压 | p95 strain norm | 27.99% | 4.65e-2 |
| 按压扫掠 | upper top x displacement | 12.84% | 2.10e-1 |
| 按压扫掠 | upper bottom x displacement | 18.88% | 4.06e-1 |
| 按压扫掠 | lower top z displacement | 57.15% | 8.60e-3 |
| 按压扫掠 | mean surface gap | 28.23% | 5.68e-2 |
| 按压扫掠 | max nodal-smoothed contact pressure | 102.92% | 3.71e2 |
| 按压扫掠 | p95 nodal-smoothed contact pressure | 99.07% | 3.59e2 |
| 按压扫掠 | max displacement norm | 14.95% | 3.28e-1 |
| 按压扫掠 | p95 von Mises | 24.97% | 4.96e1 |
| 按压扫掠 | p95 strain norm | 31.83% | 5.46e-2 |

按压工况中的 x 向位移在物理上接近零，`upper_bottom_mean_x_displacement` 的相对误差分母过小，不适合作为主指标；应看最大绝对误差。

## 终点状态

| 工况 | 求解器 | t | mean gap | max nodal pressure | p95 nodal pressure | raw sample max pressure | p95 von Mises | p95 strain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 按压 | Abaqus | 1.0 | -8.3919e-2 | 68.14 | 68.14 | - | 63.89 | 6.07e-2 |
| 按压 | SFC | 1.0 | -6.5595e-2 | 78.24 | 71.46 | 155.37 | 77.87 | 7.52e-2 |
| 按压扫掠 | Abaqus | 1.0 | -7.3840e-2 | 61.48 | 56.99 | - | 77.85 | 8.00e-2 |
| 按压扫掠 | SFC | 1.0 | -4.6621e-2 | 76.89 | 64.36 | 126.15 | 118.35 | 1.35e-1 |

## 效率和接触采样

| 工况 | Abaqus native solve | SFC solve | SFC / Abaqus |
|---|---:|---:|---:|
| 按压 | 148.44 s | 70.81 s | 2.10x 更快 |
| 按压扫掠 | 169.39 s | 67.38 s | 2.51x 更快 |

SFC backend breakdown：

| 工况 | 步数 | 平均 active samples | 最大 active samples | field update 总耗时 | field query 总耗时 |
|---|---:|---:|---:|---:|---:|
| 按压 | 1001 | 72.61 | 100 | 1.312 s | 0.134 s |
| 按压扫掠 | 1001 | 67.21 | 100 | 1.183 s | 0.112 s |

## 当前改进与剩余问题

相对上一版，主 pressure 指标不再使用高斯点 raw sample maximum，而是使用面积加权 nodal-smoothed pressure。纯按压的 pressure L2 相对误差从约 `163%` 降到约 `96%`，按压扫掠从约 `170%` 降到约 `103%`。这说明 pressure smoothing / 面积口径修正方向有效。

但 gap 和 pressure 仍是最大误差来源，说明 SFC 与 Abaqus 的 surface-to-surface contact enforcement 仍未完全同口径。下一步应继续对齐：

- Abaqus CPRESS 的从面/主面输出位置和 SFC 从面节点平滑位置；
- Abaqus surface smoothing 与边界 active-set 处理；
- 线性 pressure-overclosure 刚度在 Abaqus 接触单元中的有效面积归一化；
- 接触建立/释放阶段的 active-set 稳定与切线收敛控制。

这组结果可作为“长时双柔性体 SDF 接触可运行且有速度优势”的证据；若要作为最终投稿精度主图，还需要继续压低 pressure/gap 误差。
