# 双柔性立方体 Q4 master 接触对齐记录

本轮目标是把 SFC 曲线相对 Abaqus/Standard 的误差进一步压低，同时保持通用算法，不做针对单个算例的硬编码修正。

## 通用修正

1. 新增 `quadrilateral_master_surface_penalty_response(...)`。
   - slave 侧仍使用 C3D8 顶面 Q4 Gauss 面积分。
   - master 侧不再把 HEX8 bottom face 固定拆成两个三角形，而是在四节点 Q4 面片上做最近点 Newton 校正。
   - master Jacobian 使用四节点 Q4 shape function 分配，因此是通用的 surface-to-surface quadrilateral master payload，不是后处理滤波。

2. `quadrilateral_surface_quadrature_cache` 的一维 Gauss 积分阶数不再限制为 1/2/3 阶。
   - 现在通过 `numpy.polynomial.legendre.leggauss(order)` 支持任意正阶。
   - 这只提高面积积分能力，不改变接触律、材料或 SDF 几何精度。

3. prescribed displacement 的速度和加速度同步更新。
   - SFC 之前只强制约束位移，约束点速度/加速度由 Newmark 递推间接产生。
   - 现在使用同一半余弦 ramp 给出约束位移、速度和加速度，避免反力诊断中混入约束点惯性误差。

4. 求解和诊断分离。
   - 应力/应变恢复、VTK 输出和曲线 history 只在输出帧计算。
   - 核心求解仍然使用 `dt=0.001 s` 的完整时间步，不降低接触计算精度。

## 1 s 细网格 Q4 master 正式结果

命令配置：

```text
lower mesh: 16 x 8 x 4 C3D8
upper mesh: 8 x 8 x 3 C3D8
total_time = 1.0 s
dt = 0.001 s
closure_time = 0.2 s
quadrature_order = 3
master_surface_mode = quadrilateral_oracle
```

输出目录：

```text
results/flexible_cube_press_1s_dense_q4
```

主要误差：

| 指标 | L2 相对误差 |
|---|---:|
| lower top z displacement | 9.09% |
| upper reaction force | 7.00% |
| nominal reaction pressure | 7.00% |
| p95 displacement norm | ~0.00% |
| p95 von Mises | 8.23% |
| p95 strain norm | 7.78% |
| mean surface gap | 12.57% |

结论：位移、反力、应力、应变曲线已经进入 10% 以内；`mean surface gap` 仍为 12.57%，不能声称所有接触几何指标都小于 10%。

计时：

| 求解器 | solve wall time |
|---|---:|
| Abaqus/Standard native contact | 170.79 s |
| SFC Q4 master contact | 324.76 s |

结论：该细网格路径是当前的精度对齐路径，但不是加速路径。

## 0.2 s 中间网格筛选结果

中间网格：

```text
lower mesh: 14 x 7 x 4 C3D8
upper mesh: 7 x 7 x 3 C3D8
total_time = 0.2 s
dt = 0.001 s
master_surface_mode = quadrilateral_oracle
```

该短程 ramp 段结果全部低于 6%，且 SFC 快于 Abaqus：

| 指标 | L2 相对误差 |
|---|---:|
| lower top z displacement | 2.51% |
| mean surface gap | 3.32% |
| upper reaction force | 5.27% |
| p95 von Mises | 3.01% |
| p95 strain norm | 3.04% |

| 求解器 | solve wall time |
|---|---:|
| Abaqus/Standard native contact | 42.25 s |
| SFC Q4 master contact | 35.64 s |

但是同一中间网格扩展到 1 s 后，后期应力/应变振动误差重新超过 10%，因此不能作为最终 1 s 论文结果。

## 当前判断

本轮已经证明，误差主要不是 SDF 查询本身，而是 surface-to-surface 接触离散口径和网格分辨率问题。Q4 master payload 是必要修正；细网格可以把位移、反力、应力和应变压入 10% 以内。

剩余缺口：

- `mean surface gap` 仍为 12.57%，需要进一步改成同口径的 Abaqus COPEN/CDISP 或接触积分点 opening 对比。
- 细网格 Q4 master 路径慢于 Abaqus，需要 C++ fused Q4 contact/tangent 或 matrix-free tangent solve 才能成为加速路径。
- 中间网格短程已经兼具精度和速度，但 1 s 长程应力/应变仍不稳定，不能作为最终长程工程 claim。
