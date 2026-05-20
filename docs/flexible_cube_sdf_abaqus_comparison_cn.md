# 双柔性体 SDF 接触对比算例

本算例用于替代刚性 open driver 工况，检查 SFC Lagrangian-SDF 在“柔性体-柔性体”接触中的结果和效率。

## 工况设置

- 下体：柔性 C3D8 块，底面固定。
- 上体：柔性 C3D8 立方体，顶面施加法向位移。
- 接触：下体顶面作为 slave，上体底面作为 SDF master。
- SFC：两侧均由内部 FEM 求解，接触采用 Q4 面 Gauss 积分和 Lagrangian-SDF master 查询。
- Abaqus：同样的两个 C3D8 柔性体，frictionless linear pressure-overclosure。
- 默认初始间隙：`-0.02`，即小初始压入，用于确保短程动力学内真实进入接触。

## 运行命令

```powershell
python validation\run_flexible_cube_sdf_abaqus_comparison.py `
  --out-dir results\flexible_cube_sdf_abaqus_overlap `
  --total-time 0.04 `
  --dt 0.001 `
  --closure 0.20 `
  --initial-gap -0.02 `
  --abaqus-command C:\SIMULIA\Commands\abaqus.bat
```

## 输出文件

- `results/flexible_cube_sdf_abaqus_overlap/flexible_cube_sdf_abaqus_summary.md`
- `results/flexible_cube_sdf_abaqus_overlap/flexible_cube_sdf_abaqus_history.csv`
- `results/flexible_cube_sdf_abaqus_overlap/flexible_cube_sdf_abaqus_metrics.csv`
- `results/flexible_cube_sdf_abaqus_overlap/flexible_cube_sdf_abaqus_timing.csv`
- `results/flexible_cube_sdf_abaqus_overlap/flexible_cube_sdf_abaqus_curves.png`
- `results/flexible_cube_sdf_abaqus_overlap/abaqus_vtk/frame_manifest.csv`

## 当前结果

| 指标 | L2 相对误差 | 最大绝对误差 |
|---|---:|---:|
| lower top mean z displacement | 52.36% | 1.083e-3 |
| max displacement norm | 7.01% | 2.140e-2 |
| upper-top reaction force | 96.33% | 8.933e4 |
| p95 von Mises | 36.85% | 9.168e1 |
| p95 strain norm | 34.31% | 8.939e-2 |

效率：

| 求解器 | wall time |
|---|---:|
| Abaqus/Standard | 18.04 s |
| SFC Lagrangian-SDF two-flexible | 2.91 s |

SFC 约为 Abaqus 的 `6.2x`。

## 解释

这个算例已经满足“双柔性体接触链路存在”的要求：SFC 中 `active_samples` 非零，`contact_integral_force` 非零，并且 Abaqus 与 SFC 都由各自后端独立建模和求解。

但它还不是最终精度对齐算例。当前最大差异来自两个方面：

1. 默认使用小初始压入来保证短程接触，Abaqus 对初始 overclosure 的处理和 SFC 线性罚接触的初始响应并不完全一致。
2. 上顶面反力包含上体惯性、内力和接触传递，不等同于纯接触积分力；后续应导出 Abaqus contact pressure 或 contact force field，再与 SFC `contact_integral_force` 对齐。

下一步应把该 runner 扩展为两组：

1. 准静力双柔性压入：更长加载时间或静力步，消除惯性反力峰值，用于接触刚度单位标定。
2. 动力学双柔性接触：保留 0.001 s 步长和动态响应，用于验证时间历程和效率。
