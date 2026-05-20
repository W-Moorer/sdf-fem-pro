# 双柔性体 Lagrangian-SDF 与 Abaqus 原生接触对比

本算例用于检查 SFC Lagrangian-SDF 接触后端在“双柔性体接触”中的结果口径。Abaqus 只作为外部参考求解器；SFC 独立完成 FEM 组装、隐式动力学步进、SDF 接触查询和接触力装配。

## 通用修复

本轮修复没有引入针对单一曲线的特化参数，主要改动是通用求解语义对齐：

- 位移边界只强制位移，速度和加速度由同一 HHT/Newmark 递推产生，不再把解析速度/解析加速度直接塞进约束自由度。
- 双柔性体 C3D8 动力学路径显式使用 lumped mass，避免位移约束加速度通过 consistent mass 非对角项过强耦合到自由节点。
- Abaqus 输入请求 `CSTRESS, CDISP` 接触场输出，并导出 `CPRESS/COPEN` 系列字段。
- 接触 footprint 筛选改为基于参考底面节点集合，而不是当前最小 z 节点，避免非平面变形时漏掉真实接触区域。
- 若 broad-phase 没有选中接触采样点，则返回零接触响应；这只是候选为空时的安全处理，不改变接触精度。

## 当前非 quick 结果

命令：

```powershell
python validation\run_flexible_cube_sdf_abaqus_comparison.py `
  --out-dir results\flexible_cube_sdf_abaqus `
  --total-time 0.04 `
  --dt 0.001 `
  --closure 0.20 `
  --initial-gap 0.01 `
  --abaqus-command C:\SIMULIA\Commands\abaqus.bat
```

输出：

- `results/flexible_cube_sdf_abaqus/flexible_cube_sdf_abaqus_history.csv`
- `results/flexible_cube_sdf_abaqus/flexible_cube_sdf_abaqus_metrics.csv`
- `results/flexible_cube_sdf_abaqus/flexible_cube_sdf_abaqus_timing.csv`
- `results/flexible_cube_sdf_abaqus/flexible_cube_sdf_field_timing.csv`
- `results/flexible_cube_sdf_abaqus/flexible_cube_sdf_abaqus_curves.png`
- `results/flexible_cube_sdf_abaqus/abaqus_vtk/frame_manifest.csv`

指标：

| 指标 | L2 相对误差 | 最大绝对误差 |
|---|---:|---:|
| lower-top z displacement | 34.96% | 7.71e-5 |
| mean surface gap | 15.28% | 2.96e-3 |
| max contact pressure | 9.96% | 7.45 |
| max displacement norm | 7.01% | 2.14e-2 |
| p95 von Mises | 10.50% | 22.82 |
| p95 strain norm | 10.18% | 2.41e-2 |

时间：

| 求解器 | wall time |
|---|---:|
| Abaqus/Standard 原生接触 | 18.21 s |
| SFC Lagrangian-SDF 双柔性体 | 3.18 s |

该算例中 SFC 约为 Abaqus 的 `5.7x`。SDF field update 总耗时约 `0.50 s`，field query 总耗时约 `0.006 s`。

## 结论

修复后，SFC 与 Abaqus 在位移、gap、最大接触压力、应力和应变曲线上已经进入同量级对齐。当前仍不建议把 `upper_reaction_force` 作为主精度指标，因为它包含位移约束反力、惯性反力和质量离散差异；正式论文中应优先报告位移、gap、CPRESS/pressure、应力和应变曲线。
