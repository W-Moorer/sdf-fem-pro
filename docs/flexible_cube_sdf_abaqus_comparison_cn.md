# 双柔性体 Lagrangian-SDF 与 Abaqus 原生接触对比

本算例用于检查 SFC Lagrangian-SDF 接触后端在“双柔性体接触”中的结果口径。Abaqus 只作为外部参考求解器；SFC 独立完成 FEM 组装、隐式动力学步进、SDF 接触查询和接触力装配。

## 与 assets PDF 工况的对应关系

`assets/uzun_korsunsky_2026_fuzzycontact_sigmoidal_traction_law.pdf` 中的 Benchmark 2 / Fig. 5--6 是 combined normal and tangential loading 工况。原文设置为：

- stationary lower body：`20 x 10 x 2 mm^3`；
- moving upper body：`4.5 x 4.5 x 1 mm^3`；
- 材料：isotropic linear elastic，`E = 200 GPa`，`nu = 0.3`；
- 第一步：对上部体施加 `-2000 N` z 向分布压缩载荷，建立法向接触；
- 第二步：达到平衡后施加 `+2000 N` x 向切向载荷；
- 原文包含 friction / stick-slip，摩擦系数为 `0.5`；
- Fig. 6 的前两行展示法向压缩后的 `ux, uy, uz, sigma_xx, sigma_yy, sigma_zz`，后两行展示切向运动后的同类场。

当前 `run_flexible_cube_sdf_abaqus_comparison.py` 的默认设置对应这个 PDF 工况中的 **法向压缩接触阶段**，用于 SFC 与 Abaqus 原生无摩擦法向接触的曲线对齐。脚本现在也支持可选的 `--tangential-force` 阶段：法向压紧后释放上柔性体顶面 x 向位移约束，并施加分布式 x 向外力。这个阶段仍然是 **无摩擦线性接触**，不是完整 Fig. 6 frictional stick-slip 复现，也不是滑行摩擦验证。完整的 Fig. 6 风格“法向压缩后横向扫掠”应写成 `Fig. 6-inspired frictionless normal-to-tangential loading`：可以保留几何和加载路径，但必须明确 contact law remains frictionless。

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

可选切向载荷阶段示例：

```powershell
python validation\run_flexible_cube_sdf_abaqus_comparison.py `
  --out-dir results\flexible_cube_sdf_abaqus_tangential `
  --total-time 0.04 `
  --dt 0.001 `
  --closure 0.20 `
  --initial-gap 0.01 `
  --tangential-force 2000 `
  --tangential-start-time 0.02 `
  --tangential-ramp-time 0.02 `
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
