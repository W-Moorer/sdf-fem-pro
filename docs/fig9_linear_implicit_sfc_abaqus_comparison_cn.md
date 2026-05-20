# Figure 9 工况：Abaqus/Standard 与 SFC Lagrangian-SDF 隐式动力学对比

本算例对应论文初稿 Figure 9 的无摩擦、无阻尼、线性接触工况。它用于检查 SFC 的 Lagrangian-SDF 接触后端在同一工程接触设定下，与 Abaqus/Standard 原生接触求解的位移、接触反力、应力和应变时程是否同量级、同趋势。

## 当前对齐设置

- 下部柔性体：C3D8 中等网格，`12 x 6 x 3`。
- 上部加载体：刚性矩形 driver，先法向闭合，再横向平移。
- 接触：无摩擦，线性 pressure-overclosure，接触刚度 `4500`。
- 阻尼：不引入 Rayleigh 阻尼，不引入接触阻尼。
- Abaqus：`*Dynamic, DIRECT, NOHAF`，`nlgeom=NO`，输出 `S, E, LE`。
- SFC：lower top 为 slave，moving rigid driver 为 master，driver SDF 法向朝向 lower top。
- SFC 接触积分：lower top 使用 C3D8 顶面四边形 Q4 Gauss 积分，不再把四边形面拆成两个三角形；同时只保留投影落在当前 driver 面内的积分点，避免开口面的 edge-feature 伪接触。
- 收敛：SFC 隐式 Newmark 迭代使用 field-contact 一致切线，并以残差、位移修正、接触力变化和 active set 稳定作为收敛判据。

## 运行命令

```powershell
python validation\run_fig9_linear_implicit_sfc_abaqus_comparison.py `
  --out-dir results\fig9_linear_implicit_sfc_abaqus_aligned `
  --total-time 0.04 `
  --dt 0.001 `
  --nx 12 --ny 6 --nz 3 `
  --abaqus-command C:\SIMULIA\Commands\abaqus.bat
```

## 输出文件

- `results/fig9_linear_implicit_sfc_abaqus_aligned/fig9_linear_implicit_summary.md`
- `results/fig9_linear_implicit_sfc_abaqus_aligned/fig9_linear_implicit_history.csv`
- `results/fig9_linear_implicit_sfc_abaqus_aligned/fig9_linear_implicit_metrics.csv`
- `results/fig9_linear_implicit_sfc_abaqus_aligned/fig9_linear_implicit_timing.csv`
- `results/fig9_linear_implicit_sfc_abaqus_aligned/fig9_linear_implicit_curves.png`
- `results/fig9_linear_implicit_sfc_abaqus_aligned/abaqus_driver_reaction_history.csv`
- `results/fig9_linear_implicit_sfc_abaqus_aligned/abaqus_frame_metrics.csv`
- `results/fig9_linear_implicit_sfc_abaqus_aligned/abaqus_vtk/frame_manifest.csv`

## 当前误差

| 指标 | L2 相对误差 | 最大绝对误差 |
|---|---:|---:|
| top mean z displacement | 0.2976 | 1.920e-3 |
| max displacement norm | 0.3435 | 2.678e-2 |
| driver/contact normal force | 0.4380 | 1.491e3 |
| p95 von Mises | 0.4272 | 1.069e1 |
| p95 small-strain norm | 0.5392 | 1.409e-2 |

与上一版相比，错误来源已经从“接触方向和开口边界语义不一致”收敛为“接触离散、Abaqus 内部 surface-to-surface 平滑、以及 SFC 当前 flat-punch footprint 处理差异”。位移曲线和接触反力曲线现在同量级、同趋势，但峰值时间和峰值幅值仍有差异，因此这组结果仍不能写成 Abaqus 源级等价。

## 效率

| 求解器 | wall time |
|---|---:|
| Abaqus/Standard implicit dynamic | 20.30 s |
| SFC Lagrangian-SDF implicit | 1.64 s |

SFC 在该中等网格短程工况中约为 Abaqus/Standard 的 `12.4x`。这个数字用于说明当前 SDF 接触后端在该 scoped benchmark 下的运行开销优势；论文中仍应明确 Abaqus 还包含商业求解器启动、输入处理和 Standard 接触开销。

## 结论

这一步修正完成了四个必要对齐：

1. SFC 接触方向改为 lower top slave、rigid driver master。
2. Abaqus 改为 `nlgeom=NO`，并导出小应变 `E` 的帧级指标。
3. Abaqus driver RP reaction 与 SFC 总法向接触力进入同一条 force-time 对比。
4. SFC 接触积分改为 C3D8 顶面 Q4 Gauss quadrature，不再使用 triangle split。
5. SFC 隐式迭代启用 field-contact 一致切线和残差控制收敛。

剩余主要误差来自当前工况仍使用刚性 open driver，而目标论文应转向柔性立方体对柔性块接触；Abaqus 内部 surface-to-surface 平滑策略与 SFC 的 SDF closest-feature 场仍有差异。下一步应建立柔性上立方体的同工况模型，用上体当前 FEM 表面构造 Lagrangian-SDF/master field，再比较柔性体-柔性体接触的位移、反力、应力和应变。
