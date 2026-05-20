# Figure 9 工况：Abaqus/Standard 与 SFC Lagrangian-SDF 隐式动力学对比

本算例对应论文初稿 Figure 9 的无摩擦、无阻尼、线性接触工况。它用于检查 SFC 的 Lagrangian-SDF 接触后端在同一工程接触设定下，与 Abaqus/Standard 原生接触求解的位移、接触反力、应力和应变时程是否同量级、同趋势。

## 当前对齐设置

- 下部柔性体：C3D8 中等网格，`12 x 6 x 3`。
- 上部加载体：刚性矩形 driver，先法向闭合，再横向平移。
- 接触：无摩擦，线性 pressure-overclosure，接触刚度 `4500`。
- 阻尼：不引入 Rayleigh 阻尼，不引入接触阻尼。
- Abaqus：`*Dynamic, DIRECT, NOHAF`，`nlgeom=NO`，输出 `S, E, LE`。
- SFC：lower top 为 slave，moving rigid driver 为 master，driver SDF 法向朝向 lower top。
- SFC 接触积分：只保留投影落在当前 driver 面内的 lower-top 积分点，避免开口刚性面的 edge-feature 伪接触。
- 收敛：SFC 隐式 Newmark 迭代使用 field-contact 一致切线。

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
| top mean z displacement | 0.3073 | 1.767e-3 |
| max displacement norm | 0.2981 | 2.238e-2 |
| driver/contact normal force | 0.4582 | 1.567e3 |
| p95 von Mises | 0.4676 | 1.185e1 |
| p95 small-strain norm | 0.5819 | 1.531e-2 |

与上一版相比，错误来源已经从“接触方向和开口边界语义不一致”收敛为“接触离散、Abaqus 内部 surface-to-surface 平滑、以及 SFC 当前 flat-punch footprint 处理差异”。位移曲线和接触反力曲线现在同量级、同趋势，但峰值时间和峰值幅值仍有差异，因此这组结果仍不能写成 Abaqus 源级等价。

## 效率

| 求解器 | wall time |
|---|---:|
| Abaqus/Standard implicit dynamic | 20.09 s |
| SFC Lagrangian-SDF implicit | 1.70 s |

SFC 在该中等网格短程工况中约为 Abaqus/Standard 的 `11.8x`。这个数字用于说明当前 SDF 接触后端在该 scoped benchmark 下的运行开销优势；论文中仍应明确 Abaqus 还包含商业求解器启动、输入处理和 Standard 接触开销。

## 结论

这一步修正完成了四个必要对齐：

1. SFC 接触方向改为 lower top slave、rigid driver master。
2. Abaqus 改为 `nlgeom=NO`，并导出小应变 `E` 的帧级指标。
3. Abaqus driver RP reaction 与 SFC 总法向接触力进入同一条 force-time 对比。
4. SFC 隐式迭代启用 field-contact 一致切线。

剩余主要误差来自开口刚性 driver 的边界接触语义和 Abaqus 内部 surface-to-surface 接触平滑策略。若要继续降低误差，应优先把 SFC 的 rigid driver contact 从 open-surface SDF 进一步改成显式 bounded parametric plane contact，或构造闭合刚性 punch 几何，使 SDF closest-feature 语义与 Abaqus master surface 更接近。
