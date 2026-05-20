# Figure 9 工况：无摩擦、无阻尼、线性接触隐式动力学对比

本轮按照论文初稿 Figure 9 对应的 Fig. 6-inspired complex frictionless contact 工况，建立了一个简化但同几何语义的 Abaqus/Standard vs SFC 对比：

- 下部柔性体：C3D8 中等网格，`12 x 6 x 3`。
- 上部加载体：刚性矩形 driver，先法向闭合，再横向平移。
- 接触：无摩擦，线性 pressure-overclosure，罚刚度 `4500`。
- 阻尼：不引入 Rayleigh 阻尼，不引入接触阻尼。
- 动力学：Abaqus 使用 Standard implicit dynamic；SFC 使用当前动态 SDF / Lagrangian-SDF field-contact 后端和无阻尼 Newmark 隐式步进。
- 时间：`0.04 s`，固定步长 `0.001 s`，共 41 个输出帧。

## 运行命令

```powershell
python validation\run_fig9_linear_implicit_sfc_abaqus_comparison.py `
  --out-dir results\fig9_linear_implicit_sfc_abaqus `
  --total-time 0.04 `
  --dt 0.001 `
  --nx 12 --ny 6 --nz 3 `
  --abaqus-command C:\SIMULIA\Commands\abaqus.bat
```

## 输出文件

- `results/fig9_linear_implicit_sfc_abaqus/fig9_linear_implicit_summary.md`
- `results/fig9_linear_implicit_sfc_abaqus/fig9_linear_implicit_history.csv`
- `results/fig9_linear_implicit_sfc_abaqus/fig9_linear_implicit_metrics.csv`
- `results/fig9_linear_implicit_sfc_abaqus/fig9_linear_implicit_timing.csv`
- `results/fig9_linear_implicit_sfc_abaqus/fig9_linear_implicit_curves.png`
- `results/fig9_linear_implicit_sfc_abaqus/abaqus_vtk/frame_manifest.csv`

## 误差

| 指标 | L2 相对误差 | 最大绝对误差 |
|---|---:|---:|
| 顶面平均 z 位移 | 0.3074 | 1.641e-3 |
| 最大位移范数 | 0.3018 | 2.165e-2 |
| p95 von Mises | 0.4844 | 12.360 |
| p95 应变范数 | 0.4161 | 1.293e-2 |

位移趋势可以对上，但当前 SFC 曲线在接触建立后的响应更早、更强；应力和应变幅值偏高。这个结果支持“同一无摩擦线性接触设置下 SFC 可以完成该工况并给出同量级响应”，但还不能声称与 Abaqus/Standard 完全等价。

## 效率

| 求解器 | wall time |
|---|---:|
| Abaqus/Standard implicit dynamic | 30.19 s |
| SFC Lagrangian-SDF implicit | 0.99 s |

该中等网格工况中，SFC 完整对比路径比 Abaqus 快约 `30x`。注意这里的 SFC 仍是验证型 Python 后端；速度优势主要来自该工况的模型规模较小、SDF 接触路径轻量，以及 Abaqus 启动/输入/Standard 接触开销较高。

## 差异原因

当前误差主要来自接触离散和算法语义差异，而不是阻尼：

1. Abaqus 使用 Standard surface-to-surface 接触内部单元和接触平滑；SFC 使用动态 SDF 场上的积分点罚接触。
2. Abaqus 的线性 pressure-overclosure 与 SFC 的 `pressure_stiffness * penetration` 在量纲上对齐，但接触面积归一、内部接触单元权重和平滑策略不完全相同。
3. Figure 9 工况包含横向 shift；无摩擦时切向力应为零，但有限接触面边界、SDF normal 插值和 Abaqus 主从面处理会造成不同的局部压力分布。
4. Abaqus 隐式动态使用其内部接触收敛控制；SFC 当前验证脚本使用固定次数 Newmark/contact fixed-point 迭代，没有复刻 Abaqus 的私有收敛控制。

因此本文中这组结果更适合作为“简单无阻尼线性接触工程工况的效率/趋势对比”，而不是 Abaqus 源级等价证明。
