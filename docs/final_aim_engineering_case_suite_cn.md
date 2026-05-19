# Final-Aim 工程算例矩阵与图像规范

日期：2026-05-19

本文档把最终论文的数值算例从旧的 dynamic-field/replay 证据中重新整理出来。目标不是再增加一个零散验证脚本，而是把论文落点明确为：

```text
SFC final-aim Lagrangian/material SDF contact solve
vs
CalculiX native contact solve
```

也就是说，最终结果必须来自两个独立求解流程：

```text
SFC:
  internal FEM assembly
  + MaterialSDF
  + LagrangianSDFContactOracle
  + node-to-surface / surface-to-surface contact

CalculiX:
  generated native contact input
  + CalculiX native contact solve
  + parsed displacement/stress/strain/contact outputs
```

不再把 brute-force current-surface projection 作为论文对比路径。它最多只能作为开发期内部回归检查，不能出现在论文的效率或结果对比主线中。

## 1. 最终算例层次

最终算例应分为四组。静力学用于前期力学正确性和算力章节铺垫，论文主落点放在 3 s 动力学场景。

| 算例 | 类型 | 目的 | SFC 路径 | CalculiX 路径 | 主图 |
|---|---|---|---|---|---|
| Case 1 | 静力学变形 | 小球压/落到悬臂梁，展示柔性梁应力应变和接触区 | final-aim oracle | native contact | 3D 云图 + load/displacement 曲线 |
| Case 2a | 动力学 | 高分辨率柔性立方体自由落体到刚性平面并最终静止 | final-aim oracle | native contact | 时间历程曲线 + 逐帧 VTK |
| Case 2b | 动力学 | 高分辨率柔性小球自由落体到刚性平面并最终静止 | final-aim oracle | native contact | 时间历程曲线 + 逐帧 VTK |
| Case 3 | 动力学 | 柔性小球落到地面上柔性立方体顶部，双柔性体接触 | final-aim oracle | native contact | 时间历程曲线 + 逐帧误差云图 |
| Case 4 | 动力学/准动力工程场景 | 当前稿件 Figure 9 / 参考论文 Fig. 6 对应的组合加载工况，高分辨率重绘 | final-aim oracle | native contact | Fig. 6 风格 3D 场云图 |

所有动力学场景统一为：

```text
total_time = 3.0 s
dt = 0.001 s
gravity = 9.81
initial vertical velocity = 0
```

如果场景使用压力或载荷驱动，载荷只能作为外力或压力历史进入，不允许用强制位移伪造自由落体接触过程。对于 Case 4，如果参考文献包含切向位移加载而本文仍保持无摩擦接触，则必须明确写成“Fig. 6-inspired frictionless geometry/loading audit”，不能声称 stick-slip 或摩擦验证。

## 2. 网格密度要求

网格不能再使用旧 quick 级别。目标细度应接近参考论文 Fig. 3 的视觉密度：连续三维实体、足够密的表面三角/四边形边线、不是离散点云。

建议正式网格目标：

| 几何 | 建议单元 | 正式分辨率目标 | quick/smoke 分辨率 |
|---|---|---:|---:|
| 悬臂梁 | C3D8 或 TET4 | 长向 >= 48，厚向 >= 6 | 长向 12，厚向 2 |
| 柔性立方体 | C3D8 | 每边 >= 12，接触面 >= 144 cells | 每边 3-4 |
| 柔性小球 | TET4 或球体表面三角 + 体四面体 | 表面平均边长约 0.5 mm 等效细度 | 低阶规则化球 |
| Fig. 6 / Figure 9 工况下体 | C3D8/TET4 | 参考论文量级，接触面边长约 0.5 mm | 当前 24x12x4 可作为 smoke |

正式论文图片必须显示连续三维渲染面，不允许使用离散点云替代云图。

## 3. 输出指标

每个 SFC/CalculiX 对比场景至少输出：

```text
displacement-time 或 load-displacement 曲线
reaction/contact force 曲线
minimum gap / maximum penetration 曲线
contact energy 或接触功曲线
active contact samples / contact area 曲线
von Mises stress cloud
engineering strain norm cloud
contact pressure cloud
gap / penetration cloud
absolute error cloud: |SFC - CalculiX|
SFC solve wall time
CalculiX solve wall time
SFC core solve time
SFC contact oracle time
diagnostics/postprocess time
```

曲线中应直接画 SFC 和 CalculiX 两条实际曲线，并在图例括号中写误差，例如：

```text
SFC Lagrangian SDF (L2 err. = 2.7e-3)
CalculiX native contact
```

不要使用误差柱状图。误差可以进入：

```text
图例
曲线旁注
表格
error cloud
```

## 4. 图像风格要求

参考 `assets/uzun_korsunsky_2026_fuzzycontact_sigmoidal_traction_law.pdf`：

- Fig. 3：高密度三维网格展示，实体表面连续，网格边线细，局部几何放大，不是点云。
- Fig. 6：多面板三维场云图，白色线框显示加载/接触体，面场颜色连续，色标紧贴每个 panel 下方。

本文最终图应采用：

```text
Times / Times New Roman 字体
白底
连续三维实体面渲染
细网格边线
统一 SFC/CalculiX 色标
error cloud 单独色标
图例不遮挡曲线
子图角标不遮挡曲线或色标
不用柱状图
不用二维点云替代三维应力/应变云图
```

Case 4 的主图建议使用 4x3 风格：

```text
row 1: SFC ux / uy / uz
row 2: SFC sigma_xx / sigma_yy / sigma_zz
row 3: CalculiX ux / uy / uz 或 SFC pressure/gap/active
row 4: absolute error ux / stress / pressure
```

如果版面过密，可拆成两张图：

```text
Figure A: displacement components
Figure B: stress/strain/contact fields
```

## 5. 当前项目覆盖情况

当前仓库已有一些可复用组件，但还不能直接声称已经完成你设定的最终算例矩阵。

| 需求 | 当前状态 | 是否可直接作为最终论文证据 |
|---|---|---|
| rigid plane block/sphere drop | `run_calculix_drop_impact_comparison.py` 已有 TET4 rigid-plane drop 比较 | 不能直接用；SFC 路径还不是 final-aim oracle，云图/VTK 也需补齐 |
| 3 s, dt=0.001 动力学 | 部分脚本可设置时长和步长 | 需要重新跑正式非 quick 数据 |
| block-on-block large-area dynamic | `run_large_area_dynamic_surface_contact.py` 已有 pressure-driven dynamic-field 版本 | 不能直接用；需要迁移到 Lagrangian oracle |
| Fig. 6-inspired / current Figure 9 | `run_fig6_inspired_frictionless_contact.py` 已有 dynamic-field 静态/分阶段图 | 不能直接用；需要提高分辨率、改 final-aim oracle、补 CalculiX native solve |
| cantilever + sphere static | 当前没有完整 final-aim + CalculiX 对应 runner | 需要新增 |
| flexible sphere on flexible cube | 当前没有完整 final-aim + CalculiX 对应 runner | 需要新增 |
| no bar charts | 旧稿中仍有 backend ablation 柱状图类图片 | 需要替换为曲线/表格 |
| brute-force projection baseline | 已从 final-aim 论文定位中移除 | 只允许内部回归，不进主对比 |

## 6. 论文实验章节建议结构

最终实验章节应改成：

```text
7.1 Oracle consistency and Jacobian verification
7.2 Static sphere/cantilever contact
7.3 Dynamic drop onto rigid plane: cube and sphere
7.4 Dynamic sphere-on-flexible-cube contact
7.5 Fig. 6 / Figure 9 complex engineering contact
7.6 Efficiency and backend timing
```

其中 7.2 是前期验证；7.3-7.5 是论文主落点；7.6 只用曲线和表格，不用柱状图。

## 7. Claim Gate

只有在对应数据生成后，才能允许如下 claim：

| Claim | Gate |
|---|---|
| 任意有限变形柔性体 contact oracle | 刚体运动、大转角弯曲、柔性体动态场景都使用 final-aim oracle |
| 结果准确 | SFC vs CalculiX displacement/stress/strain/contact curves 和 cloud error 低于阈值 |
| 动力学有效 | 3 s, dt=0.001 全时间历程完成，接触后稳定段存在 |
| 效率更高 | 完整 SFC solve wall time < 完整 CalculiX wall time，且 core/contact/diagnostics timing 分离 |
| 高分辨率工程可视化 | 输出连续三维云图和 ParaView VTK/PVD 帧序列 |

不能用 kernel timing 单独冒充完整求解加速，也不能用 projection 内部参考替代 CalculiX 对比。

## 8. 下一步实现顺序

为了避免四个正式算例同时扩散，建议实现顺序为：

1. 新增统一 `final_aim_engineering_case_suite` case registry。
2. 先完成 Case 2a/2b rigid-plane drop，因为已有 CalculiX drop 脚本可迁移。
3. 再完成 Case 3 flexible sphere on flexible cube。
4. 再完成 Case 1 static cantilever。
5. 最后迁移 Case 4 Figure 9/Fig. 6-inspired 工况，并重绘高分辨率 3D 云图。

每完成一个正式 case，都必须同时生成：

```text
CSV metrics
PNG/PDF curves
VTK/PVD frames
SFC/CalculiX cloud figures
Markdown handoff
```

