# 论文图风格对齐说明

参考对象：`assets/uzun_korsunsky_2026_fuzzycontact_sigmoidal_traction_law.pdf`。

## 提取到的图形风格

- 白底、Times 风格英文字体。
- 数值算例图以连续三维表面云图为主，不使用二维离散点云作为主要证据图。
- 应力、应变、位移和接触压力图使用清晰色条，并尽量保持同类对比图的色标一致。
- 子图使用小写角标标注，角标放在图外侧或空白区域，不遮挡主几何和曲线。
- 曲线/柱状对比图的图例应放在绘图区外或图内空白区，不能覆盖曲线、阈值线或柱状结果。
- 多面板图应包含几何、载荷/接触区域、云图和对比误差，使工程力学读者可以直接审查物理场。

## 本轮对齐的图件

- `results/external_visual/figures/external_visual_fields.png`
  - 三维连续边界表面云图，包含 SFC、reference、absolute error。
- `results/external_visual/figures/dynamic_sdf_field_visualization.png`
  - SDF field 可视化，包含 current FEM surface、phi slice、validity slice、contact pressure 和 scalar-gradient normals。
- `results/external_visual/figures/surface_to_surface_quadrature_contact.png`
  - surface-to-surface quadrature contact 图，包含 slave surface、master SDF plane、七点三角形积分点、gap、contact pressure 和 normals。
- `results/native_contact_backend_ablation_quick/figures/backend_ablation_overview.png`
  - backend ablation 图，覆盖 node-to-surface、reference surface-to-surface、vectorized surface-to-surface、vectorized + batch projection build。
- `results/true_sdf_final/figures/field_speedup_vs_query_count.png`
  - 图例移到绘图区外，避免遮挡 speedup 曲线和 crossover 标记。

## 论文嵌入状态

`paper/main.pdf` 已重新编译，新增并嵌入：

- backend ablation overview；
- dynamic SDF field visualization；
- surface-to-surface quadrature contact visualization。

这些图用于视觉和工程解释，不替代 field accuracy、Jacobian finite difference 和 crossover cost model 的核心方法证据。
