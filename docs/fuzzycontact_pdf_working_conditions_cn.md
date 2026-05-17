# FuzzyContact PDF 工况解析与 SFC 复现可行性

资料来源：`assets/1-s2.0-S0301679X25009636-main.pdf`  
论文：F. Uzun and A. M. Korsunsky, "FuzzyContact: A generic contact mechanics formulation via a modified penalty method governed by sigmoidal traction law", Tribology International 215 (2026) 111468.

## 结论

PDF 中确实给出了足够多的几何、材料、载荷、网格尺度、约束和参考结果，可以作为 SFC 自主建模的工程接触算例来源。最适合当前 FEM-induced dynamic narrow-band SDF contact 路线的算例是：

1. Benchmark 1：sphere-block Hertzian patch test。
2. Benchmark 2：normal loading stage only。
3. Benchmark 3：V-indenter indentation。
4. Benchmark 4：cylinder-platen compression。
5. Benchmark 5：connected assembly mesh convergence，作为后续复杂几何/网格收敛场景。

但是不能把这些算例表述成“完整复现 FuzzyContact 方法”等价，因为 FuzzyContact 的接触律是 sigmoidal traction law，并且 Benchmark 2 的后半段包含摩擦 stick-slip；当前项目约束明确不加入 friction/self-contact/nonlinear FEM/GPU/barrier/POD/neural/Abaqus dependency。因此本文应把它们定位为：

> 使用公开论文工况自主建立 SFC 模型，验证 dynamic SDF field contact 在相同或相近工程几何、材料、载荷、约束下的接触响应、场变量和效率表现。

## 通用参数

论文第 3 节给出了五个 benchmark 的共同接触参数族：

| proximity range `a` | exponent `m` | transition parameter `b` |
| --- | ---: | --- |
| 0.5 mm | 110 | 0.05a |
| 0.25 mm | 55 | 0.1a |
| 0.125 mm | 27.5 | 0.2a |
| 0.0625 mm | 13.75 | 0.4a |

其他通用信息：

- damping coefficient：0.5 kg/s。
- Benchmark 2 的 friction coefficient：0.5。
- 文中说明 displacement、stress、strain 等结果以 VTU 形式发布在数据仓库中。

SFC 不能直接照搬 sigmoidal frictional traction law。对本文而言，这些参数主要用于理解 FuzzyContact 参考结果，不应进入核心 SDF-FEM 方法。

## Benchmark 1：Hertzian Sphere-Block Patch Test

用途：Hertz 接触、无摩擦法向接触、球-块非共形接触。

| 项 | PDF 工况 |
| --- | --- |
| 几何 | sphere radius 1 mm；rectangular block 4 x 4 x 2 mm^3 |
| 材料 | aluminium；density 2700 kg/m^3；Young's modulus 70 GPa；Poisson ratio 0.33 |
| 载荷 | sphere 上施加 -1000 N z-direction compressive load |
| 网格 | tetrahedral elements；average element size 0.25 mm |
| FuzzyContact penalty scale | 1000 N |
| 约束 | contact force quantified 后，sphere top nodes 和 lower block bottom surface 在所有平移方向 fully constrained |
| 参考量 | Hertz contact radius 0.2673 mm；total indentation 71.45 um；single-body indentation 35.7 um |
| FuzzyContact 结果 | sphere indentation 33.03 um；lower block max downward displacement 13.78 um；relative deviation 7.48% |

SFC 复现建议：

- 优先作为主工程 benchmark。
- 使用当前 dynamic narrow-band SDF field 从 block current surface 或 sphere current surface 构建场。
- 对比指标：indentation、contact radius、gap/penetration map、contact pressure surrogate、von Mises stress、displacement field、field update/query timing。
- 可以直接对比 Hertz analytical reference 和 FuzzyContact 表 1 的 single-body indentation。

## Benchmark 2：Combined Normal and Tangential Loading

用途：非匹配网格、法向接触、摩擦 stick-slip。

| 项 | PDF 工况 |
| --- | --- |
| 几何 | stationary body 20 x 10 x 2 mm^3；moving body 4.5 x 4.5 x 1 mm^3 |
| 材料 | isotropic linear elastic；Young's modulus 200 GPa；Poisson ratio 0.3 |
| 网格 | tetrahedral elements；average element size 0.5 mm；intentionally non-aligned nodes |
| 法向载荷 | moving upper body 施加 -2000 N z-direction distributed compressive load |
| 切向载荷 | equilibrium 后施加 +2000 N x-direction tangential load |
| 约束 | top surface of upper body 和 bottom surface of stationary lower body 在 FEM stage fully constrained |
| normal | N(0, 0, 1) |
| FuzzyContact penalty scale | 25 N |
| friction | coefficient 0.5 |
| 参考量 | tangential force plateau about 1000.075 N；Coulomb limit 1000 N；error < 0.008% |

SFC 复现建议：

- 当前只能严格复现 pure normal loading stage。
- 不能声称复现 tangential friction/stick-slip，因为 friction 被项目约束排除。
- 可用于验证：非匹配网格下的 dynamic SDF normal contact、pressure/contact-zone shape、field query amortization。

## Benchmark 3：V-Indenter Problem

用途：尖锐/非平面接触几何、V-indenter indentation、外部 commercial solver comparison。

| 项 | PDF 工况 |
| --- | --- |
| 几何 | V-shaped indenter from 2 x 2 mm^2 square-based prism, height 1 mm；base plate 6 x 6 x 2 mm^3 |
| 材料 | aluminium；density 2700 kg/m^3；Young's modulus 70 GPa；Poisson ratio 0.33 |
| 载荷 | V-indenter 上施加 -90 N z-direction compressive load until global equilibrium |
| 网格 | irregular tetrahedral elements；average element size 0.25 mm |
| 约束 | V-indenter top surface 和 base plate bottom surface fully constrained during FEM after contact force quantification |
| FuzzyContact penalty scale | 250 N |
| 外部参考 | ABAQUS displacement-control calibration |
| 参考量 | base top diagonal path from (-2 mm, -2 mm) to (2 mm, 2 mm)；maximum z-displacement about 0.00185 mm |
| 表 1 | ABAQUS indentation depth 1.856 um；FuzzyContact 1.876 um；relative difference 1.077% |

SFC 复现建议：

- 这是最适合展示 3D von Mises / strain / displacement cloud 的算例之一。
- 需要自主生成 V-indenter 和 base plate 的 tetra mesh；不要从 VTU 反推几何作为主流程。
- 对比指标：peak indentation、top-surface diagonal displacement curve、von Mises stress near apex、dynamic SDF contact active set、field update/query timing。
- 该算例不能用来声称 Abaqus 等价，只能作为公开论文工况下的外部参考对比。

## Benchmark 4：Cylinder-Platen Compression / Neutron Diffraction Validation

用途：圆柱压缩、实验应变参考、加载序列。

| 项 | PDF 工况 |
| --- | --- |
| 试样 | stainless-steel cylinder；height 18 mm；diameter 10 mm |
| 压头/加载板 | rectangular loading platen 10 x 10 x 2 mm^3 |
| 实验测量 | internal elastic strains in 4 x 4 x 4 mm^3 gauge volume |
| 材料 | effective E from experiment 195.223 GPa；Poisson ratio 0.3698；simulation-derived E 196.691 GPa |
| 网格 | tetrahedral elements；average element size 0.5 mm |
| penalty scale | 250 N |
| 约束 | cylinder bottom face fully constrained；top surface subjected to distributed compressive load；lateral curved surface traction-free |
| 载荷序列 | 8 loads from 2825 N to 22600 N, step 2825 N |
| 参考量 | 表 1：effective Young's modulus reference 195.223 GPa；FuzzyContact 196.691 GPa；relative difference 0.75% |

SFC 复现建议：

- 可作为工程力学风格强验证算例：位移、应力、应变云图和载荷-应变曲线。
- 当前可以做 linear elastic quasi-static compression with dynamic SDF contact。
- 如果要对实验曲线逐点对比，需要从论文图 8 或公开数据仓库中获取/数字化 axial/radial strain 数据；PDF 正文只给了有效模量汇总。

## Benchmark 5：Connected Assembly Mesh Convergence

用途：复杂连接装配、网格收敛、接触主导响应。

| 项 | PDF 工况 |
| --- | --- |
| 几何 | inner inverted T-shaped component and outer enclosure around horizontal T arm |
| 截面 | both components cross-section 1 x 1 mm^2 |
| 总域 | 13 x 9 x 1 mm^3 |
| 运动 | inner T along +y with prescribed velocity 0.001 mm/s until contact force reduces velocity to zero |
| 约束 | inner top surface 和 outer bottom surface fully constrained after contact force quantification |
| 网格尺度 | average element sizes 0.5, 0.25, 0.125, 0.0625 mm |
| penalty scales | 0.32 N, 0.129 N, 0.04063 N, 0.01223 N |
| 参考量 | all four simulations converged to total contact force 11.174 N |
| 图 9 | maximum displacement and von Mises stress converge with refinement；finest two meshes relative changes below 5% |

SFC 复现建议：

- 可以作为“接触主导 + 多网格 + SDF 更新/查询摊销”算例。
- 几何比前四个复杂，适合 Phase 后续，不宜作为第一批实现阻塞项。
- 当前若严格遵守不加摩擦/自接触，仍可做法向接触和网格收敛；不要声称复现 FuzzyContact 全部历史相关接触律。

## 可执行优先级

第一批最合理：

1. Benchmark 1 sphere-block：解析 Hertz + 公开 FuzzyContact 数值摘要双参考。
2. Benchmark 3 V-indenter：3D 应力/应变/位移云图和路径位移曲线。
3. Benchmark 4 cylinder-platen：工程圆柱压缩，应变/有效模量对比。

第二批：

4. Benchmark 2 normal-only：非匹配网格和接触区验证；摩擦段明确 unsupported。
5. Benchmark 5 connected assembly：复杂装配与网格收敛，作为大算例。

## 对论文 claim 的影响

这些 PDF 工况可以支撑：

- FEM-SDF 在柔性体接触工况中的结果准确性。
- 3D displacement / strain / von Mises / gap / active contact zone 的物理可视化。
- 接触主导场景下，通过 dynamic SDF field 避免大量 projection query 的摊销效率优势。
- 与公开论文 benchmark 的 reference scalar metrics 对齐。

这些 PDF 工况不能单独支撑：

- 完整 FuzzyContact nonlinear sigmoidal traction law 等价。
- frictional stick-slip 等价。
- self-contact。
- arbitrary non-manifold global SDF。
- production commercial solver equivalence。
- Abaqus/CalculiX/FuzzyContact 源级别一致性。

## 下一步实现建议

建议新增一个独立 validation runner：

```text
validation/run_fuzzycontact_autonomous_sfc_benchmarks.py
```

职责：

1. 自主生成对应几何和 TET4/HEX8 或 surface/volume meshes。
2. 使用 SFC 内部 FEM assembly 和 DynamicNarrowBandSDF + field_contact。
3. 输出每个 benchmark 的 CSV、VTU/VTK、PNG/PDF figure。
4. 对比公开参考量，而不是对 VTU 结果做反推拟合。
5. 对所有图使用 Times 字体。

推荐输出：

```text
results/fuzzycontact_autonomous_sfc/
  benchmark1_hertz_metrics.csv
  benchmark1_hertz_fields.vtu
  benchmark1_hertz_3d_cloud.png
  benchmark3_v_indenter_metrics.csv
  benchmark3_v_indenter_fields.vtu
  benchmark3_v_indenter_3d_cloud.png
  benchmark4_cylinder_platen_metrics.csv
  benchmark4_cylinder_platen_fields.vtu
  benchmark4_cylinder_platen_3d_cloud.png
  fuzzycontact_autonomous_sfc_summary.md
```
