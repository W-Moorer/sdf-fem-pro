# FuzzyContact 1/3/4/5 本地 SFC 全仿真对比可行性

## 判定标准

这里的“本地全仿真对比”定义为：

1. 几何、网格、材料、载荷和约束由 SFC validation runner 自主生成。
2. FEM 组装、接触检测、dynamic narrow-band SDF field 构建、field_contact 查询、罚函数接触力、位移/应力/应变后处理均在本项目内完成。
3. FuzzyContact PDF/VTU/表格只作为外部参考，不作为反推模型或求解依赖。
4. 不加入 friction、self-contact、barrier、GPU、POD、neural SDF、Abaqus core dependency。

## 总体结论

Benchmark 1、3、4 可行，适合作为第一批工程力学论文算例。Benchmark 5 可做，但不适合作为第一批严格外部等价对比，因为 PDF 给出的几何和历史接触律信息不足以唯一重建 FuzzyContact 的完整轨迹。

| Benchmark | 本地全仿真 | 外部参考强度 | 推荐程度 | 主要用途 |
| --- | --- | --- | --- | --- |
| 1 sphere-block Hertz | 可行 | 强：Hertz analytical + FuzzyContact scalar table | 第一优先级 | 接触准确性、Hertz 标量对比、SDF 查询效率 |
| 3 V-indenter | 可行 | 中强：ABAQUS/FuzzyContact indentation + path plot/VTU | 第一优先级 | 3D 应力云图、尖锐几何接触、位移路径对比 |
| 4 cylinder-platen | 可行 | 中：实验有效模量 + load sequence | 第二优先级 | 工程压缩、应变/应力云图、载荷序列 |
| 5 connected assembly | 部分可行 | 中弱：总接触力和网格收敛图，几何细节不足 | 后续优先级 | 接触主导效率、网格收敛、复杂装配展示 |

## Benchmark 1：sphere-block Hertz

可行性：高。

已有基础：

- TET4/HEX8 组装：`src/sfc/fem/assembler.py`、`src/sfc/fem/tet4.py`、`src/sfc/fem/hex8.py`。
- sphere-like TET4 生成经验：`validation/run_calculix_drop_impact_comparison.py`、`validation/run_geometric_nonlinear_vtk.py`。
- block TET4 生成：`validation/run_phase3_validation.py::structured_tet_block`。
- dynamic SDF field：`src/sfc/sdf/dynamic_narrow_band_sdf.py`。
- field contact：`src/sfc/contact/field_contact.py`。

需要补的内容：

- 生成更接近半径 1 mm sphere 的自主 TET4 mesh。
- 生成 4 x 4 x 2 mm block mesh。
- 写 two-body field-contact quasi-static solve：每个增量从 current master surface rebuild SDF，slave quadrature samples 插值查询 gap/J/force。
- 输出 3D displacement、strain norm、von Mises、gap/contact pressure、active contact zone。

可对比的参考量：

- Hertz contact radius 0.2673 mm。
- Hertz total indentation 71.45 um；single-body indentation 35.7 um。
- FuzzyContact sphere indentation 33.03 um；block downward displacement 13.78 um。

论文价值：

- 最适合证明“FEM-SDF 正确计算法向接触几何和接触反力”。
- 也是最干净的解析基准。

## Benchmark 3：V-indenter

可行性：中高。

已有基础：

- TET4/HEX8 mechanics 和 stress/strain 后处理已有。
- 项目已有 3D surface cloud plotting 经验，且图形可统一 Times 字体。
- DynamicNarrowBandSDF 对非平面 master surface 已通过 Jacobian/gradient tests。

需要补的内容：

- 自主生成 V-shaped indenter mesh：2 x 2 mm^2 square-based prism，height 1 mm。
- 自主生成 6 x 6 x 2 mm base plate。
- 明确 V-indenter 的尖角朝向和接触面定义；PDF 图 3/7 足够支持一个确定实现，但不应声称源级完全相同。
- 增加 diagonal path 采样：base top surface from (-2 mm, -2 mm) to (2 mm, 2 mm)。

可对比的参考量：

- ABAQUS indentation depth 1.856 um。
- FuzzyContact indentation depth 1.876 um。
- base top diagonal path 最大 z-displacement about 0.00185 mm。

论文价值：

- 最适合替换错误的二维点云图，展示真正三维应力/应变/位移云图。
- 能说明 SDF field 对尖锐、非平面接触几何的适用性。

## Benchmark 4：cylinder-platen

可行性：中。

已有基础：

- 圆柱体可由规则环向/轴向节点 + TET4/Delaunay 或 HEX-like subdivision 生成。
- platen 是简单 10 x 10 x 2 mm block。
- 八个载荷步可用 quasi-static incremental solve 完成。

需要补的内容：

- 自主 cylinder volume mesh。
- gauge volume 4 x 4 x 4 mm^3 内的应变平均。
- load sequence：2825 N 到 22600 N，步长 2825 N。
- 输出 axial/radial strain、effective modulus、stress/strain/displacement cloud。

可对比的参考量：

- Experimental effective Young's modulus 195.223 GPa。
- FuzzyContact effective Young's modulus 196.691 GPa。
- Poisson ratio 0.3698。

论文价值：

- 工程力学味道强，适合证明结果不是玩具例子。
- 但它对 SDF 接触优势的证明较弱，因为整体响应主要由材料压缩和边界条件决定；SDF 优势仍需由 field accuracy/Jacobian/Q* 证明。

## Benchmark 5：connected assembly

可行性：部分可行。

已有基础：

- block/structured mesh、contact sample、field query timing、stress cloud 都能复用。
- SDF field 对大量接触查询的摊销优势在这个算例上最容易体现。

主要风险：

- PDF 只给出 inverted T + outer enclosure、cross-section 1 x 1 mm^2、domain 13 x 9 x 1 mm^3、四个 mesh size、总接触力 11.174 N，但没有完整 CAD/mesh、间隙、圆角/尖角处理、初始相对位置和所有约束细节。
- FuzzyContact 使用历史相关 proximity/contact update；当前 SFC 不应照搬其 sigmoidal traction law。

可以做的版本：

- SFC 自主建立一个与论文几何同族的 T-shaped connected assembly。
- 做 mesh convergence：0.5、0.25、0.125、0.0625 mm。
- 对比 total contact force 是否接近 11.174 N，并重点报告网格收敛趋势和 SDF amortized speedup。

不能做的 claim：

- 不能称为 FuzzyContact problem 5 完整轨迹等价。
- 不能把它作为唯一外部精度基准。

论文价值：

- 适合作为“接触主导 + 大量查询 + SDF 摊销优势”的性能算例。
- 不适合作为第一批核心准确性算例。

## 当前项目缺口

要真正完成 1/3/4/5 本地全仿真，需要补三个基础模块：

1. Autonomous benchmark mesh builders：sphere、V-indenter、cylinder、T-assembly。
2. Two-body field-contact solver loop：把 `DynamicNarrowBandSDF + field_contact` 接入全局静力/动力迭代，而不是只做 replay 或单步 contact response。
3. Paper-grade 3D output：VTK/VTU + Times 字体 PNG/PDF，包含 displacement、strain norm、von Mises、gap/contact pressure、active contact zone。

## 推荐执行顺序

第一步：Benchmark 1。

- 原因：解析参考最强，工况最明确，能最快证明准确性。

第二步：Benchmark 3。

- 原因：最适合三维应力/应变云图，能修复当前论文视觉说服力问题。

第三步：Benchmark 4。

- 原因：工程压缩算例，可以增强工程力学可信度。

第四步：Benchmark 5。

- 原因：最适合效率和网格收敛，但严格等价风险最大，应放在已有 1/3/4 后面。
