# 使用 FuzzyContact 第一帧/解场 VTU 作为 SFC 模型的可行性分析

## 结论

对 Benchmark 1、3、4，可以使用 FuzzyContact 数据集中对应 `.vtu` 的第一份 displacement 文件作为 SFC 本地仿真的模型来源，但需要分清两件事：

1. **可以使用 VTU 的节点、TET4 connectivity、body split 作为几何/网格模型。**
2. **不能盲目把 VTU `points` 当参考构型。** 有些文件的 `points` 已经是变形后坐标，有些文件更像参考坐标；必须用 displacement 字段和 PDF 标称尺寸恢复或判定参考构型。

这种路线不再是“完全从 PDF 参数重新网格划分”，而是：

```text
FuzzyContact first solution VTU
-> recover/reference-check mesh geometry
-> SFC VolumeMesh
-> internal FEM assembly
-> DynamicNarrowBandSDF build from current surface
-> field_contact solve/replay
-> compare final displacement/stress/strain/contact metrics
```

它适合做论文里的 **external benchmark mesh reproduction**，但不应表述为“从零 CAD 自主建模”。如果论文想强调完全自主建模，仍应另做一个 PDF-parameter mesh generator；如果目标是尽快得到和公开数据一一对应的工程对比，第一帧 VTU 作为模型入口是可行且更稳妥的。

## 本地数据检查

本地数据位置：

```text
results/fuzzycontact_vtu_reference_full/downloads/Dataset_for_Solutions/Dataset_for_Solutions
```

Benchmark 1、3、4 的 displacement 文件均为 TET4：

| Benchmark | Body files | Nodes / cells | VTK type | 字段 |
| --- | --- | ---: | --- | --- |
| problem_1 | `displacement_bottom.vtu`, `displacement_top.vtu` | bottom 2233/9832, top 404/1526 | 10 = tetra | displacement |
| problem_3 | `base_displacement.vtu`, `indenter_displacement.vtu` | base 4873/22091, indenter 231/636 | 10 = tetra | displacement |
| problem_4 | `displacement_bottom.vtu`, `displacement_top.vtu` | bottom 10389/53833, top 1323/4800 | 10 = tetra | displacement |

这些文件的同名 stress/strain/mises 文件与 displacement 文件共享 mesh，可用于结果对比和三维云图。

## 坐标恢复规则

需要对每个 body 计算候选参考坐标：

```text
X_candidate_0 = VTU points
X_candidate_minus = VTU points - displacement
X_candidate_plus = VTU points + displacement
```

然后选择最符合 PDF 标称尺寸、固定边界和物理初始几何的候选作为 SFC `VolumeMesh.X`。

### Benchmark 1：sphere-block Hertz

检查结果：

- `problem_1/displacement_bottom.vtu`
  - `points` bbox 约为 `[-0.0013, 4.0013] x [-0.0014, 4.0012] x [-2, 0.00047]`
  - `points - displacement` bbox 恢复为精确 `0..4, 0..4, -2..0`
- `problem_1/displacement_top.vtu`
  - displacement 最大值约 `0.033`，对应论文中的 `33.03 um = 0.03303 mm`
  - `points - displacement` 更接近未变形 sphere mesh

判定：

```text
Benchmark 1 应使用 X_ref = points - displacement。
```

可行性：高。

优点：

- 可恢复 block 的精确参考尺寸。
- sphere displacement 标量与论文表 1 一致。
- 可直接使用数据集 TET4 mesh，避免自行生成 sphere 网格导致对比误差混入。

风险：

- FuzzyContact 位移字段名写成 `displacement (m)`，但数值实际与 mm/um 标度一致；SFC 对比时应统一采用 mm-N-MPa 单位体系。

## Benchmark 3：V-indenter

检查结果：

- `problem_3/base_displacement.vtu`
  - `points` bbox 精确为 `0..6, 0..6, -2..0`
  - displacement 最大约 `0.002135`，对应论文中 `~0.00185 mm` 的同量级变形
- `problem_3/indenter_displacement.vtu`
  - `points` span 为 `2 x 2 x 1`
  - z 方向初始位置约 `0.0125..1.0125`

判定：

```text
Benchmark 3 更适合使用 X_ref = points。
```

可行性：高。

优点：

- VTU points 已经很像参考几何。
- V-indenter 的准确网格和尖角几何可以直接保留。
- 可与 FuzzyContact/ABAQUS 的 indentation depth 和 diagonal displacement profile 对比。

风险：

- PDF 只给出 V-indenter 的描述和图示；如果不用 VTU mesh，自己重建尖角几何会引入歧义。
- 使用第一帧 VTU mesh 可以规避几何歧义，但论文中要称为“dataset mesh reconstruction”，不是“完全独立 CAD generation”。

## Benchmark 4：cylinder-platen

检查结果：

- `problem_4/displacement_bottom.vtu`
  - `points` bbox 约为 cylinder `diameter 10 mm, height 18 mm`
  - displacement 最大约 `0.031 mm`
- `problem_4/displacement_top.vtu`
  - `points` bbox 为 platen `10 x 10 x 1 mm`，位于 cylinder 顶部上方
  - displacement 最大约 `0.0021 mm`

判定：

```text
Benchmark 4 应优先使用 X_ref = points。
```

可行性：中高。

优点：

- cylinder/platen 几何在 VTU points 中清晰。
- TET4 mesh 已经存在，不需要重新生成圆柱体网格。
- 可输出工程力学风格的 3D displacement / strain / stress 云图。

风险：

- PDF 正文给的是八个载荷步和有效模量结果，但本地数据集中 problem_4 看起来只有一组 displacement/stress/strain 文件；如果没有逐载荷 VTU，则只能对最终/代表性载荷状态做对比。
- 若要复现实验的 load-strain 斜率，需要补充或数字化图 8 数据，否则只能对比表 1 的 effective modulus。

## 建议的 SFC 构建方式

新增一个 runner：

```text
validation/run_fuzzycontact_first_frame_sfc_models.py
```

推荐步骤：

1. 读取 problem 1/3/4 displacement VTU。
2. 合并同一 body 的 displacement/stress/strain/mises 文件。
3. 自动判定参考坐标：
   - problem 1：`points - displacement`
   - problem 3：`points`
   - problem 4：`points`
4. 构造 SFC `VolumeMesh` 和 `DeformableBody` / `MechanicsModel`。
5. 根据 PDF 设置材料、载荷和约束。
6. 每个增量用当前 master boundary 重建 `DynamicNarrowBandSDF`。
7. slave surface quadrature 通过 `field_contact` 查询 gap/J/force。
8. 输出：
   - final displacement error
   - von Mises / strain norm error
   - contact zone / active mask
   - gap / penetration field
   - SDF update/query timing
   - 3D Times-font figures

## Claim 边界

可以支持：

- 使用公开 FuzzyContact benchmark mesh 的 SFC 本地 FEM-SDF contact 仿真。
- 与 FuzzyContact 公开结果场进行三维云图和标量指标对比。
- Dynamic SDF field 在这些工程接触模型上的查询与接触响应验证。

不能支持：

- 完整复现 FuzzyContact 的 sigmoidal traction law。
- 摩擦、self-contact、barrier、GPU、neural/POD。
- 声称第一帧 VTU 是完全未变形 CAD，除非经过 `points`/`points-displacement` 判定。
- 完全独立从 PDF 参数重建几何；这是另一个更慢但更干净的路线。

## 推荐使用顺序

1. problem_1：先跑，因为 reference 最强，坐标恢复也最明确。
2. problem_3：再跑，因为最适合三维应力/应变/位移云图。
3. problem_4：第三个跑，用于工程压缩和实验背景补强。

