# Lagrangian SDF Contact Oracle 交接

日期：2026-05-19

## 背景

`docs/final_aim.md` 的目标不是继续把当前构型 `DynamicNarrowBandSDF` grid build 做到更快，而是把主方法转向：

```text
reference/material SDF phi0(X)
+ FEM deformation map chi(X,t)
+ refit reference surface patch AABBs
+ on-demand pull-back query
+ local closest-point Newton corrector
```

因此，本轮新增一个并行的新模块，保留现有 `DynamicNarrowBandSDF + field_contact` 作为兼容路径、对照路径和已验证实验路径。

## 新增代码

1. `src/sfc/sdf/material_sdf.py`
   - `MaterialSDF`
   - `MaterialSDFGrid`
   - `MaterialPatchProjection`
   - `ReferencePatchBVH`

2. `src/sfc/contact/lagrangian_sdf_oracle.py`
   - `LagrangianSDFContactOracle`
   - `LagrangianSDFQueryResult`
   - `LagrangianOracleConstraint`
   - `LagrangianOracleContactResponse`
   - `lagrangian_oracle_constraint_from_sample(...)`
   - `lagrangian_oracle_jacobian_row(...)`
   - `lagrangian_oracle_penalty_response(...)`

3. `src/sfc/fem/deformation_map.py`
   - `FEMDeformationMap`
   - `DeformationMapEvaluation`
   - TET4 affine material/current inverse and `chi(X,t)` evaluation
   - HEX8 isoparametric reference/current inverse and `chi(X,t)` evaluation

4. API exports
   - `sfc.sdf.MaterialSDF`
   - `sfc.sdf.MaterialSDFGrid`
   - `sfc.sdf.ReferencePatchBVH`
   - `sfc.fem.FEMDeformationMap`
   - `sfc.contact.LagrangianSDFContactOracle`

## 当前算法语义

`MaterialSDF` 存储参考节点、参考表面三角 patch、可选 `phi0(X)` 与 `grad_phi0(X)`，以及可选 `MaterialSDFGrid`。当没有显式 `phi0/grad_phi0` callable 或 reference grid 时，fallback 使用参考三角面片的局部有向距离。

`MaterialSDFGrid` 是参考空间 regular SDF grid，保存：

- `phi0` grid；
- finite-difference 或用户提供的 `grad_phi0`；
- validity mask；
- trilinear `query_phi(X)`；
- scalar-interpolation derivative `query_gradient(X)`。

它是 material/reference grid，不是 current-space dynamic SDF grid。

`ReferencePatchBVH` 当前是 refit AABB patch index。它不构建当前构型 SDF grid；每次 FEM 节点更新时只基于同一组参考 patch topology 更新当前 AABB：

```text
reference patch C_k
-> current vertices chi(C_k,t)
-> current AABB B_k(t)
```

`FEMDeformationMap` 提供：

- `chi(X,t)`；
- deformation gradient `F = d chi / dX`；
- material/current pull-back；
- TET4 和 HEX8 支持；
- boundary face 到所属体单元的查找。

`LagrangianSDFContactOracle.query(...)` 的流程：

1. 用 refit AABB index 取候选 reference patch；
2. 若有 active patch cache，则把 cached patch 作为候选排序提示；
3. 对每个候选 patch 做当前空间到 patch 参数域的 least-squares pull-back；
4. 如果没有 `FEMDeformationMap`，在 patch 参数域里做 Newton corrector：
   ```text
   minimize 0.5 ||x - chi(u,v)||^2
   ```
5. 如果存在 `FEMDeformationMap`，执行 final-aim KKT/Gauss-Newton corrector：
   ```text
   F(X)^T (chi(X,t) - x) + lambda grad_X phi0(X) = 0
   phi0(X) = 0
   ```
   并用
   ```text
   n = normalize(F^{-T} grad_X phi0)
   g = dot(x - chi(X*,t), n)
   ```
6. 返回：
   - signed gap `g = dot(x - chi(X*), n)`
   - current normal
   - closest current point
   - closest material point
   - face id
   - barycentric coordinates
   - master FEM node ids and shape weights

`lagrangian_oracle_penalty_response(...)` 提供 point/sample-to-oracle 的无摩擦 penalty response：

```text
E_c = 1/2 k <-g>_+^2
J_slave = N_slave n^T
J_master = -N_master n^T
f = J^T k<-g>_+
K ~= k J^T J
```

当前 corrector 使用三角 patch 参数化，因此约束 `X in Gamma0` 由 patch 参数域隐式满足。这是 `final_aim.md` 中 KKT 约束优化的第一版 surface-patch 实现。

## Cache 语义

active patch cache 只改变候选排序，不改变候选集合的 exact fallback。因此 cache 不允许改变最终最近 patch/gap/normal 结果。

## 新增测试

`tests/test_lagrangian_sdf_oracle.py`

覆盖：

1. 刚体平移/旋转下 gap 不变、normal 随刚体旋转；
2. 大转角弯曲 surface 上，oracle 与当前表面 brute-force projection 一致；
3. Newton-corrected gap 的 `dg/dx` 与 finite difference 一致；
4. active patch cache 不改变 query 结果。
5. reference `MaterialSDFGrid + FEMDeformationMap + KKT` oracle 与当前表面 projection 一致；
6. oracle query 不调用 `DynamicNarrowBandSDF.build(...)`，即不重建 current-space SDF grid；
7. oracle master/slave Jacobian 与 finite difference 一致；
8. oracle penalty response 满足 master/slave action-reaction。

局部测试结果：

```text
python -m pytest -q tests/test_lagrangian_sdf_oracle.py
8 passed in 0.64s
```

## 与现有 DynamicNarrowBandSDF 的关系

当前项目已有两条路线：

| 路线 | 状态 | 用途 |
| --- | --- | --- |
| `DynamicNarrowBandSDF + field_contact` | 成熟、已验证 | 当前论文/实验兼容路径、对照路径、field-grid 方法证据 |
| `MaterialSDF + LagrangianSDFContactOracle` | 新增原型 | `final_aim.md` 对应的新主线候选 |

不要再把 C++ tangent/field build 优化误认为 `final_aim.md` 的核心实现。那些优化仍属于 current-space dynamic field pipeline。

## 当前仍未完成

1. 高阶 tricubic/B-spline `phi0(X)`；
2. 真正树结构 BVH 或 spatial hash，而不仅是 vectorized AABB patch index；
3. 双柔性体 patch-pair optimization；
4. oracle-based surface-to-surface quadrature vectorized/C++ backend；
5. 与 CalculiX/SFC 工程算例的完整 oracle 路径对比；
6. 论文标题、摘要、方法、实验主线重写。

## 论文主线判断

如果坚持 `docs/final_aim.md`，论文主方法应从：

```text
dynamic narrow-band SDF field rebuilt from the current FEM surface
```

调整为：

```text
deformation-aware / Lagrangian SDF contact oracle
```

现有 dynamic field 结果可以保留，但应降级为：

- baseline；
- ablation；
- compatibility path；
- current-space field comparison。

不能再把每步 current-space SDF rebuild 写成最终主方法。
