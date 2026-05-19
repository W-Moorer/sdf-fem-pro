# Lagrangian SDF Contact Oracle 交接

日期：2026-05-19

## 目标定位

`docs/final_aim.md` 指向的最终主线不是“每步重建 current-space SDF grid”，而是：

```text
reference/material SDF phi0(X)
+ FEM deformation map chi(X,t)
+ refit reference surface patch bounds
+ cheap pull-back SDF query
+ constrained closest-point Newton corrector
+ FEM contact force/Jacobian scatter
```

因此，本模块保留已有 `DynamicNarrowBandSDF + field_contact` 作为兼容、对照和消融路径，同时新增 deformation-aware / Lagrangian SDF contact oracle 作为 final-aim 主方法候选。

## 已实现代码

1. `src/sfc/sdf/material_sdf.py`
   - `MaterialSDF`
   - `MaterialSDFGrid`
   - `MaterialPatchProjection`
   - `ReferencePatchBVH`

2. `src/sfc/fem/deformation_map.py`
   - `FEMDeformationMap`
   - `DeformationMapEvaluation`
   - TET4 affine `chi(X,t)`、pull-back、deformation gradient
   - HEX8 isoparametric `chi(X,t)`、pull-back、deformation gradient

3. `src/sfc/contact/lagrangian_sdf_oracle.py`
   - `LagrangianSDFContactOracle`
   - `LagrangianSDFQueryResult`
   - `LagrangianPullbackQueryResult`
   - `LagrangianPatchPairQueryResult`
   - `LagrangianOracleConstraint`
   - `LagrangianOracleContactResponse`
   - `lagrangian_oracle_constraint_from_sample(...)`
   - `lagrangian_oracle_jacobian_row(...)`
   - `lagrangian_oracle_penalty_response(...)`
   - `lagrangian_oracle_penalty_response_batch(...)`
   - `lagrangian_patch_pair_query(...)`

4. API exports
   - `sfc.sdf.MaterialSDF`
   - `sfc.sdf.MaterialSDFGrid`
   - `sfc.sdf.ReferencePatchBVH`
   - `sfc.fem.FEMDeformationMap`
   - `sfc.contact.LagrangianSDFContactOracle`
   - `sfc.contact.LagrangianPullbackQueryResult`
   - `sfc.contact.LagrangianPatchPairQueryResult`

## Material SDF

`MaterialSDF` 保存参考节点、参考边界三角 patch、可选 `phi0(X)`、可选 `grad_phi0(X)`、可选 `MaterialSDFGrid` 和窄带半径。当没有显式 callable 或 grid 时，fallback 使用参考三角面的局部有向距离。

`MaterialSDFGrid` 是参考空间 regular SDF grid，不是 current-space dynamic SDF grid。它支持：

- `phi0` grid；
- finite-difference 或用户提供的 `grad_phi0`；
- validity mask；
- trilinear `query_phi(X)`；
- cubic tensor-product Lagrange `query_phi(X)`；
- unnormalized scalar derivative `query_raw_gradient(X)`；
- normalized `query_gradient(X)`。

三次插值路径用于高阶 material SDF 查询。它不构建当前构型 SDF 场。

## Reference Patch Index

`ReferencePatchBVH` 现在是 refit AABB + spatial hash patch index。它不重建 current-space SDF grid；每次 FEM 节点更新时只更新同一组 reference patch 的当前 AABB：

```text
reference patch C_k
-> current vertices chi(C_k,t)
-> current AABB B_k(t)
-> spatial hash refit
```

`cached_face_id` 只改变候选排序，不允许改变候选集合的 exact fallback 结果。因此 active patch cache 不会改变最终 gap/normal。

## Oracle Query

精确 query 路径：

1. 使用 refit patch index 取候选 reference patch；
2. cached patch 只作为排序提示；
3. 对候选 patch 做 current-space pull-back 初值；
4. 无 `FEMDeformationMap` 时，在三角 patch 参数域中做 closest-point Newton；
5. 有 `FEMDeformationMap` 时，执行 KKT/Gauss-Newton corrector：

```text
F(X)^T (chi(X,t) - x) + lambda grad_X phi0(X) = 0
phi0(X) = 0
```

6. 输出：
   - signed gap `g = dot(x - chi(X*), n)`；
   - current normal `n = normalize(F^{-T} grad_X phi0)`；
   - closest current point；
   - closest material point；
   - face id；
   - barycentric coordinates；
   - master FEM node ids and shape weights。

## Cheap Pull-Back Query

新增 `query_pullback(...)` 和 `query_pullback_gap_normal(...)`，对应 final-aim 两级算法的低成本阶段：

```text
X ~= chi^{-1}(x,t)
g_tilde = phi0(X) / ||F^{-T} grad_X phi0(X)||
n_tilde = normalize(F^{-T} grad_X phi0(X))
```

这一路径用于粗查询、初值、诊断或后续分层策略；它不替代精确 closest-point Newton corrector。

## Contact Response

`lagrangian_oracle_penalty_response(...)` 提供 point/sample-to-oracle 的无摩擦 penalty response：

```text
E_c = 1/2 k <-g>_+^2
J_slave = N_slave n^T
J_master = -N_master n^T
f = J^T k<-g>_+
K ~= k J^T J
```

`lagrangian_oracle_penalty_response_batch(...)` 是稳定的批量入口。目前内部仍复用精确 scalar oracle，但调用点已经固定，后续可以无接口变更地下沉到 C++ fused backend。

## Patch-Pair Query

新增 `lagrangian_patch_pair_query(...)`，用于双柔性体 surface patch-pair 最近点查询。当前实现枚举给定 patch id 集合，并使用三角形-三角形最近点几何核返回：

- closest point on body A；
- closest point on body B；
- material point A/B；
- face id A/B；
- barycentric A/B；
- gap；
- normal。

这对应 final-aim 中“对重要接触区域、深穿透区域、边缘区域启用 patch-pair 优化”的第一版实现。

## 测试

新增和保留的测试位于 `tests/test_lagrangian_sdf_oracle.py`，覆盖：

1. 刚体平移/旋转下 gap 不变、normal 随刚体旋转；
2. 大转角弯曲 surface 中 oracle 与当前表面 brute-force projection 一致；
3. Newton-corrected gap 的 `dg/dx` 与 finite difference 一致；
4. active patch cache 不改变 query 结果；
5. reference `MaterialSDFGrid + FEMDeformationMap + KKT` oracle 与当前表面 projection 一致；
6. oracle query 不调用 `DynamicNarrowBandSDF.build(...)`；
7. cheap pull-back query 在 affine plane case 中与解析结果一致；
8. oracle master/slave Jacobian 与 finite difference 一致；
9. oracle penalty response 满足 master/slave action-reaction；
10. cubic material SDF interpolation 与三次多项式及其导数一致；
11. reference patch spatial hash 与 exact AABB radius filter 一致；
12. batch query 与 scalar loop 一致；
13. batch penalty response 与 scalar response 一致；
14. patch-pair query 在平行 surface case 中返回正确 gap/normal。

验证结果：

```text
python -m pytest -q tests/test_lagrangian_sdf_oracle.py
14 passed in 0.75s

pytest -q
324 passed in 642.18s (0:10:42)
```

新的 final-aim 局部测试策略见 `docs/final_aim_test_scope.md`。当只修改 Lagrangian/material SDF oracle 及其 validation runner 时，推荐使用：

```text
pytest -q -m final_aim
```

本轮局部测试结果：

```text
pytest -q -m final_aim
16 passed, 310 deselected in 25.92s
```

## 已迁移算例

新增 `validation/run_final_aim_lagrangian_contact_cases.py`，将已有 Phase-10 小型 TET4/HEX8 静力和动力接触算例迁移到 final-aim 路径：

```text
MaterialSDF -> LagrangianSDFContactOracle -> oracle penalty response
```

它不构建 current-space SDF grid，不调用旧 projection query 路径。参考解为同一 penalty law 下的 analytic rigid plane。

运行命令：

```text
python validation/run_final_aim_lagrangian_contact_cases.py --quick --out-dir results/final_aim_lagrangian_contact_quick
```

输出：

- `results/final_aim_lagrangian_contact_quick/final_aim_lagrangian_contact_cases.csv`
- `results/final_aim_lagrangian_contact_quick/final_aim_lagrangian_contact_samples.csv`
- `results/final_aim_lagrangian_contact_quick/final_aim_lagrangian_contact_history.csv`
- `results/final_aim_lagrangian_contact_quick/final_aim_lagrangian_contact_summary.md`

quick 结果：

| Case | Status | Max gap error | Force rel. error | Max penetration | Reaction z |
| --- | --- | ---: | ---: | ---: | ---: |
| `static_linear_tet4_lagrangian_oracle` | passed | 0.000000e+00 | 0.000000e+00 | 2.000000e-02 | 1.200000e+01 |
| `dynamic_linear_tet4_lagrangian_oracle` | passed | 0.000000e+00 | 0.000000e+00 | 1.451057e-02 | 7.018813e+00 |
| `static_linear_hex8_lagrangian_oracle` | passed | 0.000000e+00 | 0.000000e+00 | 2.000000e-02 | 1.200000e+01 |
| `dynamic_linear_hex8_lagrangian_oracle` | passed | 0.000000e+00 | 0.000000e+00 | 1.272863e-02 | 7.637180e+00 |

## 与 DynamicNarrowBandSDF 的关系

当前项目有两条路线：

| 路线 | 状态 | 用途 |
| --- | --- | --- |
| `DynamicNarrowBandSDF + field_contact` | 已验证 | current-space field 兼容路径、对照路径、消融和已有实验基础 |
| `MaterialSDF + LagrangianSDFContactOracle` | final-aim 核心原型 | deformation-aware material SDF contact oracle 主方法候选 |

如果论文坚持 `final_aim.md`，主方法应从：

```text
dynamic narrow-band SDF field rebuilt from the current FEM surface
```

调整为：

```text
deformation-aware / Lagrangian material SDF contact oracle
```

已有 dynamic field 结果可以作为 baseline、ablation、compatibility path 或 current-space field comparison，但不应再作为最终主方法。

## 仍不应声称

- friction；
- self-contact；
- nonlinear FEM 主求解器已完整覆盖；
- GPU 实时；
- barrier contact；
- neural/POD/data-driven SDF；
- Abaqus 依赖的核心求解；
- production BVH superiority；
- arbitrary non-manifold robust global SDF。

## 后续工程化

仍建议继续做，但它们不阻塞当前 final-aim 原型成立：

1. C++ fused oracle backend；
2. matrix-free tangent 与预条件求解；
3. patch-pair 候选由 broad phase 自动生成；
4. oracle-based surface-to-surface quadrature；
5. 与 CalculiX/SFC 工程算例的完整 oracle 路径对比；
6. 论文标题、摘要、方法和实验主线按 Lagrangian material SDF oracle 重写。
