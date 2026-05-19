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
   - `MaterialPatchProjection`
   - `ReferencePatchBVH`

2. `src/sfc/contact/lagrangian_sdf_oracle.py`
   - `LagrangianSDFContactOracle`
   - `LagrangianSDFQueryResult`

3. API exports
   - `sfc.sdf.MaterialSDF`
   - `sfc.sdf.ReferencePatchBVH`
   - `sfc.contact.LagrangianSDFContactOracle`

## 当前算法语义

`MaterialSDF` 存储参考节点、参考表面三角 patch、可选 `phi0(X)` 与 `grad_phi0(X)`。当没有显式 `phi0/grad_phi0` callable 时，fallback 使用参考三角面片的局部有向距离。

`ReferencePatchBVH` 当前是 refit AABB patch index。它不构建当前构型 SDF grid；每次 FEM 节点更新时只基于同一组参考 patch topology 更新当前 AABB：

```text
reference patch C_k
-> current vertices chi(C_k,t)
-> current AABB B_k(t)
```

`LagrangianSDFContactOracle.query(...)` 的流程：

1. 用 refit AABB index 取候选 reference patch；
2. 若有 active patch cache，则把 cached patch 作为候选排序提示；
3. 对每个候选 patch 做当前空间到 patch 参数域的 least-squares pull-back；
4. 在 patch 参数域里做 Newton corrector：
   ```text
   minimize 0.5 ||x - chi(u,v)||^2
   ```
5. 返回：
   - signed gap `g = dot(x - chi(X*), n)`
   - current normal
   - closest current point
   - closest material point
   - face id
   - barycentric coordinates

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

局部测试结果：

```text
python -m pytest -q tests/test_lagrangian_sdf_oracle.py
4 passed in 0.50s
```

## 与现有 DynamicNarrowBandSDF 的关系

当前项目已有两条路线：

| 路线 | 状态 | 用途 |
| --- | --- | --- |
| `DynamicNarrowBandSDF + field_contact` | 成熟、已验证 | 当前论文/实验兼容路径、对照路径、field-grid 方法证据 |
| `MaterialSDF + LagrangianSDFContactOracle` | 新增原型 | `final_aim.md` 对应的新主线候选 |

不要再把 C++ tangent/field build 优化误认为 `final_aim.md` 的核心实现。那些优化仍属于 current-space dynamic field pipeline。

## 仍未完成

1. 高阶/体素 `phi0(X)` grid 存储与插值；
2. 体单元 `chi(X,t)` 反解，而不仅是 surface triangle patch affine map；
3. KKT 形式的 `phi0(X)=0` constrained Newton corrector；
4. 真正树结构 BVH 或 spatial hash，而不仅是 vectorized AABB patch index；
5. 双柔性体 patch-pair optimization；
6. oracle-based contact force/Jacobian assembly；
7. 与 CalculiX/SFC 工程算例的完整 oracle 路径对比；
8. 论文标题、摘要、方法、实验主线重写。

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
