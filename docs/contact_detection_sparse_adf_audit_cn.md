# 接触检测与稀疏 ADF 审计报告

## 结论

结合 `legacy_abaqus_prototype/contact_sdf` 中的检测代码和当前 `src/sfc` 的 Lagrangian-SDF / surface-contact 路径，当前与 Abaqus 对不上的风险确实更可能集中在接触检测和接触计算口径，而不是单纯的材料参数或时间积分参数。

更具体地说：

1. `legacy` 的粗检测和细检测流程可以作为“候选检测思路”参考，但不能作为最终接触计算理论。
2. 稀疏 ADF / hashed SDF 适合做窄带候选、快速近似距离和缓存，不适合直接输出最终 gap、normal、master payload 和接触 Jacobian。
3. 当前 SFC 主线已经比 `legacy` 更接近正确理论，因为最终 gap/normal/payload 来自当前构型 closest feature 或 secondary-normal constrained projection；但仍有几个会直接造成 active set、压力、应力和应变误差的漏洞。
4. 齿轮算例中最需要继续定位的是 active set 覆盖、secondary-line release、surface-to-surface constraint averaging、pressure smoothing 和输出口径，而不是继续调罚刚度。

## legacy 检测流程

相关文件：

- `legacy_abaqus_prototype/contact_sdf/pre_processing.py`
- `legacy_abaqus_prototype/contact_sdf/contact_test.py`
- `legacy_abaqus_prototype/contact_sdf/gradient_sdf.py`
- `legacy_abaqus_prototype/contact_sdf/sdf_predict.py`
- `legacy_abaqus_prototype/contact_sdf/contact_force.py`

### 粗检测

`pre_processing.py` 中 `make_contact_detector(...)` 的每步流程是：

```text
Abaqus-derived DOF / displacement
-> current vertices
-> global AABB overlap
-> predicted SDF field
-> ContactDetectorSurface.detect(...)
```

第一层粗检测是两物体当前顶点的全局 AABB：

```text
min1/max1 with broad_tol
min2/max2 with broad_tol
if separated: no contact
```

`contact_test.py` 中 `ContactDetectorSurface.detect(...)` 又做了第二层 AABB：

```text
sample-side triangle AABB
vs
other-body global AABB
```

这个粗检测能排除明显远离的面片，但有三个限制：

- 它不是 face-to-face 或 patch-to-patch broad phase，只是“采样侧面片 vs 对方全局盒子”。
- 它没有 swept AABB，因此高速动力学或大步长下可能漏掉步内穿越。
- 它依赖 `broad_tol`，该值如果不是由网格尺寸、最大步内位移和接触容差推导，就会变成经验参数。

### 细检测

`ContactDetectorSurface.detect(...)` 的细检测流程是：

```text
select sample side: 2in1 or 1in2
-> build barycentric samples on sample triangles
-> query other-side SDF by trilinear interpolation
-> active if phi <= sdf_tol
-> normal = normalized interpolated SDF gradient
-> deduplicate by rounded position
```

关键代码位置：

- 面片采样：`contact_test.py` 中 `_build_barycentric_table(...)`
- SDF 查询：`trilinear_val_grad_uniform(...)`
- active 判定：`valid_mask = isfinite(vals) & (vals <= sdf_tol)`
- 去重：`_dedup_pairs_by_position(..., eps=1e-4)`

这个流程的问题是：它把 SDF 插值阈值直接当成最终细检测。对工程接触求解来说这不够，因为最终接触约束需要：

- 当前构型 closest feature；
- master face id；
- barycentric / Q4 natural coordinates；
- 当前法向；
- master-side sensitivity；
- 面积权重；
- 与接触方向兼容的 active/release 逻辑。

`legacy` 细检测只给出 `phi_other` 和插值梯度 normal，没有可靠 payload，因此无法严格支撑：

```text
dg/dx_master = -N_master n^T
```

## 稀疏 ADF 理论风险

`gradient_sdf.py` 中的 `HashedGradientSDF` 存储窄带体素：

```text
psi_j = phi(v_j)
g_j = grad(phi)(v_j)
```

查询采用 voxel-center Taylor 近似：

```text
d(p) = psi_j + (p - v_j)^T g_j
```

这是一种稀疏 ADF / hashed narrow-band SDF 的近似查询。它适合：

- 快速判断查询点是否接近参考表面；
- 做 candidate scoring；
- 减少 dense SDF 存储；
- 为后续 exact projection 提供初值。

但它不适合作为最终接触 gap：

1. 它是体素中心一阶 Taylor 近似，不是当前构型 closest-feature projection。
2. hash miss 会返回 `NaN`，除非显式启用 dense fallback；如果 fallback 没有进入最终检测链，就会形成漏检。
3. `SDFPredictorInverse` 和 `SDFPredictorTaylor` 通过 kNN 位移插值或法向 Taylor 更新预测变形 SDF，这对大转角、大剪切、曲率变化和局部接触区不够严格。
4. 只存 `psi` 和 `grad` 不存 closest face id、barycentric、normal payload，不能直接装配 master Jacobian。
5. 若用于最终接触力，会把检测近似误差直接传入压力、应力和应变场。

因此，稀疏 ADF 的正确定位应该是：

```text
sparse ADF / hash / BVH
    -> broad phase / candidate / cache / initial guess
current-surface closest feature or normal-tube constrained projection
    -> final gap / normal / payload / Jacobian
surface-to-surface quadrature
    -> force / tangent / pressure output
```

## legacy force path 的理论漏洞

`contact_force.py` 暴露了几个不能迁移到当前主线的问题：

1. gap 符号和当前 SFC 主约定不同。当前 SFC 使用正 gap 表示分离、负 gap 表示穿透；legacy 中使用 `g = -phi0`。
2. 代码注释中已经指出 predictor-corrector 迭代中理论上应加入：

   ```text
   g_current = -phi0 + dot(du_point - du_other, n)
   ```

   但实际代码保留了 `g = -phi0`。这会使步内 Newton / corrector 阶段的 gap 与位移修正脱钩。
3. contact tangent 的符号在注释中也不确定，不能作为 Abaqus 对齐依据。
4. 该路径包含阻尼和摩擦形式，不符合当前无摩擦、线性罚函数或 hard-contact 对齐阶段的最小理论闭环。

## 当前 SFC 路径的优点

当前核心路径已经避免了 `legacy` 中最严重的问题：

- `src/sfc/sdf/material_sdf.py` 中 `ReferencePatchBVH` refit 当前 AABB，但保留参考 patch 拓扑。
- `src/sfc/contact/lagrangian_sdf_oracle.py` 的 query 会从候选 patch 中做当前构型 closest-feature 或 KKT corrector。
- `query_master_sensitivity(...)` 明确返回 master node weights 和 `-N_master n`。
- `src/sfc/contact/lagrangian_surface_contact.py` 已经支持 secondary-normal projection、normal-compatible query、Q4 closest-feature payload 和 Q4 surface quadrature。

这说明当前问题不是“还在用 legacy 那种 SDF 阈值当最终接触”，而是 surface-to-surface 口径还没有完全对齐 Abaqus。

## 当前仍可能导致误差的漏洞

### 1. 候选半径仍可能造成 active recall 下降

`LagrangianSDFSurfaceContactGeometry._prepare_sampling(...)` 用 master face centroid KDTree 和 face bounding radius 做候选筛选：

```text
candidate radius = search_radius + master_max_radius
slave face tube = search_radius + slave_radius + master_max_radius
```

如果 `search_radius` 只按局部接触容差设置，而没有包含：

- 最大步内闭合距离；
- secondary-normal line projection 的可能长度；
- 主从面片代表长度；
- Abaqus finite-sliding 接触搜索区；

则粗检测会提前过滤掉真实接触候选。此前区域诊断已经显示，在齿轮接触建立/释放窗口，SFC active recall 曾明显下降，这与候选覆盖不足或 release 口径不一致高度相关。

### 2. secondary-line projection 需要 closest-feature closed/open 保护

当前 `secondary_normal_projection_samples(...)` 是最接近 Abaqus surface-to-surface 的方向，因为 Abaqus 的接触方向来自 secondary surface constraint region 的平均法向。但该路径有一个固有风险：

```text
沿 secondary normal 的射线可能打到远处或背向 facet
```

因此 hard distance limit 不能无条件把大 overclosure 释放成 open。正确逻辑是：

```text
如果 secondary-line hit closed 且全局 closest feature 也 closed:
    保留 overclosure
如果 secondary-line hit 超限但全局 closest feature open:
    释放该远程 line hit
```

当前代码已经朝这个方向修过，但仍需要通过逐帧 active precision/recall 和 gap/pressure error 来确认。

### 3. Q4 / triangle split 口径不一致会影响压力和应力

Abaqus surface-to-surface 对 C3D8 面是四边形面片和 constraint region 口径。若 SFC 在某些路径中仍把面拆成三角形：

- 面积权重可能不同；
- active set 可能抖动；
- pressure peak 会偏；
- 应力/应变峰值会被放大或削弱。

因此 HEX8/C3D8 工况应优先走：

```text
Q4 slave quadrature
Q4 master closest-feature payload
Q4 area integration
```

而不是 triangle split 近似。

### 4. constraint averaging 仍未完全等价 Abaqus

Abaqus surface-to-surface 不是简单 node-to-surface，也不是独立 quadrature penalty 点求和。它在 secondary surface 周围区域施加平均意义的约束，接触方向也来自区域平均 normal。

SFC 当前的 `slave_node_region_participation` 已经比单点接触更接近 Abaqus，但仍可能缺失：

- secondary node-centered constraint region；
- region area；
- region average normal；
- region-level gap；
- active/release 与 Newton iteration 的耦合；
- pressure smoothing / nodal output semantics。

这会解释“位移曲线较近，但应力/应变和 pressure trend 不一致”的现象。

### 5. active set 接受准则还不是 Abaqus severe-discontinuity 口径

Abaqus/Standard 接触不是每步只重建一次 active set。它会处理 open-to-closed / closed-to-open severe discontinuity，并检查 penetration/contact force compatibility。

SFC 当前已有 active signature 和 residual 收敛诊断，但仍不是完整 Abaqus-style：

```text
trial contact status
-> severe-discontinuity loop
-> equilibrium Newton loop
-> contact compatibility residual
-> cutback/retry
-> accepted state
```

如果 active set 在接触建立和释放窗口提前接受，就会造成局部应力/应变峰值不对。

## 本轮增加的诊断修正

本轮没有改核心求解器，只改了一个验证脚本：

- `validation/run_source_gear_vtk_regional_alignment.py`

修正内容：

1. active precision/recall/Jaccard 的最小值只在真正有接触的 frame 上统计，避免无接触帧污染判断。
2. summary 增加 `Worst Active-Mask Overlap Frames`，直接列出 SFC-only / Abaqus-only active node。
3. summary 增加 `Stress/Strain Error Active-Set Context`，把最大应力/应变误差和同一时刻的 active overlap 放在一起。

这个改动的目的不是改变结果，而是更明确判断误差到底来自：

```text
候选漏检 / active recall 低
还是
SFC false active 多
还是
active set 基本对上但 pressure/stress 口径不一致
```

用已有 VTK manifest 重跑后处理：

```text
python validation/run_source_gear_vtk_regional_alignment.py \
  --sfc-manifest results/source_gear_secondary_line_limit_0004/sfc_vtk/sfc_manifest.csv \
  --abaqus-manifest results/source_gear_abaqus_penalty_full_stride2_match_step_0004/abaqus_vtk_contact_full_check/abaqus_contact_full_manifest.csv \
  --time-tolerance 1e-9 \
  --out-dir results/source_gear_secondary_line_limit_0004/regional_alignment_contact_audit
```

关键结果：

| time | SFC active | Abaqus active | precision | recall | Jaccard | 诊断 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.00028 | 161 | 549 | 0.695652 | 0.204007 | 0.187291 | Abaqus active 区域大量漏检，对应 p95 应力/应变最大误差 |
| 0.00030 | 160 | 504 | 0.537500 | 0.170635 | 0.148789 | active recall 很低，对应 full-field p95 误差峰值 |
| 0.00034 | 163 | 241 | 0.226994 | 0.153527 | 0.100817 | 既有漏检，也有 SFC-only false active |
| 0.00038 | 86 | 0 | 0.000000 | 0.000000 | 0.000000 | Abaqus 已释放，SFC 仍保留 false active |
| 0.00040 | 248 | 44 | 0.177419 | 1.000000 | 0.177419 | SFC 过度保持 active，压力/penetration 口径不一致 |

最大应力/应变误差与 active set 错配直接同帧出现：

- `abaqus_active/strain_norm_nodeavg` 最大 p95 误差 `88.255%` 出现在 `t=0.00028`，active recall 只有 `0.204007`。
- `full/von_mises_nodeavg` 最大 p95 误差 `51.462%` 出现在 `t=0.00030`，active recall 只有 `0.170635`。

这说明齿轮当前误差不是“位移积分已经坏掉”或“材料刚度单纯不对”，而是接触建立/释放窗口的 active set 和 pressure/penetration 口径没有对齐。

## 本轮通用修复

本轮进一步修改了：

- `validation/run_flexible_gear_implicit_lagrangian_sdf_comparison.py`

修复内容：

```text
secondary-normal line projection 的候选半径
>= closest-feature broad-phase 半径
```

原因是 secondary-normal line projection 的 `search_radius` 只负责枚举可能主面片，不负责最终接触接受。最终接受仍由：

```text
mesh-derived line-distance limit
hard finite-sliding gate
global closest-feature closed/open guard
```

决定。因此扩大候选枚举不会降低几何精度，也不会改变最终 gap/normal/payload 的定义；它只避免真实 Abaqus-active 主面片在粗检测阶段被提前过滤。

当前全齿轮默认数值变为：

```text
closest-feature radius  = 0.0027711314770112426
secondary tracking radius = 0.0027711314770112426
secondary line limit = 0.0008633433402562334
secondary hard line limit = 0.0017266866805124668
```

也就是说，候选 tube 更保守，但最终 line-hit 接受 tube 仍保持原 mesh-derived 精度口径。

## 下一步定位顺序

建议按以下顺序继续，不要先调参数：

1. 用新区域诊断重跑已有 SFC/Abaqus VTK 对比，查看最大应力/应变误差帧的 active precision/recall/Jaccard。
2. 若 recall 低，优先修 broad candidate radius / finite-sliding search tube / active patch cache，而不是改 SDF gap。
3. 若 precision 低，优先修 secondary-line release / remote facet rejection / contact status acceptance。
4. 若 active overlap 已经高但压力误差大，修 surface-to-surface constraint region averaging、Q4 area integration 和 pressure smoothing。
5. 若 pressure 对齐但应力/应变仍偏，修 Abaqus-style stress/strain nodal averaging 和 RP/MPC 虚功口径。

## 最终判断

你的怀疑是合理的：当前误差很可能有一部分来自接触检测和接触计算，尤其是 active set 生命周期、secondary-line constrained projection 和 surface-to-surface averaging，而不是 SDF 查询精度本身。

但不能回退到 `legacy` 的“采样点 SDF 阈值检测 + 插值梯度法向 + 简化 penalty force”。那条路缺少 closest-feature payload 和一致 Jacobian，理论上无法支撑当前论文目标。正确路线是把稀疏 ADF 作为候选加速层，把最终接触约束继续放在当前构型 closest-feature / secondary-normal constrained projection 和 surface-to-surface quadrature 上。
