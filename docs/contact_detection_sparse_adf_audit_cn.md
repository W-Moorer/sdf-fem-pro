# 接触检测与稀疏 ADF 审计报告

## 结论

结合 `legacy_abaqus_prototype/contact_sdf` 中的检测代码和当前 `src/sfc` 的 Lagrangian SDF / dynamic SDF 接触路径，当前齿轮算例的主要风险不在“有没有 SDF”这一层，而在下面三层：

1. 粗检测和细检测的角色边界必须更清楚：稀疏 ADF / hash SDF 适合做候选和近似距离预筛，不能作为最终接触 gap、normal、Jacobian 的唯一来源。
2. 细检测必须回到当前构型的 closest-feature / secondary-normal constrained projection，并携带 master payload，才能支撑接触力、切线和 master-side sensitivity。
3. 当前与 Abaqus 的剩余误差更像接触状态、constraint averaging、secondary-line release 语义不完全一致，而不是单纯 broad phase 漏检。

因此，师弟 legacy 代码对“检测链条”有参考价值，但不能直接迁移成主接触计算。它缺少 surface-to-surface 约束平均、接触状态收敛、master-side sensitivity 和一致切线语义。

## legacy 检测流程

入口主要在：

- `legacy_abaqus_prototype/contact_sdf/pre_processing.py`
- `legacy_abaqus_prototype/contact_sdf/contact_test.py`
- `legacy_abaqus_prototype/contact_sdf/gradient_sdf.py`
- `legacy_abaqus_prototype/contact_sdf/contact_force.py`

其流程可以概括为：

```text
Abaqus-derived DOF / displacement
-> current vertices
-> global AABB overlap
-> predicted SDF field
-> candidate face AABB overlap
-> barycentric surface samples
-> trilinear / Taylor SDF query
-> threshold: phi <= sdf_tol
-> deduplicate by rounded point
-> active_pairs
-> penalty force / approximate stiffness
```

### 粗检测

`make_contact_detector(...)` 中先把两体顶点变形到当前构型，然后做全局 AABB：

```text
min1/max1 with broad_tol
min2/max2 with broad_tol
if separated: no contact
```

`ContactDetectorSurface.detect(...)` 里又对采样侧每个三角形做 face AABB，与另一物体全局 AABB 比较，筛掉明显不可能接触的面片。

这个粗检测是合理的第一层筛选，但它只能判断“可能相交”，不能证明最终接触约束正确。

### 细检测

legacy 的细检测不是 current-surface closest point，而是：

1. 在采样侧三角形上生成 barycentric sample。
2. 对所有 sample point 批量查询另一侧 SDF。
3. 用 `phi <= sdf_tol` 判定接触。
4. 法向取 SDF 插值梯度。
5. 按 rounded position 去重，只保留 `abs(phi_other)` 更小的记录。

这里的关键漏洞是：它把 SDF 插值结果当成了最终细检测结果。对于接触力和切线来说，这还不够，因为 active sample 需要明确的 closest feature、barycentric payload、normal compatibility、面积权重和 master-side sensitivity。

## 稀疏 ADF / hashed SDF 理论风险

`gradient_sdf.py` 中的 `HashedGradientSDF` 是一种稀疏窄带存储：

```text
d(p) = psi_j + (p - v_j)^T grad_j
```

其中 `j` 是查询点所在 voxel 的中心。这个模型有明显用途：

- 用 hash/block 结构减少窄带 SDF 存储；
- 使查询近似 O(1)；
- 适合 broad phase 或 candidate scoring。

但它也有理论限制：

1. 它是 voxel-center Taylor 近似，不是严格的 closest-feature projection。
2. 对大转角、大剪切、曲率变化和边/角接触，单 voxel gradient 可能不能给出正确法向。
3. hash miss 或窄带外查询需要 fallback，否则会漏候选。
4. 只存 `psi` 和 `grad`，没有 closest face id、barycentric、master normal payload，不能直接给 `dg/dx_B = -N_B n^T`。
5. 如果用于最终 gap，会把“检测近似误差”直接传入接触力和应力场。

所以稀疏 ADF 可以是候选加速结构，但不能替代最终细检测。最终细检测必须从当前 FEM 表面重建真实 closest feature 或做 normal-tube constrained projection。

## legacy force path 的问题

`contact_force.py` 里也暴露了几个重要问题：

1. gap 符号和当前项目主约定不同。legacy 里使用 `g = -phi0`，当前 SFC 路径使用正 gap 表示分离、负 gap 表示穿透。
2. 代码注释承认 predictor-corrector 迭代中通常应加入：

   ```text
   g_current = -phi0 + dot(du_point - du_other, n)
   ```

   但实际保留了 `g = -phi0`。这会让步内迭代里的 gap 与位移修正脱钩。

3. 刚度矩阵符号和 residual 口径在注释中也不确定，不适合作为 Abaqus 对齐路径。
4. 该路径包含阻尼/摩擦型项，不符合当前无摩擦、线性罚函数或 hard contact 对齐阶段的最小理论闭环。

因此 legacy force path 只能作为历史参考，不能作为当前论文或核心后端的理论依据。

## 当前 SFC 路径的相对改进

当前项目已经比 legacy 路径更接近可发表/可验证接触后端：

- `src/sfc/sdf/dynamic_narrow_band_sdf.py`：当前构型 dynamic narrow-band SDF field，projection 只用于 field build，query 使用插值。
- `src/sfc/sdf/narrow_band_grid.py`：有 scalar phi 插值导数，能保证 slave Jacobian 与 `query_phi` 的导数一致。
- `src/sfc/sdf/material_sdf.py`：参考构型 material SDF / patch BVH。
- `src/sfc/contact/lagrangian_sdf_oracle.py`：refit 当前 AABB 后做当前构型 patch projection / KKT corrector。
- `src/sfc/contact/lagrangian_surface_contact.py`：surface samples、secondary-normal projection、contact payload、active aggregation 和 reduced Jacobian。

这说明当前问题不是“仍然只有 legacy 那种 SDF 阈值检测”。当前已经有更强的细检测结构。

## 当前仍存在的漏洞

### 1. secondary-line hard tube 的释放语义

在 source gear case 中，误差峰值出现在接触建立/释放窗口。Abaqus `.msg` 显示这段存在 open-to-closed / closed-to-open severe discontinuity iterations，并且之后会做 penetration/contact-force compatibility convergence。

当前 SFC 之前的 hard tube 逻辑存在一个通用漏洞：如果 secondary-normal line hit 距离超过 hard limit，就可能直接把负 gap 释放成正 gap。对于齿轮齿面这类有限滑移接触，如果全局 closest feature 仍然是 closed，这种释放会提前丢掉真实过闭合约束。

本轮已经把逻辑改成：

```text
只有当 secondary-line hit 超出 hard limit 且 global closest feature 是 open，
才释放该 line hit；
如果 line hit closed 且 closest feature 也 closed，则保留 overclosure。
```

这是通用修正，不是针对单个齿轮调参。

### 2. active set 生命周期未完全对齐 Abaqus

Abaqus/Standard 的 contact convergence 不是每步简单重算 active set 后求一次平衡。它会把接触开闭作为 severe discontinuity iteration 处理，并在 equilibrium iteration 里检查 contact compatibility。

当前 SFC 虽然已有 active signature、hard-contact residual 和若干 Newton 迭代，但还没有完整等价的：

- open/closed status trial；
- contact status severe-discontinuity loop；
- penetration/contact-force compatibility convergence；
- cutback / retry；
- active set 稳定后再接受增量。

这会导致曲线位移能对上，但应力/应变在接触状态切换窗口跳动。

### 3. constraint averaging 和压力场语义仍需对齐

Abaqus surface-to-surface contact 会做 constraint region averaging / smoothing。当前 SFC 里的 aggregation mode 已经向这个方向靠近，但仍可能在以下方面不一致：

- slave node participation 与 element-face pressure averaging 的口径；
- master-side closest payload 的 Q4/triangle split 口径；
- contact pressure 从 sample/face 到 node/element 的输出映射；
- RP reaction 与 contact traction 的等效虚功口径。

这些不一致会直接放大应力/应变误差，即使位移曲线看起来接近。

### 4. sparse ADF 不能作为最终 claims

如果论文或代码叙事把 hashed sparse ADF 写成最终接触计算，就会有理论漏洞。正确写法应是：

```text
sparse ADF / hash / BVH: candidate detection and acceleration
current-surface projection / normal-tube inversion: final gap, normal, payload
surface-to-surface quadrature: force and tangent integration
```

## 对当前齿轮误差的判断

结合本轮新增 `.msg` 解析诊断，Abaqus 在同一时间窗口内有大量 contact status iteration 和 compatibility convergence。当前 SFC 的 p95 stress/strain 最大误差约出现在 `t=0.00030` 附近，这正是接触状态变化窗口。

因此当前更可能的问题排序是：

1. contact status / active set 接受准则未完全按 Abaqus severe-discontinuity + compatibility semantics 实现；
2. secondary-normal line projection 的 release / hard-tube 策略仍需用 closest-feature closed/open 状态约束；
3. surface-to-surface constraint averaging 和 pressure smoothing 口径未完全一致；
4. RP reaction / stress output mapping 仍有输出语义差异；
5. broad phase 候选漏检不是首要矛盾，但需要用候选 miss audit 排除。

## 建议的下一步验证

不要继续只看最终位移曲线。应增加一个“粗检测/细检测审计”：

```text
for each accepted frame:
  1. collect all slave quadrature/sample points
  2. run current SFC broad candidates
  3. run sparse ADF candidate scoring if available
  4. run exact current-surface closest-feature reference
  5. compare:
     - candidate miss rate
     - false active rate
     - false open rate
     - gap max/mean error
     - normal angle error
     - closest face id stability
     - active set Jaccard vs Abaqus CPRESS/COPEN/CSTATUS
```

这应放在 `validation/` 下，不能把 `legacy_abaqus_prototype` import 到 `src/sfc`。

## 修复优先级

1. 保留稀疏 ADF / hash / BVH 为候选加速，不把它作为最终接触 law。
2. 对每个 active sample，最终 gap/normal/payload 必须来自当前构型 closest feature 或 secondary-normal constrained projection。
3. 补 Abaqus-style contact status loop：open/closed trial、severe discontinuity、compatibility residual、active set convergence、必要 cutback。
4. 统一 surface-to-surface quadrature 和 pressure averaging 口径，再比较应力/应变。
5. 最后优化 C++ fused candidate/projection/contact response；不要在理论未对齐时先优化速度。

## 本轮代码侧已完成的小修正

本轮已对 `src/sfc/contact/lagrangian_surface_contact.py` 做了一个通用修正：

- secondary-line hard limit 不再无条件释放远距离 line hit；
- 只有当 global closest feature 也是 open 时才释放；
- 如果 line hit 和 closest feature 都是 closed，则保留该 overclosure。

对应测试覆盖：

- scalar secondary-normal projection；
- batch secondary-normal projection；
- open closest feature 不产生伪 overclosure；
- closed line hit past hard limit 仍保留。

这个修正不能单独保证齿轮全流程误差进入 10%，但它排除了一个明确的“真实 closed contact 被 hard tube 提前释放”的细检测漏洞。
