# Source gear Abaqus theory-alignment diagnosis

本文档记录当前齿轮 `Abaqus/Standard native contact` 与 `SFC Lagrangian-SDF` 对比中应力/应变误差的定位结论。这里的 Abaqus 只作为外部理论与验证参考，不进入 `src/sfc` 核心求解依赖。

## 已定位并修复的问题

1. **secondary-line 远距离线交点误激活**
   - 现象：释放阶段 Abaqus `CPRESS=0`，但 SFC 仍存在负 gap 和高应力。
   - 原因：SFC 的 secondary-normal line projection 只用 broad phase 找候选面，一旦法向线与候选三角形相交，即使线向距离很大也会产生压缩约束。
   - 修复：新增 `secondary_line_distance_limit`，超出接受 tube 的线投影不再产生压缩罚力；默认距离由局部接触面代表尺寸和目标 overclosure 推导，不按单一曲线调参。
   - 覆盖测试：`secondary_normal_projection_releases_remote_line_intersection` 和 C++ batch 对应测试。

2. **`nlgeom=YES` 下非线性残差使用参考线性切线**
   - 现象：启用 finite StVK 内部力后，0.00040 s 出现应力爆炸，Newton residual 很大但状态被接受。
   - 原因：SFC 使用当前有限变形 StVK residual，但 Newton 线性化仍使用 `K_red` 参考刚度。
   - 修复：`finite_stvk_visual` 模式下装配当前 StVK tangent，并通过有限 RP visual Jacobian 映射到 reduced 坐标作为当前迭代切线。
   - 结果：0.00040 s 爆应力消失，末段 p95 node-averaged von Mises 误差降到约 `0.25%`。

3. **source-drive 非线性迭代控制不可配置**
   - 修复：新增 `--source-max-iterations`，用于固定步长 source-drive 对齐实验。

4. **缺少上一接受步锚点的 secondary finite-sliding tracking**
   - 现象：当前构型瞬时投影会在相邻齿面/邻近面片之间跳转；在中段接触滑移时，SFC 的 active contact samples 与压力分布过早收缩。
   - 理论依据：Abaqus finite-sliding surface-to-surface 接触使用 path-based tracking，而不是每个增量只做一次无历史的最近特征投影。
   - 修复：新增 `secondary_path_tracking`。每个 secondary 约束点保存上一接受步的 master face 与 barycentric anchor；下一步在该 face 及其邻域内优先更新 normal-line 约束，再回退到瞬时投影。该策略由接触路径历史和网格邻域驱动，不按某条误差曲线调参。
   - 覆盖测试：`secondary_path_tracking_keeps_previous_anchor_face` 和 batch 路径对应测试。

5. **line-distance gate 误裁剪真实闭合接触**
   - 现象：全齿面 `0.00020 s` 对比中，位移误差仍很小（约 `0.34%`），但 p95 node-averaged von Mises / equivalent strain 误差跳到约 `37.6%`。同一时刻 SFC active samples 从 `220` 掉到 `204`，法向力从 `191 N` 掉到 `116 N`，而 Abaqus active contact node count 继续增加。
   - 原因：`secondary_line_distance_limit` 最初直接把 `abs(gap)` 超出局部代表长度的负 gap 翻成正 gap。齿轮工况中默认 limit 为 `0.0008633`，而真实压入到 `0.00020 s` 已达到同一量级，因此有效接触约束被误释放。
   - 修复：line-distance gate 只释放“secondary line 命中远处面片，但真实 closest-feature 仍为分离”的伪接触；若 closest-feature 也已闭合，则保留 secondary-normal 线投影 gap。该修复同时覆盖标量 Python 路径和 C++ batch 路径。
   - 覆盖测试：`secondary_normal_projection_preserves_closed_line_intersection_past_limit` 和 `secondary_normal_projection_sample_arrays_preserve_closed_line_intersection_past_limit`。

6. **secondary tracking tube 过窄导致候选面提前丢失**
   - 现象：修复 line-distance gate 后，最小 gap 不再被截断，但 `0.00020 s` p95 应力误差仍约 `36.5%`，active samples 仍然偏少。
   - 原因：secondary-normal 路径的默认 contact search radius 仍为 `1e-4`，只对应初始间隙/接触分辨率，而不是 Abaqus finite-sliding surface-to-surface 的局部 constraint region。随着齿面切向滑移和压入增长，有效主面候选在进入精确 closest-feature 检查前已被 broad phase 裁掉。
   - 修复：默认 secondary tracking radius 改为 `max(初始法向包络, 局部接触面代表长度, 1e-4)`。在当前全齿面工况中，该值从 `1e-4` 增至 `0.0008633433402562334`。最终是否形成接触仍由 secondary-normal projection 与 closest-feature open-clearance release 决定，不降低几何精度。
   - 验证：`results/source_gear_tracking_radius_fullpatch_00020/sfc_vs_abaqus_history_metric_errors.csv` 中，`0.00020 s` 位移误差 `0.395%`，p95 node-averaged von Mises / equivalent strain 误差 `1.42%`，max node-averaged von Mises / equivalent strain 误差 `3.55%`，mean node-averaged stress/strain 误差 `1.98%`。接触压力 max / mean active 误差分别为 `7.89% / 4.04%`。

## 历史未通过的误差区间

以下结论对应修复第 5-6 点之前的旧实现。它们解释了本轮继续定位的出发点，不再代表当前 `0.00020 s` 验证结果。

完整 0.0004 s、旧版 `finite_stvk_visual + consistent tangent + secondary_line_distance_limit` 后：

- 位移最大误差：约 `3.21%`。
- 末段 p95 / max node-averaged von Mises 误差：约 `0.25% / 0.75%`。
- 但 `0.00020-0.00032 s` 中段 p95 stress 仍低估，最大约 `50.9%`。

关键对照：

- 关闭 line-distance gate 只改善 0.00030-0.00032，不能解决 0.00020-0.00028。
- 关闭 `slave_node_region` 聚合几乎不改变中段误差。
- 使用 `calculix` 风格 TET4 mass 几乎不改变中段误差。
- 使用 master closest-feature normal 能改善 0.00020 s，但会损害 0.00016-0.00018 s，不是全局正确解。

## 理论层面的剩余差异

Abaqus/Standard 文档指出：

- surface-to-surface contact 在 slave node 附近有限区域内以平均意义施加约束；
- finite-sliding surface-to-surface 的默认 tracking 是 path-based；
- 有限滑移时 active contact constraints 的连接关系会随切向相对运动变化；
- surface-to-surface 约束通常涉及多个 slave nodes 和更多 master-side load-transfer nodes。

当前 SFC 仍是：

```text
current configuration
-> secondary/master projection
-> current sample payload
-> penalty force
```

还不是 Abaqus-style：

```text
previous accepted contact state
-> increment path tracking
-> anchor/constraint region update
-> slave-region average normal
-> master participation factors
-> pressure-overclosure enforcement
```

因此，当时的剩余误差应定位为 **finite-sliding path-based surface-to-surface contact tracking、constraint participation factors 与 tracking tube 口径尚未完全对齐**，不是 SDF gap 查询本身的错误。

本轮已经实现 path-based tracking 的第一层、participation 聚合、active-set stability 选项、closed-feature line gate 修正和局部 constraint-region tracking radius。当前已验证到 `0.00020 s`：位移、p95/max/mean 应力应变和接触压力指标均低于 10%。完整 `0.00040 s` 仍需用新实现重跑确认。

后续补充实现：

- `*_participation` 接触聚合模式：对线性 pressure-overclosure 样本组，用正压力权重形成 slave/master participation factors，并选择等效 area/gap，使聚合后的总法向力和罚能与原分布式样本一致。
- 数组级 `*_participation` 聚合路径：保留 C++/batch contact query，不再为了 surface-to-surface 约束区域聚合退回逐对象 ContactSample 路径。
- `--source-contact-active-set-stability`：source-drive 非线性迭代只有在 residual、correction 和 active contact signature 均满足条件时才提前接受；用于避免接触状态仍在跳变时误判为收敛。

## 当前结果文件

- `results/source_gear_contact_release_velocity_diagnostic.csv`
- `results/source_gear_contact_line_limit_diagnostic.csv`
- `results/source_gear_finite_stvk_consistent_tangent_0004/sfc_vs_abaqus_vtk_metric_errors.csv`
- `results/source_gear_finite_stvk_no_line_gate_00032/sfc_full_gear_lagrangian_sdf_history.csv`
- `results/source_gear_finite_stvk_master_closest_0002/sfc_full_gear_lagrangian_sdf_history.csv`

## 下一步应实现

1. 新增 path-based finite-sliding contact tracking workspace。
   - 已完成第一层：face/barycentric anchor tracking。
   - 已完成第二层：线性罚函数样本组的 participation factors 和 active-set stability gate。
   - 已完成第三层：participation factors 的数组级 batch 聚合，避免 source-drive 接触对齐路径退回 Python 对象装配。
   - 剩余：完成齿轮 0.00020/0.00032/0.00040 s 分阶段验收，并根据误差决定是否需要 C++ fused participation/tangent。
2. 每个 slave constraint 保存上一接受步的 anchor/main face、自然坐标、参与节点和状态。
3. 在当前增量内沿 slave constraint center 的路径更新 anchor，而不是只做当前构型瞬时投影。
4. 将 constraint participation factors 用于 master-side force/Jacobian 分配。
5. 用 Abaqus `CPRESS/COPEN/RF/U/S/LE` 逐步验收：
   - 先 0.00020 s；
   - 再 0.00032 s；
   - 最后完整 0.00040 s。

## 本轮验证

- 相关测试：`pytest -q tests\test_flexible_gear_implicit_lagrangian_sdf.py -k "array_participation or participation or active_signature or secondary_path_tracking or secondary_normal_projection or secondary_line_distance_limit or source_drive"`
- 结果：`24 passed, 38 deselected`。
- full gear 最短 smoke 对比在 240 s 限时内未完成，原因是当前路径触发 full gear finite-StVK tangent 和大规模接触诊断；它不作为精度验收证据。
