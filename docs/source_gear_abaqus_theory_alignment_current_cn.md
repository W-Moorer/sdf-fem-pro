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

## 当前仍未通过的误差区间

完整 0.0004 s、`finite_stvk_visual + consistent tangent + secondary_line_distance_limit` 后：

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

因此，剩余误差应定位为 **finite-sliding path-based surface-to-surface contact tracking 与 constraint participation factors 尚未实现**，不是 SDF gap 查询本身的错误。

## 当前结果文件

- `results/source_gear_contact_release_velocity_diagnostic.csv`
- `results/source_gear_contact_line_limit_diagnostic.csv`
- `results/source_gear_finite_stvk_consistent_tangent_0004/sfc_vs_abaqus_vtk_metric_errors.csv`
- `results/source_gear_finite_stvk_no_line_gate_00032/sfc_full_gear_lagrangian_sdf_history.csv`
- `results/source_gear_finite_stvk_master_closest_0002/sfc_full_gear_lagrangian_sdf_history.csv`

## 下一步应实现

1. 新增 path-based finite-sliding contact tracking workspace。
2. 每个 slave constraint 保存上一接受步的 anchor/main face、自然坐标、参与节点和状态。
3. 在当前增量内沿 slave constraint center 的路径更新 anchor，而不是只做当前构型瞬时投影。
4. 将 constraint participation factors 用于 master-side force/Jacobian 分配。
5. 用 Abaqus `CPRESS/COPEN/RF/U/S/LE` 逐步验收：
   - 先 0.00020 s；
   - 再 0.00032 s；
   - 最后完整 0.00040 s。

