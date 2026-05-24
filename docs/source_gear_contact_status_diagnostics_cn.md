# 源齿轮接触状态诊断记录

本文档记录 Abaqus/Standard 原生接触与 SFC Lagrangian-SDF 接触在全齿轮动力学工况中的逐步定位结果。Abaqus 只作为外部理论与验证参考，不进入 `src/sfc` 核心求解路径。

## 当前最佳结果

当前最佳 SFC 结果目录：

```text
results/source_gear_hardtube_default_00040
```

对应设置：

- full gear TET4: `38884` nodes / `176486` elements
- Abaqus reference: `results/source_gear_abaqus_penalty_full_stride2_match_step_0004`
- SFC step: `dt = 1.0e-5 s`, duration `0.00040 s`
- contact: Lagrangian-SDF secondary-normal line projection
- averaging: `slave_node_region_participation`
- finite-sliding hard tube: `1.7266866805124668e-3`

诊断输出：

```text
results/source_gear_hardtube_contact_status_diagnostics
```

关键结果：

- release-stage mismatch frames: `0`
- final p95 node-averaged von Mises error: `0.71%`
- final p95 equivalent strain error: `0.71%`
- full-window max p95 stress/strain error: `28.69%` at `t = 0.00030 s`

这说明此前 `t = 0.00038 s` 的巨大假接触释放错误已经被修复；当前剩余问题集中在中段 `0.00022-0.00032 s` 的接触压力分布和应力波幅值。

## 已修复的问题

### 1. 远端 secondary-normal line 伪捕获

旧路径允许 secondary-normal line 在局部 finite-sliding 区域之外命中远端主面，并且在 closest-feature 仍显示轻微闭合时继续产生压力。这导致 Abaqus 已释放接触时，SFC 仍保留大 active set。

修复：

- `LagrangianSDFSurfaceContactGeometry.secondary_line_hard_distance_limit`
- 默认由局部接触面特征长度和目标过闭合尺度导出
- 超出 hard tube 的 line hit 一律释放为正 gap

验证：

- 原始 `source_gear_abq_dt_current_settings_00040`: max p95 error `195.32%`, release mismatch `1`
- 修复后 `source_gear_hardtube_default_00040`: max p95 error `28.69%`, release mismatch `0`

## 已排除的方向

### 1. 单纯放大 hard tube 不可用

验证目录：

```text
results/source_gear_hardtube3_00040
```

把 hard tube 从 `2x` 放宽到 `3x` 后：

- mid-contact 某些帧短暂改善
- release-stage mismatch 回到 `1`
- max p95 error 上升到 `108.52%`

结论：hard tube 不能继续靠放宽解决，否则会重新引入远端伪接触。

### 2. 按 Abaqus deck 行顺序反转主从面不可用

验证目录：

```text
results/source_gear_abaqus_slave_order_00040
```

将 SFC 改成 `gear1_slave` 后：

- max p95 error 从 `28.69%` 升到 `44.16%`
- release-stage mismatch 回到 `1`

结论：当前外部 Abaqus 输出和 SFC 几何法向/面定义在数值上更接近 `gear2_slave`；主从面顺序不是剩余误差主因。

### 3. HHT alpha 改成 `-0.05` 不可用

验证目录：

```text
results/source_gear_hht_tf_alpha_m005_00040
```

结果：

- max p95 error `128.75%`
- release-stage mismatch `1`

结论：当前 Abaqus 参考输出与 SFC 的 `alpha=-0.41421` 口径更接近；把 SFC 改成 transient-fidelity alpha 会破坏已有动态响应对齐。

### 4. signed-status participation 和 area-average 未改善中段误差

新增通用可选模式：

```text
slave_node_region_signed_participation
```

验证目录：

```text
results/source_gear_signed_status_participation_00040
results/source_gear_region_area_average_00040
```

结果：

- signed-status participation: max p95 error `50.57%`
- area-average: max p95 error `50.65%`

结论：剩余误差不是简单的区域 signed-average release 或面积平均法向即可解决。Abaqus 的 surface-to-surface penalty 还包含更细的压力平滑、约束区域权重和接触状态更新语义。

## 当前根因判断

目前可以较明确地排除：

- RP 角速度单位错误
- 全局位移/RP 运动错误
- SDF 距离查询本身错误
- 步长不一致
- 单纯 HHT alpha 不一致
- 单纯主从面反转
- 单纯 signed-average release

剩余误差最可能来自：

1. Abaqus surface-to-surface penalty 的接触压力平滑口径；
2. slave constraint region 的有效面积和权重分配；
3. active set 在接触峰值附近的区域连续性；
4. SFC 当前 participation 聚合对局部压力峰的权重仍偏强。

## 下一步

下一阶段不应继续调接触刚度或 SDF 查询。应专门实现并验证 Abaqus-style surface-to-surface penalty 的约束区域压力平滑：

1. 从局部 slave nodal constraint region 构造区域面积；
2. 对同一约束区域内的压力使用面积/形函数一致权重投影；
3. active status 使用区域连续性和 pressure-overclosure law 一致更新；
4. 保持现有 hard finite-sliding tube，防止远端伪捕获；
5. 每次修改后用 `run_source_gear_contact_status_diagnostics.py` 验证：
   - release-stage mismatch must remain `0`
   - full-window p95 stress/strain error must drop below `10%`
