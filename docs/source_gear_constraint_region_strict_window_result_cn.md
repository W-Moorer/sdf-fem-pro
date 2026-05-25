# 全齿轮 constraint-region 严格同步窗口结果

本轮使用新的 `slave_node_region_constraint` 接触聚合口径重跑全齿轮 source-drive 同步窗口。

## 命令

第一次尝试使用当前全部默认值，包括放大的 secondary tracking radius。该运行在 20 分钟内没有完成第一步，也没有写出 history：

```text
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py ^
  --drive-mode source_inp ^
  --contact-mode penalty ^
  --duration 0.0001 ^
  --dt 1e-5 ^
  --active-faces-per-body 0 ^
  --history-frame-stride 2 ^
  --abaqus-vtk-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0004\abaqus_vtk_contact_full_check\abaqus_contact_full_manifest.csv ^
  --out-dir results\source_gear_constraint_region_strict_00010
```

结论：当前放大的 broad-phase tracking radius 在全齿轮第一步上候选量过大，仍需工程优化或分层候选缓存。该失败不作为精度证据。

随后为了隔离验证新的 `slave_node_region_constraint` 接触理论口径，保持此前严格同步窗口使用的搜索半径：

```text
--source-contact-search-radius 0.0001
```

最终运行命令：

```text
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py ^
  --drive-mode source_inp ^
  --contact-mode penalty ^
  --duration 0.0001 ^
  --dt 1e-5 ^
  --active-faces-per-body 0 ^
  --history-frame-stride 2 ^
  --source-contact-search-radius 0.0001 ^
  --abaqus-vtk-manifest results\source_gear_abaqus_penalty_full_stride2_match_step_0004\abaqus_vtk_contact_full_check\abaqus_contact_full_manifest.csv ^
  --out-dir results\source_gear_constraint_region_radius1e4_00010
```

## 输出

- `results/source_gear_constraint_region_radius1e4_00010/sfc_full_gear_lagrangian_sdf_history.csv`
- `results/source_gear_constraint_region_radius1e4_00010/sfc_vs_abaqus_history_metric_errors.csv`
- `results/source_gear_constraint_region_radius1e4_00010/full_gear_lagrangian_sdf_summary.csv`
- `results/source_gear_constraint_region_radius1e4_00010/full_gear_lagrangian_sdf_summary.md`

## 设置

```text
nodes/elements = 38884 / 176486
active contact faces = 11560 / 11560
duration = 0.0001 s
dt = 1e-5 s
increments = 10
history stride = 2
contact averaging = slave_node_region_constraint
contact direction = secondary_average
contact projection = secondary_line
pair order = gear2_slave
search radius = 0.0001
HHT alpha = -0.41421
```

## 精度结果

与 Abaqus VTK manifest 同步时间点比较，`0--0.0001 s` 窗口内最大误差：

| 指标 | 最大相对误差 | 时间 |
| --- | ---: | ---: |
| max displacement norm | 0.262238% | 0.0001 |
| p95 von Mises nodeavg | 1.103768% | 0.00004 |
| p95 equivalent elastic strain nodeavg | 1.103768% | 0.00004 |
| max von Mises nodeavg | 2.981877% | 0.00004 |
| max equivalent elastic strain nodeavg | 2.981877% | 0.00004 |

这说明在 10 步严格同步窗口内，位移、p95 应力和 p95 应变已经明显低于 10% 验收线。

接触压力输出仍需谨慎解释：

```text
mean active contact pressure nodeavg max rel. error = 16.190928%
```

压力误差仍高于 10%，说明 CPRESS/COPEN 输出语义和接触压力投影还需要继续对齐。当前结果支持位移/应力/应变曲线对齐，但不应声称全压力场已经完全对齐。

## 收敛和效率

```text
SFC wall time = 1011.514586 s
source converged steps = 1
source iteration-limit steps = 9
source unstable accepted steps = 0
source sparse CG count = 149
source sparse CG iterations = 22218
```

因此，当前结论是：

```text
精度窗口通过；
效率和非线性收敛尚未通过。
```

下一步不应继续改 SDF 几何查询，而应处理：

1. full gear broad-phase 候选缓存/剪枝，否则默认 tracking radius 过慢；
2. contact force increment convergence 和 cutback；
3. constraint-region CPRESS/COPEN 输出投影；
4. matrix-free / preconditioned contact tangent，减少 9/10 步 iteration-limit。
