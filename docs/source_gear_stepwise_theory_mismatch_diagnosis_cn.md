# 源齿轮 Abaqus/SFC 应力应变误差逐层定位

## 当前可复现实验

对比对象来自同一个源齿轮输入：

`commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp`

短程诊断设置：

- duration: `0.0002 s`
- dt: `1e-5 s`
- material: `E=2.05e11`, `nu=0.28`, `rho=7850`
- RP1 angular velocity: `52.36 rad/s`
- RP2 torque: `50`
- SFC: Lagrangian SDF surface contact
- SFC contact law: linear penalty, `k=5e9`

当前同口径最佳结果来自：

`results/source_gear_tensor_nodeavg_abaqus_face_order_probe_0002`

Abaqus linear penalty reference 来自：

`results/source_gear_abaqus_penalty_tensor_nodeavg_0002`

最终帧误差：

| 指标 | 相对误差 |
| --- | ---: |
| max displacement magnitude | `0.751%` |
| p95 node-averaged von Mises | `16.573%` |
| mean node-averaged von Mises | `8.343%` |
| max node-averaged von Mises | `5.308%` |
| object 1 p95 node-averaged von Mises | `27.361%` |
| object 2 p95 node-averaged von Mises | `10.577%` |
| RP2 rotation | `0.740%` |

结论：整体位移和 RP 运动已经基本对齐，但局部 p95 应力/应变仍未达到 `<10%` 验收线。

## 已修正或已排除的问题

### 1. Abaqus C3D4 surface face 顺序已修正

Abaqus C3D4 surface face 节点顺序为：

- S1: `1-2-3`
- S2: `1-4-2`
- S3: `2-4-3`
- S4: `3-4-1`

已将解析器从原来的集合等价三角面改为 Abaqus face 顺序，并增加测试：

`tests/test_flexible_gear_explicit_sdf_comparison.py::test_c3d4_surface_entries_use_abaqus_face_node_order`

修正后齿轮短程结果几乎不变，说明它不是当前误差主因，但它是必要的通用理论修正，因为 face normal 会影响接触符号、normal compatibility 和 surface-to-surface 投影语义。

### 2. C++ normal-compatible candidate fallback 已修正

新增并测试了 C++ batch kernel：

`closest_points_indexed_faces_normal_compatible`

它只在候选面内搜索 normal-compatible master face。如果候选区内不存在兼容面，或最近兼容面超过 fallback distance，则返回无效样本；Python 桥接层会过滤 `face_id < 0`，避免以前可能把无效样本误映射到最后一个 master face。

对应测试：

- `test_normal_compatible_sample_arrays_match_sample_payload_when_cpp_available`
- `test_normal_compatible_sample_arrays_drop_incompatible_cpp_candidates`

严格 normal-compatible 路径运行结果：

`results/source_gear_tensor_nodeavg_compatible_cpp_clipped_probe_0002`

| 指标 | 相对误差 |
| --- | ---: |
| max displacement magnitude | `6.541%` |
| p95 node-averaged von Mises | `19.524%` |
| mean node-averaged von Mises | `9.317%` |
| object 1 p95 node-averaged von Mises | `29.907%` |
| object 2 p95 node-averaged von Mises | `15.607%` |

结论：这个修正是必要的鲁棒性修正，但严格 normal-compatible 并不是齿轮误差的主修复路径。

### 3. HARD/LINEAR contact law 不是短程主因

已生成 Abaqus linear penalty deck：

`results/source_gear_abaqus_penalty_tensor_nodeavg_0002/gear_contact_source_penalty.inp`

其中使用：

`*Surface Behavior, pressure-overclosure=LINEAR`

并设定 `k=5e9`。在 `0.0002 s` 短程窗口内，它与原 HARD reference 的主要指标几乎一致。因此当前误差不能主要归因于 HARD/LINEAR 接触律差异。

### 4. 质量矩阵口径不是主因

使用 `tet4_mass_kind=calculix` 重跑：

`results/source_gear_tensor_nodeavg_calculixmass_probe_0002`

| 指标 | 相对误差 |
| --- | ---: |
| max displacement magnitude | `0.801%` |
| p95 node-averaged von Mises | `16.659%` |
| object 1 p95 node-averaged von Mises | `28.170%` |
| object 2 p95 node-averaged von Mises | `10.576%` |

与当前 consistent mass 基本一致，因此质量离散不是当前主因。

### 5. 应力输出口径基本已对齐

将 Abaqus VTK 位移场输入 SFC 的 `linear_corotated` 应力恢复后，得到的 p95 von Mises 与 Abaqus ODB 导出的 node-averaged stress 很接近：

| 指标 | SFC stress recovery on Abaqus U | Abaqus ODB S |
| --- | ---: | ---: |
| full p95 | `7.3349e6` | `7.3506e6` |
| object 1 p95 | `6.0866e6` | `6.1002e6` |
| object 2 p95 | `8.5654e6` | `8.5947e6` |

结论：不是单纯的“应力恢复公式/云图输出格式”导致误差。剩余误差来自 SFC 自己算出的局部位移梯度和接触载荷分布。

## 当前最明确的根因

新增 regional VTK 诊断后，最终帧结果为：

| region | metric | p95 rel. error | field RMSE rel. | correlation |
| --- | --- | ---: | ---: | ---: |
| full | displacement magnitude | `0.738%` | `0.717%` | `0.9999996` |
| full | von Mises nodeavg | `16.573%` | `27.591%` | `0.9329` |
| full | contact pressure nodeavg | `0.000%` | `49.515%` | `0.8734` |
| Abaqus active | von Mises nodeavg | `29.072%` | `47.043%` | `0.2425` |
| Abaqus active | contact pressure nodeavg | `0.246%` | `32.157%` | `0.7803` |
| active union | von Mises nodeavg | `27.811%` | `49.940%` | `0.3240` |
| active union | contact pressure nodeavg | `0.397%` | `49.515%` | `0.6075` |

active-set Jaccard overlap:

`0.7089`

这个结果说明：

1. 位移场整体几乎完全相关，但 active contact 区域的应力相关性很低。
2. 接触压力的 max/p95 可以看起来接近，但空间分布 RMSE 很大。
3. SFC 和 Abaqus 的 active contact patch 并没有完全落在同一片节点区域。
4. 因此当前主因是 surface-to-surface contact enforcement/output semantics 没有对齐，特别是 constraint-region averaging、pressure smoothing、active-set 边界和 master/slave 等效面积权重。

## 不是当前主因的方向

当前证据不支持继续优先修改：

- SDF 最近点查询精度；
- 单纯调接触罚刚度；
- 单纯改质量矩阵；
- 单纯改 stress/strain 云图输出；
- 单纯加阻尼或降低时间步精度。

这些方向不能解释“全局位移几乎对齐，但 active 区域压力/应力空间分布不对”的现象。

## 下一层应修复的理论实现

下一步应实现通用的 Abaqus-style surface-to-surface contact constraint-region averaging：

1. slave 侧用节点/区域约束区域，而不是独立三角采样点直接决定 active set。
2. pressure/overclosure 应在同一 constraint region 中做面积权重平均。
3. master 侧 payload 应按投影区域的等效 shape weights 回传，而不是只由局部最近三角样本决定。
4. active-set release/stick 应基于区域约束状态稳定性，而不是单个 sample 的瞬时 gap。
5. 诊断指标必须同时报告标量曲线和空间场指标：pressure RMSE、pressure correlation、active-set Jaccard、active-region stress RMSE。

这属于理论口径对齐，不是针对齿轮算例调参；修复后应先重跑 `0.0002 s` 短程齿轮窗口，确认 pressure spatial metrics 改善，再进入长时程和效率对比。
