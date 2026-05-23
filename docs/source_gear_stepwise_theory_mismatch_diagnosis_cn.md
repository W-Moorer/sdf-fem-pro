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

## 本轮新增定位：secondary surface normal 是主导理论差异

Abaqus surface-to-surface contact 的接触方向来自 secondary surface constraint region 的平均法向，而不是简单使用 main/master surface 的 closest-feature normal。SFC 原先使用 master SDF normal 作为接触方向，这会让整体 RP/位移基本对齐，但局部接触压力方向和齿根应力分布偏离。

新增通用选项：

- `--source-contact-direction master`
- `--source-contact-direction secondary_average`
- `--source-contact-projection closest_feature`
- `--source-contact-projection secondary_plane`
- `--source-contact-projection secondary_line`

其中 `secondary_average + closest_feature` 表示：候选和 gap 仍来自 closest-feature SDF，但接触力/Jacobian 方向改为 secondary surface 平均法向。这个不是调参，而是 Abaqus surface-to-surface 接触方向口径对齐。

短程齿轮结果：

| 设置 | full displacement p95 error | full von Mises p95 error | active-union von Mises p95 error | pressure active-union RMSE |
| --- | ---: | ---: | ---: | ---: |
| master normal + slave_node | `0.738%` | `16.573%` | `27.811%` | `49.515%` |
| secondary_average + slave_node_region | `0.348%` | `10.161%` | `13.879%` | `49.173%` |
| secondary_average + secondary_plane gap | `0.225%` | `8.528%` | `38.131%` | `110.612%` |

结论：

1. secondary surface normal 是当前最大有效修正，能把 full-field p95 von Mises 从 `16.573%` 降到 `10.161%`。
2. 单纯把 master-normal gap 除以 `n_master dot n_secondary` 虽然让 full-field p95 低于 10%，但 pressure field 和 active-region stress 明显恶化，不能作为最终验收解。
3. 真正更接近 Abaqus 的做法应是沿 secondary normal 做 main surface line projection：只有投影点落在 main triangle 内时才使用 secondary-direction gap，否则回退 closest-feature gap。
4. 已实现 Python 版 `secondary_normal_projection_samples(...)` 并通过单元测试，但全齿轮 11,560 接触面短程运行超时；下一步需要把 secondary-line projection 下沉到 C++/candidate fused backend，避免 Python 候选循环成为瓶颈。

因此当前可接受的阶段性修复是 `secondary_average + closest_feature`，它证明理论方向正确但仍未完成 `<10%` 总体验收。下一步应实现 C++ secondary-line projection，并继续以 pressure RMSE、pressure correlation、active-set Jaccard 和 active-region stress error 作为硬指标。

## 本轮新增验证：C++ secondary-line projection 不是最终可接受解

本轮已把 Python 参考版 `secondary_normal_projection_samples(...)` 下沉为 C++ indexed projection kernel：

- C++ kernel：`closest_points_indexed_faces_secondary_normal(...)`
- Python wrapper：`closest_points_indexed_faces_secondary_normal(...)`
- field/contact 几何入口：`secondary_normal_projection_sample_arrays(...)`
- source gear runner 入口：`--source-contact-projection secondary_line`

短程齿轮窗口验证：

```text
python validation/run_flexible_gear_full_lagrangian_sdf_comparison.py \
  --out-dir results/source_gear_tensoravg_secondary_line_cpp_probe_0002 \
  --active-faces-per-body 11560 \
  --duration 0.0002 \
  --dt 0.0001 \
  --drive-mode source_inp \
  --pressure-stiffness 5.0e9 \
  --contact-mode penalty \
  --history-frame-stride 1 \
  --write-sfc-vtk \
  --vtk-frame-stride 1 \
  --source-stress-postprocess linear_corotated \
  --source-contact-averaging slave_node_region \
  --source-contact-kinematics finite_rp_corotated \
  --source-contact-normal-filter opposing \
  --source-contact-direction secondary_average \
  --source-contact-projection secondary_line \
  --source-contact-pair-order gear2_slave \
  --source-internal-kinematics corotated_rp \
  --source-rotating-inertia finite_kinematic \
  --abaqus-vtk-manifest results/source_gear_abaqus_penalty_tensor_nodeavg_0002/abaqus_vtk/abaqus_manifest.csv
```

结果：

| 设置 | SFC wall time | full displacement p95 error | full von Mises p95 error | active-union von Mises p95 error | pressure active-union RMSE | active-set Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secondary_average + secondary_line C++ | `71.095 s` | `29.343%` | `7.382%` | `37.407%` | `45.329%` | `0.808` |

这个结果不能作为最终修复，原因是：

1. C++ 下沉解决了 Python 超时问题，但没有解决理论误差。
2. full-field von Mises p95 虽然降到 `<10%`，但 displacement p95 升到 `29.343%`。
3. active-union von Mises p95 升到 `37.407%`，说明接触区应力趋势更差。
4. pressure RMSE 仍在 `45%` 量级，说明压力空间分布仍未与 Abaqus 对齐。

因此当前定位更加明确：误差不应继续通过改变 SDF 查询或 gap 投影来修；下一层应对齐 Abaqus surface-to-surface enforcement 的 constraint-region pressure/overclosure averaging、active status averaging 和等效面积权重。`secondary_line` 只能作为诊断和后续可选 projection kernel，不能作为验收路径。

## 本轮新增验证：constraint-region averaging 口径

新增两个通用 averaging 口径用于定位 Abaqus-style surface-to-surface enforcement：

- `*_constraint`：先对 constraint region 做 signed gap 平均，再决定 active 状态。
- `*_area_average`：保持正穿透面积积分，但 normal / slave support / master payload 使用区域面积平均，而不是穿透量加权。

短程齿轮窗口结果：

| 设置 | full displacement p95 error | full von Mises p95 error | active-union von Mises p95 error | pressure active-union RMSE | active-set Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| secondary_average + slave_node_region | `0.348%` | `10.161%` | `13.879%` | `49.173%` | `0.709` |
| secondary_average + slave_node_region_constraint | `30.019%` | `9.257%` | `53.979%` | `49.274%` | `0.757` |
| secondary_average + slave_node_region_area_average | `29.985%` | `8.986%` | `54.025%` | `54.960%` | `0.732` |
| secondary_average + opposing_search + slave_node_region | `30.287%` | `7.471%` | `37.891%` | `54.105%` | `0.786` |

这些结果说明：

1. signed-average 或 area-average support 会让 full-field p95 von Mises 看起来低于 10%，但同时造成位移误差约 30%，active contact 区域应力和 pressure RMSE 明显恶化。
2. opposing-search 候选筛选也提高了 active-set overlap，但破坏了整体位移和接触载荷。
3. 因此当前最可信的阶段性路径仍是 `secondary_average + closest_feature + slave_node_region`。
4. 剩余误差更可能来自 Abaqus 内部 contact pressure smoothing / constraint enforcement stiffness / active status transition 的组合口径，而不是简单的 signed gap averaging 或候选筛选。

下一层不应接受这些“全场 p95 过线但 active 区错误”的模式。应继续定位 Abaqus penalty enforcement 的等效接触刚度和区域压力平滑：用一个最小法向压入 benchmark 先对齐 RF--penetration 曲线，再把该 enforcement 口径接回齿轮工况。

## 最小 LINEAR penalty 法向压入 benchmark

本轮新增：

- `validation/run_abaqus_linear_penalty_normal_indentation.py`
- `tests/test_abaqus_linear_penalty_normal_indentation.py`

该 benchmark 用两个 C3D8 block 只定义接触面：

1. 下块所有节点固定；
2. 上块所有节点做刚性法向位移；
3. 接触使用 Abaqus `*Surface Behavior, pressure-overclosure=LINEAR`；
4. 因为两块体都不发生体变形，所以理论关系应为

```text
F_n = k_p A max(closure - initial_gap, 0)
```

运行命令：

```text
python validation/run_abaqus_linear_penalty_normal_indentation.py \
  --out-dir results/linear_penalty_normal_indentation_abaqus \
  --run-abaqus \
  --timeout 300
```

结果：

| 指标 | 数值 |
| --- | ---: |
| pressure stiffness | `5.0e9` |
| initial gap | `0.02` |
| contact area | `1.0` |
| closures | `0, 0.01, 0.02, 0.025, 0.03` |
| SFC wall time | `0.010497 s` |
| Abaqus analysis wall time | `10.363180 s` |
| Abaqus export wall time | `0.396958 s` |
| max RF / normal-force relative error | `0.000000%` |
| max CPRESS relative error | `0.000000%` |

关键逐步结果：

| closure | penetration | Abaqus force | SFC force | Abaqus CPRESS | SFC CPRESS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.025` | `0.005` | `2.5e7` | `2.5e7` | `2.5e7` | `2.5e7` |
| `0.030` | `0.010` | `5.0e7` | `5.0e7` | `5.0e7` | `5.0e7` |

结论：

1. SFC 的 Lagrangian-SDF Q4 payload penalty force 积分与 Abaqus `pressure-overclosure=LINEAR` 的局部 RF--penetration / CPRESS 口径已经完全一致。
2. 因此齿轮短程窗口中剩余的 stress/strain/pressure 空间分布误差不应继续归因于线性 penalty 刚度单位或基本力积分公式。
3. 下一步应把定位集中到复杂齿面上的 active constraint region 构造：Abaqus 在曲面齿面上的 secondary surface constraint region、pressure smoothing / status transition / 等效输出区域，与当前 SFC 的 `slave_node_region` 仍不完全等价。
