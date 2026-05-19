# SDF 后端工程优化交接

日期：2026-05-19

## 本轮目标

在不降低接触几何精度、不改变 field interpolation 查询语义的前提下，降低大面积 surface-to-surface 动力学算例中的 SDF field update 开销。

## 已实现内容

1. 新增 `RequiredPointSDFWorkspace`
   - 文件：`src/sfc/sdf/dynamic_narrow_band_sdf.py`
   - 作用：对固定接触查询结构复用 narrow-band grid 数组、required trilinear grid nodes 和候选投影分组。
   - 查询端仍返回普通 `DynamicNarrowBandSDF` 对象，`field_contact` 仍只使用插值查询。
   - projection 仍只发生在 field construction 阶段。

2. 大面积动力学 runner 接入 workspace
   - 文件：`validation/run_large_area_dynamic_surface_contact.py`
   - controlled-motion 与 pressure-driven 两条 SFC runner 均改为：
     `RequiredPointSDFWorkspace.build(...) -> field_contact`
   - 输出 history/summary 中的 `field_path` 已同步更新。

3. projection payload 批量化小优化
   - 文件：`src/sfc/sdf/dynamic_surface_sdf.py`
   - 将最佳三角形法向从 Python list comprehension 改为批量 `_unit_triangle_normals(...)`。
   - 文件：`src/sfc/sdf/dynamic_narrow_band_sdf.py`
   - 将 grid-node normal 归一化改为 `_unit_rows(...)`，减少 Python 循环和临时对象。

4. 可调 batch threshold
   - `DynamicSurfaceConfig.sdf_batch_projection_threshold`
   - CLI：`--sdf-batch-projection-threshold`
   - 当前 quick/non-quick 默认值：`250000`，用于在 all-faces batch 和 candidate-group batch 之间切换。

## 精度保持检查

命令：

```bash
python validation/run_large_area_dynamic_surface_contact.py --quick --skip-calculix --out-dir results/large_area_pressure_dynamic_quick_optimized --band-radius 2.0 --spacing 0.35
```

与优化前 `results/large_area_pressure_dynamic_quick` 对比：

| 量 | 最大绝对差 |
| --- | ---: |
| upper mean z displacement | `7.666143856e-08` |
| lower top mean z displacement | `1.259636854e-08` |
| min gap | `1.618082707e-07` |
| max penetration | `1.618082707e-07` |
| normal force | `6.037104808e-02` |
| contact energy | `9.626351986e-03` |

这些差异处于浮点路径和候选批处理顺序引起的数值噪声范围内；接触 gap、穿透、位移和力学响应没有方法层面的改变。

## 速度结果

quick pressure-driven large-area dynamic case：

| 版本 | complete SFC wall time | mean field update | mean field query/contact |
| --- | ---: | ---: | ---: |
| 优化前 | `1.166161e+01 s` | `4.231009e-01 s/step` | `2.937867e-03 s/step` |
| 本轮优化后 | `7.378983e+00 s` | `2.499411e-01 s/step` | `3.151644e-03 s/step` |

相对优化前：

- 完整 SFC quick wall time 降低约 `36.7%`。
- field update 均值降低约 `40.9%`。
- field query/contact assembly 基本不变，因为本轮主要优化 field construction。

## 测试

命令：

```bash
pytest -q
```

结果：

```text
303 passed in 646.60s (0:10:46)
```

新增/更新测试：

- `tests/test_dynamic_narrow_band_sdf.py::test_required_point_workspace_matches_build_required_points`
- `tests/test_large_area_dynamic_surface_contact.py::test_large_area_dynamic_surface_contact_quick_outputs`

## 尚未实现但建议继续做

1. consistent contact tangent
   - 当前仍是 fixed-point/Newmark 迭代结构。
   - 真正 consistent tangent 需要把 field-contact Jacobian 进入全局切线矩阵，并做有限差分回归；这会减少迭代次数，但属于求解器层改造。

2. 编译化 projection kernel
   - 当前仍是 NumPy/Python 批处理。
   - 下一步可用 Numba 或 C++/pybind11 实现同一 closest-feature projection kernel，保持几何结果一致但进一步降低 field update 时间。

3. accepted-state exact field reuse
   - 当前 fixed-point 最后一次 projection 对应 correction 前状态；若直接复用会改变 history 中 accepted-state gap 的严格语义。
   - 在 consistent tangent 或显式 accepted-state field 缓存完成后，才能无歧义地消除 record/iteration 之间的 accepted-state rebuild。

## 论文表述边界

本轮优化支持的表述是：

> The Python SDF backend reduces field-construction overhead by reusing required-grid-node workspaces and batch payload normalization, while preserving interpolation-only field queries and contact accuracy.

暂不应声称：

- 已完成 production-scale BVH；
- 已完成 C++ production compiled kernel；
- 已完成 consistent tangent；
- 完整 1 s non-quick pressure-driven dynamic solve 已经快于 CalculiX。

## 2026-05-19 追加优化

本轮继续优化的目标是：不改变接触 gap、payload、力装配和时间积分精度，只降低工程实现开销。

### 新增 compiled backend

1. SDF field-population projection
   - 新增：`src/sfc/sdf/_numba_projection.py`
   - 接入：`RequiredPointSDFWorkspace(projection_backend="auto")`
   - 默认可用时走 `compiled_padded_aabb`：
     - 对当前 boundary faces 计算 padded AABB；
     - 在编译循环中筛选候选面；
     - 对候选面执行同一 closest-feature projection；
     - 若某个 grid node 没有 AABB 候选，则回退 all-face exact search。
   - 该路径仍只用于 field construction，query path 仍是 interpolation-only。

2. required grid node 去重
   - `_required_trilinear_corner_indices(...)` 从 `np.unique(axis=0)` 改为 linear grid index 去重。
   - 同一 formal 尺寸查询下，该步骤从约 `0.11 s` 降到约 `0.005 s`。
   - required nodes 集合不变。

3. field-contact response
   - 新增：`src/sfc/contact/_numba_field_contact.py`
   - `surface_to_surface_field_penalty_response_vectorized(...)` 可用时使用 compiled gap interpolation + slave/master force accumulation。
   - 输出仍为 `FieldSurfaceContactResponse`，force/gap 与原 vectorized 路径一致。

4. 压力载荷缓存
   - pressure-driven runner 中 top-pressure 空间分布不变，仅压力标量随时间变化。
   - 已预计算 unit-pressure nodal force，每个迭代只乘当前 pressure。
   - 与逐次重新积分压力面力相比，位移、gap、force、energy 保持一致。

5. 线性求解缓存
   - Newmark effective matrix 在该线性动力学 runner 中固定。
   - 已改为 `scipy.sparse.linalg.factorized(...)`，迭代中复用 LU solve。

### 关键验证

测试：

```bash
pytest -q tests/test_field_surface_to_surface_contact.py \
  tests/test_large_area_dynamic_surface_contact.py \
  tests/test_dynamic_narrow_band_sdf.py::test_compiled_padded_aabb_projection_matches_all_faces_in_band
```

结果：

```text
7 passed
```

完整 1.0 s SFC 压力驱动大面积接触：

```bash
python validation/run_large_area_dynamic_surface_contact.py \
  --skip-calculix \
  --out-dir results/large_area_pressure_dynamic_1s_dt001_compiled \
  --total-time 1.0 \
  --dt 0.001 \
  --band-radius 8.0 \
  --spacing 0.35 \
  --nx 36 --ny 36 --nz 2 \
  --driver-nx 36 --driver-ny 36 \
  --frame-stride 50
```

结果：

| 指标 | 数值 |
| --- | ---: |
| complete SFC solve wall time | `1.910167e+02 s` |
| mean field update | `2.152489e-02 s/step` |
| sum record field update | `2.154642e+01 s` |
| mean field query/contact | `1.636655e-02 s/step` |
| sum record field query/contact | `1.638291e+01 s` |
| max active samples | `18144` |
| active steps | `957 / 1001` |

与旧 pressure-driven 1.0 s 结果对比：

| 版本 | complete SFC solve wall time |
| --- | ---: |
| 优化前 | `1.791773e+03 s` |
| 本轮优化后 | `1.910167e+02 s` |

提升约 `9.4x`，但仍未快于之前同场景 CalculiX native-contact wall time `5.362540e+01 s`。

### 当前瓶颈结论

现在已经不能再说主要瓶颈是记录步的 field update。完整 1.0 s case 中，recorded field update + recorded field query 约 `37.9 s`，而 complete SFC solve 为 `191.0 s`。差额主要来自每个时间步内部 fixed-point contact iteration 的重复 field build/query。

因此，若要在完整动态求解 wall time 上超过 CalculiX，下一步必须做：

1. consistent contact tangent，减少 fixed-point rebuild 次数；
2. C++/pybind11 或 C++ extension 版 spatial-hash + projection + contact accumulation，继续压低每次 iteration 的 field build/query；
3. accepted-state response/field reuse，但必须保证 accepted state 的 response 对应最终位移，不能复用 correction 前的近似响应。

### 当前不能声称

本轮后仍不能声称：

- 完整 1.0 s pressure-driven dynamic solve 已经快于 CalculiX；
- fixed-point iteration overhead 已经解决；
- 已完成 C++ production backend；
- 已完成 consistent tangent。
