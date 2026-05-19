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
- 已完成 C++/Numba compiled kernel；
- 已完成 consistent tangent；
- 完整 1 s non-quick pressure-driven dynamic solve 已经快于 CalculiX。
