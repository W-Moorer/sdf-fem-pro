# SDF 后端工程优化交接

日期：2026-05-19

## 本轮目标

在不降低接触精度、不改变 dynamic narrow-band SDF field 查询语义的前提下，继续降低大面积 surface-to-surface 动力学接触算例的完整 SFC solve time。用户明确要求进入下一层工程化：

- 真正 C++ fused extension，而不是仅依赖 Python/Numba wrapper；
- matrix-free contact tangent，不能显式装配全局 `Kc`；
- core solve timing 与 diagnostics/postprocess timing 分离；
- accepted-state response reuse，但 response 必须对应最终 accepted state。

## 已实现内容

1. 可选 C++ pybind11 后端
   - 新增：`src/sfc/_cpp_sdf_contact.cpp`
   - 新增：`src/sfc/sdf/_cpp_projection.py`
   - 新增：`src/sfc/contact/_cpp_field_contact.py`
   - 新增：`setup.py`
   - 更新：`pyproject.toml` 增加 build-time `pybind11`
   - 本地已通过 `python setup.py build_ext --inplace` 编译出 `sfc._sfc_cpp`。

2. C++ projection 与 C++ field-contact
   - C++ projection 提供 all-face 和 padded-AABB exact-fallback closest-feature projection。
   - C++ field-contact 提供 field interpolation gap、scalar-gradient slave force、payload-based master force 的 fused loop。
   - Projection 默认仍优先使用已有稳定并行 compiled backend；C++ projection 保留为显式可测后端，避免串行 C++ projection 使当前最快路径倒退。
   - Field-contact 查询/力装配默认优先使用 C++ 后端；失败时回退 Numba。

3. Matrix-free contact tangent
   - 新增：`FieldContactMatrixFreeStiffness`
   - `surface_to_surface_field_penalty_response_vectorized(..., matrix_free_stiffness=True)` 返回 matrix-free `J^T W J` 算子。
   - 算子与显式 `Kc = sum_i k A_i J_i^T J_i` 数值等价，但不装配全局 `Kc`。
   - 大面积动态 runner 新增 `--matrix-free-contact-tangent`，通过 SciPy `LinearOperator + CG + effective-matrix preconditioner` 求解。

4. Timing 拆分
   - `SFCDynamicResult` 新增：
     - `core_solve_seconds`
     - `diagnostics_seconds`
   - summary 现在同时报告：
     - complete SFC dynamic solve wall time；
     - core solve time；
     - diagnostics/accepted-response time。

5. Accepted-state response reuse
   - runner 不再把最后一次 correction 前的 response 当作 accepted response。
   - 每个 accepted state 都在最终位形处重新评估一次 response，然后下一帧 history 直接复用该 response。
   - 因此 `accepted_response_reused=true` 不改变 history 中 gap/force/energy 的定义。

## 验证结果

### 单元等价测试

命令：

```bash
python -m pytest -q tests/test_dynamic_narrow_band_sdf.py::test_cpp_projection_backend_matches_numpy_batch_when_built tests/test_field_surface_to_surface_contact.py::test_cpp_field_contact_backend_matches_vectorized_force_when_built tests/test_field_surface_to_surface_contact.py::test_vectorized_surface_response_matches_reference_force
```

结果：

```text
3 passed in 0.49s
```

补充局部测试：

```bash
python -m pytest -q tests/test_large_area_dynamic_surface_contact.py tests/test_field_surface_to_surface_contact.py tests/test_dynamic_narrow_band_sdf.py
```

结果：

```text
29 passed in 33.42s
```

### 1.0 s 大面积压力驱动动力学算例

命令：

```bash
python validation/run_large_area_dynamic_surface_contact.py \
  --skip-calculix \
  --out-dir results/large_area_pressure_dynamic_1s_dt001_cppcontact_reuse \
  --total-time 1.0 \
  --dt 0.001 \
  --band-radius 8.0 \
  --sdf-candidate-padding 1.0 \
  --spacing 0.35 \
  --nx 36 --ny 36 --nz 2 \
  --driver-nx 36 --driver-ny 36 \
  --frame-stride 50 \
  --newmark-iterations 3
```

结果：

| 指标 | 数值 |
| --- | ---: |
| complete SFC solve wall time | `1.118422e+02 s` |
| core solve time | `8.249577e+01 s` |
| diagnostics/accepted-response time | `1.900239e+01 s` |
| mean field update | `1.354637e-02 s/step` |
| mean field query/contact | `4.168860e-03 s/step` |
| quadrature samples per step | `18144` |
| max active samples | `18144` |
| accepted response reused | `1000 / 1001` frames |

与上一轮严格不降精度路径 `results/large_area_pressure_dynamic_1s_dt001_pad1` 对比：

| 指标 | 上一轮 | 本轮 |
| --- | ---: | ---: |
| complete SFC solve wall time | `1.567543e+02 s` | `1.118422e+02 s` |
| mean field update | `1.300819e-02 s/step` | `1.354637e-02 s/step` |
| mean field query/contact | `1.663970e-02 s/step` | `4.168860e-03 s/step` |

历史量逐点一致：

| 量 | 最大差异 |
| --- | ---: |
| upper mean z displacement | `0.0` |
| lower top mean z displacement | `0.0` |
| min gap | `0.0` |
| max penetration | `0.0` |
| normal force | `0.0` |
| contact energy | `0.0` |

结论：本轮严格路径相对上一轮从 `156.75 s` 降到 `111.84 s`，约 `1.40x`；相对最初 Python/NumPy 路径 `1791.77 s`，约 `16.0x`。

## Matrix-free tangent 结果

0.1 s formal-size 测试中，matrix-free tangent 与非 tangent 路径数值接近：

| 量 | 0.1 s final 对比 |
| --- | ---: |
| upper mean z displacement 差异 | `1.53e-09` |
| min gap 差异 | `3.43e-09` |
| normal force 差异 | `1.64e-03` |

但当前 Python `LinearOperator + CG` 实现更慢：

| 路径 | 0.1 s complete SFC wall |
| --- | ---: |
| C++ field-contact + fixed-point 3 iterations | `2.283296e+01 s` |
| matrix-free tangent 2 iterations | `5.066909e+01 s` |

结论：matrix-free tangent 的数学接口已经建立，且不显式装配全局 `Kc`；但当前 Python/CG 版本还不是性能路径。下一步若要让 tangent 成为加速路径，需要把 matrix-free matvec 和 Krylov/preconditioner 进一步下沉到 C++，或使用更合适的块预条件与收敛准则。

## CalculiX 对比状态

已有同一工况历史参考：

| 路径 | wall time |
| --- | ---: |
| CalculiX native contact reference | `5.362540e+01 s` |
| SFC 本轮严格路径 | `1.118422e+02 s` |

本轮尝试在新输出目录单独重跑 CalculiX，但 `ccx` 在 360 s 超时；因此不把这次超时结果用于性能 claim。当前严谨结论仍是：

- SFC 相对自身上一轮有明确提速；
- SFC 完整 1 s solve time 仍未快过已有 CalculiX 完成记录；
- 不能声称“完整 solve time 已经超过 CalculiX”。

## 当前可支持的论文表述

可以写：

> The optimized backend reduces the complete 1 s SFC pressure-driven dynamic solve from 156.75 s to 111.84 s without changing the displacement, gap, force, or contact-energy histories. The acceleration comes primarily from the C++ field-contact interpolation/force kernel and accepted-state response reuse for diagnostics.

不能写：

- 完整 SFC solve time 已经快于 CalculiX；
- matrix-free tangent 已经带来整体加速；
- C++ projection 是当前默认最快 field update 路径；
- production-scale BVH/C++ solver 已完成。

## 下一步要真正超过 CalculiX

1. 把 matrix-free tangent 的 matvec 和预条件求解继续下沉到 C++，避免 Python `LinearOperator` 调度和大数组构造。
2. 将 required-node generation、projection payload fill、field query、force accumulation 进一步合并成一个 C++ fused step，减少 Python 对象和数组往返。
3. 对 dynamic solve 区分论文 reporting：
   - 完整 solve wall time；
   - core solve time；
   - diagnostics/postprocess time。
4. 保持精度门槛：任何减少迭代、放宽 CG 容差、跳过 accepted-state reevaluation 的方案，都不能作为“不降精度”的默认路径。

## 2026-05-19 追加：C++ tangent matvec 与 deferred diagnostics

本轮继续推进两个不改变接触结果的工程化点。

### 已实现

1. C++ matrix-free contact tangent matvec
   - 新增 C++ 入口：`contact_stiffness_matvec(...)`
   - Python 包装：`src/sfc/contact/_cpp_field_contact.py`
   - `FieldContactMatrixFreeStiffness.matvec(...)` 在 C++ extension 可用时自动调用 C++ matvec。
   - 数学形式仍是同一个 `K_c v = J^T W J v`，没有显式装配全局 `Kc`。

2. Deferred diagnostics / postprocess
   - 新增配置：`DynamicSurfaceConfig.defer_diagnostics`
   - 新增 CLI：`--defer-diagnostics`
   - 开启后，求解循环不再为 history/VTK 额外重建 accepted-state response。
   - 接触诊断在求解结束后对保存的 accepted states 逐帧后处理生成，因此 gap、force、energy 历史量不变。

### 验证

针对性测试：

```bash
python -m pytest -q tests/test_field_surface_to_surface_contact.py::test_cpp_contact_stiffness_matvec_matches_reference_when_built tests/test_field_surface_to_surface_contact.py::test_vectorized_surface_response_matches_reference_force tests/test_large_area_dynamic_surface_contact.py::test_large_area_dynamic_surface_contact_deferred_diagnostics_outputs
```

结果：

```text
3 passed in 2.55s
```

0.1 s formal-size deferred diagnostics 与原路径对比：

| 量 | 最大差异 |
| --- | ---: |
| upper mean z displacement | `0.0` |
| lower top mean z displacement | `0.0` |
| min gap | `0.0` |
| max penetration | `0.0` |
| normal force | `0.0` |
| contact energy | `0.0` |

0.1 s matrix-free C++ matvec 与原 matrix-free 路径对比：

| 量 | 最大差异 |
| --- | ---: |
| upper mean z displacement | `2.78e-17` |
| lower top mean z displacement | `1.73e-18` |
| min gap | `2.48e-15` |
| max penetration | `2.48e-15` |
| normal force | `2.91e-11` |
| contact energy | `3.64e-12` |

### 1.0 s 大面积压力驱动动力学算例

命令：

```bash
python validation/run_large_area_dynamic_surface_contact.py \
  --skip-calculix \
  --out-dir results/large_area_pressure_dynamic_1s_dt001_deferdiag \
  --total-time 1.0 \
  --dt 0.001 \
  --band-radius 8.0 \
  --sdf-candidate-padding 1.0 \
  --spacing 0.35 \
  --nx 36 --ny 36 --nz 2 \
  --driver-nx 36 --driver-ny 36 \
  --frame-stride 50 \
  --newmark-iterations 3 \
  --defer-diagnostics
```

结果：

| 指标 | 数值 |
| --- | ---: |
| complete command wall including output/postprocess | `1.046569e+02 s` |
| SFC solve-loop wall excluding deferred diagnostics | `8.005022e+01 s` |
| inner correction solve time | `7.952349e+01 s` |
| diagnostics/postprocess time | `1.587470e+01 s` |
| mean field update | `1.092996e-02 s/step` |
| mean field query/contact | `3.465488e-03 s/step` |

与 `results/large_area_pressure_dynamic_1s_dt001_cppcontact_reuse` 历史量逐点一致：

| 量 | 最大差异 |
| --- | ---: |
| upper mean z displacement | `0.0` |
| lower top mean z displacement | `0.0` |
| min gap | `0.0` |
| max penetration | `0.0` |
| normal force | `0.0` |
| contact energy | `0.0` |

### 当前结论

Deferred diagnostics 将可用于论文中的工程求解器计时口径：

- 完整命令时间：`104.66 s`
- 求解循环时间：`80.05 s`
- 后处理/诊断时间：`15.87 s`

这一步没有改变接触结果，但仍未让 SFC solve-loop 快于已有 CalculiX `53.63 s`。剩余主要瓶颈仍是每个时间步内的重复 field rebuild/contact iteration。下一步若要继续压到 CalculiX 以下，需要把 tangent 迭代控制和预条件求解继续 C++ 化，而不仅仅是 C++ matvec。
