# Phase-13 正式实验与图件交接

## 1. 论文 claim 口径

当前稿件不再把后端描述成只有 surface-to-surface。正式口径是：

- `node_to_surface_field_penalty_response`：保留的 node/sample-to-surface 接触积分路径；
- `surface_to_surface_field_penalty_response`：参考 surface-to-surface 三角形积分路径；
- `surface_to_surface_field_penalty_response_vectorized`：论文对齐 CalculiX native contact 的向量化 surface-to-surface 路径；
- dynamic narrow-band SDF field 仍然是主方法，projection 只用于 field construction、显式 refinement 或 timing/reference。

因此主线应写成：

```text
dynamic SDF field + closest-feature sensitivity
+ node-to-surface / surface-to-surface field contact backends
+ optimized field construction/query backend
```

而不是只写：

```text
surface-to-surface contact backend
```

## 2. 已运行的正式实验

### True dynamic SDF field

命令：

```bash
python validation/run_true_dynamic_sdf_field_validation.py --out-dir results/true_sdf_final
```

关键输出：

- `results/true_sdf_final/field_accuracy.csv`
- `results/true_sdf_final/field_eikonal.csv`
- `results/true_sdf_final/field_contact_jacobian.csv`
- `results/true_sdf_final/field_timing.csv`
- `results/true_sdf_final/field_crossover.csv`
- `results/true_sdf_final/true_dynamic_sdf_summary.md`
- `results/true_sdf_final/figures/*.png`
- `results/true_sdf_final/figures/*.pdf`

关键结果：

| spacing | max phi error | max normal error | max Eikonal residual | Q* |
| ---: | ---: | ---: | ---: | ---: |
| 0.100 | `1.835893e-03` | `5.751473e-02` | `6.784055e-03` | `822` |
| 0.075 | `1.132281e-03` | `4.769023e-02` | `4.546733e-03` | `1669` |
| 0.050 | `5.114950e-04` | `3.600289e-02` | `4.743522e-03` | `4544` |

### 三维场云图与 SDF 可视化

命令：

```bash
python validation/run_external_visual_validation.py --out-dir results/external_visual
```

关键输出：

- `results/external_visual/figures/external_visual_fields.png`
- `results/external_visual/figures/external_visual_fields.pdf`
- `results/external_visual/figures/dynamic_sdf_field_visualization.png`
- `results/external_visual/figures/dynamic_sdf_field_visualization.pdf`
- `results/external_visual/vtk/*.vtu`

说明：

- 当前图是连续三维边界面渲染，不是二维点云。
- SFC/reference/error 三列使用匹配色标。
- 字体通过 Matplotlib Times-style serif 配置生成。

### Independent CalculiX native contact vs SFC true-field contact

命令：

```bash
python validation/run_fuzzycontact_calculix_native_contact_trajectory.py --case problem_1 --sfc-backend surface_to_surface_vectorized --quadrature-order 7 --sdf-spacing-scale 0.25 --sfc-max-iterations 4 --out-dir results/native_contact_formal_problem1 --timeout 900
python validation/run_fuzzycontact_calculix_native_contact_trajectory.py --case problem_3 --sfc-backend surface_to_surface_vectorized --quadrature-order 7 --sdf-spacing-scale 0.25 --sfc-max-iterations 4 --out-dir results/native_contact_formal_problem3 --timeout 900
python validation/run_fuzzycontact_calculix_native_contact_trajectory.py --case problem_4 --sfc-backend surface_to_surface_vectorized --quadrature-order 7 --sdf-spacing-scale 0.25 --sfc-max-iterations 4 --out-dir results/native_contact_formal_problem4 --timeout 1200
```

关键结果：

| problem | max disp err | max strain err | max stress err | max VM err | SFC time | CalculiX time | CCX/SFC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| problem 1 | `1.380272e-02` | `1.805590e-02` | `1.630772e-02` | `1.599641e-02` | `1.870 s` | `2.872 s` | `1.536` |
| problem 3 | `2.301961e-02` | `3.961379e-02` | `3.697091e-02` | `3.304457e-02` | `0.628 s` | `3.778 s` | `6.015` |
| problem 4 | `6.558312e-04` | `2.191263e-03` | `1.725421e-03` | `1.294581e-03` | `12.678 s` | `9.343 s` | `0.737` |

结论：

- 三个场景全部通过 `5%` native solver agreement gate。
- problem 1 和 problem 3 中 SFC 更快。
- problem 4 中 SFC 慢于 CalculiX，原因是 field update 成本占主导；论文不能写“所有场景都更快”。

### Solver-level contact-dominated timing

命令：

```bash
python validation/run_true_field_solver_timing.py --out-dir results/true_field_solver_timing_formal
```

关键输出：

- `results/true_field_solver_timing_formal/solver_step_timing.csv`
- `results/true_field_solver_timing_formal/solver_crossover.csv`
- `results/true_field_solver_timing_formal/figures/solver_step_timing_breakdown.png`
- `results/true_field_solver_timing_formal/figures/solver_step_speedup.png`
- `results/true_field_solver_timing_formal/figures/solver_query_crossover.png`

结果：

- `16/16` accuracy rows passed。
- `16/16` total-step acceleration rows passed。
- projection/field total-step speedup 范围约为 `7.51x` 到 `24.17x`。
- 最小 measured query crossover \(Q^\ast\)：`24.610`。

### Backend ablation

命令：

```bash
python validation/run_native_contact_backend_ablation.py --quick --case problem_1 --quadrature-order 7 --sdf-spacing-scale 0.25 --sfc-max-iterations 4 --out-dir results/native_contact_backend_ablation_quick
```

关键输出：

- `results/native_contact_backend_ablation_quick/backend_ablation.csv`
- `results/native_contact_backend_ablation_quick/figures/backend_ablation_runtime.png`
- `results/native_contact_backend_ablation_quick/figures/backend_ablation_endpoint_metrics.png`

说明：

- 该 ablation 是代表性工程分层图，不替代 non-quick 正式速度 claim。
- non-quick 全四后端 ablation 的瓶颈是非向量化 reference surface-to-surface 路径，20 分钟内未完成。
- 正式速度 claim 仍以 `results/true_field_solver_timing_formal` 和三个 native-contact formal case 为准。

## 3. 稿件更新

已更新：

- `paper/main.tex`
- `paper/manuscript.md`

更新内容：

- 贡献点改为同时包含 node-to-surface 和 surface-to-surface 后端。
- 实现章节列出三个 field-contact backend。
- timing 表更新为最新 non-quick 结果。
- 新增 native CalculiX/SFC solver comparison。
- 新增 solver-level timing 和 backend ablation 小节。
- `paper/main.pdf` 已重新生成。

## 4. 当前支持的 claim

支持：

- true dynamic narrow-band SDF field 存在；
- field query 使用 interpolation；
- scalar field derivative 和 finite difference 一致；
- field-contact Jacobian 通过 slave/master FD；
- native CalculiX contact 与 SFC true-field contact 在三个报告场景中通过 `5%` gate；
- contact-dominated TET4/HEX8 static/dynamic timing 在测得 \(Q^\ast\) 后支持加速；
- 后端同时包含 node-to-surface 和 surface-to-surface 路径。

不支持：

- 所有场景都快于 CalculiX；
- friction；
- self-contact；
- nonlinear FEM 作为主方法；
- GPU；
- barrier contact；
- production BVH superiority；
- arbitrary non-manifold global SDF。
