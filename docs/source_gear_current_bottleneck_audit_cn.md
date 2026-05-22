# source gear penalty 当前瓶颈审计

本文档记录 `commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp`
源齿轮工况在当前 SFC Lagrangian-SDF penalty 路径下的短程验证状态。

## 运行口径

- 几何、材料、TET4 网格、RP-MPC、RP1 角速度、RP2 力矩均来自源 `gear_contact.inp`。
- 只调整接触和积分口径：SFC 使用 frictionless linear penalty contact，HHT 隐式动力学对齐 Abaqus/Standard 口径。
- 时间步：`dt = 1.0e-5 s`。
- 当前短程验证：`duration = 1.0e-4 s`，共 10 steps。
- VTK：scalar-only，应力/应变云图隔步输出，`vtk_frame_stride = 2`。

命令：

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --duration 0.0001 `
  --dt 0.00001 `
  --pressure-stiffness 5e9 `
  --write-sfc-vtk `
  --vtk-frame-stride 2 `
  --vtk-scalars-only `
  --history-frame-stride 2 `
  --abaqus-vtk-manifest results\flexible_gear_source_penalty_abaqus_stride2_10steps_rpdiag\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_10steps_current
```

## 当前结果

结果目录：

```text
results/source_gear_penalty_10steps_current
```

SFC timing：

- complete wall time: `121.907727 s`
- residual/contact sampling time: `38.180717 s`
- sparse CG linear correction time: `68.540804 s`
- history stress/strain diagnostics time: `7.444579 s`
- VTK cloud output time: `7.586990 s`
- sparse CG solves: `30`
- sparse CG iterations: `16578`
- direct fallback count: `0`

VTK output：

- SFC VTK frame count: `6`
- frame stride: `2`
- scalar-only frame size: about `22 MB/frame`

末帧 SFC vs Abaqus VTK metric error：

- max displacement magnitude relative error: `4.071%`
- node-averaged von Mises relative error: `2.135%`
- node-averaged equivalent elastic strain relative error: `2.135%`
- RP1 prescribed rotation relative error: `3.39e-6%`
- RP2 rotation relative error: `3.738%`
- RP2 angular velocity relative error: `3.856%`

分对象误差：

- gear 1 displacement relative error: `0.090%`
- gear 1 node-averaged von Mises relative error: `2.135%`
- gear 2 displacement relative error: `4.071%`
- gear 2 node-averaged von Mises relative error: `0.339%`

## 已验证但未保留的优化尝试

以下尝试没有改变物理精度，但短程实测没有收益或变慢，因此没有保留到源码：

- C++/Python 外层的 static sampling template cache：2-step wall time 没有下降。
- 直接 reduced-force assembly：数学上等价，但 Python 累加路径比现有 full-force + sparse projection 慢。
- 3x3 block-Jacobi contact tangent preconditioner：CG iteration 略降，但预条件构造和应用成本更高。
- `np.savetxt` 批量 ASCII VTK writer：字段不变，但写出时间比现有逐行 writer 更慢。

## 已保留的有效改动

- SFC 应力/应变动画闪动已通过 corotated elastic postprocess 和 fixed color ranges 处理。
- VTK 使用 scalar-only 可显著降低单帧体积。
- VTK node-averaged scalar 计算已向量化，2-step VTK 写出时间由约 `2.42 s` 降至约 `1.16 s`。
- force-only penalty contact 只 scatter active samples 的非零接触力，数值指标保持不变。
- TET4 reference data 和 mass assembly 已向量化，setup 成本降低。

## 当前瓶颈判断

10-step 结果显示，完整求解 wall time 的主导项已经不是云图写出，而是：

```text
sparse CG linear correction: 68.54 s / 121.91 s
contact sampling + C++ closest projection: 38.18 s / 121.91 s
```

因此，继续优化必须进入更底层的工程路径：

1. C++ fused source-penalty contact step，减少 Python 往返和 repeated sample/projection 调度。
2. 更强的 C++/matrix-free 预条件求解器，降低 `source_sparse_cg_iterations`。
3. projection kernel 的 candidate pruning / active patch cache，但必须保留 exact fallback，不降低接触精度。
4. 将 diagnostics/VTK 与 core solve timing 分离报告；完整 wall time 仍可保留作为工程端到端成本。

## 更新：quadrature-point 级接触管裁剪

新增的 contact sampling 路径在已有 face-level conservative tube 之后，再对每个 slave
quadrature point 做一次保守 tube 检查：

```text
distance(point, nearest master face bounding sphere)
<=
search_radius + max_master_face_radius
```

若不满足该条件，该 quadrature point 不可能落入当前接触搜索管，因此不送入
closest-feature projection kernel。该裁剪不改变进入搜索管内点的 closest-point 计算、
gap、normal、payload 或 penalty law。

10-step 复测目录：

```text
results/source_gear_penalty_10steps_qp_cull
```

相同 Abaqus manifest 下的末帧误差变为：

- max displacement magnitude relative error: `1.195%`
- node-averaged von Mises relative error: `0.0736%`
- node-averaged equivalent elastic strain relative error: `0.0736%`
- RP2 rotation relative error: `1.149%`
- RP2 angular velocity relative error: `1.244%`

timing 变为：

- complete wall time: `96.933245 s`
- residual/contact sampling time: `23.988831 s`
- sparse CG linear correction time: `59.342516 s`
- history stress/strain diagnostics time: `7.326365 s`
- VTK cloud output time: `6.115524 s`
- sparse CG solves: `27`
- sparse CG iterations: `15090`

该结果同时改善了接触搜索成本和 Abaqus 对齐误差，说明原 face-level tube
对大三角 slave face 过宽，会把远离当前接触管的积分点送入 projection/contact
响应。点级 tube 裁剪更接近有限滑移接触的局部搜索语义。

## 更新：base-LU 预条件 sparse CG

source-drive penalty 路径中的动态基矩阵

```text
K_base = M / (beta dt^2) + (1 + alpha) K_ref
```

在固定时间步、固定 RP-MPC 约束下保持不变。因此当前实现对 free-DOF
`K_base` 做一次 sparse LU 分解，并将其作为 CG 的左预条件器来求解

```text
(K_base + J^T W J) delta = rhs
```

接触切线项 `J^T W J`、CG operator、残差验收和 tolerance 均保持不变；
如果 LU 分解失败，代码会回退到原来的 diagonal preconditioner。

10-step 复测目录：

```text
results/source_gear_penalty_10steps_base_lu_pc
```

timing 变为：

- complete wall time: `46.009660 s`
- residual/contact sampling time: `23.179764 s`
- sparse CG linear correction time: `5.242211 s`
- one-time base LU factorization time: `4.376137 s`
- history stress/strain diagnostics time: `7.150304 s`
- VTK cloud output time: `5.915829 s`
- sparse CG solves: `27`
- sparse CG iterations: `56`

相同 Abaqus manifest 下的末帧误差保持在同一水平：

- max displacement magnitude relative error: `1.195%`
- node-averaged von Mises relative error: `0.0736%`
- node-averaged equivalent elastic strain relative error: `0.0736%`
- RP2 rotation relative error: `1.149%`
- RP2 angular velocity relative error: `1.244%`

相比未使用 base-LU 预条件的 qp-cull 路径，10-step wall time 从
`96.93 s` 降至 `46.01 s`，CG iteration 从 `15090` 降至 `56`。

## 更新：source step 匹配入口和隔帧输出语义

`validation/run_flexible_gear_full_lagrangian_sdf_comparison.py` 现在提供更明确的
`--match-source-step` 入口，作为既有 `--use-source-timing` 的别名。启用后，SFC
会直接采用 `gear_contact.inp` 中 `*Dynamic` 行解析出的时间设置：

```text
initial dt = 1.0e-5 s
duration   = 5.0e-2 s
min dt     = 1.0e-10 s
max dt     = 5.0e-5 s
```

source-drive 路径仍强制使用 SFC 线性罚函数接触：

```text
--drive-mode source_inp --contact-mode penalty
```

并且默认隔帧保存：

```text
--history-frame-stride 2
--vtk-frame-stride 2
```

本轮 smoke 验证目录：

```text
results/source_gear_penalty_2steps_match_step_fields_smoke
```

验证结果：

- affected pytest: `21 passed, 4 deselected`
- SFC smoke wall time: `12.549438 s`
- smoke increments: `2`
- SFC VTK frame count: `2`
- `sfc_vtk_frame_stride`: `2`
- `sfc_history_frame_stride`: `2`
- stress/strain postprocess: `corotated_body_elastic_residual`
- rotation unit: `radian`

summary 现在显式记录：

- `sfc_match_source_step`
- `sfc_duration`
- `sfc_dt`
- `sfc_increment_count`
- `source_dynamic_initial_dt`
- `source_dynamic_duration`
- `source_dynamic_min_dt`
- `source_dynamic_max_dt`

后续全程命令应使用：

```powershell
python validation/run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --match-source-step `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --pressure-stiffness 5e9 `
  --write-sfc-vtk `
  --vtk-frame-stride 2 `
  --vtk-scalars-only `
  --history-frame-stride 2 `
  --abaqus-vtk-manifest results/flexible_gear_source_penalty_abaqus_stride2_10steps_rpdiag/abaqus_vtk/abaqus_manifest.csv `
  --out-dir results/source_gear_penalty_full_match_step
```

注意：`--match-source-step` 会触发 `0.05 s / 1e-5 s = 5000` 个 SFC 增量。按当前
10-step `46.01 s` 的实测速度线性外推，完整端到端 wall time 仍可能达到数小时；
因此在启动完整全程前，下一层优化重点应继续放在 source-drive residual/contact
sampling 和 VTK/history 输出开销，而不是改变接触精度或时间步。

## 更新：indexed candidate projection + exact fallback

source-drive compiled contact sampling 现在新增一条 C++ indexed projection 路径：

```text
per-query nearby face candidates
-> C++ closest projection over candidate faces
-> exact all-face fallback if candidate result is outside search radius
```

候选面只用于减少第一轮投影核的扫描面数；如果候选投影没有在 `search_radius`
内命中，则立即回退到全量 master faces 投影。因此该优化不改变 gap、normal、
closest-feature payload 或罚函数接触律。为了进一步减少重复工作，compiled batch
路径在使用 indexed projection 时不再重建未使用的 oracle BVH；它仍然用当前
master 坐标构造保守的 centroid tree 来枚举候选面。

新增/更新测试：

```text
tests/test_dynamic_narrow_band_sdf.py::test_cpp_indexed_projection_matches_all_faces_with_exact_fallback_when_built
tests/test_rp_mpc_and_lagrangian_surface_contact.py
tests/test_flexible_gear_implicit_lagrangian_sdf.py
tests/test_flexible_gear_source_penalty_deck.py
```

验证结果：

```text
43 passed, 4 deselected
```

10-step source-drive penalty 复测目录：

```text
results/source_gear_penalty_10steps_indexed_no_bvh_refit
```

与上一轮 base-LU 结果相比：

| metric | base-LU only | indexed projection |
|---|---:|---:|
| complete wall time | `46.009660 s` | `42.523000 s` |
| residual/contact sampling | `23.179764 s` | `19.167068 s` |
| sparse CG linear correction | `5.242211 s` | `5.251324 s` |
| one-time base LU | `4.376137 s` | `4.345154 s` |
| history diagnostics | `7.150304 s` | `6.887624 s` |
| VTK output | `5.915829 s` | `6.735551 s` |
| sparse CG iterations | `56` | `56` |

数值结果保持一致：

- final active contact samples: `900`
- final min gap: `-7.522933791806687e-04`
- final normal force: `76.72803452060242`
- final max displacement norm: `2.2541615995428984e-04`
- final p95 von Mises: `1.1876012449640669e+07`
- final p95 equivalent elastic strain: `4.9435108733463595e-05`

相同 Abaqus VTK manifest 下的末帧误差仍为：

- max displacement magnitude relative error: `1.195%`
- node-averaged von Mises relative error: `0.0736%`
- node-averaged equivalent elastic strain relative error: `0.0736%`
- RP2 rotation relative error: `1.149%`
- RP2 angular velocity relative error: `1.244%`
