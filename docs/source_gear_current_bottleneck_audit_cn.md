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
