# final_aim.md 完整实现路线

## 总判断

`docs/final_aim.md` 的目标不是继续把每个时间步的 current-space SDF grid 重建得更快，而是把主方法推进为：

```text
参考/材料空间 SDF 或参考表面 patch
-> 当前 FEM 变形映射 chi(X,t)
-> 当前变形表面上的按需几何 oracle
-> Abaqus-style node-to-surface 与 surface-to-surface 接触约束
-> 内部 FEM 求解器
-> 与 Abaqus 作为外部参考的结果/效率对比
```

这条路线必须满足两个边界：

1. Abaqus 只能作为理论和外部验证参考，不能进入 `src/sfc` 的矩阵装配、力装配、时间积分或接触评价。
2. 最终 gap、normal、Jacobian 必须来自当前 FEM 表面几何，通过局部投影、normal-tube inversion 或 Lagrangian SDF oracle 得到，不能回到旧的每查询 spatial-hash projection 主路径。

当前仓库已经具备部分基础：`MaterialSDF`、`LagrangianSDFContactOracle`、RP-MPC、HHT/增量控制、constraint-region 接触验证、gear runner 和若干 Abaqus 对齐脚本。还没有完全闭合的是：full gear 动力接触的稳定收敛、Abaqus-style surface constraint region 的完整切线、压力/反力输出口径、以及在保证精度下超过 Abaqus 的工程化求解效率。

## 第一层：几何 oracle 固化

目标：把 `final_aim.md` 中的 Lagrangian SDF oracle 作为稳定基础，而不是临时 validation 工具。

需要保持或补足：

- `src/sfc/sdf/material_sdf.py`
  - 保存参考 SDF、参考表面 patch、参考窄带 cell/patch。
  - 提供批量查询所需的结构化数组，不在 query 中创建大量 Python 对象。
- `src/sfc/contact/lagrangian_sdf_oracle.py`
  - 输入当前 FEM 节点、slave query 点、master patch/cache。
  - 返回 `gap`、`normal`、closest material coordinate、master shape weights、face id、Jacobian payload。
  - cache 只能作为 hint，必须保留全局或 BVH fallback，不能牺牲接触精度。

验收测试：

```powershell
pytest -q tests/test_lagrangian_sdf_oracle.py tests/test_contact_query_equivalence.py
```

必须通过的性质：

- 刚体平移/旋转下 gap 和 normal 正确。
- 有限变形下 oracle 与当前表面几何一致。
- master 侧 Jacobian 符号为负，action-reaction 不破坏。
- active patch cache 不改变结果，只改变搜索顺序和速度。

## 第二层：保留 node-to-surface，但不让它代表全文主线

目标：node-to-surface 是兼容路径和小规模验证路径，不是 full gear surface-to-surface 对齐的最终证据。

需要明确：

- node-to-surface 接触可用于点接触、小球落地、早期调试和接口验证。
- 对齿轮、块体滑移、大面积接触，应使用 surface-to-surface / constraint-region 路径。
- 论文中不能把 node-to-surface 结果当成 Abaqus `TYPE=SURFACE TO SURFACE` 的等价结果。

验收测试：

```powershell
pytest -q tests/test_rp_mpc_and_lagrangian_surface_contact.py tests/test_abaqus_sphere_drop_short_validation.py
```

## 第三层：Abaqus-style surface-to-surface constraint region

目标：对齐 Abaqus surface-to-surface 的核心语义：slave 节点附近有限区域平均、平均法向、沿 secondary/slave normal 搜索 master anchor、压力/穿透按 constraint region 恢复。

需要实现或保持：

- `src/sfc/contact/constraint_region.py`
  - secondary/slave node region 构造。
  - 区域面积、区域平均法向、区域 shape weights。
  - active set、active area、region count、区域平均 gap。
- `src/sfc/contact/tracking_state.py`
  - accepted-state face id/barycentric/path cache。
  - path tracking 先查上一帧及邻域，再 fallback。
  - 输出 cache hit、face switch、barycentric drift、active-region Jaccard。
- master payload
  - T3/TET4 表面三角形路径已可用。
  - Q4/HEX8 面片需要保持原始四边形 quadrature，不应只靠 triangle split 代表 Abaqus surface-to-surface。

接触力口径：

```text
g_I = average_region_gap(I)
n_I = average_region_normal(I)
p_I = k * <-g_I>_+
f_slave += integral_region N_s p_I n_I dA
f_master -= interpolated_master_payload p_I n_I dA
```

压力云图口径：

- `contact_secondary_pressure_nodeavg`
- `contact_secondary_penetration_nodeavg`
- `contact_secondary_active_node`

不要优先比较 nodal CPRESS；先比较：

- total normal force
- contact virtual work
- contact energy
- active area
- active region count

验收测试：

```powershell
pytest -q tests/test_two_block_sliding_region_validation.py tests/test_tooth_patch_region_constraint_validation.py tests/test_source_gear_secondary_line_alignment_gate.py
```

## 第四层：隐式动力学和非线性收敛控制

目标：不要用固定迭代次数“算完就接受”，而要使用 Abaqus-style convergence/cutback 口径。

需要实现或保持：

- HHT/generalized-alpha 隐式动力学。
- residual norm 收敛。
- displacement correction 收敛。
- contact force increment 收敛。
- active set 稳定性。
- line search。
- cutback/retry。
- failed trial 不污染 accepted state、path cache、contact lifecycle。

必须避免：

- iteration-limit 后默认接受。
- 为某个算例调 contact stiffness 或 smoothing。
- 用阻尼掩盖接触不收敛。

验收测试：

```powershell
pytest -q tests/test_increment_control.py tests/test_calculix_aligned_backend.py tests/test_contact_validation_gates.py
```

## 第五层：Abaqus theory alignment，而不是调参对齐

目标：每个与 Abaqus 对齐的功能都要先对齐理论和输出语义。

优先级：

1. `*MPC, BEAM` RP kinematics。
2. RP reaction by equivalent virtual work。
3. HHT/Newmark 状态定义和有效切线。
4. surface-to-surface constraint averaging。
5. contact pressure recovery output semantics。
6. stress/strain output location：单元中心、节点平均或同一采样点，不能混用。

小算例验证顺序：

```text
单 RP-MPC 运动学
-> 双体 RP 动力无接触
-> 双柔性块法向压入
-> 双柔性块横向扫掠
-> cropped gear tooth patch
-> full gear 10 steps
```

对应测试：

```powershell
pytest -q tests/test_abaqus_rp_mpc_global_assembly_alignment.py tests/test_abaqus_dual_rp_mpc_dynamics_alignment.py tests/test_abaqus_finite_rotation_hub_mpc_alignment.py
```

## 第六层：full gear 严格同步验证

目标：先不要追求全时长，先让 full gear 10 个严格同步步通过 gate。

建议命令：

```powershell
python validation/run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --out-dir results/flexible_gear_full_strict10_source_region `
  --drive-mode source_inp `
  --duration 0.0025 `
  --dt 0.00025 `
  --active-faces-per-body 16 `
  --contact-mode penalty `
  --source-contact-averaging slave_node_region_constraint `
  --source-contact-direction secondary_average `
  --source-contact-projection secondary_line `
  --source-secondary-path-tracking `
  --source-contact-active-set-stability `
  --source-stress-postprocess finite_stvk_visual `
  --source-internal-kinematics finite_stvk_visual `
  --source-rotating-inertia finite_kinematic `
  --history-frame-stride 1 `
  --vtk-frame-stride 2 `
  --write-sfc-vtk `
  --vtk-scalars-only
```

先看这些输出：

- `sfc_full_gear_entry_gate.csv`
- `sfc_source_active_set_line_search_gate.csv`
- `sfc_source_convergence_gate.csv`
- `sfc_path_tracking_gate.csv`
- `sfc_contact_total_gate.csv`
- `sfc_full_gear_lagrangian_sdf_history.csv`

10 步验收条件：

- accepted step count >= 10。
- iteration-limit accepted count = 0。
- unstable accepted count = 0。
- final time equals duration。
- active set line-search gate passed。
- path tracking gate passed。
- contact total priority gate passed。
- stress/strain 曲线趋势与 Abaqus 对齐后再看云图。

若这层失败，优先修：

1. constraint-region consistent tangent。
2. contact active set cutback/line search。
3. path tracking cache fallback。
4. RP reaction/output 口径。

不要先改 SDF 查询。

## 第七层：0.003 s、0.05 s 和全时长验证

10 步通过后按以下顺序扩展：

```text
0.003 s strict sync
-> 0.05 s Abaqus/SFC displacement-stress-strain comparison
-> full duration
```

每层都要生成：

- displacement 曲线。
- strain 曲线。
- stress/von Mises 曲线。
- total normal force。
- contact virtual work/energy。
- active area/region count。
- SFC/Abaqus 同步 VTK 帧，隔帧保存即可。

不使用柱状图作为主证据。误差写在曲线图图例或表格中。

## 第八层：效率优化，只能在精度 gate 通过后进行

效率目标不是牺牲精度，而是减少重复工作和 Python 开销。

优先级：

1. persistent workspace
   - 复用 query buffers、region buffers、force buffers、sparse pattern。
2. active patch/cache reuse
   - 上一 accepted state 的 master face、barycentric、region topology 可作为 hint。
3. sparse factor/preconditioner reuse
   - 接触拓扑不变时复用线性求解资源。
4. consistent constraint-region tangent
   - 把非线性迭代从 16 次降到 2 到 4 次。
5. C++ fused contact/tangent kernel
   - 合并 candidate、payload、gap、force、tangent matvec。
6. matrix-free/preconditioned tangent solve
   - 避免每步完整装配和分解大矩阵。

效率报告必须区分：

- full wall time。
- core solve time。
- contact oracle/search time。
- tangent/linear solve time。
- diagnostics/postprocess/VTK time。

论文中不能用 backend breakdown 冒充完整 solve speedup。

## 第九层：论文主线

如果按照 `final_aim.md` 继续推进，论文主线应调整为：

```text
deformation-aware / Lagrangian SDF contact oracle
+ closest-feature FEM sensitivities
+ Abaqus-style surface-to-surface constraint regions
+ internal FEM implicit dynamics
+ external Abaqus validation
```

dynamic narrow-band current-grid SDF 可以作为兼容路径或 ablation，但不能再抢占 final_aim 的主方法叙事。

论文可支持的 scoped claims：

- 任意有限变形柔性体的当前表面接触 gap/normal/Jacobian 可由 Lagrangian SDF oracle 按需查询。
- node-to-surface 与 surface-to-surface 两条接触离散路径都保留，但工程接触主证据采用 surface-to-surface constraint region。
- 与 Abaqus 的对齐基于理论和输出语义，而不是调参。
- 在大量接触采样和可复用拓扑下，SFC 有机会通过 oracle/cache/fused tangent 降低接触查询和求解开销。

暂不应宣称：

- 通用生产级 Abaqus 替代。
- 摩擦、自接触、拓扑变化。
- 任意非流形几何全局 SDF 鲁棒性。
- 所有齿轮工况全时长已超过 Abaqus。

## 下一步执行顺序

立即下一步不是继续修改 SDF 查询，而是：

1. 跑 full gear 10-step strict sync，确认现有 gate 是否真实闭合。
2. 若 10-step 不闭合，先修 constraint-region tangent / active-set cutback / path tracking。
3. 10-step 通过后再扩展到 0.003 s 和 0.05 s。
4. 精度和趋势进入门槛后，才进入 C++ fused / matrix-free / factor reuse 优化。
5. 每一层都提交对应结果、测试和 supported/unsupported claims。
