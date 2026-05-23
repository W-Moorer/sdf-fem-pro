# Abaqus 理论口径对齐分析：柔性齿轮接触工况

## 目的

当前齿轮工况的位移曲线可以在早期局部对齐，但应力、应变和接触状态仍存在明显偏差。下一步不应继续按曲线调参，而应从 Abaqus/Standard 的理论口径开始逐项对齐。

本文档基于 Abaqus 文档、当前 Abaqus deck 和 SFC 运行结果，列出理论差异和修复优先级。

## 当前 Abaqus deck 的实际口径

参考 deck：

`results/source_gear_abaqus_penalty_full_stride2_match_step_0030/abaqus_run/gear_contact_source_penalty.inp`

关键设置：

- `*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE`
- 从属/主面顺序：`_PickedSurf16, _PickedSurf15`
- `*Surface Behavior, pressure-overclosure=LINEAR`
- 线性接触刚度：`5.0e9`
- `*Friction` 为 `0.`
- `*Step, nlgeom=YES`
- `*Dynamic`
- 固定时间增量：`1e-5, 3e-3, 1e-5, 1e-5`
- `*MPC`：两个 `BEAM` 约束，将齿轮 hub 节点与 RP 绑定
- RP1：速度边界驱动，主要为 `UR3` 角速度
- RP2：`Cload` 施加 6 号自由度力矩 `50.`

Abaqus `.dat` 报告的隐式动力学口径：

- `DYNAMIC APPLICATION PACKED SCHEME IS MODERATE DISSIPATION`
- `HILBER-HUGHES-TAYLOR (HHT) TIME INTEGRATOR IS USED`
- `ALPHA = -0.414214`
- `BETA = 0.500000`
- `GAMMA = 0.914214`
- `INITIAL ACCELERATIONS WILL NOT BE CALCULATED`
- `CONTACT IMPACT/RELEASE COMPUTATIONS ARE SKIPPED`
- `LARGE DISPLACEMENT THEORY WILL BE USED`

## Abaqus 文档给出的理论含义

### 1. 隐式动力学 / HHT

Abaqus/Standard 直接积分动力学在 moderate dissipation 下使用 HHT 时间积分。Abaqus 文档说明，负的 HHT `alpha` 会引入数值阻尼，`alpha = 0` 才是无阻尼的梯形 Newmark 方法。

当前 Abaqus deck 明确给出 `alpha=-0.414214`，因此 SFC 对齐时必须使用同一组 HHT 参数，而不是默认 Newmark `alpha=0`。

理论影响：

- SFC 的 `previous` / HHT 历史力必须匹配 Abaqus 的 step 起始语义。
- Abaqus 报告该工况不计算初始加速度，因此 SFC 不应在 step 0 自动求初始平衡加速度。
- 当前已按该口径使用 `alpha=-0.414214`，并将初始加速度置零。

### 2. `*MPC, BEAM` 与 RP 动力学

Abaqus 文档说明 BEAM MPC 在两个节点之间提供刚性梁约束；在隐式动力学中，MPC 对位移严格施加，而速度和加速度由积分算子从位移关系导出。对非线性 MPC，在几何非线性分析中速度和加速度只近似满足约束，可能出现高频加速度振荡。

理论影响：

- SFC 不能简单把 hub 约束视为小转角线性 kinematic map 后再额外叠加离心残差。
- 当前探针表明，加入自写 centripetal residual/tangent 后，RP2 角速度在 `0.0012 s` 从 Abaqus 的约 `2393 rad/s` 偏到 `12244 rad/s`，直接造成 p95 von Mises 尖峰。
- 因此当前齿轮对齐路径应先保持 `source_rotating_inertia=none`，因为 Abaqus 的 BEAM MPC 动力学惯性效应已经通过非线性约束和 HHT 位移更新语义体现，而不是通过额外节点离心残差独立添加。

本轮修复：

- 修正了 centripetal tangent 中错误作用到 z 分量的径向项。
- 增加了 reduced tangent 有限差分测试，防止该诊断路径继续污染理论判断。
- 该路径仍只应作为诊断，不应作为当前 Abaqus 对齐默认策略。

### 3. Surface-to-surface 接触离散

Abaqus 文档指出，surface-to-surface 接触不是单点 node-to-surface 约束，而是在从属节点附近的区域内以平均意义施加接触约束；接触方向基于从属面附近区域的平均法向。文档也指出 surface-to-surface 通常比 node-to-surface 给出更准确的应力和压力，且力不会过度集中在单个从属节点。

理论影响：

- SFC 当前的 `slave_node` / `slave_face` averaging 只是近似 Abaqus 的 constraint region averaging。
- 仍缺失更完整的 Abaqus-style 从属面约束区域构造：每个 secondary node 周围的 patch、平均 normal、区域面积、主面 closest-feature 和约束权重应作为一个统一 constraint region，而不是简单把 tri3/quadrature 样本拆成节点 tributary。
- 当前接触 active node 在若干时刻与 Abaqus 不一致，例如 `0.0014 s` 附近 Abaqus active node 为 0，而 SFC 仍有约 218 个 active node。这说明主要误差源仍是接触约束区域和 contact status/release 口径，而不是 SDF 查询本身。

### 4. 线性罚接触

Abaqus 文档说明，`PRESSURE-OVERCLOSURE=LINEAR` 定义线性 pressure-overclosure 关系；若没有接触面积，pressure 可解释为 force，但当前工况是 element-based surface-to-surface，因此应按面积积分压力。

理论影响：

- SFC 的 `p = k * penetration` 和 `force = integral p n dA` 方向是对的。
- 但要对齐 Abaqus，应保证面积权重、平均约束区域、active/release 逻辑和接触方向一致。
- 不应通过单一工况调 `k` 来消除误差，因为 `k=5e9` 已明确来自 deck。

### 5. Surface smoothing / faceted surface

Abaqus 文档说明 surface-to-surface 通常比 node-to-surface 更能处理接触压力噪声，但 faceted surface 的接触压力仍可能出现显著偏差；对于轴对称、球面、环面等几何，Abaqus 可使用 surface smoothing / geometric correction，但该功能不能用于 orphan mesh。

理论影响：

- 当前齿轮是 orphan mesh 形式导入，Abaqus deck 中没有显式 geometric correction。
- 因此 SFC 不应加入人工滤波式平滑来“美化”应力，而应实现与 faceted surface-to-surface 约束区域一致的平均口径。

### 6. 应力/应变输出位置

Abaqus 文档说明 Abaqus/Standard 可在积分点、单元质心、外推节点、节点平均等位置输出单元变量；对 Mises 等派生量，Abaqus 会先平均外推后的张量分量，再计算派生量。

理论影响：

- 当前 SFC 的 p95 von Mises 若来自单元常应变/节点简单平均，而 Abaqus manifest 来自 nodal averaged tensor-then-invariant，就会产生应力曲线偏差。
- 下一步应把 SFC 输出改成 Abaqus-style：TET4 单元应力张量外推到节点、按 averaging region 做张量平均，再计算 Mises 和等效应变。
- 在这个口径对齐前，不应把 p95 应力误差完全归因于接触算法。

## 当前结果给出的诊断结论

已完成的中程探针：

`results/source_gear_penalty_contact_centripetal_tangent_abaqus_alpha_probe_0014`

结果：

- `SFC wall time = 538.36 s`
- 最大位移相对误差约 `46.06%`
- p95 von Mises / strain 在 `0.0012 s` 出现尖峰，误差约 `1298.74%`
- RP2 角速度在 `0.0012 s`：SFC `12244 rad/s`，Abaqus `2393 rad/s`

结论：

- 额外 centripetal residual/tangent 不是当前理论对齐路径，应关闭。
- 早期位移能对上，说明 RP 边界和基本 HHT 口径并非主要问题。
- 应力/应变趋势不对的主要候选源是：
  1. Abaqus-style surface-to-surface constraint region averaging 尚未完整实现；
  2. contact status/release 口径不一致；
  3. SFC stress/strain postprocess 不是 Abaqus 的 nodal averaged tensor-then-invariant 口径；
  4. `*MPC, BEAM` 反力/约束虚功输出口径仍需单独校验，但它不是当前应力尖峰的主因。

## 本轮输出口径修正

历史 CSV 以前主要报告单元级：

```text
p95_von_mises
p95_equivalent_elastic_strain
```

而 Abaqus VTK manifest 使用 node-averaged 标量：

```text
p95_von_mises_nodeavg
p95_equivalent_elastic_strain_nodeavg
```

这会让曲线对比混合不同统计口径。现在 SFC history row 和 summary 也输出 node-averaged 曲线指标，用于和 Abaqus manifest 对齐。

短程验证：

`results/source_gear_penalty_contact_history_nodeavg_smoke_0004`

对比 `results/source_gear_abaqus_penalty_full_stride2_match_step_0030/abaqus_vtk/abaqus_manifest.csv`：

| time (s) | max displacement rel. error | p95 nodeavg von Mises rel. error | p95 nodeavg equiv. strain rel. error | active nodes SFC/Abaqus |
|---:|---:|---:|---:|---:|
| 0.0002 | 0.452% | 39.798% | 39.798% | 324 / 466 |
| 0.0004 | 3.721% | 1.930% | 1.930% | 44 / 44 |

解释：

- `0.0004 s` 处，在相同 node-averaged 口径下，应力/应变已经进入 2% 以内。
- `0.0002 s` 处仍有约 40% 应力/应变误差，且 active node 数量不同。这进一步支持“下一步应修 contact constraint region / active status，而不是调材料或接触刚度”的判断。
- 该修正只改变诊断和输出口径，不改变求解物理，也不降低接触精度。

## 本轮 constraint-region 接触修正

当前 `slave_node` / `slave_face` 聚合以前采用：

```text
g_region = mean(g_l)
p_region = k max(-g_region, 0)
```

这会让同一个 surface-to-surface constraint region 内的正间隙样本抵消负间隙样本。对于线性 pressure-overclosure law，更一致的离散应保持正穿透量面积积分：

```text
penetration_region = mean(max(-g_l, 0))
g_region = -penetration_region
```

若 region 内没有任何穿透样本，则继续使用 signed mean clearance。active region 的法向也改成按 `area * penetration` 加权；open region 仍用面积加权。这是通用的线性罚接触面积积分修正，不改变 SDF 查询、不调接触刚度、不降低几何精度。

短程验证：

`results/source_gear_penalty_contact_positive_region_smoke_0004`

| time (s) | max displacement rel. error | p95 nodeavg von Mises rel. error | p95 nodeavg equiv. strain rel. error | active nodes SFC/Abaqus |
|---:|---:|---:|---:|---:|
| 0.0002 | 0.391% | 39.646% | 39.646% | 336 / 466 |
| 0.0004 | 3.655% | 1.803% | 1.803% | 65 / 44 |

对比上一轮 nodeavg-only 结果，该修正小幅改善了 `0.0002 s` 的 active count 和应力误差，并略微改善 `0.0004 s` 的应力/应变误差；但 early active status 仍未对齐。

同时测试了一个 HHT 初始外载假设：

`results/source_gear_penalty_contact_hht_step_load_smoke_0004`

该假设把初始 HHT history balance 改为包含 step-start CLOAD。结果反而使位移误差恶化到 `4.619% / 5.852%`，因此已撤回，不进入主线。

## 本轮 split diagnostics 和残差收敛排查

新增 SFC contact diagnostics，将原先合并的 contact fields 拆成：

```text
contact_*_nodeavg              # total: slave + master
contact_slave_*_nodeavg        # slave side only
contact_master_*_nodeavg       # master side only
```

对应新增 history/manifest 指标包括：

```text
active_contact_slave_node_count
active_contact_master_node_count
min_contact_slave_gap_node
min_contact_master_gap_node
p95_contact_slave_pressure_nodeavg
p95_contact_master_pressure_nodeavg
```

短程验证：

`results/source_gear_penalty_contact_split_diagnostics_smoke_0004`

| time (s) | Abaqus active | SFC total | SFC slave | SFC master | Abaqus min gap | SFC min gap | p95 nodeavg stress error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0002 | 466 | 336 | 173 | 163 | -1.670e-3 | -8.11e-4 | 39.646% |
| 0.0004 | 44 | 65 | 32 | 33 | -9.73e-5 | -1.11e-4 | 1.803% |

结论：

- `0.0002 s` 的差异不是 total/slave/master 统计混淆导致；SFC 两侧 active 都偏少，且最小 gap 约为 Abaqus 的一半。
- `0.0004 s` 的 gap 和应力已基本对齐，说明问题集中在接触建立初期。

还将 source-drive 接触迭代从“只看位移修正量”改成“位移修正量 + 力残差”双准则，以接近 Abaqus 非线性迭代收敛思想。

短程验证：

`results/source_gear_penalty_contact_residual_convergence_smoke_0004`

| time (s) | iterations | residual norm | p95 nodeavg stress error |
|---:|---:|---:|---:|
| 0.0002 | 4 | 2.34e-8 | 39.646% |
| 0.0004 | 4 | 2.99e-8 | 1.803% |

结论：

- 残差收敛已显著改善，但 early stress 误差没有变化。
- 因此 `0.0002 s` 误差不是 Newton 提前停止造成的，下一步应继续对齐 Abaqus contact status/COPEN/CPRESS 的 surface region 定义或接触建立阶段的输出语义。

## 当前 0.003 s 全程结果

为满足阶段目标，已使用当前 SFC source-drive 设置跑到 `0.003 s`：

```text
results/source_gear_penalty_contact_history_only_0030
```

运行设置：

- source deck：`commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp`
- drive：`source_inp`
- contact：linear penalty，`pressure_stiffness = 5e9`
- HHT：`alpha = -0.414214`
- time step：`dt = 1e-5`
- duration：`0.003 s`
- contact averaging：`slave_node`
- contact kinematics：`finite_rp_corotated`
- internal kinematics：`corotated_rp`
- rotating inertia diagnostic：`none`

输出：

- SFC history：`sfc_full_gear_lagrangian_sdf_history.csv`
- 对齐误差：`sfc_vs_abaqus_history_nodeavg_errors.csv`
- 曲线图：`sfc_vs_abaqus_history_nodeavg_curves.png`
- 中断恢复：`source_checkpoint.npz`

误差摘要：

| metric | max rel. error | final rel. error | mean rel. error |
|---|---:|---:|---:|
| max displacement | 68.521% | 3.916% | 11.066% |
| p95 nodeavg von Mises | 83.770% | 83.770% | 49.799% |
| p95 nodeavg equivalent strain | 83.770% | 83.770% | 49.799% |

结论：

- 当前实现已经能完成 `0.003 s` 全程 history-only 求解，但应力/应变曲线未达 `<=10%` 验收。
- 位移最终误差为 `3.916%`，说明整体 RP 驱动和最大位移量级不是主要问题。
- 应力/应变最终误差为 `83.770%`，说明 SFC 仍严重低估 Abaqus 后半段齿轮 2 的应力增长。
- 该误差不能再用输出统计口径、Newton 残差或 total/slave/master active 统计解释。更可能的理论缺口是：
  1. Abaqus `*MPC, BEAM` 有限转动约束在 `nlgeom=YES` 下的应力传递口径；
  2. 当前 SFC `corotated_rp` 参考线性刚度把大转动弹性应变剥离过多；
  3. Abaqus 后半段 p95 stress 主要出现在 gear 2，而 SFC 的 source-drive 内部力仍使用 reference-linear/corotated 近似；
  4. 接触压力虽然影响局部波动，但不是后半段 80% 级应力低估的唯一原因。

因此下一步应转向有限转动 RP-MPC / `nlgeom=YES` 内部应力口径对齐，而不是继续只改 SDF 查询或 contact active 统计。

## 下一步理论对齐优先级

### 第一优先级：不要使用 centripetal residual 作为默认

保留已修复的 centripetal tangent 测试作为诊断工具，但正式齿轮对齐使用：

```text
--source-rotating-inertia none
```

### 第二优先级：实现 Abaqus-style surface-to-surface constraint region

目标不是调 `pressure_smoothing_factor`，而是按 Abaqus 文档描述实现：

- secondary node-centered averaging region；
- region 内 slave surface 平均 normal；
- region area；
- master closest-feature payload；
- region-level gap；
- region-level consistent force/tangent；
- contact release/status 与 HHT iteration 收敛耦合。

这一步应替代当前 `slave_node` 的简单 tributary split。

### 第三优先级：对齐 stress/strain 输出口径

新增 Abaqus-style postprocess：

- 单元积分点/质心张量；
- 外推到节点；
- 按 element type/material/section averaging region 做张量平均；
- 再计算 von Mises 和 equivalent elastic strain；
- 对比同一对象、同一节点集合、同一 p95 指标。

### 第四优先级：单独验证 BEAM MPC / RP reaction

设计无接触双齿轮 RP-MPC 动力小算例：

- RP1 给定角速度；
- RP2 给定力矩；
- 输出 RP2 rotation/velocity/acceleration/reaction；
- 不引入接触；
- 先确认 SFC 的 reduced mass、HHT 和 MPC 虚功口径与 Abaqus 一致。

完成后再回到完整接触齿轮。

## 参考文档

- Abaqus/Standard direct-integration implicit dynamics / HHT：
  https://abaqus-docs.mit.edu/2017/English/SIMACAEANLRefMap/simaanl-c-dynamic.htm
- Abaqus/Standard contact constraint enforcement：
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAEITNRefMap/simaitn-c-contactconstraints.htm
- Abaqus `*SURFACE BEHAVIOR`：
  https://abaqus-docs.mit.edu/2017/English/SIMACAEKEYRefMap/simakey-r-surfacebehavior.htm
- Abaqus/Standard contact formulations：
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAEITNRefMap/simaitn-c-contactpairform.htm
- Abaqus/Standard contact pairs：
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAEITNRefMap/simaitn-c-contactpair.htm
- Abaqus surface smoothing：
  https://abaqus-docs.mit.edu/2017/English/SIMACAEITNRefMap/simaitn-c-smoothsurfaces.htm
- Abaqus output to data/results files：
  https://abaqus-docs.mit.edu/2017/English/SIMACAEOUTRefMap/simaout-c-printfile.htm
- Abaqus general multi-point constraints：
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAECSTRefMap/simacst-c-mpc.htm
