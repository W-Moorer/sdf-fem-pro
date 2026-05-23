# 柔性齿轮应力/应变误差理论诊断

## 目的

当前目标是把 `commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp` 的齿轮工况在 SFC Lagrangian-SDF 路径中对齐到 Abaqus/Standard：相同模型驱动、线性罚函数接触、无摩擦、同一时间步和总时长，并比较位移、应力、应变曲线与效率。

已有 `0.003 s` 全程结果显示：

- 位移末值误差约 `3.916%`；
- p95 node-averaged von Mises 与等效弹性应变末值误差约 `83.770%`；
- SFC 最终仍有较多 active contact nodes，而 Abaqus 接触状态随时间释放/重建；
- SFC 后半段应力基本维持在 `4-5 MPa`，Abaqus p95 stress 增长到约 `27 MPa`。

因此问题不能再简单归因于 SDF 查询误差或绘图口径，而应从 Abaqus 的 `nlgeom=YES`、HHT、`*MPC, BEAM`、有限转动惯性和接触状态语义上继续对齐。

## 本轮新增诊断

新增脚本：

```text
validation/analyze_source_gear_checkpoint_stress_modes.py
```

该脚本读取 SFC accepted checkpoint：

```text
results/source_gear_penalty_contact_history_only_0030/source_checkpoint.npz
```

并与 Abaqus VTK manifest 对比：

```text
results/source_gear_abaqus_penalty_full_stride2_match_step_0030/abaqus_vtk/abaqus_manifest.csv
```

输出：

```text
results/source_gear_checkpoint_stress_modes/source_gear_checkpoint_stress_modes.csv
results/source_gear_checkpoint_stress_modes/source_gear_checkpoint_stress_modes.md
```

诊断比较了同一个 SFC accepted state 下三种应力/应变解释：

| mode | max displacement error | p95 VM nodeavg error | p95 equivalent strain nodeavg error |
|---|---:|---:|---:|
| raw_small_strain_linearized_mpc | 383.930% | 83.770% | 83.770% |
| linear_corotated_elastic_residual | 3.916% | 83.770% | 83.770% |
| finite_stvk_visual_configuration | 3.916% | 83.828% | 83.828% |

结论：只改变应力后处理口径不能消除误差。即使用有限变形 StVK 在有限 RP 视觉构型上重算应力，误差仍约 `83.8%`。这说明 accepted trajectory 本身缺少 Abaqus `nlgeom=YES` 有限转动动力学效应，而不是单纯云图后处理问题。

## 本轮新增有限转动虚功映射

新增内部工具：

```text
_source_drive_finite_visual_jacobian(...)
```

它计算有限 RP 视觉映射的解析 Jacobian：

```text
J_visual = d x_visual / d q
```

并加入有限差分测试：

```text
test_source_drive_finite_visual_jacobian_matches_finite_difference
```

测试已通过。该映射用于诊断 `J_visual.T @ f_int(x_visual)`，符合等效虚功投影原则，不是针对单个算例调参。

## 有限 StVK 残差短程试验

短程试验目录：

```text
results/source_gear_finite_stvk_internal_smoke_0004
```

设置：

- source: `gear_contact.inp`
- duration: `0.0004 s`
- dt: `1e-5 s`
- contact: linear penalty, stiffness `5e9`
- HHT alpha: `-0.414214`
- contact kinematics: `finite_rp_corotated`
- contact averaging: `slave_node`
- internal kinematics: `finite_stvk_visual`

结果：

- `0.0002 s`：位移误差约 `0.391%`，p95 stress 误差约 `39.959%`，与旧路径量级相近；
- `0.0004 s`：应力严重过大，p95 stress 相对误差约 `53055%`；
- Newton 残差在 `0.0004 s` 约 `4.006e4`，说明该模式没有可靠收敛。

结论：不能把有限变形 StVK 内力直接接入残差而继续使用固定线性切线。这会形成不一致的非线性动力学求解，不能作为正式对齐路径。

## 有限运动学惯性短程试验

本轮新增有限运动学惯性诊断路径：

```text
--source-rotating-inertia finite_kinematic
```

对应 reduced inertia 形式为：

```text
J(q)^T M [J(q) qddot + Jdot(q, qdot) qdot]
```

并返回 `J(q)^T M J(q)` 作为 Newmark 加速度项的质量切线。该路径不改变接触刚度、不改变材料参数，也不是针对单个曲线调参。

短程试验目录：

```text
results/source_gear_finite_kinematic_inertia_smoke_0004
```

设置：

- source: `gear_contact.inp`
- duration: `0.0004 s`
- dt: `1e-5 s`
- contact: linear penalty, stiffness `5e9`
- HHT alpha: `-0.414214`
- contact kinematics: `finite_rp_corotated`
- contact averaging: `slave_node`
- contact normal filter: `opposing`
- internal kinematics: `corotated_rp`
- rotating inertia: `finite_kinematic`

关键结果：

| time | max displacement error | p95 VM nodeavg error | p95 equivalent strain nodeavg error | active nodes SFC/Abaqus |
|---:|---:|---:|---:|---:|
| 0.0002 | 0.391% | 39.620% | 39.620% | 336 / 466 |
| 0.0004 | 3.655% | 5.014% | 5.014% | 65 / 44 |

结论：

- 有限运动学惯性在 `0.0004 s` 把 p95 应力/应变维持在 `<=10%` 验收范围内；
- 但 `0.0002 s` 的接触建立阶段仍约 `39.6%`，且 SFC active nodes 和最大穿透仍低于 Abaqus；
- 因此早期误差不是应力后处理造成，也不是仅靠有限惯性可以解决，下一层应对齐 Abaqus surface-to-surface constraint region / active status 口径。

法向过滤探针：

```text
results/source_gear_finite_kinematic_no_normal_filter_probe_0002
```

去掉 `opposing` 法向过滤后，`0.0002 s` active nodes 从 `336` 提高到 `415`，更接近 Abaqus 的 `466`，但 p95 stress 误差仍约 `40.10%`。这说明单纯放宽法向过滤不能解决应力误差，真正缺口仍是接触约束区域和压力/穿透场的定义。

## 当前理论差异定位

最主要差异不是 SDF 查询，而是有限转动动力学：

1. 当前 SFC source-drive 主路径用 reference-linear / corotated residual，使大转动下的整体旋转几乎不产生内部应力；
2. Abaqus `nlgeom=YES + *MPC, BEAM` 下，RP 有限转动、转矩驱动、HHT 更新和惯性项耦合，会产生旋转惯性/约束传递相关的应力增长；
3. SFC 当前质量矩阵仍主要是线性 reduced mass `T.T M T`，没有完整使用有限运动学下的 `J(q).T M J(q)`；
4. 当前额外 centripetal residual 是诊断路径，曾导致 RP2 速度发散，说明仅补一个离心项而不同步有限运动学质量、HHT history 和一致切线是不正确的；
5. 应力/应变差异后半段由动力学路径主导，接触 active mismatch 会影响局部压力和短时波动；有限运动学惯性修正后，后半段短程误差已能进入 10% 内，但接触建立初期仍需继续对齐 constraint region。

## 下一步正确修复方向

下一步不应调接触刚度、阻尼或 SDF 查询。应实现 Abaqus-style finite-kinematic RP-MPC dynamics：

```text
x = x(q)
J(q) = dx/dq
xdot = J(q) qdot
xddot = J(q) qddot + Jdot(q, qdot) qdot
```

动力学残差应使用：

```text
J(q)^T M [J(q) qddot + Jdot(q, qdot) qdot]
+ J(q)^T f_int(x(q))
- J(q)^T f_ext
- J(q)^T f_contact
```

同时需要一致或半一致切线：

```text
J^T K_int J
+ K_inertia
+ K_contact
```

这样才是理论对齐 Abaqus `nlgeom=YES` 的通用路径。完成该层后，再回到 `0.003 s` 全程对比应力/应变和效率。

## 验证命令

```text
python validation/analyze_source_gear_checkpoint_stress_modes.py --source commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp --checkpoint results/source_gear_penalty_contact_history_only_0030/source_checkpoint.npz --abaqus-manifest results/source_gear_abaqus_penalty_full_stride2_match_step_0030/abaqus_vtk/abaqus_manifest.csv --out-dir results/source_gear_checkpoint_stress_modes --duration 0.003
pytest -q tests/test_flexible_gear_implicit_lagrangian_sdf.py -k "finite_visual_jacobian or corotated or source_drive or checkpoint or centripetal"
pytest -q tests/test_flexible_gear_implicit_lagrangian_sdf.py -k "finite_kinematic_inertia or finite_visual_jacobian or centripetal"
```

测试结果：

```text
10 passed, 24 deselected
5 passed, 31 deselected
```
