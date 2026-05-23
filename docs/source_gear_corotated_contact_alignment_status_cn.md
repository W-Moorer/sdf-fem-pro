# source gear 对齐当前状态

## 已完成的通用修正

本轮没有调接触刚度，也没有修改 SDF 最近点查询精度，而是补了三项 Abaqus 理论口径相关的通用修正：

1. **有限 RP 转动接触几何**
   - 新增 `source_contact_kinematics=finite_rp_corotated`。
   - 接触查询使用有限转动后的当前表面位置，而不是把累计 RP 角度当成小转角位移。
   - 这修复了 `0.002--0.003 s` 后段 SFC 接触样本完全消失的问题。

2. **主/从表面法向兼容过滤**
   - 新增 `source_contact_normal_filter=opposing`。
   - 仅保留 `n_master dot n_slave <= 0` 的相对朝向候选。
   - 这避免齿面曲率下最近但不相对的侧面/背面三角片造成假穿透。

3. **corotated RP elastic residual 矩阵**
   - 新增 `source_internal_kinematics=corotated_rp`。
   - 内部弹性残差使用 `u_elastic = Tq - theta_z e_z x (X - X_RP)`。
   - 纯 RP z 转动不再产生弹性残差。

相关测试：

```text
pytest -q tests\test_flexible_gear_implicit_lagrangian_sdf.py tests\test_flexible_gear_source_penalty_deck.py -k "source_penalty or source_drive or corotated or averaging or normal_compatibility"
19 passed, 18 deselected
```

## 对齐结果

短程 `0.0001 s` 中，使用

```text
finite_rp_corotated
opposing normal filter
HHT alpha = -0.25 或 -0.3
```

可使位移、p95 von Mises、等效应变误差进入 10% 以内。

但是完整 `0.003 s` 长程中，`alpha=-0.3` 结果失败：

```text
latest displacement p95 rel. error: 3.918%
latest von Mises p95 rel. error: 83.772%
latest strain norm p95 rel. error: 87.496%
latest active contact nodes: SFC 337, Abaqus 88
```

对应文件：

```text
results/source_gear_penalty_contact_0030_sfc_finite_rp_opposing_alpha_m0p3
results/source_gear_penalty_contact_0030_finite_rp_opposing_alpha_m0p3_sequence_alignment
results/source_gear_penalty_contact_0030_finite_rp_opposing_alpha_m0p3_diagnostics
```

`alpha=-0.25 + corotated_rp internal` 长程在 `0.0014 s` 已表现出同样趋势，因此已停止该 run：

```text
time = 0.0014 s
SFC p95 von Mises = 4.657849e6
Abaqus p95 von Mises = 8.360519e6
```

## 当前误差来源判断

接触 gap 已经不是主要误差源。短程中 SFC 和 Abaqus 的最小 gap 基本一致：

```text
t = 0.00010 s
SFC min gap    = -2.86e-4
Abaqus min gap = -2.86e-4
```

长程应力趋势仍不一致的主要原因是：

1. Abaqus `nlgeom=YES` 下的高速转动柔性体包含有限转动动力学效应。
2. 当前 SFC source-drive 后端虽然可做有限转动接触几何和 corotated stress 输出，但还没有完整的有限转动惯性项。
3. 对高速齿轮，Abaqus 应力随 RP2 角速度和接触历史继续增长；SFC 的 corotated 线性弹性应力停留在约 `1.3e7` max，缺少旋转坐标系下的离心/陀螺惯性贡献。

因此，继续调接触刚度、SDF 查询或法向平滑不能解决这个问题。

## 下一步必须实现

要让 full gear 长程应力/应变和 Abaqus 对齐，需要进入下一层理论实现：

1. 有限转动 RP-MPC 动力学不再只作为后处理/接触几何映射。
2. 在 corotated body frame 中加入转动惯性项：
   - 离心项；
   - 陀螺项；
   - Euler 加速度项；
   - 与 RP 角速度/角加速度一致的等效体力或惯性残差。
3. 相应更新 tangent / matrix-free solve，使 Newton 残差包含这些惯性项。
4. 重新跑 source gear：
   - 先 `0.0001 s`；
   - 再 `0.001 s`；
   - 最后 `0.003 s`。

当前结论：SDF 接触几何口径已经明显改善，但 full gear 应力/应变还不能通过 `<10%` 验收。
