# Abaqus 等价 SDF 接触框架实现进展

本轮将 source-drive 齿轮验证路径的默认接触聚合从点级正穿透参与口径改为 Abaqus-style constraint-region 口径。

## 改动

默认值由：

```text
source_contact_averaging = slave_node_region_participation
```

改为：

```text
source_contact_averaging = slave_node_region_constraint
```

影响文件：

- `validation/run_flexible_gear_implicit_lagrangian_sdf_comparison.py`
- `validation/run_flexible_gear_full_lagrangian_sdf_comparison.py`
- `validation/check_source_gear_secondary_line_alignment.py`

## 理论含义

`slave_node_region_participation` 的行为更接近：

```text
先在局部 sample 上计算 < -g_i >_+
再把正压力积分到 slave-node region
```

这仍然会在区域平均 gap 为开时保留局部 sample 压力。

`slave_node_region_constraint` 的行为更接近 Abaqus surface-to-surface 文档中的 finite secondary region average：

```text
G_r = (1 / A_r) integral_R psi_r(s) g(s) dA
p_r = k < -G_r >_+
```

也就是说，接触区域是否闭合先由 signed region gap `G_r` 决定，而不是由区域内某个 quadrature/sample 点的局部穿透决定。

## 已加测试

新增测试覆盖：

1. 当局部 sample 有穿透但区域平均 gap 为正时，constraint-region 口径不产生压力。
2. 当区域平均 gap 为负时，压力和能量来自 `A_r k < -G_r >_+`，而不是来自 pointwise penetration integral。

这一步只是最小 Abaqus-equivalent contact operator 的第一层。后续仍需补：

- contact force increment convergence；
- cutback；
- 同口径 CPRESS/COPEN 输出；
- 齿面 patch 与全齿轮多步误差验证。
