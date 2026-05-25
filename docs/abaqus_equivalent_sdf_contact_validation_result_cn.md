# Abaqus 等价 SDF 接触框架本轮验证结果

本轮实现了第一层 Abaqus-equivalent constraint-region 接触口径，并运行三个 Abaqus 外部对比工况。

## 实现变更

source-drive 齿轮路径默认接触聚合改为：

```text
source_contact_averaging = slave_node_region_constraint
```

该模式使用区域 signed-average gap：

```text
G_r = (1 / A_r) integral_R psi_r(s) g(s) dA
p_r = k < -G_r >_+
```

而不是先对每个 sample 做 pointwise positive overclosure 再聚合。

## 测试

```text
pytest -q tests\test_flexible_gear_implicit_lagrangian_sdf.py -k "constraint_region or array_constraint_region or secondary_contact_tracking_radius or default_contact_search_radius or secondary_line_distance_limit" tests\test_source_gear_secondary_line_alignment_gate.py
```

结果：

```text
6 passed, 69 deselected
```

## Abaqus 对比工况

### 1. 线性 pressure-overclosure 法向压入

命令：

```text
python validation\run_abaqus_linear_penalty_normal_indentation.py --run-abaqus --out-dir results\abaqus_equivalent_sdf_linear_penalty_indentation --timeout 600
```

输出：

- `results/abaqus_equivalent_sdf_linear_penalty_indentation/linear_penalty_normal_indentation_summary.md`
- `results/abaqus_equivalent_sdf_linear_penalty_indentation/abaqus_vs_sfc_linear_penalty_normal_indentation_errors.csv`

结果：

```text
max RF/normal-force relative error = 0.000000%
max CPRESS output-mean relative difference = 0.000000%
SFC wall time = 0.013571 s
Abaqus analysis wall time = 10.434412 s
```

结论：线性罚函数接触律和法向力口径在该标定算例中与 Abaqus 对齐。

### 2. 双柔性块 surface-to-surface HARD contact

命令：

```text
python validation\run_abaqus_lagrangian_sdf_hard_contact_alignment.py --out-dir results\abaqus_equivalent_sdf_hard_blocks --timeout 600
```

输出：

- `results/abaqus_equivalent_sdf_hard_blocks/lagrangian_sdf_hard_contact_summary.md`
- `results/abaqus_equivalent_sdf_hard_blocks/abaqus_vs_sfc_lagrangian_sdf_hard_contact_errors.csv`
- `results/abaqus_equivalent_sdf_hard_blocks/sfc_contact_quadrature_diagnostics.csv`

结果：

```text
upper top U3 relative error = 0.000001%
lower top U3 relative error = 2.258479%
upper top RF3 relative error = 2.378846%
COPEN mean absolute error = 6.177705e-04
CPRESS mean relative error = 3.902983%
SFC analysis wall time = 2.393700e-03 s
Abaqus analysis wall time = 8.062733 s
```

结论：平面双柔性体接触中的位移、反力和平均压力均进入小误差范围。

### 3. 倾斜 tooth-patch surface-to-surface HARD contact

命令：

```text
python validation\run_abaqus_lagrangian_sdf_tooth_patch_hard_contact.py --out-dir results\abaqus_equivalent_sdf_tooth_patch --timeout 600
```

输出：

- `results/abaqus_equivalent_sdf_tooth_patch/lagrangian_sdf_tooth_patch_hard_contact_summary.md`
- `results/abaqus_equivalent_sdf_tooth_patch/abaqus_vs_sfc_lagrangian_sdf_tooth_patch_hard_contact_errors.csv`
- `results/abaqus_equivalent_sdf_tooth_patch/sfc_contact_quadrature_diagnostics.csv`

结果：

```text
upper top normal-displacement relative error = 0.000001%
lower top normal-displacement relative error = 2.184243%
upper top normal-reaction relative error = 0.249169%
COPEN mean absolute error = 5.959372e-04
CPRESS mean relative error = 2.719382%
SFC analysis wall time = 4.575000e-04 s
Abaqus analysis wall time = 8.041627 s
```

结论：非轴对齐 tooth-like patch 中，Q4 slave quadrature、Q4 master closest-feature payload 和 hard-contact pressure-compliance enforcement 可以与 Abaqus 对齐。

## 当前限制

这三个算例验证的是：

```text
contact law
surface-to-surface region pressure semantics
Q4 payload
normal-direction reaction
small tooth-like patch
```

尚未证明：

```text
全齿轮 10 步误差 < 10%
全齿轮长程效率优于 Abaqus
完整 contact force increment convergence / cutback 已实现
```

下一步应将 `slave_node_region_constraint` 默认路径用于全齿轮 10 步严格同步重算，并检查 displacement / von Mises / equivalent strain 是否全部进入 `<10%`。
