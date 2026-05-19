# CalculiX 同工况小球落地对比

## 目的

本次验证使用同一个 Abaqus/Standard implicit sphere-drop 输入工况，生成 CalculiX 原生接触输入并运行 `ccx`，用于观察开源求解器 CalculiX 在该工况下相对 Abaqus/Standard 和当前 SFC Lagrangian SDF 曲线的表现。

该验证不改变 SFC core。Abaqus 和 CalculiX 均作为外部参考，不参与 SFC 的矩阵装配、时间积分或接触计算。

## 工况

- 输入：`commercial_software_comparison/abaqus_flexible_body_rigid_plane_implicit/sphere_drop/abaqus_run/sphere_drop_implicit.inp`
- 几何：C3D4 柔性小球 + 刚性平面
- 材料：`E = 5.0e7`, `nu = 0.3`, `rho = 1200`
- 重力：`9.81 m/s^2`
- 接触：frictionless linear penalty, normal stiffness `5.0e9`
- 对比窗口：`0.12 s`
- 输出步长：`0.001 s`
- CalculiX：使用自动增量，因为 fixed direct increment 在首次接触时触发 CalculiX impact cutback 后停止

## 运行命令

```powershell
python validation/run_calculix_same_sphere_drop_comparison.py `
  --duration 0.12 `
  --dt 0.001 `
  --hht-alpha -0.3 `
  --out-dir results/calculix_same_condition_sphere_drop_012_auto
```

## 结果文件

- `results/calculix_same_condition_sphere_drop_012_auto/same_condition_z_cm.png`
- `results/calculix_same_condition_sphere_drop_012_auto/same_condition_min_gap.png`
- `results/calculix_same_condition_sphere_drop_012_auto/same_condition_normal_force.png`
- `results/calculix_same_condition_sphere_drop_012_auto/same_condition_open_source_metrics.csv`
- `results/calculix_same_condition_sphere_drop_012_auto/same_condition_calculix_history.csv`

## 主要数值结果

| Source | Metric | Value |
| --- | --- | ---: |
| SFC Lagrangian SDF | max z-CM error vs Abaqus | `4.2523e-3 m` |
| SFC Lagrangian SDF | RMS z-CM error vs Abaqus | `1.8442e-3 m` |
| SFC Lagrangian SDF | first contact time | `0.075 s` |
| CalculiX/ccx | max z-CM error vs Abaqus | `1.1778e-2 m` |
| CalculiX/ccx | RMS z-CM error vs Abaqus | `3.9674e-3 m` |
| CalculiX/ccx | first contact time | `0.0645 s` |
| Abaqus/Standard | first contact time | `0.0660 s` |

## 观察

CalculiX 的首次接触时间与 Abaqus/Standard 更接近，但接触后的反弹明显更高，导致 0.12 s 窗口内的位移轨迹误差大于当前 SFC Lagrangian SDF 曲线。

这说明 CalculiX 在该输入转换下可以作为开源原生接触参考，但不能直接当作 Abaqus/Standard 的等价替代。后续若要把 CalculiX 纳入正式论文对比，应继续对齐以下项：接触阻尼/耗散、HHT 参数、接触切步策略、面接触积分和接触输出口径。

## 测试

```powershell
pytest -q tests/test_calculix_same_sphere_drop_comparison.py tests/test_abaqus_sphere_drop_short_validation.py
```

结果：`9 passed`。
