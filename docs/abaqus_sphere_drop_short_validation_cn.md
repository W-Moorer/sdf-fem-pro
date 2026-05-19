# Abaqus 小球掉落刚性平面短程验证

## 验证目的

本阶段使用已有 Abaqus/Explicit 小球掉落刚性平面外部结果，对 SFC 侧的拉格朗日 SDF 接触路径做短程一致性验证。Abaqus 仅作为外部参考结果；SFC 不读取 Abaqus ODB，不把 Abaqus 作为核心依赖。

SFC 验证路径为：

```text
Abaqus .inp 工况解析
-> SFC C3D4/TET4 柔性小球重建
-> MaterialSDF 刚性平面
-> LagrangianSDFContactOracle 间隙查询
-> 与 Abaqus VTK 帧历史对齐
```

## 工况范围

- Abaqus 输入：`commercial_software_comparison/abaqus_flexible_body_rigid_plane/sphere_drop/abaqus_run/sphere_drop_explicit.inp`
- 柔性体：C3D4 小球，267 个节点，528 个四面体单元
- 刚性体：R3D4 刚性平面
- 材料：`E = 5.0e7`，`nu = 0.30`，`rho = 1200`
- 重力：`9.81 m/s^2`
- 初始间隙：`0.02 m`
- Abaqus 输出间隔：`0.001 s`
- 本次短程验证：`0.05 s`

由于初始间隙为 20 mm，按自由落体估计首次接触时间为：

```text
t_contact ~= 0.0638550857 s
```

因此 `0.05 s` 短程验证处于接触前阶段。它验证的是工况重建、重力动力学、Abaqus VTK 对齐和拉格朗日 SDF 间隙查询一致性，不声称已经完成冲击接触阶段验证。

## 结果

运行命令：

```text
python validation/run_abaqus_sphere_drop_short_validation.py --duration 0.05 --dt 0.001 --out-dir results/abaqus_sphere_drop_short
```

关键结果：

| 指标 | 数值 |
| --- | ---: |
| 最大质心高度绝对误差 | `5.595140492448136e-07 m` |
| RMS 质心高度误差 | `1.7807100550218418e-07 m` |
| 最大最小间隙绝对误差 | `5.590436568236185e-07 m` |
| SFC 接触激活 | `false` |
| Abaqus 接触激活 | `false` |
| SFC 短程 wall time | `1.5962249003350735 s` |

输出文件：

- `results/abaqus_sphere_drop_short/abaqus_sphere_drop_short_sfc_history.csv`
- `results/abaqus_sphere_drop_short/abaqus_sphere_drop_short_abaqus_history.csv`
- `results/abaqus_sphere_drop_short/abaqus_sphere_drop_short_comparison.csv`
- `results/abaqus_sphere_drop_short/abaqus_sphere_drop_short_metrics.csv`
- `results/abaqus_sphere_drop_short/abaqus_sphere_drop_short_z_cm.png`
- `results/abaqus_sphere_drop_short/abaqus_sphere_drop_short_min_gap.png`
- `results/abaqus_sphere_drop_short/abaqus_sphere_drop_short_summary.md`

## 当前结论

短程接触前阶段通过：SFC 从 Abaqus `.inp` 直接重建的小球自由落体轨迹和 Abaqus/Explicit VTK 结果在 `1e-6 m` 量级内一致，拉格朗日 SDF oracle 给出的平面间隙和 Abaqus 几何间隙一致。

下一步如果要验证真正接触阶段，需要将时间延长到 `0.064 s` 之后，并使用接触阶段的 SFC 动力学求解器进行 Abaqus native contact 对齐。
