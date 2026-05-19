# Abaqus 小球掉落刚性平面全程验证

## 验证目的

本阶段把短程 `0.05 s` 验证扩展到 Abaqus/Explicit 小球掉落刚性平面的完整 `3.0 s` 轨迹。SFC 侧不读取 Abaqus ODB，不调用 Abaqus 求解能力；它从 Abaqus `.inp` 中重建 C3D4 小球网格、材料、重力和刚性平面，然后使用内部 FEM 矩阵和拉格朗日 SDF 接触 oracle 独立求解。

SFC 路径为：

```text
Abaqus .inp 工况解析
-> SFC C3D4/TET4 体网格重建
-> 内部质量矩阵 / 刚度矩阵 / 重力装配
-> MaterialSDF 刚性平面
-> LagrangianSDFContactOracle 接触间隙和法向
-> 隐式 Newmark + frictionless penalty contact
-> 与 Abaqus/Explicit VTK 全时程对比
```

## 工况

- 输入文件：`commercial_software_comparison/abaqus_flexible_body_rigid_plane/sphere_drop/abaqus_run/sphere_drop_explicit.inp`
- 参考结果：`commercial_software_comparison/abaqus_flexible_body_rigid_plane/sphere_drop/vtk/frame_*.vtk`
- 总时长：`3.0 s`
- 时间步长：`0.001 s`
- 小球：267 节点，528 个 C3D4/TET4 单元
- 刚性平面：R3D4 平面
- 材料：`E = 5.0e7`，`nu = 0.30`，`rho = 1200`
- 重力：`9.81 m/s^2`
- 初始间隙：`0.02 m`
- SFC 接触路径：`MaterialSDF + LagrangianSDFContactOracle`
- SFC 接触刚度：`1.0e9`

## 运行命令

```text
python validation/run_abaqus_sphere_drop_full_validation.py --duration 3.0 --dt 0.001 --contact-stiffness 1.0e9 --output-stride 1 --out-dir results/abaqus_sphere_drop_full
```

相关测试：

```text
pytest -q tests/test_abaqus_sphere_drop_full_validation.py tests/test_abaqus_sphere_drop_short_validation.py tests/test_abaqus_flexible_body_rigid_plane.py
```

测试结果：`12 passed`。

## 关键结果

| 指标 | 数值 |
| --- | ---: |
| SFC 首次接触时间 | `0.064 s` |
| Abaqus 首次接触时间 | `0.06400582939386368 s` |
| 首次接触时间误差 | `5.829393863676646e-06 s` |
| 最大质心高度绝对误差 | `2.4672423556769102e-02 m` |
| RMS 质心高度误差 | `1.1088715974366383e-02 m` |
| 最大最小间隙绝对误差 | `2.411631547911104e-02 m` |
| 最大 von Mises 绝对误差 | `1.2093576519933087e+06` |
| 最大应变范数绝对误差 | `5.0675070135701196e-02` |
| SFC solve wall time | `117.26165200024843 s` |
| Newton 总迭代数 | `6027` |
| Newton 失败步数 | `0` |

输出文件：

- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_sfc_history.csv`
- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_abaqus_history.csv`
- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_comparison.csv`
- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_metrics.csv`
- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_z_cm.png`
- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_min_gap.png`
- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_von_mises.png`
- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_strain_norm.png`
- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_active_contact.png`
- `results/abaqus_sphere_drop_full/abaqus_sphere_drop_full_summary.md`

## 当前结论

全程验证已经跑通：SFC 可以在同一 Abaqus 小球掉落刚性平面工况下完成独立的拉格朗日 SDF 接触动力学求解，并和 Abaqus 全部 VTK 帧对齐比较。最关键的首次接触时间误差为 `5.83e-6 s`，说明几何重建、重力时程和拉格朗日 SDF 平面接触触发位置是对齐的。

但是，`3.0 s` 全程轨迹和应力/应变结果还没有达到可作为论文“高精度等价”证据的程度。主要原因不是 SDF 查询本身，而是接触和动力学模型仍不完全一致：

1. Abaqus 使用 Explicit general contact 和 hard normal contact；
2. Abaqus 输入中包含 bulk viscosity；
3. 当前 SFC full runner 使用 frictionless penalty contact；
4. 当前 SFC 未显式匹配 Abaqus/Explicit 的稳定时间积分、接触耗散和 hard-contact 反弹处理；
5. 罚接触刚度会影响反弹高度、接触持续时间和应力峰值。

因此，本轮结果可以作为“全程验证流水线已建立”的证据，不能直接作为“全程轨迹与 Abaqus 高精度等价”的论文证据。下一步应做接触律和耗散对齐：匹配 Abaqus hard contact / bulk viscosity 的等效 SFC 接触响应，或在论文中明确将该算例定位为外部参考诊断而非等价证明。

## 接触律和耗散对齐进展

已在 `validation/run_abaqus_sphere_drop_full_validation.py` 中加入显式参数化的耗散项：

- `contact_damping`：法向接触 dashpot，仅在接触激活点且相对法向速度为闭合时产生耗散；
- `mass_damping`：质量比例耗散，用于模拟 Abaqus/Explicit bulk viscosity 对冲击后振荡的等效耗散；
- `stiffness_damping`：刚度比例耗散；
- `damping_start_time`：耗散起始时间。为了不破坏自由落体接触前轨迹，推荐从首次接触时刻附近启动。

对齐运行命令：

```text
python validation/run_abaqus_sphere_drop_full_validation.py --duration 3.0 --dt 0.001 --contact-stiffness 1.0e9 --mass-damping 0.5 --damping-start-time 0.064 --contact-damping 0.0 --stiffness-damping 0.0 --output-stride 1 --out-dir results/abaqus_sphere_drop_full_aligned_dissipation
```

对齐后关键结果：

| 指标 | 未对齐 | 接触后耗散对齐 | 变化 |
| --- | ---: | ---: | ---: |
| 首次接触时间误差 | `5.829e-6 s` | `5.829e-6 s` | 保持 |
| 最大质心高度误差 | `2.467e-2 m` | `1.664e-2 m` | 降低约 `32.6%` |
| RMS 质心高度误差 | `1.109e-2 m` | `5.466e-3 m` | 降低约 `50.7%` |
| 最大最小间隙误差 | `2.412e-2 m` | `1.680e-2 m` | 降低约 `30.3%` |
| full z L2 error | `6.075e-1` | `2.995e-1` | 降低约 `50.7%` |
| full min-gap L2 error | `5.996e-1` | `2.996e-1` | 降低约 `50.0%` |
| Newton failed steps | `0` | `0` | 保持 |
| SFC wall time | `117.26 s` | `115.88 s` | 基本持平 |

当前可支持的结论是：接触律/耗散对齐显著改善了全程位移和间隙轨迹，同时不破坏首次接触时刻。尚不能声称应力/应变峰值已经与 Abaqus/Explicit 完全等价，因为 Abaqus hard contact、bulk viscosity、显式稳定积分和当前 SFC penalty contact 的局部冲击应力仍不完全一致。
