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
