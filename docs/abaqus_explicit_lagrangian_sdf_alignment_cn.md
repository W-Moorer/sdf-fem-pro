# Abaqus/Explicit 积分器调研与 SFC Lagrangian-SDF 对齐结果

## Abaqus/Explicit 算法口径

官方文档将 Abaqus/Explicit 描述为基于对角 lumped mass 的显式中心差分直接积分。核心更新为：

```text
a^n = M_lumped^{-1} (F_ext^n + F_contact^n - F_int^n)
v^{n+1/2} = v^{n-1/2} + dt a^n
u^{n+1} = u^n + dt v^{n+1/2}
```

初始半步速度使用

```text
v^{-1/2} = v^0 - 0.5 dt a^0
```

本次 SFC 对齐实现只采用该核心显式积分器：lumped mass、固定步长、无 HHT/Newmark、无 Newton、无接触切线、无 bulk/contact damping。

参考文档：

- Abaqus Theory Guide, Explicit dynamic analysis: https://docs.software.vt.edu/abaqusv2024/English/SIMACAETHERefMap/simathe-c-expdynamic.htm
- Abaqus Getting Started, Explicit time integration: https://docs.software.vt.edu/abaqusv2024/English/SIMACAEGSARefMap/simagsa-c-ovwtime.htm
- Abaqus Analysis Guide, Explicit Dynamic Analysis: https://docs.software.vt.edu/abaqusv2024/English/SIMACAEANLRefMap/simaanl-c-expdynamic.htm

## SFC 实现

修改文件：

- `validation/run_abaqus_sphere_drop_full_validation.py`
- `tests/test_abaqus_sphere_drop_full_validation.py`

新增能力：

- `--integrator explicit`
- lumped-mass central difference
- fixed-step explicit update
- node-sample Lagrangian-SDF rigid-plane fast path
- vectorized stress/strain summary for explicit output frames

该路径仍然是 SFC 内部 FEM 装配和 Lagrangian-SDF 接触求解，不读取 Abaqus ODB，不使用 Abaqus 矩阵/力文件。

## 对齐工况

参考数据：

```text
results/abaqus_explicit_linear_no_contact_damping_fine_1s/dt_5em06
```

设置：

- total time: `1.0 s`
- fixed dt: `5e-6 s`
- output interval: `0.001 s`
- contact: frictionless linear penalty
- contact stiffness: `5.0e9`
- bulk viscosity: `0, 0`
- contact damping fraction: `0`

SFC 命令：

```text
python validation/run_abaqus_sphere_drop_full_validation.py --inp results/abaqus_explicit_linear_no_contact_damping_fine_1s/dt_5em06/abaqus_run/sphere_drop_explicit_linear_dt_5em06.inp --vtk-dir results/abaqus_explicit_linear_no_contact_damping_fine_1s/dt_5em06/vtk --duration 1.0 --dt 5e-6 --integrator explicit --contact-integration node --output-stride 200 --out-dir results/sfc_explicit_lagrangian_sdf_abaqus_linear_dt5e6_1s_node_fast
```

## 结果

| Metric | Value |
| --- | ---: |
| SFC solve wall time | `28.662549200002104 s` |
| Abaqus/Explicit analysis wall time | `54.84324390022084 s` |
| Abaqus/Explicit total wall time with VTK/energy extraction | `109.08434669999406 s` |
| SFC speedup vs Abaqus analysis | `1.91x` |
| SFC speedup vs Abaqus total | `3.81x` |
| max center-z error | `2.620857548957384e-4 m` |
| RMS center-z error | `9.513284595429342e-5 m` |
| max min-gap replay error | `8.251592139528283e-4 m` |
| explicit steps | `200000` |
| Newton iterations | `0` |

主曲线：

```text
results/sfc_explicit_lagrangian_sdf_abaqus_linear_dt5e6_1s_node_fast/abaqus_sphere_drop_full_z_cm.png
```

## 结论

当前实现已经做到：

- 与 Abaqus/Explicit 同类核心显式积分器；
- 无阻尼、无摩擦、线性 penalty 工况；
- SFC Lagrangian-SDF 独立求解；
- 位移曲线视觉上高度重合；
- SFC core solve 时间低于 Abaqus analysis 时间。

但不能声称逐点完全相同。剩余误差主要来自 Abaqus general contact 的内部接触离散、输出帧重建、接触力/接触面积后处理定义，以及商业求解器的内部约束/接触修正细节。论文表述应写成“closely matches the Abaqus/Explicit trajectory while reducing solve time”，不应写成 source-level equivalence。
