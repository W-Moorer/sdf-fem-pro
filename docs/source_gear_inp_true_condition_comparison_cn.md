# `gear_contact.inp` 齿轮旋转工况对比

本报告只针对真正的 source-drive 齿轮旋转工况，即：

```text
commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp
```

不是上一轮的 closure active-patch 长程算例。

## 已定位的数据

### Abaqus 长程数据

已有 Abaqus source-drive VTK manifest：

```text
results/flexible_gear_source_inp_compare/abaqus_vtk/abaqus_manifest.csv
```

时间范围：

```text
0.0 s -> 0.05000000074505806 s
```

### Abaqus penalty 短窗口参考

已有更接近前期 SFC penalty/source-step 口径的 Abaqus manifest：

```text
results/source_gear_abaqus_penalty_full_stride2_match_step_0030/abaqus_vtk/abaqus_manifest.csv
```

时间范围：

```text
0.0 s -> 0.003000000026077032 s
```

### SFC source-drive 数据

当前已有最长 source-drive SFC 数据：

```text
results/source_gear_penalty_contact_history_only_0030/sfc_full_gear_lagrangian_sdf_history.csv
```

时间范围：

```text
0.0002 s -> 0.003 s
```

最新优化后的 constraint-region/full-surface SFC 数据目前只到：

```text
0.0001 s
```

因此，本次没有把 SFC 伪装成已经完成 `0.05 s` source-drive 全流程；只在真实重叠时间段内比较。

## 0.003 s 对比结果

使用：

```text
SFC:
results/source_gear_penalty_contact_history_only_0030/sfc_full_gear_lagrangian_sdf_history.csv

Abaqus:
results/source_gear_abaqus_penalty_full_stride2_match_step_0030/abaqus_vtk/abaqus_manifest.csv
```

输出：

```text
results/source_gear_inp_true_condition_comparison/source_gear_penalty_step0030_curves.png
results/source_gear_inp_true_condition_comparison/source_gear_penalty_step0030_summary.csv
```

| 指标 | SFC final | Abaqus final | final relative error | L2 relative error |
|---|---:|---:|---:|---:|
| max displacement norm | `4.739842e-02` | `4.561225e-02` | `3.92%` | `9.42%` |
| max node-averaged von Mises | `1.249109e+07` | `4.273199e+07` | `70.77%` | `56.19%` |
| max node-averaged strain norm | `6.374938e-05` | `3.907966e-04` | `83.69%` | `74.15%` |

## 结论

位移曲线在 `0.003 s` 窗口内可以认为已经达到可比较状态，末端误差约 `3.92%`。

但是应力、应变曲线仍明显不对齐。SFC 的最大节点平均 von Mises 和应变范数远低于 Abaqus，说明当前 source-drive 齿轮旋转工况尚不能支撑“应力应变准确对齐”的论文结论。

这不是 SDF 查询本身一个单点误差能够解释的问题，更可能来自：

1. 接触压力分布与 Abaqus surface-to-surface 约束平均口径仍不一致；
2. source-drive 长程使用的 SFC 数据不是最新 constraint-region/full-surface 路径；
3. Abaqus manifest 中的 strain 使用 `LE`/log strain 口径，而 SFC history 中是内部线性/节点平均 strain norm，仍存在输出语义差异；
4. 齿面局部接触压力和应力集中没有被当前 SFC source-drive 长程结果正确传递到体应力场。

## 当前不能支持的 claim

不能写：

```text
SFC 在 gear_contact.inp 齿轮旋转工况中位移、应力、应变均与 Abaqus 长程结果一致。
```

当前最多只能写：

```text
在已有 0.003 s source-drive 窗口内，SFC 位移响应与 Abaqus penalty/source-step 参考基本一致；但应力和应变仍存在显著差异，需要继续对齐接触压力分布、应力输出口径和最新 constraint-region 长程路径。
```

## 下一步

要完成真正的 `0.05 s gear_contact.inp` 对比，需要用最新 constraint-region/full-surface source-drive 路径从 checkpoint 续跑到 `0.05 s`，并输出同口径的：

- max / p95 displacement;
- max / p95 node-averaged von Mises;
- max / p95 node-averaged strain norm;
- contact pressure;
- active contact area;
- RP reaction / equivalent virtual-work reaction。

按当前最新 full-surface 10-step 实测，直接跑满 `0.05 s` 是长任务，不能用 closure active-patch 结果替代。
