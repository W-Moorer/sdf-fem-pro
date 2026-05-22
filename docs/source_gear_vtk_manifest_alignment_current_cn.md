# Source gear VTK manifest alignment current result

本记录对应商业软件齿轮工况 `commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp`。

## 工况口径

- 原始 Abaqus 步：`*Dynamic`，`1e-05, 0.05, 1e-10, 5e-05`。
- 当前 SFC 建模/驱动：读取同一 `gear_contact.inp` 的网格、材料、RP-MPC hub、G1 绕 z 角速度和 G2 绕 z 扭矩。
- 当前 SFC 接触：`source_penalty` 罚函数接触。
- 当前 SFC 输出：VTK 隔位/隔步保存由 runner 参数控制；动画使用 node-averaged stress/strain 和固定全局色标。

## 当前可核验基线

当前使用已有 100-step SFC 与 Abaqus VTK manifest 做同步长对比：

```powershell
python validation\run_source_gear_vtk_manifest_alignment.py `
  --sfc-manifest results\source_gear_penalty_100steps_stride10_p95\sfc_vtk\sfc_manifest.csv `
  --abaqus-manifest results\source_gear_penalty_abaqus_100steps_stride10_p95\abaqus_vtk\abaqus_manifest.csv `
  --out-dir results\source_gear_penalty_100steps_stride10_p95_alignment_current
```

输出：

- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/sfc_vs_abaqus_vtk_metric_errors.csv`
- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/sfc_vs_abaqus_vtk_metric_curves.png`
- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/animation_fixed_color_ranges.csv`
- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/paraview_fixed_range_animation.py`
- `results/source_gear_penalty_100steps_stride10_p95_alignment_current/source_gear_vtk_manifest_alignment_summary.md`

末帧误差：

| 指标 | 相对误差 |
| --- | ---: |
| displacement magnitude | 3.927% |
| p95 node-averaged von Mises | 20.832% |
| p95 node-averaged equivalent elastic strain | 20.832% |

## 结论

该结果证明当前输出流水线已经能从 SFC/Abaqus 的 VTK 云图字段生成位移、应力、应变曲线，并给出固定色标 ParaView 脚本，解决动画因逐帧自动缩放造成的应力/应变闪动问题。

但这还不是最终验收所需的 0.05 s 全时长结果。全时长仍需要按同一 `gear_contact.inp` 的 `0.05 s / 1e-5 s` 跑完 SFC，并与 Abaqus 全程 VTK manifest 做同一入口的对比。
