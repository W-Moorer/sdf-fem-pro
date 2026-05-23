# 源齿轮目标最新状态

## 目标口径

目标仍然是参照：

`commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp`

来设置当前模型的驱动和建模，只调整接触方式和积分计算方式，运行相同时间与时间步长，保存 Abaqus/SFC 各自应力应变云图，并对比位移、应力、应变曲线。

源 deck 的关键设置已经由 `validation/audit_source_gear_goal_status.py` 自动读取：

- `*Dynamic`: initial dt `1e-5 s`, total time `0.05 s`, min dt `1e-10 s`, max dt `5e-5 s`;
- `nlgeom=YES`;
- Gear 1 RP dof 6 angular velocity: `52.36 rad/s`;
- Gear 2 RP dof 6 torque: `50`;
- contact: `HARD`, `SURFACE TO SURFACE`;
- BEAM MPC count: `2`。

## 当前已有证据

当前已有可视化和曲线证据覆盖到约 `0--2 ms`：

- SFC VTK frames: `101`；
- Abaqus contact VTK frames in latest comparison window: `50`；
- paired curve frames: `100`；
- SFC VTK 使用 `linear_corotated` stress/strain postprocess；
- SFC VTK manifest 中 rotation unit 为 `radian`；
- SFC/Abaqus 都已有 contact pressure/status 相关 VTK 字段。

当前合并曲线中：

- displacement max p95 relative error: `4.82%`；
- von Mises max p95 relative error: `68.16%`；
- equivalent elastic strain max p95 relative error: `68.16%`；
- strain norm max p95 relative error: `76.44%`。

## 关键缺口

当前可复用 Abaqus penalty ODB 并不是完整 `0.05 s` 参考。

核验脚本读取：

`results/source_gear_abaqus_penalty_full_stride2_match_step_0020/abaqus_run/gear_contact_source_penalty.sta`

得到：

- completed increment: `200`;
- completed time: `0.002 s`;
- status: analysis completed successfully, but only for this shortened deck.

因此当前 `results/source_gear_abaqus_penalty_full_stride2_match_step_0020` 只能作为短程 `0--2 ms` 诊断参考，不能证明原始目标要求的完整 `0.05 s` 对比已经完成。

我尝试从该 ODB 导出 `2--4 ms` 分窗 VTK，只得到 1 帧，进一步确认 ODB 不包含超过 `2 ms` 的后续帧。

## 已新增自动核验

新增脚本：

`validation/audit_source_gear_goal_status.py`

输出：

`results/source_gear_goal_status/source_gear_goal_status.csv`

当前核验项：

| Requirement | Status |
| --- | --- |
| source deck settings parsed | PASS |
| source drive units radian | PASS |
| SFC VTK stress/strain frames exist | PASS |
| Abaqus VTK contact frames exist | PASS |
| Abaqus reference duration matches source | MISSING |
| paired displacement/stress/strain curves exist | PASS |
| full source duration covered | MISSING |
| output frame stride documented | PASS |
| stress/strain alignment currently acceptable | MISSING |

## 下一步

要继续满足原始目标，需要先生成真正全时长的 Abaqus penalty reference：

```powershell
python validation\run_flexible_gear_source_penalty_abaqus.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --out-dir results\source_gear_abaqus_penalty_full_stride2_match_step_0050 `
  --pressure-stiffness 5e9 `
  --frame-stride 2 `
  --dt 0.00001 `
  --duration 0.05 `
  --require-source-timing `
  --scalars-only
```

同时 SFC 需要从当前 checkpoint 继续运行到 `0.05 s`：

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --source commercial_software_comparison\abaqus_flexible_body_gear_contact\gear_contact.inp `
  --drive-mode source_inp `
  --contact-mode penalty `
  --tet4-mass-kind consistent `
  --active-faces-per-body 0 `
  --active-patch-radius-factor 1.0 `
  --duration 0.05 `
  --dt 0.00001 `
  --pressure-stiffness 5e9 `
  --write-sfc-vtk `
  --vtk-frame-stride 2 `
  --vtk-scalars-only `
  --history-frame-stride 2 `
  --source-checkpoint results\source_gear_penalty_contact_incremental_check_sfc\source_drive_checkpoint.npz `
  --resume-source-checkpoint `
  --source-checkpoint-stride 20 `
  --out-dir results\source_gear_penalty_contact_incremental_check_sfc
```

注意：按当前性能估算，完整 `0.05 s` 需要 `5000` 个固定时间步；这不是短作业。当前 `0--2 ms` 的 SFC wall time 已经约 `344.7 s`，Abaqus `0--2 ms` 报告 wall time 约 `3305 s`。全时长运行必须按长任务处理。
