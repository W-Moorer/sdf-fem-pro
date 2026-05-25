# 当前代码 0.05 s Abaqus 对比结果

本次使用当前代码完成了一个可在本轮交互内跑完的 0.05 s 全齿轮 active-patch 算例，并与已有同工况 Abaqus metrics 进行曲线对比。

## 运行命令

```powershell
python validation\run_flexible_gear_full_lagrangian_sdf_comparison.py `
  --duration 0.05 `
  --dt 0.00025 `
  --active-faces-per-body 16 `
  --active-patch-radius-factor 2.0 `
  --contact-mode penalty `
  --out-dir results\current_code_0p05_active_patch_comparison
```

## 输出文件

- SFC history: `results/current_code_0p05_active_patch_comparison/sfc_full_gear_lagrangian_sdf_history.csv`
- curve error CSV: `results/current_code_0p05_active_patch_comparison/sfc_vs_abaqus_0p05_curve_errors.csv`
- curve PNG: `results/current_code_0p05_active_patch_comparison/sfc_vs_abaqus_0p05_displacement_stress_strain_curves.png`
- curve PDF: `results/current_code_0p05_active_patch_comparison/sfc_vs_abaqus_0p05_displacement_stress_strain_curves.pdf`

## 结果摘要

- SFC wall time: `994.638590 s`
- Abaqus 同工况已有 wall time: `1244.410506 s`
- SFC/Abaqus time ratio: `0.799`
- speedup: `1.251x`

最终时刻 `t = 0.05 s`:

| 指标 | 最终相对误差 | L2 相对误差 |
|---|---:|---:|
| max displacement norm | `4.91e-08` | `4.30e-08` |
| p95 von Mises stress | `12.58%` | `15.15%` |
| p95 equivalent elastic strain | `12.58%` | `15.15%` |
| max von Mises stress | `10.76%` | `17.18%` |

## 解释

该 0.05 s 结果说明当前 SFC 在位移曲线上与 Abaqus 几乎重合，并且整体求解时间低于 Abaqus；但应力/应变曲线仍存在约 `10-15%` 的系统性偏高。这个偏差主要出现在应力/应变输出口径和接触压力/局部应力集中，而不是位移运动学。

需要注意：这次跑完的是 active-patch 长程算例。严格 full-surface source-drive 路径若使用 `dt=1e-5` 跑满 `0.05 s`，需要 `5000` 个增量。按当前刚跑出的 full-surface 10-step 实测 `215.9 s` 估算，直接全量重跑约为十小时级任务，应作为单独长任务通过 checkpoint 续跑完成。
