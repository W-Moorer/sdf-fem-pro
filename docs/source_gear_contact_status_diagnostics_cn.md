# 齿轮工况接触状态诊断记录

本记录用于定位 Abaqus/Standard 原生接触与 SFC Lagrangian-SDF 接触在全齿轮动力学工况中的剩余误差。Abaqus 仅作为外部理论与验证参考，不进入 `src/sfc` 核心求解路径。

## 本轮新增诊断

新增脚本：

```text
validation/run_source_gear_contact_status_diagnostics.py
```

输入：

```text
SFC history CSV
Abaqus VTK contact manifest
Abaqus .sta increment table
```

输出：

```text
results/source_gear_contact_status_diagnostics/source_gear_contact_status_diagnostics.csv
results/source_gear_contact_status_diagnostics/source_gear_contact_status_diagnostics.md
```

该脚本逐帧比较：

- Abaqus 实际增量大小；
- Abaqus severe discontinuity / equilibrium iteration；
- SFC Newton iteration / residual；
- SFC active contact node count；
- Abaqus `CSTATUS/CPRESS` 导出的 active contact node count；
- SFC / Abaqus max contact pressure；
- SFC / Abaqus min gap / COPEN；
- p95 node-averaged von Mises 与 equivalent strain 误差。

## 关键发现

Abaqus `.sta` 显示该基准的实际增量为：

```text
dt = 1.0e-5 s
```

因此我按 `dt=1.0e-5 s` 重新运行 SFC 到 `0.00020 s` 和 `0.00040 s`。结论如下。

### 0.00020 s

结果目录：

```text
results/source_gear_abq_dt_current_settings_00020
```

`0.00020 s` 时：

- 位移相对误差：`0.395%`
- p95 node-averaged von Mises / strain 误差：`1.116%`
- max node-averaged von Mises / strain 误差：`4.673%`
- mean node-averaged von Mises / strain 误差：`2.719%`
- max contact pressure 误差：`7.893%`
- mean active contact pressure 误差：`4.040%`

这说明前半程运动学、应力应变和接触压力仍然在 10% 内。

### 0.00040 s

结果目录：

```text
results/source_gear_abq_dt_current_settings_00040
```

完整时间窗仍未通过。诊断脚本输出：

```text
max p95 stress/strain relative error: 195.317% at t=0.00038
max active-count relative delta: 35100.000% at t=0.00038
release-stage mismatch frames: 1
```

在 `0.00038 s`：

- Abaqus active contact node count：`0`
- Abaqus max CPRESS：`0`
- SFC active contact node count：`351`
- SFC max contact pressure：约 `1.58e7`
- SFC min gap：约 `-3.15e-3`

这说明剩余误差不是由 RP 角速度单位、全局位移、质量矩阵或 SDF 距离查询单独造成的，而是释放阶段的 Abaqus-style contact status / pressure-overclosure enforcement 口径没有对齐。

## 已排除的错误方向

1. **简单把 participation 约束改成 signed-average release 不可用。**
   该试验在 `0.00002 s` 就破坏了原本已对齐的位移和应力，因此不能作为通用修复保留。

2. **只把 SFC 步长改成 Abaqus `.sta` 中的 `1e-5 s` 不足以修复。**
   前半程仍对齐，但释放阶段仍保留虚假 active set。

3. **问题不应继续定位为 SDF query 精度。**
   当前诊断显示 gap/pressure 的错误集中在 contact status release 语义，而不是当前构型 SDF 距离值本身。

## 下一步修复目标

下一层应对齐 Abaqus surface-to-surface finite-sliding penalty contact 的状态更新语义：

1. contact status 必须按当前 secondary constraint region 更新，而不是由任意远处 line intersection 继续保持闭合；
2. released/open constraint 不能继续通过全局 closest feature 产生压缩压力；
3. re-capture 必须发生在 Abaqus-style finite-sliding tracking region 内，而不是全局任意候选；
4. 修复后必须同时降低：
   - release-stage mismatch count；
   - active contact node count delta；
   - p95 stress/strain error。

当前验收门应使用：

```text
python validation/run_source_gear_contact_status_diagnostics.py \
  --sfc-history results/source_gear_abq_dt_current_settings_00040/sfc_full_gear_lagrangian_sdf_history.csv \
  --abaqus-manifest results/source_gear_abaqus_penalty_full_stride2_match_step_0004/abaqus_vtk_contact_full_check/abaqus_contact_full_manifest.csv \
  --abaqus-sta results/source_gear_abaqus_penalty_full_stride2_match_step_0004/abaqus_run/gear_contact_source_penalty.sta \
  --out-dir results/source_gear_contact_status_diagnostics
```
