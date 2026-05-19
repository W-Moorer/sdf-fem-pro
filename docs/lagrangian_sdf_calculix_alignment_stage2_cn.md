# SFC 拉格朗日 SDF 与 CalculiX 同算例对齐阶段二结果

## 阶段目标

本阶段进入用户指定的第二步：

```text
同一 TET4/HEX8 静力/动力算例
-> SFC MaterialSDF + LagrangianSDFContactOracle 独立计算
-> CalculiX native *CONTACT PAIR 独立计算
-> 对齐位移、应变、应力曲线和误差
```

本阶段不做时间优势 claim。只有精度对齐通过之后，才允许进入完整 wall time 对比。

## 新增 runner

脚本：

```text
validation/run_final_aim_calculix_alignment.py
```

核心设置：

- SFC 路径：`MaterialSDF + LagrangianSDFContactOracle + lagrangian_oracle_penalty_response`
- CalculiX 路径：原生 `*CONTACT PAIR, TYPE=SURFACE TO SURFACE`
- 静力：位移控制压缩接触
- 动力：初速度接触冲击
- 元素：TET4/C3D4 与 HEX8/C3D8
- 接触平面：CalculiX shell midsurface 下移半厚度，使 `SPOS` 物理接触面与 SFC 的 `z=0` 平面对齐
- SFC 接触离散：面面积权重 quadrature，而不是第一阶段的 nodal samples

## 正式运行命令

```powershell
python validation\run_final_aim_calculix_alignment.py --timeout-seconds 240 --out-dir results\final_aim_calculix_alignment_existing_cases_partial
```

输出：

- `results/final_aim_calculix_alignment_existing_cases_partial/final_aim_calculix_alignment_history.csv`
- `results/final_aim_calculix_alignment_existing_cases_partial/final_aim_calculix_alignment_errors.csv`
- `results/final_aim_calculix_alignment_existing_cases_partial/final_aim_calculix_alignment_commands.csv`
- `results/final_aim_calculix_alignment_existing_cases_partial/final_aim_calculix_alignment_plots.csv`
- `results/final_aim_calculix_alignment_existing_cases_partial/final_aim_calculix_alignment_summary.md`

## CalculiX 完成状态

| Case | SFC completed | CalculiX completed | Notes |
| --- | --- | --- | --- |
| static TET4 | true | true | 完整对齐 |
| static HEX8 | true | true | 完整对齐 |
| dynamic TET4 | true | false | CalculiX 在接触冲击后 cutback 停止；已解析 `.dat` 中的 partial trajectory |
| dynamic HEX8 | true | false | CalculiX 在接触冲击后 cutback 停止；已解析 `.dat` 中的 partial trajectory |

因此本阶段支持：

- 静力 TET4/HEX8 的完整 CalculiX native contact 对齐；
- 动力 TET4/HEX8 的 SFC 完整运行；
- 动力 TET4/HEX8 的 CalculiX partial trajectory 诊断。

本阶段不支持：

- non-quick 动力 TET4/HEX8 的完整 CalculiX trajectory-equivalence claim；
- solver-level timing superiority claim。

## 误差结果

来自：

```text
results/final_aim_calculix_alignment_existing_cases_partial/final_aim_calculix_alignment_errors.csv
```

### 静力 TET4

| Metric | L2 rel. | Max abs. | Final rel. | Peak rel. |
| --- | ---: | ---: | ---: | ---: |
| mean_uz | 3.712e-03 | 3.983e-04 | 7.170e-04 | 7.170e-04 |
| max_abs_u | 1.389e-16 | 1.388e-17 | 0 | 0 |
| max_von_mises | 2.606e-01 | 8.762e-01 | 1.120e-02 | 1.120e-02 |
| max_strain_norm | 2.414e-01 | 4.853e-03 | 3.509e-02 | 3.509e-02 |

### 静力 HEX8

| Metric | L2 rel. | Max abs. | Final rel. | Peak rel. |
| --- | ---: | ---: | ---: | ---: |
| mean_uz | 3.799e-03 | 4.068e-04 | 8.467e-04 | 8.467e-04 |
| max_abs_u | 2.773e-16 | 2.776e-17 | 0 | 0 |
| max_von_mises | 2.617e-01 | 7.188e-01 | 1.270e-02 | 1.270e-02 |
| max_strain_norm | 2.431e-01 | 4.149e-03 | 3.151e-02 | 3.151e-02 |

静力曲线的 L2 stress/strain 相对误差偏大，主要来自接触刚激活附近的小值区；终值/峰值误差更能说明最终云图量级是否对齐。静力最终应力峰值误差约 1.1% 到 1.3%，最终应变峰值误差约 3.2% 到 3.5%。

### 动力 TET4 partial trajectory

| Metric | L2 rel. | Max abs. | Final rel. | Peak rel. |
| --- | ---: | ---: | ---: | ---: |
| mean_uz | 2.617e-02 | 3.518e-03 | 4.865e-02 | 5.188e-02 |
| max_abs_u | 2.402e-02 | 3.393e-03 | 4.661e-02 | 4.882e-02 |
| max_von_mises | 2.013e-01 | 8.290e-01 | 2.413e-01 | 2.969e-01 |
| max_strain_norm | 1.706e-01 | 4.030e-03 | 1.796e-01 | 2.536e-01 |

### 动力 HEX8 partial trajectory

| Metric | L2 rel. | Max abs. | Final rel. | Peak rel. |
| --- | ---: | ---: | ---: | ---: |
| mean_uz | 1.204e-02 | 1.829e-03 | 2.279e-02 | 2.626e-02 |
| max_abs_u | 1.267e-02 | 2.021e-03 | 2.668e-02 | 2.854e-02 |
| max_von_mises | 1.789e-01 | 6.790e-01 | 3.145e-01 | 3.145e-01 |
| max_strain_norm | 1.783e-01 | 3.246e-03 | 2.692e-01 | 2.692e-01 |

动力结果只能作为 partial diagnostic。CalculiX 原生接触未完成完整动力轨迹，因此不能用这些行支持完整动力等价或时间优势。

## quick 完整性检查

quick 模型下四个 CalculiX native contact cases 均完成：

```powershell
python validation\run_final_aim_calculix_alignment.py --quick --timeout-seconds 180 --out-dir results\final_aim_calculix_alignment_quick_aligned
```

quick 对齐结果：

- 静力位移 L2 误差约 `5.7e-4` 到 `6.0e-4`
- 静力最终应力峰值误差约 `1.1%` 到 `1.3%`
- 动力位移 L2 误差约 `5.4e-4` 到 `1.6e-3`
- 动力应力/应变曲线仍有 `20%` 左右相位/峰值误差

quick 结果证明 runner、输入卡、解析和曲线输出链条有效；non-quick 动力仍需要接触时间积分对齐。

## 结论

当前第二阶段已经补上：

1. 同算例 CalculiX native contact 输入生成；
2. SFC 拉格朗日 SDF 独立运行；
3. CalculiX 独立运行；
4. 位移、应变、应力曲线输出；
5. L2、最大绝对、终值、峰值误差表；
6. 曲线图，不使用柱状图；
7. 明确区分 completed 与 partial trajectory。

但还不能进入时间优势阶段。原因是：

- non-quick 动力 CalculiX 原生接触没有完成完整轨迹；
- 动力 stress/strain 曲线误差仍未达到可作为最终论文 claim 的程度；
- 只有静力 TET4/HEX8 已达到较好的最终场量级对齐。

下一步应继续做动力精度对齐，而不是比较时间：

1. 使用 CalculiX 可稳定完成的动力时间积分设置，或降低冲击强度但必须同步修改 SFC；
2. 对齐 HHT/Newmark 参数、质量矩阵形式、contact damping/impact rule；
3. 对齐 CalculiX generated contact spring 与 SFC surface quadrature 的采样权重；
4. 完整完成 non-quick 动力轨迹后，再进入 wall time 对比。
