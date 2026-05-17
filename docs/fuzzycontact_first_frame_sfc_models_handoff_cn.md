# FuzzyContact 1/3/4 第一帧 SFC 模型交接

## 工作内容

已按顺序完成：

1. Benchmark 1：sphere-block Hertz。
2. Benchmark 3：V-indenter。
3. Benchmark 4：cylinder-platen。

新增脚本：

```text
validation/run_fuzzycontact_first_frame_sfc_models.py
```

新增测试：

```text
tests/test_fuzzycontact_first_frame_sfc_models.py
```

该脚本从 FuzzyContact VTU 解场文件构造 SFC 模型入口，使用当前项目的：

```text
DynamicNarrowBandSDF.build_required_points(...)
-> field_contact interpolation query
```

来计算接触 gap、normal、active sample、penalty contact force 和 field update/query timing。

## 重要边界

这一步是 **first-frame external benchmark mesh reproduction**，不是完整 FuzzyContact 求解器等价复现。

原因：

- FuzzyContact 数据集公开的是 VTU solution fields，不是原始 FEniCS input deck。
- FuzzyContact 使用 sigmoidal traction law，当前 SFC 使用 zero-penetration penalty contact。
- 在 FuzzyContact reference displacement state 下，表面可能仍有正间隙但存在 proximity traction；SFC penalty law 只有负 gap 才激活。

因此脚本输出两个接触状态：

1. `reference_displacement`：使用 FuzzyContact VTU 位移场对应的当前构型，检查 SDF gap 和场查询。
2. `sfc_closed_contact`：在同一第一帧模型上沿法向闭合间隙，使 SFC penalty contact 激活，检查 SDF-contact force 路径。

## 参考坐标规则

| Case | 参考坐标 |
| --- | --- |
| problem_1 bottom/top | `points - displacement` |
| problem_3 base/indenter | `points` |
| problem_4 bottom/top | `points` |

该规则写入：

```text
results/fuzzycontact_first_frame_sfc/fuzzycontact_first_frame_models.csv
```

## 运行命令

```powershell
python validation/run_fuzzycontact_first_frame_sfc_models.py `
  --quick `
  --problems problem_1,problem_3,problem_4 `
  --source-dir results\fuzzycontact_vtu_reference_full\downloads\Dataset_for_Solutions `
  --out-dir results\fuzzycontact_first_frame_sfc `
  --max-contact-samples 16 `
  --max-master-faces 160
```

## 输出文件

```text
results/fuzzycontact_first_frame_sfc/
  fuzzycontact_first_frame_models.csv
  fuzzycontact_first_frame_field_contact.csv
  fuzzycontact_first_frame_samples.csv
  fuzzycontact_first_frame_plots.csv
  fuzzycontact_first_frame_summary.md
  figures/problem_1_first_frame_fields_3d.png
  figures/problem_1_first_frame_fields_3d.pdf
  figures/problem_3_first_frame_fields_3d.png
  figures/problem_3_first_frame_fields_3d.pdf
  figures/problem_4_first_frame_fields_3d.png
  figures/problem_4_first_frame_fields_3d.pdf
```

## 模型重建结果

| Case | Nodes | Elements | Reference rule | 外部参考量 |
| --- | ---: | ---: | --- | --- |
| problem_1_bottom_to_top | 2637 | 11358 | `points_minus_displacement` | FuzzyContact sphere indentation depth = 0.03303 mm |
| problem_3_base_to_indenter | 5104 | 22727 | `points` | FuzzyContact V-indenter indentation depth = 0.001876 mm |
| problem_4_bottom_to_top | 11712 | 58633 | `points` | FuzzyContact effective Young's modulus = 196691 MPa |

## 场变量对比

用公开 VTU 位移场进行 SFC TET4 线弹性后处理，与 VTU stress/mises 场的最大 von Mises 值比较：

| Case | Master max von Mises rel. error | Slave max von Mises rel. error |
| --- | ---: | ---: |
| problem_1_bottom_to_top | 5.93% | 1.16% |
| problem_3_base_to_indenter | 3.56% | 33.68% |
| problem_4_bottom_to_top | 21.97% | 14.86% |

这些指标说明第一帧模型和线弹性后处理与公开场数据在同一量级，problem 1 最强，problem 3 的 indenter 局部峰值误差较大，problem 4 中等。

## SDF field-contact 结果

| Case | State | phi min | Active samples | Field update | Query time/sample | Force norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| problem_1_bottom_to_top | reference_displacement | 5.896993e-02 | 0 | 0.670982 s | 0.341100 ms | 0 |
| problem_1_bottom_to_top | sfc_closed_contact | -1.249507e-02 | 5 | 0.671031 s | 0.364156 ms | 36.7041 |
| problem_3_base_to_indenter | reference_displacement | 1.123241e-01 | 0 | 1.070337 s | 0.371137 ms | 0 |
| problem_3_base_to_indenter | sfc_closed_contact | -1.999992e-02 | 4 | 1.049235 s | 0.342075 ms | 14.3975 |
| problem_4_bottom_to_top | reference_displacement | 6.005441e-02 | 0 | 1.160744 s | 0.345806 ms | 0 |
| problem_4_bottom_to_top | sfc_closed_contact | -5.000000e-02 | 16 | 1.153424 s | 0.340400 ms | 80.3160 |

## 图像

三张主图均为三维面片云图，不是二维点云：

- `problem_1_first_frame_fields_3d.png`
- `problem_3_first_frame_fields_3d.png`
- `problem_4_first_frame_fields_3d.png`

每张图包含：

1. displacement magnitude；
2. strain norm；
3. von Mises stress。

Matplotlib 字体配置为 Times-style serif：

```text
Times New Roman, Times, Nimbus Roman, STIXGeneral, DejaVu Serif
```

更新：图像渲染已从 quick face sampling 改为连续边界面渲染，并关闭三角形边线；因此新版 PNG/PDF 不再显示离散三角片。

## 当前支持的 claim

支持：

- 可以用公开 FuzzyContact 第一帧/解场 VTU 构造 SFC 本地 TET4 benchmark 模型。
- problem 1/3/4 均能走通 `DynamicNarrowBandSDF + field_contact` 插值查询路径。
- reference displacement state 和 SFC closed-contact state 都有可核验的 gap/timing/force 输出。
- 三维 displacement/strain/von Mises 云图已经生成。

不支持：

- 完整 FuzzyContact sigmoidal traction-law 等价。
- 完整全局静力/动力求解轨迹等价。
- friction/self-contact/barrier/GPU/POD/neural/Abaqus 相关 claim。
- 用 FuzzyContact 云图证明 SDF 加速；SDF 加速仍需由 field update/query timing 和 Q* 成本模型证明。
