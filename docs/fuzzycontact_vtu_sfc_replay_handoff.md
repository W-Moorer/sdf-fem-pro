# FuzzyContact VTU 到 SFC Replay 映射交接

## 目的

本次工作把公开 FuzzyContact VTU 数据集作为外部三维场数据源，下载并解析其中的 `displacement`、`strain`、`stress` 场，然后把这些场映射到 SFC 的 true dynamic narrow-band SDF replay 样本上。

这条验证线只作为外部工程场数据审计，不把 FuzzyContact 变成核心依赖，也不声明与 FuzzyContact 完整非线性接触求解轨迹等价。

## 数据来源

- 数据集：Dataset of FuzzyContact
- 页面：https://data.mendeley.com/datasets/tdgkkjyr2f/3
- DOI：10.17632/tdgkkjyr2f.3
- 许可：CC BY 4.0
- 本地下载包：`results/fuzzycontact_vtu_reference_full/downloads/Dataset_for_Solutions.zip`

脚本会过滤 zip 中的 `__MACOSX` 和 `._*.vtu` AppleDouble 伪文件，只解析真正的 `.vtu`。

## 实现内容

- 新增 VTU 文件分类：按 `problem`、`scenario`、`body` 和 field kind 识别数据。
- 合并同一 body 上共享网格的：
  - `displacement`
  - `strain`
  - `stress`
  - `von Mises`，如果数据集中提供
- 自动识别 paired-body replay：
  - `bottom -> top`
  - `base -> indenter`
- 从 master 接触侧表面构建 `DynamicNarrowBandSDF`。
- 对 slave 接触侧表面采样，查询：
  - SFC interpolated gap `phi`
  - scalar-field gradient normal
  - closest-feature payload
- 把外部 VTU 场映射到 replay 样本：
  - slave displacement magnitude
  - slave strain norm
  - slave stress norm
  - slave von Mises，如果存在
  - master payload-interpolated displacement magnitude
  - master payload-interpolated strain norm
  - master payload-interpolated stress norm
  - master payload-interpolated von Mises，如果存在

## 运行命令

```powershell
python validation/run_fuzzycontact_vtu_reference.py `
  --source-dir results/fuzzycontact_vtu_reference_full/downloads/Dataset_for_Solutions `
  --max-files 18 `
  --max-replay-groups 3 `
  --max-replay-samples 80 `
  --out-dir results/fuzzycontact_sfc_replay_p1_p2
```

另行运行压头-基体案例：

```powershell
python validation/run_fuzzycontact_vtu_reference.py `
  --source-dir results/fuzzycontact_vtu_reference_full/downloads/Dataset_for_Solutions/Dataset_for_Solutions/problem_3 `
  --max-files 8 `
  --max-replay-groups 1 `
  --max-replay-samples 80 `
  --out-dir results/fuzzycontact_sfc_replay_problem3
```

## 本次输出

- `results/fuzzycontact_sfc_replay_p1_p2/fuzzycontact_dataset_metadata.csv`
- `results/fuzzycontact_sfc_replay_p1_p2/fuzzycontact_vtu_inventory.csv`
- `results/fuzzycontact_sfc_replay_p1_p2/fuzzycontact_vtu_field_summary.csv`
- `results/fuzzycontact_sfc_replay_p1_p2/fuzzycontact_sfc_replay_samples.csv`
- `results/fuzzycontact_sfc_replay_p1_p2/fuzzycontact_sfc_replay_summary.csv`
- `results/fuzzycontact_sfc_replay_p1_p2/fuzzycontact_vtu_plots.csv`
- `results/fuzzycontact_sfc_replay_p1_p2/fuzzycontact_vtu_reference_summary.md`
- `results/fuzzycontact_sfc_replay_p1_p2/figures/fuzzycontact_vtu_preview.png`
- `results/fuzzycontact_sfc_replay_p1_p2/figures/fuzzycontact_sfc_replay_samples.png`
- `results/fuzzycontact_sfc_replay_problem3/fuzzycontact_sfc_replay_samples.csv`
- `results/fuzzycontact_sfc_replay_problem3/fuzzycontact_sfc_replay_summary.csv`
- `results/fuzzycontact_sfc_replay_problem3/figures/fuzzycontact_sfc_replay_samples.png`

## 映射结果

| group | samples | failed | phi min | phi mean | phi max | build seconds | query seconds/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| `problem_1:default:bottom->top` | 80 | 0 | -0.421737 | -0.008976 | 0.411959 | 68.047966 | 0.0002496 |
| `problem_2:normal:bottom->top` | 80 | 0 | -3.590820 | 3.634815 | 4.831986 | 60.060265 | 0.0002263 |
| `problem_2:normal_and_tangential:bottom->top` | 80 | 0 | -4.828072 | -3.199228 | 3.559411 | 54.085697 | 0.0002379 |
| `problem_3:default:base->indenter` | 80 | 0 | -0.515028 | 0.022022 | 0.541956 | 65.503039 | 0.0002350 |

共生成 320 个 SFC replay 样本。每个样本包含 SDF gap、normal、closest-feature payload 映射后的 master 场值，以及 slave 侧 VTU 场值。

## Claim 边界

本次结果支持：

- FuzzyContact VTU 可以作为免费外部三维场数据源解析。
- 外部 `displacement / strain / stress` 场已经映射到 SFC dynamic-SDF replay 样本。
- replay 查询使用 SDF field interpolation；projection 仍只用于 field build。
- 结果可用于论文中的外部三维场可视化和工程场审计。

本次结果不支持：

- 与 FuzzyContact 完整非线性接触求解轨迹等价。
- friction/self-contact/barrier contact 的主方法 claim。
- 以 FuzzyContact 证明 SDF 加速；SDF 加速仍应由 field accuracy、Jacobian FD 和 `Q*` 成本模型证明。
- 通用生产级 FEM 接触求解器 claim。
