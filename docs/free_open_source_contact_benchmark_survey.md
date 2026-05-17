# 免费/开源 FEM 接触 Benchmark 调研

## 结论

已经从项目中移除 NAFEMS/SimScale rounded-punch 路线。原因是：公开
SimScale 页面只暴露了曲线图片，底层 NAFEMS benchmark 报告和权威参考数据
并没有以可公开复现、可直接引用的形式开放。因此当前项目不应该把该算例作为
论文证据。

## 推荐候选

| 来源 | 已公开内容 | 对 FEM-SDF 论文的适配度 | 推荐用途 |
| --- | --- | --- | --- |
| SfePy `two_bodies_contact.py` | 两个弹性体的 penalty contact 开源示例，包含生成网格和 HDF5/VTK 输出。 | 很适合作为公开、可复现的两体接触参考。 | 作为第一条免费外部参考线：比较 gap、接触状态、位移场和压力/接触区 replay。 |
| FuzzyContact Mendeley dataset | CC BY 4.0 数据集，包含多个接触条件下的 VTU 解文件，字段包括位移、应变、应力。 | 很适合三维云图对比，因为参考场已经是标准 VTU 格式。 | 用于三维视觉审计：位移、应变、应力和接触区域对比。 |
| CalculiX 官方/用户示例 | 开源 `.inp` 示例；项目内已有 `contactenergy`、`contact1/3/6`、`ball`、`scheibe2f2f` 和 block contact 相关输入/结果。 | 已经部分接入；适合作为外部 replay/reference，不应成为核心依赖。 | 在引入更丰富的公开数据集前，继续作为主要工程外部参考线。 |
| Code_Aster validation cases | 公开验证手册和 `astest` 输入，包含 Hertz/contact、板料成形、压痕、滑动垫、梁接触等。 | 是公开验证定义的重要来源；部分算例有解析解或求解器参考。 | 优先考虑 frictionless Hertz/indentation；摩擦相关案例只适合补充材料。 |
| FEBio TestSuite / Model Repository | FEBio 是开源求解器，测试套件/模型库中有生物力学接触示例。 | 可用于非线性生物力学接触可视化 benchmark，但需要解析 FEBio 文件和输出。 | 可作为补充外部 replay 候选，不建议作为第一优先级。 |

## 候选细节

### 1. SfePy Two-Body Contact

来源：
`https://sfepy.org/doc/examples/linear_elasticity-two_bodies_contact.html`

这是一个两个弹性体之间的 penalty contact 示例。页面给出完整可运行命令，
并会写出 `output/contact/two_bodies.h5`，同时提供位移场查看命令和非线性
求解日志。

推荐接入方式：

- 将 SfePy 作为可选外部参考运行；
- 导出变形后的两个物体表面，以及 gap/contact-region 数据；
- 从当前 master surface 重建本文的 dynamic SDF field；
- 用 field interpolation replay 同一批 slave samples；
- 比较 gap、active contact set、位移场和接触压力。

这是最推荐的下一步目标，因为它开源、小型、可复现，而且直接对应本文的
two-body contact 设定。

### 2. FuzzyContact Dataset

来源：
`https://data.mendeley.com/datasets/tdgkkjyr2f/3`

该数据集包含来自 FEniCS-based contact formulation 的接触力学解文件。
它提供多个 case 的 VTU 文件，字段包括 displacement、strain 和 stress
components。

推荐接入方式：

- 将 VTU 文件作为 validation-only artifact 下载；
- 解析几何、位移、应变和应力场；
- 根据变形后的表面构造 dynamic SDF field；
- 输出 FEM-SDF replay 与数据集参考场的并排三维云图。

这是当前最强的视觉验证候选，因为参考场已经是标准可视化格式，不需要重新
从图片或封闭报告中数字化。

### 3. CalculiX Examples

来源：

- `https://github.com/calculix/examples`
- `https://github.com/calculix/CalculiX-Examples`

项目已经把若干 CalculiX-style case 作为可选外部验证和 replay 输入使用，
包括 `contactenergy`、`contact1`、`contact3`、`contact6`、`ball`、
`scheibe2f2f`。这些案例仍然有价值，因为它们开源、可运行，也容易在本地
复现或 replay。

推荐接入方式：

- 继续保持 CalculiX 只是 optional validation；
- 用 `contactenergy` 做静态 C3D8 contact-law / contact-energy replay；
- 用 block-plane / block-block trajectory 做动态外部 replay；
- `ball` 或 `scheibe2f2f` 只有在指标通过 claim gate 后再进入论文主证据。

### 4. Code_Aster Validation Cases

来源：

- `https://biba1632.gitlab.io/code-aster-manuals/docs/validation/v_toc.html`
- `https://code-aster.org/V2/doc/v11/en/man_v/v6/v6.03.154.pdf`

Code_Aster validation catalog 包含很多接触案例，例如 axisymmetric Hertz
contact、sheet-metal stamping、extrusion、beam contact、spherical indentation
以及 frictional / X-FEM contact。SSNP154 文档是可公开访问的 NAFEMS
contact-friction benchmark 文档，并说明 frictionless cylinder/roller case
有解析解。

推荐接入方式：

- 优先从 frictionless Hertz / indentation 开始，因为解析比较更干净，
  也符合当前项目约束；
- 主文避免摩擦和复杂非线性材料 claim；
- Code_Aster 输入和文档只作为外部验证参考，不作为核心依赖。

### 5. FEBio

来源：

- `https://github.com/febiosoftware/FEBio`
- `https://github.com/febiosoftware/TestSuite`
- `https://repo.febio.org/modelRepo/UserProjects`

FEBio 是 MIT 许可的开源有限元求解器，面向非线性生物力学 FEM 分析，
包含接触能力。它的模型库和测试套件可以提供比较真实的可变形接触案例。

推荐接入方式：

- 仅作为 optional external replay；
- 优先选择输出字段清楚的案例，例如 displacement、stress、contact gap、
  contact pressure；
- 不要让 FEBio 成为核心依赖。

## 替代计划

1. 从 manuscript 和 codebase 中移除 NAFEMS/SimScale rounded-punch benchmark。
   已完成。
2. 增加 SfePy `two_bodies_contact.py` 作为免费外部参考线。
   已接入为 `validation/run_external_contact_solver_comparison.py`。当前脚本让
   SfePy 完成外部 two-body penalty contact 求解，然后用最终变形 master
   surface 构造 `DynamicNarrowBandSDF.build_required_points(...)` 真场，并通过
   `query_phi(...)` 插值 replay slave contact samples；不再走旧的 projection
   query replay 路径。
3. 增加 FuzzyContact VTU dataset 作为三维场云图参考线。
   已新增 `validation/run_fuzzycontact_vtu_reference.py`。默认只抓取公开元数据并
   写出数据集 DOI、许可证、文件清单；如果用户提供 `--source-dir` 或显式
   `--download --extract`，脚本会解析 VTU、统计 displacement/stress/strain 等
   字段，并生成三维表面云图预览。该数据集较大，不作为 core dependency。
4. 保留 CalculiX `contactenergy` 和 C3D8 trajectory replay 作为已有工程参考。
   保持 optional validation/replay 定位，不进入 core solver 依赖链。
5. 后续评估 Code_Aster frictionless Hertz/indentation，作为解析接触 benchmark。
   暂不实现；优先选择无摩擦 Hertz/indentation，避免引入 friction 或复杂非线性
   主方法 claim。

## 当前执行状态

| 项目 | 状态 | 证据文件 | 论文定位 |
| --- | --- | --- | --- |
| NAFEMS/SimScale rounded punch | 已移除 | `paper/main.tex`、`paper/manuscript.md` 中不再引用 | 不作为论文证据 |
| SfePy two-body contact | 已迁移到 true dynamic SDF field replay | `validation/run_external_contact_solver_comparison.py`、`tests/test_external_contact_solver_comparison.py`、`tests/test_open_source_contact_benchmark_plan.py` | 免费外部接触状态/gap 参考 |
| FuzzyContact VTU dataset | 已新增可选元数据/VTU 解析与三维预览脚本 | `validation/run_fuzzycontact_vtu_reference.py`、`tests/test_fuzzycontact_vtu_reference.py` | 三维位移/应变/应力云图参考线 |
| CalculiX contactenergy/C3D8 | 保留已有路径 | `validation/run_calculix_contactenergy_replay.py`、`validation/run_c3d8_contact_trajectory_validation.py` | 工程外部 replay，不证明 SDF 加速 |
| Code_Aster frictionless Hertz/indentation | 后续候选 | 公开 validation catalog 和 SSNP154 文档 | 解析/公开 benchmark 候选 |

## Claim 策略

完成接入后可以支持：

- 针对具体 case 的开源外部接触 replay；
- 与外部变形表面的 field-query gap 一致性；
- 当参考场可用时，展示 stress/strain/displacement 三维视觉审计；
- 加速 claim 只来自项目内部 timing 和实测 `Q*`。

不能支持：

- 在没有开放参考数据的情况下宣称通过 NAFEMS 官方 benchmark；
- friction、self-contact、nonlinear FEM as main method、GPU、barrier contact、
  neural SDF 或 POD claim；
- 超出具体 replay/validation case 的外部求解器等价性。
