# 最新 source gear 0.003 s 对比与误差来源分析

## 运行设置

本次对比使用最新路径：

- contact averaging: `slave_node_region_constraint`
- contact direction: `secondary_average`
- contact projection: `secondary_line`
- material visualization/internal stress path: `finite_stvk_visual`
- rotating inertia: `finite_kinematic`
- contact mode: `source_penalty`
- drive mode: `source_inp`
- duration: `0.003 s`
- time step: `1e-5 s`
- increments: `300`
- Abaqus VTK manifest: `results/source_gear_abaqus_penalty_full_stride2_match_step_0030/abaqus_vtk/abaqus_manifest.csv`
- SFC output: `results/source_gear_constraint_region_optimized_0030/`

SFC nonlinear solve status:

- converged steps: `300 / 300`
- iteration-limit steps: `0`
- sparse CG solves: `905`
- sparse CG iterations: `94567`
- SFC wall time: `14300.65 s`

## 主要对比结果

| Metric | SFC final | Abaqus final interp | Final relative error | L2 relative error | L_inf relative error |
|---|---:|---:|---:|---:|---:|
| max displacement norm | 4.7285866e-02 | 4.5612250e-02 | 3.67% | 7.59% | 11.52% |
| p95 von Mises nodeavg | 2.2852238e+07 | 2.7060075e+07 | 15.55% | 22.99% | 33.63% |
| max von Mises nodeavg | 4.5362142e+07 | 4.2731986e+07 | 6.16% | 22.62% | 34.76% |
| p95 equivalent strain nodeavg | 9.5124761e-05 | 1.1264031e-04 | 15.55% | 22.99% | 33.63% |
| max equivalent strain nodeavg | 1.8882453e-04 | 1.7787623e-04 | 6.16% | 22.62% | 34.76% |
| mean active contact pressure nodeavg | 3.5123315e+06 | 3.4643564e+05 | 913.85% | 174.28% | 230.65% |

对比曲线：

- `results/source_gear_constraint_region_optimized_0030/latest_constraint_region_vs_abaqus_0030_curves.png`
- `results/source_gear_constraint_region_optimized_0030/latest_constraint_region_vs_abaqus_0030_curves.pdf`

## 为什么仍然对不上

### 1. 位移已经基本对齐，说明全局运动学不是主要矛盾

最终最大位移误差为 `3.67%`，L2 误差为 `7.59%`。这说明：

- RP/MPC 主运动；
- 有限转动运动学；
- 显式/隐式时间窗口中的驱动角速度；
- 全局刚度量级；

已经进入可接受区间。当前最主要问题不是“齿轮整体怎么动”，而是接触局部如何分布压力并传递到应力场。

### 2. p95 应力/应变仍超过 10%，主要来自局部接触压力分布差异

最大 von Mises 和最大等效应变终值误差约 `6.16%`，但 p95 von Mises / p95 strain 终值误差为 `15.55%`。这说明 SFC 的极值水平已经接近 Abaqus，但高应力区域的面积、排序分布或局部扩散方式仍不同。

这类误差通常不是单个 SDF gap 的符号错误，而是接触区域约束平均、接触压力平滑、slave/master region 选择和 nodal pressure recovery 共同造成的分布差异。

### 3. active contact node count 和 pressure 明显不一致，是最直接的证据

最终 active contact node count：

- SFC: `447`
- Abaqus: `88`

最终 mean active contact pressure：

- SFC: `3.51e6`
- Abaqus: `3.46e5`

这说明 SFC 当前把更大的节点区域判为 active，同时 nodal pressure recovery 的尺度也和 Abaqus 不一致。应力/应变曲线不对齐，很大概率由这一层传导造成。

更具体地说，当前 SFC 的 `slave_node_region_constraint` 已经比早期单点/单三角口径更接近 Abaqus，但还没有完全复现 Abaqus surface-to-surface contact 的 constraint region averaging、pressure smoothing、nodal pressure extrapolation 和 contact output semantics。

### 4. SDF 查询本身不是当前首要问题

本次路径已经使用 `secondary_line` 和 constraint-region 约束，且位移结果已接近 Abaqus。如果 SDF gap/normal 的基本几何方向错误，位移曲线不会达到当前误差水平。

因此下一步不应优先继续修改 SDF 查询公式，而应对齐 Abaqus 风格的：

- constraint region 权重；
- surface-to-surface pressure integration；
- contact pressure smoothing；
- active set averaging；
- CPRESS/COPEN 输出恢复口径；
- RP reaction / contact force 的等效虚功定义。

### 5. 计算效率仍未通过

本次 0.003 s、300 步窗口耗时 `14300.65 s`。虽然 300 步全部收敛，但 CG 总迭代数为 `94567`，说明当前主要耗时已经不是单次接触检测，而是 hard/penalty contact tangent solve 的线性迭代和预条件效率。

下一层优化应放在：

- full gear contact candidate/region/payload workspace 复用；
- consistent constraint-region tangent；
- matrix-free hard/penalty contact tangent solve；
- Schur/RP reduced preconditioner；
- 稀疏预条件器复用；
- fused C++ contact residual/tangent workspace。

## 与旧路径相比的改进

旧 source-drive 对比中，应力/应变误差曾达到：

- max nodeavg von Mises final error: `70.77%`
- max nodeavg strain final error: `83.69%`

本次最新路径降低到：

- max nodeavg von Mises final error: `6.16%`
- max nodeavg strain final error: `6.16%`
- p95 stress/strain final error: `15.55%`

这说明 `slave_node_region_constraint + secondary_line + finite_stvk_visual + finite_kinematic` 的方向是正确的，但还没有完成 Abaqus contact output/enforcement semantics 的最后对齐。

## 下一步修复优先级

1. 对齐 Abaqus-style constraint region pressure averaging 和 active set averaging。
2. 对齐 CPRESS/COPEN 的 nodal recovery 口径，避免用 SFC 内部 nodal pressure 与 Abaqus 输出语义直接混比。
3. 在相同 active/contact region 上比较总法向接触力、虚功和接触能，先判断力学等价，再判断 pressure 云图等价。
4. 实现真正 consistent constraint-region tangent，减少 CG 和非线性迭代开销。
5. 将 full gear 0.003 s 窗口扩展前，先让 p95 应力/应变进入 `<10%` 且 contact force/energy 口径一致。
