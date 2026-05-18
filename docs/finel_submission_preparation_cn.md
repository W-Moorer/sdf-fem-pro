# FINEL 投稿版本梳理

依据来源：

- FINEL 官方 Aims and scope：https://www.sciencedirect.com/journal/finite-elements-in-analysis-and-design
- FINEL 官方 Guide for authors：https://www.sciencedirect.com/journal/finite-elements-in-analysis-and-design/publish/guide-for-authors

## 1. 期刊匹配

*Finite Elements in Analysis and Design* 的 scope 强调 finite element computational models 的 development、advanced use、mathematical/theoretical foundations、computer implementation，以及 practical but challenging numerical demonstration。

当前稿件最匹配 FINEL 的角度是：

```text
finite-element contact computational methodology
+ current-space dynamic narrow-band SDF field
+ closest-feature FEM sensitivity payload
+ node/surface field-contact implementation
+ numerical validation and timing
```

不应把稿件包装成：

```text
graphics SDF paper
production contact solver paper
CalculiX reproduction paper
general nonlinear contact framework
```

## 2. 审稿人会看的核心问题

| 审稿问题 | 当前证据 | 风险 |
| --- | --- | --- |
| 是否是 FEM 方法创新，而不是应用现有 FEM？ | closest-feature payload、field Jacobian、动态 SDF field | 中低 |
| 是否有可复现 implementation？ | `src/sfc/sdf/*`, `src/sfc/contact/field_contact.py`, validation scripts | 中低 |
| 是否有 numerical demonstration？ | field accuracy、Jacobian FD、CalculiX native contact、solver timing | 低 |
| 是否过度宣称？ | scope section 已限制 claim | 低 |
| 是否缺少更强 baseline？ | 没有 production BVH/mortar/mortar-SDF 对比 | 中 |
| 是否规模还偏小？ | problem 4 与 solver timing 已补强，但仍是 Python 原型 | 中 |

## 3. 推荐论文结构

FINEL 版本主文建议保持如下结构：

1. Introduction
2. Related Work
3. FEM-Induced Dynamic Narrow-Band SDF Field
4. Field-Interpolated Contact Formulation
5. Algorithm and Cost Model
6. Implementation
7. Experiments
8. Scope of Claims
9. Supplementary Context
10. Limitations
11. Conclusion

其中第 7 节必须突出：

- field accuracy and Eikonal residual；
- field-contact Jacobian finite difference；
- material-space SDF baseline；
- amortized timing and crossover；
- independent CalculiX native-contact comparison；
- solver-level timing and backend ablation；
- external 3D visual FEM validation。

## 4. 当前创新点强度判断

### 强项

- 把 projection kernel 内化为 field construction，而不是主查询。
- 每个 SDF grid node 存 closest-feature payload，使 master-side sensitivity 有来源。
- slave Jacobian 用 scalar trilinear field derivative，理论一致性比插值 normal 更好。
- 同时保留 node-to-surface 和 surface-to-surface 后端，避免 claim 过窄。
- 实验链条包含 field、Jacobian、native external comparison、solver-level timing。

### 弱项

- 不是完整 nonlinear FEM 主方法。
- 没有 friction/self-contact。
- 没有 production BVH 或 mortar contact 的直接大规模比较。
- Python 原型性能存在 field update 主导场景，problem 4 中慢于 CalculiX。

## 5. 投稿策略

推荐首投 FINEL，而不是 CMAME/IJNME，原因：

- 当前稿件更像 finite-element method implementation and validation，而不是顶刊级广义数值理论突破。
- FINEL 明确接受 computer implementation、code validation、visualization 和 practical challenging numerical demonstration。
- 当前证据足以支撑 scoped method contribution，但还不足以支撑 CMAME/IJNME 的更高理论和规模要求。

## 6. 投稿前最后三项

1. 用正式作者信息替换 anonymous title page。
2. 决定是否公开数据/代码 DOI。
3. 对 references 做 DOI/格式审计，确保 Elsevier 数字引用风格完整。
