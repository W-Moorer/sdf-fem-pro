# SFC 拉格朗日 SDF 既有论文算例阶段一结果

## 阶段目标

本阶段只验证当前项目论文中已有的小型 TET4/HEX8 静力与动力接触算例，是否能够通过新的 final-aim 路径完成 SFC 拉格朗日 SDF 计算：

```text
MaterialSDF -> LagrangianSDFContactOracle -> oracle penalty response
```

本阶段不做 CalculiX 精度对齐，也不做 solver-level 加速 claim。CalculiX 对比应在这些 SFC 算例稳定完成之后进入下一阶段。

## 已完成运行

命令：

```powershell
python validation\run_final_aim_lagrangian_contact_cases.py --out-dir results\final_aim_lagrangian_contact_existing_paper_cases
```

输出：

- `results/final_aim_lagrangian_contact_existing_paper_cases/final_aim_lagrangian_contact_cases.csv`
- `results/final_aim_lagrangian_contact_existing_paper_cases/final_aim_lagrangian_contact_samples.csv`
- `results/final_aim_lagrangian_contact_existing_paper_cases/final_aim_lagrangian_contact_history.csv`
- `results/final_aim_lagrangian_contact_existing_paper_cases/final_aim_lagrangian_contact_summary.md`

## 结果

| Case | Regime | Element | Wall time (s) | Max gap error | Force rel. error | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `static_linear_tet4_lagrangian_oracle` | static | TET4 | 4.561830e-02 | 0.000000e+00 | 0.000000e+00 | passed |
| `dynamic_linear_tet4_lagrangian_oracle` | dynamic | TET4 | 1.105413e+00 | 0.000000e+00 | 0.000000e+00 | passed |
| `static_linear_hex8_lagrangian_oracle` | static | HEX8 | 1.109900e-02 | 0.000000e+00 | 0.000000e+00 | passed |
| `dynamic_linear_hex8_lagrangian_oracle` | dynamic | HEX8 | 1.252897e+00 | 0.000000e+00 | 0.000000e+00 | passed |

总 wall time：`2.415027e+00` s。

## 结论

这一步证明了当前论文已有的 TET4/HEX8 静力/动力小型接触算例已经可以在 SFC 拉格朗日 SDF 路径下完整完成，并且与同一 penalty law 的解析平面参考在 gap、force、trajectory 层面一致。

支持的阶段性 claim：

- 既有 TET4/HEX8 静力/动力算例可以迁移到 `MaterialSDF + LagrangianSDFContactOracle`。
- 该路径不构建 current-space SDF grid。
- 该路径不调用 legacy dynamic surface SDF / projection query 作为主方法。
- 在这些既有小型算例上，gap 与 force 对齐到数值零误差。

不支持的 claim：

- 还不能声称与 CalculiX 位移/应变/应力曲线对齐。
- 还不能声称 solver-level 时间优于 CalculiX。
- 还不能声称 commercial sphere-cantilever 工况已经高效或高精度对齐。

## 本轮实现上的关键优化

本轮同时修复了拉格朗日 SDF oracle 的一个主要性能问题：

- `ReferencePatchBVH` 现在保存 tight patch AABB，并用 tight AABB 距离下界进行精确候选排序和 branch-and-bound 截断。
- 原有 cell AABB 仍保留 padding 作为 broad-phase 覆盖范围，因此不通过缩小搜索半径换速度。
- `lagrangian_oracle_penalty_response()` 的 active force/tangent 改为一次性 COO/CSR 装配，避免逐约束 `J.T @ J` 稀疏矩阵累加。

commercial sphere-cantilever 的局部 profiling 显示：

- 50 samples contact response：从约 `19.34` s 降到约 `0.127` s。
- 完整 1104 samples contact response：约 `3.31` s。

这说明慢点主要是实现层候选剪枝与 Python 稀疏装配问题，而不是拉格朗日 SDF 理论路线本身。

## 下一阶段

下一阶段应按用户要求进入 CalculiX 同算例对比：

1. 用相同 TET4/HEX8 几何、边界、载荷、材料生成 CalculiX native contact 输入。
2. 独立运行 CalculiX 和 SFC 拉格朗日 SDF。
3. 对齐位移、应变、应力曲线和云图。
4. 分析误差来源并完成精度对齐。
5. 精度通过之后，再比较完整 wall time；不能用 field/kernel timing 冒充 solver-level 加速。
