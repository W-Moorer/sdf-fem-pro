# Final-Aim 局部测试范围

日期：2026-05-19

当前研发重心已经转到：

```text
MaterialSDF + FEMDeformationMap + LagrangianSDFContactOracle
```

因此，当修改范围只涉及 final-aim 新模块、对应 validation runner 或对应文档时，默认使用局部测试，不再每次跑完整 `pytest -q`。

## 推荐命令

```text
pytest -q -m final_aim
```

当前覆盖：

- `tests/test_lagrangian_sdf_oracle.py`
- `tests/test_final_aim_lagrangian_contact_cases.py`
- `tests/test_final_aim_complex_surface_validation.py`

本轮结果：

```text
pytest -q -m final_aim
16 passed, 310 deselected in 25.92s
```

## 算例验证命令

```text
python validation/run_final_aim_lagrangian_contact_cases.py --quick --out-dir results/final_aim_lagrangian_contact_quick
```

输出：

- `results/final_aim_lagrangian_contact_quick/final_aim_lagrangian_contact_cases.csv`
- `results/final_aim_lagrangian_contact_quick/final_aim_lagrangian_contact_samples.csv`
- `results/final_aim_lagrangian_contact_quick/final_aim_lagrangian_contact_history.csv`
- `results/final_aim_lagrangian_contact_quick/final_aim_lagrangian_contact_summary.md`

## 何时仍需全量测试

只有在以下情况才建议跑完整 `pytest -q`：

- 修改旧 dynamic field、field_contact、projection kernel 或通用 FEM assembly；
- 修改公共 mesh/topology/element backend；
- 修改会影响旧 validation runner 的共享 API；
- 准备 release、投稿归档或远端合并前需要完整回归。

否则，final-aim 日常开发以 `pytest -q -m final_aim` 为准。
