# Q4 master C++ matrix-free tangent 加速记录

## 目的

细网格 Q4 master 接触路径用于和 Abaqus/Standard 对齐精度，但原实现会显式装配全局接触刚度 `Kc`，再做直接稀疏求解。该路径在 `16x8x4` 下柔性块和 `8x8x3` 上柔性块、3x3 Q4 高斯接触积分、1 s / 0.001 s 工况中比 Abaqus 慢。

本次优化不改变接触物理、时间步、罚刚度、Q4 面积分、主从面定义或误差指标，只降低实现开销。

## 实现

- `src/sfc/_cpp_sdf_contact.cpp`
  - 新增 `quadrilateral_master_penalty_response(...)`。
  - 在 C++ 中完成 Q4 master closest patch Newton、gap、normal、Q4 shape payload 和 penalty force 累加。
  - 复用已有 C++ `solve_contact_tangent_pcg(...)` matrix-free 接触切线求解器。

- `src/sfc/contact/field_contact.py`
  - `quadrilateral_master_surface_penalty_response(...)` 新增 `matrix_free_stiffness`。
  - Q4 master payload 被映射成通用 `FieldContactMatrixFreeStiffness` 行格式：
    `slave Q4 weights + one master Q4 payload + normal + area scale`。

- `validation/run_flexible_cube_sdf_abaqus_comparison.py`
  - 新增 `--cpp-contact-tangent-solver`。
  - HHT/Newmark 切线从
    `M*c0 + (K + Kc)*(1+alpha)`
    改为等价的 matrix-free solve：
    `M*c0 + K*(1+alpha) + Kc*(1+alpha)`。
  - 若 C++ PCG 未收敛，回退到原显式 `Kc` 直接求解，不降低结果精度。

## 验证

相关测试：

```text
pytest -q tests/test_field_surface_to_surface_contact.py tests/test_flexible_cube_sdf_abaqus_comparison.py
24 passed in 0.88s
```

正式对比工况：

- lower mesh: `16x8x4` C3D8
- upper mesh: `8x8x3` C3D8
- duration: `1.0 s`
- time step: `0.001 s`
- Q4 quadrature: `3x3`
- contact stiffness: `4500.0`
- Abaqus reference: reused from `results/flexible_cube_press_1s_dense_q4`
- SFC optimized output: `results/flexible_cube_press_1s_dense_q4_cpp_tangent`

## 结果

| Path | Complete solve wall time |
|---|---:|
| Abaqus/Standard native contact | `170.789 s` |
| SFC old dense Q4 direct tangent | `355.553 s` |
| SFC C++ Q4 matrix-free tangent | `56.981 s` |

SFC 相对 Abaqus 的完整求解速度约为 `2.997x`，相对旧 dense Q4 direct tangent 约为 `6.239x`。

误差保持不变：

| Metric | L2 relative error |
|---|---:|
| top mean z displacement | `9.09%` |
| upper reaction force | `7.00%` |
| nominal contact pressure | `7.00%` |
| p95 von Mises | `8.23%` |
| p95 strain norm | `7.78%` |
| mean surface gap | `12.57%` |

结论：Q4 master 精度路径已经从“比 Abaqus 慢”变为“在相同误差口径下明显快于 Abaqus”。mean surface gap 仍略高于 10%，这是对齐问题，不是本次加速改动引入的精度损失。
