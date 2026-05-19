# CalculiX 接触前曲线对齐

## 问题

同工况小球落地对比中，CalculiX 和 Abaqus/Standard 在接触前阶段几乎重合，而旧 SFC 曲线在接触前偏高。检查历史文件后确认，偏差不是由 SDF gap 或几何映射造成，而是因为旧对齐配置把 `mass_damping=12` 从 `t=0` 开始施加。

该工况在接触前是纯重力自由落体。CalculiX/Abaqus 输入没有接触前阻尼，因此质心轨迹应满足：

```text
z(t) = z0 - 0.5 * g * t^2
```

## 修改

在 `validation/run_abaqus_sphere_drop_full_validation.py` 中新增显式阻尼启动策略：

```text
--damping-start-policy time
--damping-start-policy contact-estimate
```

其中 `contact-estimate` 使用初始最小间隙和重力计算解析首次接触估计：

```text
t_contact = sqrt(2 * initial_gap / g)
```

然后将有效阻尼启动时间设为：

```text
max(requested_damping_start_time, t_contact)
```

这样后接触阶段仍可使用耗散参数，但接触前阶段不允许阻尼污染自由落体曲线。

## 验证命令

```powershell
pytest -q tests/test_abaqus_sphere_drop_full_validation.py tests/test_calculix_same_sphere_drop_comparison.py tests/test_abaqus_sphere_drop_short_validation.py
```

结果：`12 passed`。

重新生成 0.12 s SFC/Abaqus 曲线：

```powershell
python validation/run_abaqus_sphere_drop_full_validation.py `
  --inp commercial_software_comparison/abaqus_flexible_body_rigid_plane_implicit/sphere_drop/abaqus_run/sphere_drop_implicit.inp `
  --vtk-dir commercial_software_comparison/abaqus_flexible_body_rigid_plane_implicit/sphere_drop/vtk `
  --duration 0.12 `
  --dt 0.001 `
  --contact-stiffness 5.0e9 `
  --mass-damping 12.0 `
  --damping-start-policy contact-estimate `
  --integrator hht `
  --hht-alpha -0.3 `
  --contact-integration surface `
  --quadrature-order 3 `
  --out-dir results/abaqus_sphere_drop_implicit_hht_surface_precontact_aligned_012
```

重新叠加 CalculiX/Abaqus/SFC 曲线：

```powershell
python validation/run_calculix_same_sphere_drop_comparison.py `
  --duration 0.12 `
  --dt 0.001 `
  --hht-alpha -0.3 `
  --sfc-history results/abaqus_sphere_drop_implicit_hht_surface_precontact_aligned_012/abaqus_sphere_drop_full_sfc_history.csv `
  --out-dir results/calculix_same_condition_sphere_drop_012_precontact_aligned
```

## 接触前误差

以 `t <= 0.064 s` 作为接触前窗口：

| 配置 | max z-CM error | RMS z-CM error |
| --- | ---: | ---: |
| 旧配置，阻尼从 `t=0` 生效 | `3.9799e-3 m` | `1.5240e-3 m` |
| 新配置，`contact-estimate` | `3.1070e-4 m` | `1.7918e-4 m` |

首次接触时间也从旧配置的 `0.075 s` 改为 `0.065 s`，接近 Abaqus/Standard 的 `0.066 s` 和 CalculiX 的 `0.0645 s`。

## 结论

CalculiX 对这里的直接指导是：接触前阶段必须保持无阻尼自由落体，耗散只能在接触事件附近或之后介入。当前修改把这一点显式编码为 validation 策略，避免为了后接触反弹对齐而破坏接触前位移曲线。
