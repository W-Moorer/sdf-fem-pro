# Abaqus-style 积分耗散扫描

## 目的

为分析小球落地工况中 SFC 反弹高度偏高的问题，本阶段对 SFC 隐式积分设置做了 0.12 s 参数扫描，并与同一 Abaqus/Standard implicit sphere-drop 参考曲线比较。

本扫描保持接触前自由落体对齐，即所有带阻尼的配置都使用：

```text
--damping-start-policy contact-estimate
```

这保证阻尼不在接触前生效。

## 运行命令

```powershell
pytest -q tests/test_abaqus_sphere_drop_full_validation.py `
  tests/test_abaqus_sphere_drop_integrator_sweep.py `
  tests/test_abaqus_sphere_drop_short_validation.py
```

结果：`13 passed`。

```powershell
python validation/run_abaqus_sphere_drop_integrator_sweep.py `
  --duration 0.12 `
  --dt 0.001 `
  --out-dir results/abaqus_sphere_drop_integrator_sweep_012
```

## 输出

- `results/abaqus_sphere_drop_integrator_sweep_012/integrator_sweep_summary.csv`
- `results/abaqus_sphere_drop_integrator_sweep_012/integrator_sweep_summary.md`

## 主要结果

| Case | Integrator | HHT alpha | Mass damping | Adaptive | Rebound max-z error |
| --- | --- | ---: | ---: | --- | ---: |
| `newmark_no_damping` | Newmark | `0` | `0` | false | `1.2124e-2 m` |
| `hht_tf_alpha_m005` | HHT | `-0.05` | `0` | false | `1.1914e-2 m` |
| `hht_alpha_m010` | HHT | `-0.10` | `0` | false | `1.1750e-2 m` |
| `hht_alpha_m030` | HHT | `-0.30` | `0` | false | `1.1453e-2 m` |
| `hht_md_alpha_m041421` | HHT-MD-like | `-0.4142` | `0` | false | `1.1473e-2 m` |
| `bwe` | Backward Euler | `0` | `0` | false | `6.9646e-3 m` |
| `hht_alpha_m030_mass12` | HHT | `-0.30` | `12` | false | `5.0174e-3 m` |
| `hht_alpha_m030_mass12_adaptive` | HHT | `-0.30` | `12` | true | `5.0191e-3 m` |

## 结论

1. 单独调 HHT `alpha` 不能充分压低反弹高度。`alpha=0` 到 `alpha=-0.4142` 的变化只把 rebound max-z error 从约 `1.21e-2 m` 降到约 `1.15e-2 m`。

2. Backward Euler 的数值耗散更强，能把 rebound max-z error 降到约 `6.96e-3 m`，但仍不如接触后质量阻尼配置。

3. 当前最佳 0.12 s 配置是：

```text
integrator = hht
hht_alpha = -0.3
mass_damping = 12
damping_start_policy = contact-estimate
contact_integration = surface
```

其 rebound max-z error 为 `5.0174e-3 m`。

4. 当前 adaptive increments 并未降低反弹高度：

```text
fixed:    rebound error = 5.0174e-3 m, solve wall = 10.05 s
adaptive: rebound error = 5.0191e-3 m, solve wall = 17.80 s
```

这说明目前的自适应切步只细分了接触事件，但没有对接触冲量或能量释放施加 Abaqus-style impact balance，因此不能单独解决反弹偏高。

## 对后续实现的指导

下一步不应继续盲目减小时间步，而应实现或加强：

- 接触 impact/release 事件定位；
- 接触冲量 `∫ f_contact dt` 对齐；
- 只在闭合速度为负时生效的法向接触阻尼；
- 基于接触能量/冲量的 step acceptance；
- Abaqus-style `IMPACT=AVERAGE TIME` 近似。

这些机制比单纯调 HHT `alpha` 或自动切步更接近 Abaqus/Standard 在该工况中表现出的低反弹行为。
