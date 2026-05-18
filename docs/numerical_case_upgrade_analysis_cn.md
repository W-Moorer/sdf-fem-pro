# 数值算例升级分析

## 当前问题

1. 目前 `paper/main.pdf` 中的准静力 native-contact 位移曲线只有 4 个点。原因是 `validation/run_fuzzycontact_calculix_native_contact_trajectory.py` 中正式工况的 `closures` 只使用了 `(0.25, 0.5, 0.75, 1.0)` 四个加载比例。这适合作为开发验证，但不适合作为论文中的正式加载曲线。

2. 目前 C3D8 动力学时间历程只覆盖约 `0.08 s`。代码依据是 `validation/run_c3d8_contact_trajectory_validation.py::build_c3d8_model(...)`，其中 `quick=True` 默认 `total_time=0.08`，非 quick 默认 `total_time=0.12`。这对于动力学论文图太短，不能体现接触振荡、回弹、能量变化或长时稳定性。

3. 当前动力学 C3D8 网格仍偏粗。Phase-9 动态图主要使用 `resolution=1` 的 C3D8 块体，更多是在证明程序链条和 SFC/CalculiX 时间历程一致性，不足以形成工程力学论文里的“细网格三维接触模型”视觉证据。

4. 参考文献 `assets/uzun_korsunsky_2026_fuzzycontact_sigmoidal_traction_law.pdf` 的 Fig. 6 是一个更像工程力学论文的复杂工况：法向压缩后进行切向加载，并输出 `ux, uy, uz, sigma_xx, sigma_yy, sigma_zz` 三维云图。该工况适合借鉴，但原文包含摩擦/切向阻力和 stick-slip 叙事，而当前项目约束明确不加入 friction。

## 测试策略修正

后续区分三类改动：

- 修改 `src/sfc` 核心算法、接触公式、SDF 查询路径、FEM 装配或时间积分时，必须跑 `pytest -q`。
- 修改验证脚本、绘图脚本或 benchmark runner 时，优先跑相关脚本和对应小范围测试；只有影响核心行为时才跑全量 `pytest -q`。
- 只修改模型参数、论文图片、CSV、PDF 或文档时，不跑全量 `pytest -q`，只做图像/PDF渲染检查和必要的算例命令检查。

## 建议升级方案

### 1. 准静力加载步数

将 native-contact 三个准静力工况从 4 个 closure steps 改为至少 11 个，论文正式结果建议 21 个：

```text
load factors = 0.00, 0.05, ..., 1.00
```

图中继续使用 `Closure (mm)` 或归一化加载因子作为横轴，图例中保留最大相对误差。这样曲线会从“几个点”变成真正的加载路径。

### 2. 动力学时间长度

新增正式长时动力学 runner，而不是复用 quick 开发图：

```text
total_time = 3.0 s
dt = 0.005 s 或 0.01 s
output_stride = 5 到 10
```

推荐先用 `dt=0.01 s` 形成 300 个真实步，若 SFC/CalculiX 曲线仍平滑一致，再将关键工况加密到 `dt=0.005 s`。横轴必须是 `Time (s)`，不再使用 closure。

### 3. 网格加密

建议正式论文图至少包含一组细网格：

| 工况 | 当前口径 | 建议正式口径 |
| --- | --- | --- |
| C3D8 block-plane dynamic | `resolution=1` | `resolution=4` 或 `6` |
| C3D8 block-block dynamic | `resolution=1` | `resolution=4` 或 `6` |
| sphere/plate quasi-static | driver `res=14` | `res=24` 或 `32` |
| V-indenter quasi-static | driver `res=16` | `res=32` 或更高 |
| cylinder compression | driver `res=10` | `res=20` |

更细网格会提高 SDF field build 成本，因此必须同时报告：场更新时间、查询时间、总时间、以及与 CalculiX/native projection reference 的对比。

### 4. Fig. 6 风格复杂工况

可以新增一个“Fig. 6-inspired normal-to-tangential motion”工况，但必须收紧 claim：

参考文献工况要点：

- 下部固定体：`20 x 10 x 2 mm^3`
- 上部移动体：`4.5 x 4.5 x 1 mm^3`
- 平均 TET 元素尺寸：约 `0.5 mm`
- 先施加法向压缩：`Fz = -2000 N`
- 再施加切向加载：`Fx = +2000 N`
- Fig. 6 展示两阶段的 `ux, uy, uz, sigma_xx, sigma_yy, sigma_zz`

但原文 Fig. 6 的物理结果依赖摩擦/切向阻力。当前项目不能加入 friction，所以建议采用以下无摩擦版本：

```text
Stage A: 法向压缩，建立接触区；
Stage B: 对上部体施加规定的切向位移或斜向运动，同时继续使用 frictionless SDF normal contact；
Output: ux, uy, uz, sigma_xx, sigma_yy, sigma_zz, von Mises, gap, pressure, active contact mask。
```

这个工况可以证明：

- 变形体在复杂非匹配网格和滑移几何下仍能由当前表面重建 SDF；
- SDF 查询路径能稳定跟随移动接触区；
- surface-to-surface quadrature contact 输出的位移、应力、gap 和 pressure 云图物理可解释；
- 接触主导时，field query 的摊销优势仍然存在。

不能声称：

- 复现 Fig. 6 的 stick-slip；
- 复现摩擦力平台；
- 证明摩擦接触；
- 证明切向接触阻力。

若必须严格复现 Fig. 6 的摩擦响应，则需要放开“不加入 friction”的硬约束；否则只能做 Fig. 6-inspired 几何和加载路径，而不是 Fig. 6 friction physics replication。

## 推荐执行顺序

1. 先改准静力 native-contact runner：加载步从 4 个增加到 21 个，重新生成三条准静力曲线。
2. 新增长时动力学 runner：`3 s`，C3D8 `resolution=4` 起步，输出 `z displacement / force / energy / min gap / active contact` 时间历程。
3. 生成细网格 C3D8 云图：位移、应变、应力、gap、pressure、active mask。
4. 增加 Fig. 6-inspired 无摩擦复杂工况，重点输出三维云图和 SDF field 可视化，而不是 stick-slip claim。
5. 最后再将论文实验段落改成：
   - SDF field accuracy；
   - Jacobian FD；
   - 准静力多步 native-contact；
   - 3 秒动力学时间历程；
   - Fig. 6-inspired complex contact；
   - backend/timing ablation。

