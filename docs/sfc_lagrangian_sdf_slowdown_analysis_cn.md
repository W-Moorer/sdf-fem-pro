# SFC 拉格朗日 SDF commercial sphere-cantilever 慢点分析

## 结论

当前 commercial sphere-cantilever 工况中，SFC 拉格朗日 SDF 跑得慢主要不是理论路线问题，而是实现层仍是 Python reference path：

1. `LagrangianSDFContactOracle.query_many()` 并未真正批量化，只是逐点调用 `query()`。
2. 每个 sample 的候选 patch 数量过大，约 512 到 528 个，而 beam top 总面数只有 960 个。
3. `lagrangian_oracle_penalty_response()` 逐 sample 构造 Python dataclass 约束、逐约束构造 1 行稀疏 Jacobian，并逐项累加 `J.T @ J`。
4. 隐式 Newmark 每步最多 5 次 Newton 迭代，每次都重新做 contact response 并重新 `spsolve` 全局切线。
5. commercial 模型约 1104 个 sphere contact samples、11556 个总自由度；逐 sample oracle 查询成本被 Newton 循环放大。

这会导致拉格朗日 SDF 的“避免每步重建 current SDF grid”优势没有被释放，反而被 Python 查询和稀疏装配开销淹没。

## 实测定位

commercial input deck:

`commercial_software_comparison/abaqus_sphere_cantilever/abaqus_run/sphere_cantilever_explicit.inp`

解析得到：

- beam: 2745 nodes, 1920 C3D8R elements
- sphere: 1107 nodes, 2208 C3D4 elements
- beam top material surface: 960 triangle patches
- sphere contact samples: 1104
- total DOFs in coupled SFC solve: 11556

局部 profiling 显示 FEM 装配不是主瓶颈：

| Operation | Time |
| --- | ---: |
| sphere stiffness | 0.294 s |
| sphere mass | 0.237 s |
| beam stiffness | 1.780 s |
| beam mass | 1.601 s |
| MaterialSDF/oracle build/refit | 0.121 s |
| sphere sample generation | 0.036 s |

真正异常的是 contact response。即使没有 active contact，只取前若干 samples：

| Samples | Response time | Time/sample | Active |
| ---: | ---: | ---: | ---: |
| 1 | 0.801 s | 0.801 s | 0 |
| 5 | 1.941 s | 0.388 s | 0 |
| 10 | 3.658 s | 0.366 s | 0 |
| 20 | 8.000 s | 0.400 s | 0 |
| 50 | 19.342 s | 0.387 s | 0 |

按 1104 samples 外推，单次 contact response 约为 7 min。一个 0.05 s、dt = 0.001 的短程仿真有 51 个时间点，每步 Newton 最多 5 次 response，因此当前实现必然跑不动。

单点 oracle 查询的候选 patch 数量也过大：

| Sample | Candidates | Query time |
| ---: | ---: | ---: |
| 0 | 512 | 0.613 s |
| 1 | 512 | 0.392 s |
| 10 | 528 | 0.404 s |
| 20 | 528 | 0.376 s |
| 100 | 512 | 0.386 s |

这说明当前 `ReferencePatchBVH` 的 ball query 在该工况下没有形成有效的局部候选集。

## 代码层原因

关键位置：

- `src/sfc/contact/lagrangian_sdf_oracle.py:144`
  `query_many()` 返回 tuple comprehension，本质是 Python for-loop。
- `src/sfc/contact/lagrangian_sdf_oracle.py:291`
  `query()` 对所有候选 patch 逐个执行 closest-point correction。
- `src/sfc/contact/lagrangian_sdf_oracle.py:555`
  penalty response 先把所有 samples 逐个转成 constraint objects。
- `src/sfc/contact/lagrangian_sdf_oracle.py:584`
  stiffness 使用 `stiffness = stiffness + ...` 逐约束稀疏矩阵累加。
- `validation/run_commercial_sphere_cantilever_short_comparison.py:791`
  每步最多 5 次 Newton，每次重新 contact response。
- `validation/run_commercial_sphere_cantilever_short_comparison.py:800`
  每次 Newton 都对全局 tangent 子矩阵调用 `spsolve`。

## 为什么这是实现问题

拉格朗日 SDF 的名义优势是：

- reference/material SDF 不需要每步重建 current-space phi grid；
- 每步只 refit deformed patches / AABB；
- 查询应复用 patch topology、sample topology、candidate cache 和 shape weights；
- contact response 应按数组批量输出 gap、normal、master weights，再一次性累积 force/tangent。

当前实现虽然在概念上没有重建 current SDF grid，但查询和装配仍是逐点逐面 Python 参考实现，因此没有发挥这一优势。

## 不降低精度的优化方向

这些优化不改变接触离散、gap 定义、normal 定义、Jacobian 定义或 penalty stiffness，只改变执行方式：

1. 真正的 batch oracle。
   预生成 sample points、sample weights、face ids、area weights；oracle 返回数组形式的 `gap/normals/master_face/master_barycentric`。

2. 更强的 exact candidate pruning。
   用真正 BVH traversal 或 cell-linked list，把候选 patch 从 500+ 降到局部几十个或更少；保留 exact nearest fallback，不能靠任意缩小 search radius 换速度。

3. 向量化 contact force/tangent。
   用 `np.add.at` 或 COO 一次性构造 active rows/cols/vals，避免 per-constraint `J.T @ J` 和 sparse addition。

4. Matrix-free tangent solve。
   不显式装配全局 `Kc`，实现 `K_eff x + J^T W J x` matvec，并配合预条件 Krylov 或复用 elastic effective matrix factorization。

5. C++ fused backend。
   把 `refit -> candidate lookup -> closest patch -> gap/normal -> force/tangent matvec` 下沉到 C++，Python 只负责传入当前节点数组和接收 force/history。

6. persistent workspace。
   复用 query buffers、candidate buffers、force buffers、COO index buffers 和 solver workspace，减少每步临时数组和 Python 对象分配。

## 优先级

1. 先实现 `lagrangian_oracle_penalty_response_vectorized()`，保持数学结果不变，验证与 scalar path bitwise/near-equal。
2. 再实现 `LagrangianOracleWorkspace`，缓存 samples、area weights、offsets、candidate hints 和输出 buffer。
3. 然后实现 C++ fused query/contact response。
4. 最后做 matrix-free tangent + 预条件求解，减少 Newton 每次全局直接分解成本。

只有完成 2 到 4，SFC 拉格朗日 SDF 才可能在完整 solve time 上稳定超过 CalculiX，而不是只在理论更新成本上占优。
