# 接触检测与接触计算理论不一致审计

本文档记录当前 SFC Lagrangian-SDF 齿轮接触路径与 Abaqus/Standard surface-to-surface 接触语义之间的主要理论差异。结论是：当前误差很可能来自接触算法层，而不是单纯来自 SDF 查询精度。

## 当前证据

最近的区域对齐结果显示：

- 全局位移误差已经较小，末帧 max displacement norm 约为 3.21%。
- 但接触建立/释放窗口内的应力、应变误差很大，`t = 0.00028--0.00034 s` 附近 p95 stress/strain error 可达 70%--88%。
- active set 覆盖不稳定，例如 `t = 0.00030 s` 时 SFC active node count 为 160，Abaqus active node count 为 504，recall 约 0.1706；`t = 0.00038 s` 时 Abaqus active 为 0，但 SFC 仍有 active。
- 刚体/RP 位移曲线能接近，说明主问题不是全局运动单位或时间步长，而是接触压力区域、active status 和局部应力传递。

因此，剩余误差应优先从接触理论实现排查，而不是继续调 SDF 查询半径、接触刚度或阻尼。

## legacy_abaqus_prototype 粗检测/细检测流程的定位

legacy 路径大致是：

```text
Abaqus/外部位移
-> 当前节点坐标
-> 全局 AABB / sparse ADF 候选
-> face AABB / barycentric samples
-> 局部 SDF / Taylor 预测
-> phi <= tolerance 进入 active pairs
-> 近似 penalty force
```

这个流程适合作为候选生成、可视化或 replay 诊断，但不能作为最终接触理论：

- 它的 SDF/ADF 是稀疏候选场或预测场，不应直接决定最终 gap/Jacobian。
- active 判断偏向点采样，缺少 surface-to-surface constraint region 的平均开闭状态。
- 细检测如果只返回点级 gap，无法自然得到 Abaqus 风格的压力平滑、约束区域面积和等效反力语义。

因此，legacy 代码可以提供粗检测经验，但不能直接迁入主接触算法作为最终接触求解。

## 稀疏 ADF / Lagrangian SDF 理论边界

稀疏 ADF 或 material-space SDF 在柔性体任意变形下只能承担以下角色：

```text
候选缓存
初值
参考 patch 索引
normal-tube / closest-feature 搜索加速
```

最终接触量必须来自当前构型：

```text
g = phi_current(x)
n = current contact normal
J_slave = N_slave n^T
J_master = -N_master n^T
```

如果稀疏 ADF 的预测 gap 直接参与最终 active set 或最终 penalty force，就会在大变形、曲面滑移和齿面接触中产生错误。当前主路径已经基本转为 current-surface closest-feature / secondary-normal projection，但仍存在以下理论缺口。

## 理论缺口 1：点级穿透积分不等于 Abaqus surface-to-surface 约束区域

当前默认路径使用三角面 `tri3` 积分点，之后通过 `slave_node_region_participation` 等模式聚合。这个口径本质上是：

```text
先在局部 quadrature/sample 点判断穿透
再把正压力按面积和形函数权重合并到 slave node region
```

这与 Abaqus surface-to-surface 的约束区域语义仍有差异。Abaqus 更接近：

```text
先定义 secondary constraint region
在该区域上计算平均 clearance / overclosure
再由 pressure-overclosure law 决定该区域是否闭合
最后将压力按区域权重分配
```

这一区别会导致两个问题：

1. 某个区域内只有局部 sample 穿透时，SFC 仍可能产生压力；Abaqus 可能认为区域平均 gap 仍为开。
2. 局部 sample 的压力峰值会被过度保留，直接放大应力/应变尖峰。

已有 `*_signed_participation` 模式说明了这个方向，但之前的结果表明仅做 signed status 不足以解决全部误差，因为还缺少压力平滑和约束区域权重的完整一致性。

## 理论缺口 2：secondary-normal line projection 与 closest-feature guard 混合后，active status 不够统一

当前路径为了接近 Abaqus surface-to-surface，引入 secondary normal line projection：

```text
slave quadrature point
-> secondary averaged normal
-> line hit master facet
-> line distance gate
-> closest-feature open/closed guard
```

这个方向是对的，但现在 line hit、closest-feature guard 和最终 active status 还不是同一个约束区域上的单一函数。结果是：

- 粗检测可能找到了 line hit；
- closest-feature guard 又可能把它释放或替换；
- 区域聚合时又根据 sample-level gap 重新决定压力；
- active set convergence 用的是离散 sample signature。

理论上应改为每个 constraint region 有一个明确的接触状态函数：

```text
G_r(q) = averaged normal clearance over secondary constraint region r
active iff G_r(q) < tolerance/status threshold
```

候选 line projection 和 closest-feature 只能服务于这个 `G_r(q)` 的构造，不能各自独立改变 active status。

## 理论缺口 3：接触切线仍是冻结几何的罚函数切线

当前 penalty tangent 主要是：

```text
K_c = J^T k A J
```

其中 `J` 来自当前 sample 的 normal、slave weights 和 master payload。这个切线冻结了：

- normal 对位移的变化；
- master closest-feature / barycentric payload 对位移的变化；
- secondary constraint region 权重和平均 normal 的变化；
- active status 平滑/切换；
- 压力平滑对相邻区域的耦合。

对于位移主导的 RP 曲线，这种近似可能仍可收敛；但应力、应变对接触压力分布非常敏感，冻结几何切线会造成接触建立阶段的压力振荡和局部应力峰值偏差。

## 理论缺口 4：active set 收敛判据不是 Abaqus-style contact convergence

当前 source-drive penalty 路径使用 residual、correction 和 exact active signature 稳定性判断。问题是：

- exact sample signature 对网格细节过于敏感；
- 没有使用 Abaqus 风格的 contact force residual、penetration tolerance、status tolerance 和 cutback 组合；
- 达到最大迭代后仍会接受该步，只记录诊断；
- penalty source-drive 路径没有真正的自动增量/cutback 闭环。

这会导致接触建立/释放窗口中残余接触状态被带入下一步，进而造成应力/应变曲线跳动。

## 理论缺口 5：压力输出与求解压力不是同一个口径

当前 VTK/CSV 中的 `contact_pressure_nodeavg`、`contact_penetration_nodeavg` 是从 sample 或聚合 sample 后处理得到。Abaqus 的 CPRESS/COPEN 是接触算法内部 constraint region / secondary-node 语义下的输出。

如果求解使用一种聚合口径，而输出又按 node average 重新解释，会出现：

```text
位移曲线可对齐
但 pressure/stress/strain 云图和 p95 指标对不上
```

后续必须把求解用的 constraint pressure 和输出用的 pressure 使用同一个区域定义。

## 需要的通用修复方向

下一步不应继续调参，而应实现一个真正的 Abaqus-style surface-to-surface constraint-region 层：

1. **构造 secondary constraint regions**

   对每个 secondary node 或 surface region，收集相邻 slave facets，并计算区域面积：

   ```text
   A_r = integral_R N_r dA
   ```

2. **定义区域平均 gap**

   ```text
   G_r(q) = (1 / A_r) integral_R N_r(x) g(x, q) dA
   ```

   这里的 `g(x, q)` 仍由当前构型 Lagrangian-SDF / normal-line closest-feature payload 得到。

3. **用区域 gap 决定 active status**

   ```text
   active_r iff G_r(q) < g_tol
   p_r = k < -G_r(q) >_+
   ```

   不能用“区域内任一点穿透”直接决定整个区域闭合。

4. **用同一约束区域分配虚功**

   ```text
   delta W_c = sum_r A_r p_r delta G_r
   ```

   由此得到 slave/master nodal force 和 consistent tangent。

5. **压力平滑必须来自形函数/区域权重**

   不允许引入 case-specific smoothing factor。压力平滑应来自 mesh shape functions、constraint region area 和 surface topology。

6. **active convergence 改成接触残差口径**

   每步应同时检查：

   ```text
   equilibrium residual
   displacement correction
   contact force increment
   penetration/status tolerance
   active region stability
   ```

   失败时 cutback，而不是接受不稳定步。

7. **输出语义与求解语义统一**

   VTK/CSV 中的 CPRESS/COPEN 对应同一个 `G_r, p_r, A_r`，再投影到节点或单元。

## 为什么这不是降低精度

上述修复不会放宽接触几何，也不会降低 SDF 查询精度。它改变的是从点级几何查询到 surface-to-surface 接触约束的离散化方式：

```text
当前点级 sample pressure
-> 理论一致的 constraint-region pressure
```

SDF/closest-feature 仍给出当前构型 gap、normal 和 master payload；只是 active status、压力分布和切线改为有限元接触理论中的区域约束口径。

## 后续验收顺序

建议按以下顺序修复并验证：

1. 平面双柔性块压入：RF/U、pressure、active region 必须和 Abaqus 对齐。
2. 齿面局部 patch：对齐 contact active region、pressure 和 p95 stress/strain。
3. 全齿轮 10 步：位移、应力、应变误差均小于 10%。
4. 全齿轮长程：在同一时间步和同一输出口径下比较效率。

在第 1 步没有通过前，不应继续用全齿轮长算例作为主要调试工具。
