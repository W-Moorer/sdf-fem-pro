# 源齿轮算例应力/应变趋势不一致诊断

## 当前结论

当前源齿轮工况中，位移曲线接近 Abaqus 不能说明应力/应变已经对齐。这个工况由 RP/BEAM MPC、给定角速度和给定转矩强约束主导，整体位移和转角天然更容易贴合；应力/应变则由接触压力历史、接触释放时序、几何非线性、材料内力线性化和场输出恢复口径共同决定，因此更敏感。

目前应力/应变趋势差异不是 SDF 最近点查询本身的首要证据，而是接触力学口径和输出口径尚未与 Abaqus 完全一致。

## 直接证据

源 Abaqus deck `commercial_software_comparison/abaqus_flexible_body_gear_contact/gear_contact.inp` 的关键设置为：

- `*MPC, BEAM`：两个齿轮 hub 与 RP 通过 BEAM MPC 约束。
- `*Surface Behavior, pressure-overclosure=HARD`。
- `*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE`。
- `*Step, name=Step-1, nlgeom=YES, inc=2000`。
- `*Dynamic`：`1e-05,0.05,1e-10,5e-05`。
- Gear 1 RP dof 6 给定角速度 `52.36`，单位是弧度每秒。
- Gear 2 RP dof 6 施加转矩 `50`。
- 材料为 `E=2.05e11, nu=0.28, rho=7850`。

当前 SFC 源齿轮路径的关键设置为：

- `solve_sfc_source_drive_pair(...)` 使用完整双柔性 TET4 网格和 RP/hub reduced assembly。
- 接触为 `contact_mode = source_penalty`，即线性罚函数接触路径。
- `material_linearization = reference_linear`，全局内力使用参考线性刚度路径。
- master surface 为三角面 `MaterialSDF.from_triangle_surface(...)`，slave 采样为 `quadrature="tri3"`。
- SFC 输出应力/应变默认使用 `linear_corotated` 后处理：先扣除 RP 刚体转动的线性近似，再从残余弹性位移恢复小应变线弹性应力。
- VTK 曲线比较中，SFC 的 `strain_norm_nodeavg` 与 Abaqus 的 `logarithmic_strain_norm_nodeavg` 被放在同一图上，但这两个量并不是完全相同的应变定义。

当前 `0--2 ms` 区间合并对齐结果显示：

| 指标 | 最大 p95 相对误差时刻 | SFC p95 | Abaqus p95 | p95 相对误差 |
| --- | ---: | ---: | ---: | ---: |
| displacement magnitude | `0.48 ms` | `5.340e-3` | `5.094e-3` | `4.82%` |
| von Mises stress | `1.98 ms` | `4.367e6` | `1.372e7` | `68.16%` |
| equivalent elastic strain | `1.98 ms` | `1.818e-5` | `5.710e-5` | `68.16%` |
| Abaqus active-region von Mises | `0.50 ms` | `2.731e6` | `3.637e5` | `650.78%` |

这说明整体位移低误差与局部应力/应变高误差同时存在。后者不是动画色标或角速度单位问题。

## 为什么位移能对上

1. 位移主要受 RP 运动学边界控制。Gear 1 的 RP dof 6 被直接给定角速度，Gear 2 的 hub 也由 BEAM MPC 强约束到 RP。因此整体位移、转角和大尺度刚体运动主要由输入边界决定。
2. 接触压力分布即使有明显差异，也未必立刻显著改变整体位移范数。局部齿面压力可以相差很多，但全局位移 p95 仍可能只有几个百分点误差。
3. 当前比较时间窗很短，`0--2 ms` 内惯性主导和强约束主导更明显，局部应力波尚未通过整体位移充分反映出来。

因此，位移对齐只是说明 RP/MPC 驱动、时间单位、网格坐标和大尺度运动大体正确；不能证明接触功、接触压力和内部应力已经对齐。

## 为什么应力/应变趋势不对

### 1. Abaqus 是 `nlgeom=YES`，SFC 源齿轮路径仍是参考线性内力

Abaqus deck 启用了几何非线性；SFC 当前源齿轮快速路径在求解中使用 `reference_linear`，即参考刚度矩阵和小应变线性内力。虽然 SFC 后处理做了 `linear_corotated` 以避免把刚体转动误读为应变，但这不是 Abaqus `nlgeom=YES` 的完整有限变形内力、几何刚度和应力更新。

结果是：整体转动可以被扣除并显示得合理，但局部接触下的应力波、齿根弯曲应力和接触区等效应变趋势可能与 Abaqus 的几何非线性路径不同。

### 2. HARD surface-to-surface 与当前线性罚函数接触不是同一个接触约束

源 deck 是 `HARD` surface-to-surface。当前对比为了可控和可输出，SFC 使用线性 penalty 接触。即使 Abaqus 侧也转成 penalty 对照，Abaqus/Standard 的 surface-to-surface 接触仍包含更复杂的约束平均、接触状态判断、压力平滑、主从面区域处理和接触释放逻辑。

当前诊断中已经看到接触状态不一致：

- `0.94--0.98 ms` 附近 Abaqus active contact nodes 已经为 `0`，SFC 仍有数百个 active nodes 且压力在 `3.6e6--4.1e6` 量级。
- 到 `1.98 ms`，两边 active nodes 都为 `0`，但 Abaqus 的 p95 von Mises 仍明显高于 SFC。这说明差异已经转化为历史效应：前序接触冲量、接触功和应力波不同，后续即使没有当前接触，内部应力场仍继续分叉。

这也是“趋势都不对”的核心原因：动态接触中的应力不是当前 gap 的瞬时函数，而是前序接触力历史的结果。

### 3. 接触面积和采样口径仍不完全等价

SFC 当前 `LagrangianSDFSurfaceContactGeometry` 对 TET4 边界三角面做 `tri3` 采样，面积权重来自三角面面积。Abaqus surface-to-surface 接触不是简单的逐三角 Gauss 点罚函数叠加；它会进行 surface constraint averaging 和 contact region 级别处理。

因此即使相同 gap 法向，积分面积、active set 边界和压力峰值也可能不同。应力峰值对压力分布最敏感，所以比位移更早、更大幅度地分叉。

### 4. 应变指标口径不一致

当前脚本中存在三个不同应变口径：

- SFC `strain_norm_nodeavg`：从 SFC 恢复的小应变张量或 corotated 残余应变求范数。
- Abaqus `logarithmic_strain_norm_nodeavg`：来自 Abaqus `LE` 的对数应变范数。
- `equivalent_elastic_strain_nodeavg`：从 von Mises stress 反推的等效弹性应变，本质上与 von Mises 同步，并不是独立应变验证。

所以 `strain_norm` 曲线本身不能作为跨软件强结论。论文中若展示应变，应优先统一为同一种定义：例如都用单元积分点小应变，或都用 Abaqus `LE` 对应的有限应变口径；否则趋势差异可能部分来自定义差异。

### 5. 节点平均会放大局部接触边界差异

SFC 和 Abaqus 都输出 node-averaged scalar，但节点平均规则不一定一致。接触区附近 TET4 单元应力近似常值，节点平均时一个高应力单元是否参与、权重如何给、是否跨 body/region 平滑，都会显著影响 max 和 p95 曲线。

这解释了为什么 active-region 应力误差会出现非常大的瞬时值，而全场位移仍平滑。

### 6. HHT/隐式动力学和自动增量口径仍未完全等价

SFC 使用 HHT/Newmark 近似 Abaqus/Standard direct integration，但 Abaqus 的隐式动力学与自动增量、非线性迭代、接触状态收敛和 cutback 耦合。当前 SFC 源齿轮路径是固定 `dt` 以及 reduced modified Newton / CG 接触修正。即使 `dt` 一样，接触建立和释放时的有效平衡点也可能不同。

动态应力波对接触时刻和冲量非常敏感，这会造成后续应力趋势差异。

## 最可能的主因排序

1. 接触压力历史和释放时序不同：这是当前数据已经直接显示的主因。
2. SFC 源齿轮路径求解内力为 `reference_linear`，而 Abaqus deck 是 `nlgeom=YES`。
3. surface-to-surface 接触约束平均/压力平滑/active set 口径不同。
4. 应变和应力场输出位置与节点平均口径不同。
5. HHT/自动增量/非线性收敛控制仍不是 Abaqus 源级等价。

## 修复优先级

1. 先不要继续只看位移，也不要继续调阻尼或调接触刚度。应先输出并对齐接触功、接触力、CPRESS/COPEN、active area 和 release time。
2. 将源齿轮 SFC 求解路径从 `reference_linear` 升级到有限变形一致内力路径，或者把 Abaqus 对照改成 `nlgeom=NO` 做纯线性理论验证。不能让 `nlgeom=YES` 对 `reference_linear` 直接承担论文主结论。
3. 单独建立 surface-to-surface 接触标定：同一 TET4 齿面 patch，比较 Abaqus 与 SFC 的 pressure distribution、active set 和法向接触力，确认面积权重和约束平均一致后再跑完整齿轮。
4. 应力/应变比较先改成单元中心或积分点口径，再做节点平均云图。节点平均云图用于视觉展示，曲线验收应优先用 element-center p95 / volume-mean / contact-patch mean。
5. 对应变曲线，停用混合 `strain_norm` 主指标；主文改用 von Mises stress 与统一定义的 equivalent elastic strain，`LE` 对比放入“输出口径说明”。
6. 加能量审计：internal energy、kinetic energy、contact work、external work。应力趋势分叉发生在哪个时刻，应由能量和接触冲量定位，而不是只看云图。

## 对论文表述的影响

当前结果可以说明：

- 源 deck 驱动单位、RP/MPC 大尺度运动和 VTK 对齐流程基本正确。
- SFC 与 Abaqus 的位移曲线在强约束齿轮工况下接近。
- 现有应力/应变曲线不能作为“完整齿轮接触应力等价”的支持证据。

当前不能声称：

- SFC 对 Abaqus 源齿轮工况的应力/应变趋势已经对齐。
- 线性罚函数 SFC 接触已经等价于 Abaqus `HARD` surface-to-surface contact。
- `reference_linear` 源齿轮快速路径已经等价于 Abaqus `nlgeom=YES`。

下一步应把问题定位为“接触压力历史与应力恢复口径对齐”，而不是继续从 SDF 查询精度入手。
