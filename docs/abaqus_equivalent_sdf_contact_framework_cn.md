# Abaqus 等价的 SDF 引导 surface-to-surface 接触理论框架

本文档目的不是复刻 Abaqus 闭源实现，而是根据 Abaqus 公开理论文档和计算接触力学文献，重新推导一个可在 SFC 中实现的、与 Abaqus/Standard surface-to-surface 正常接触语义等价的 SDF 接触框架。

适用范围：

- 无摩擦正常接触；
- deformable-deformable FEM 接触；
- finite sliding；
- linear penalty 或 hard contact 的 penalty approximation；
- implicit dynamics / HHT；
- SDF 只作为当前构型几何 oracle，不作为独立物理接触律。

不包含：

- friction；
- self-contact；
- cohesive contact；
- thermal/electrical coupling；
- Abaqus 源代码级内部经验参数复刻。

## 1. 文献和官方理论依据

### 1.1 Abaqus surface-to-surface contact

Abaqus/Standard 文档明确区分 node-to-surface 与 surface-to-surface。关键点是：

- node-to-surface 抵抗 secondary nodes 穿透 main surface；
- surface-to-surface 在 secondary nodes 附近的有限区域上以平均意义施加接触；
- surface-to-surface 通常比 node-to-surface 给出更平滑、更准确的接触压力和应力；
- finite-sliding surface-to-surface 可以使用 main-secondary tracking；
- contact force redistribution upon sliding 有平滑阶次控制。

对应官方资料：

- Abaqus, "Contact Formulations in Abaqus/Standard":
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAEITNRefMap/simaitn-c-contactpairform.htm
- Abaqus, "Generally Applicable Contact Controls in Abaqus/Standard":
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAEITNRefMap/simaitn-c-contactcontrolsstd.htm

这说明 SFC 当前不能只使用：

```text
g_i = phi(x_i)
p_i = k < -g_i >_+
```

作为最终 surface-to-surface 接触。这个是点接触或 quadrature-point contact，不是 Abaqus-style surface-to-surface constraint region。

### 1.2 Abaqus normal contact enforcement

Abaqus/Standard 正常接触支持：

- direct enforcement；
- penalty method；
- augmented Lagrange method。

对于 finite-sliding surface-to-surface hard contact，Abaqus/Standard 默认使用 penalty method。线性罚函数可以看作 hard contact 的刚性近似，也可以作为 linear pressure-overclosure law。

对应官方资料：

- Abaqus, "Contact Constraint Enforcement Methods in Abaqus/Standard":
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAEITNRefMap/simaitn-c-contactconstraints.htm
- Abaqus, "Contact Pressure-Overclosure Relationships":
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAEITNRefMap/simaitn-c-normalinteraction.htm
- Abaqus Theory Guide, "Contact pressure definition":
  https://abaqus.uclouvain.be/English/SIMACAETHERefMap/simathe-c-contactpress.htm

### 1.3 Abaqus implicit dynamics and nonlinear solution

Abaqus/Standard direct integration dynamics 默认使用 Hilber-Hughes-Taylor operator，除非使用 quasi-static application。非线性方程用 Newton / modified Newton / quasi-Newton，收敛同时检查残差和修正量。自动步长会在收敛困难时 cutback；动态问题还有 half-increment residual 和接触状态变化相关的步长控制。

对应官方资料：

- Abaqus, "Implicit Dynamic Analysis Using Direct Integration":
  https://abaqus-docs.mit.edu/2017/English/SIMACAEANLRefMap/simaanl-c-dynamic.htm
- Abaqus Theory Guide, "Nonlinear solution methods in Abaqus/Standard":
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAETHERefMap/simathe-c-nonlinearsol.htm
- Abaqus, "Convergence Criteria for Nonlinear Problems":
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAEANLRefMap/simaanl-c-convergcriteria.htm
- Abaqus, "Automatic incrementation control in Abaqus/Standard":
  https://docs.software.vt.edu/abaqusv2024/English/SIMACAEGSARefMap/simagsa-c-nlnautomatic.htm

### 1.4 计算接触力学文献

经典接触力学文献支持将 surface-to-surface contact 写成变分接触势、接触约束积分、罚函数或增广拉格朗日形式。与本文最相关的是 segment-to-segment / mortar 思路：接触不是单点事件，而是在接触面区域上弱式施加非穿透约束。

代表性文献：

- Wriggers, *Computational Contact Mechanics*, Springer.
- Laursen, *Computational Contact and Impact Mechanics*, Springer.
- Puso and Laursen, "A mortar segment-to-segment contact method for large deformation solid mechanics", CMAME, 2004.
- Puso and Laursen, "A mortar segment-to-segment frictional contact method for large deformations", CMAME, 2004.
- Fischer and Wriggers, "Frictionless 2D contact formulations for finite deformations based on the mortar method", CMAME, 2006.
- Maas et al., "A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in FEBio", ASME J. Biomech. Eng., 2018:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/

这些文献不说明 Abaqus 的闭源实现细节，但给出了与 Abaqus surface-to-surface 平均约束一致的弱式理论基础。

## 2. 当前 SDF 接触理论缺漏

当前 SDF 接触容易写成：

```text
g_i = phi_B(x_i^A, q_B)
n_i = grad phi_B(x_i^A, q_B) / |grad phi_B|
p_i = k < -g_i >_+
f_c = sum_i A_i p_i n_i
```

这个理论有五个缺漏。

### 2.1 点级 gap 不等于 surface-to-surface 约束

Abaqus surface-to-surface 的核心是 finite secondary region average，而不是单个点独立 active。点级 gap 会造成：

- 局部一个 quadrature point 穿透就产生压力；
- active set 受三角面切换影响强；
- pressure 和 stress 出现 spikes；
- 位移可能接近，但应力/应变不稳定。

### 2.2 SDF normal 不等于接触约束区域法向

SDF normal 来自主面最近特征。对于 faceted gear tooth：

```text
n = n_master_face
```

会在面、边、顶点附近跳变。Abaqus surface-to-surface 通过平均区域、接触方向、force redistribution smoothing 和可选 surface smoothing 减小这种跳变。SFC 必须区分：

```text
geometric closest-feature normal
constraint-region contact direction
output pressure direction
```

### 2.3 master-side sensitivity 不完整

常用近似：

```text
dg/dx_A = N_A n^T
dg/dx_B = -N_B n^T
```

只在 closest feature、normal、barycentric payload 冻结时成立。完整线性化还包括：

```text
dn/dq
d xi*/dq
d weights_region/dq
d active_status/dq
```

如果忽略这些项，Newton 可以在位移上收敛，但接触压力峰值和应力云图会出现偏差。

### 2.4 active set 不应由 sample signature 单独决定

Abaqus 非线性收敛会结合：

- force residual；
- displacement correction；
- constraint residual；
- contact status discontinuity；
- automatic cutback；
- contact impact velocity/acceleration compatibility。

SFC 当前如果只要求 sample-level active signature 稳定，会过度离散，容易在齿面滑移时产生跳变。

### 2.5 求解压力和输出压力必须同口径

如果求解使用 quadrature sample pressure，而 VTK/CSV 输出再做 node average，则和 Abaqus CPRESS/COPEN 不同口径。必须定义：

```text
COPEN_r = G_r
CPRESS_r = p_r
```

再用同一个区域权重投影到节点或单元。

## 3. Abaqus 等价 SDF contact operator

设 secondary body 为 A，main body 为 B。当前构型：

```text
x_A(X_A, q_A)
x_B(X_B, q_B)
Gamma_A(q_A), Gamma_B(q_B)
```

SDF oracle：

```text
phi_B(x, q_B)
n_B(x, q_B) = grad_x phi_B / |grad_x phi_B|
payload_B(x) = (face_id, xi*, N_B(xi*), n_B)
```

这里 `phi_B` 可以来自：

- dynamic narrow-band SDF field；
- Lagrangian SDF oracle + current-surface corrector；
- current closest-feature projection；
- normal-tube inversion。

但它只负责几何量，不直接决定最终 pressure。

### 3.1 Secondary constraint region

对每个 secondary constraint region `r`，定义非负权重函数：

```text
psi_r(s) >= 0
```

其中 `s` 是 secondary surface 参数。`psi_r` 应满足局部 partition-of-unity 或近似 partition-of-unity：

```text
sum_r psi_r(s) = 1
```

区域面积：

```text
A_r(q_A) = integral_{Gamma_A} psi_r(s) dA
```

实现上可以取：

- secondary node patch；
- slave element face patch；
- surface-to-surface averaging patch；
- element-order smoothing patch。

对 TET4 triangular faces，最基本可实现版本是 secondary-node patch：

```text
psi_r = nodal shape function N_r on incident slave faces
A_r = sum_{faces e incident to r} integral_{face e} N_r dA
```

这比单个 quadrature point 更接近 Abaqus 文档中的 secondary-node nearby averaging region。

### 3.2 区域平均 gap

定义点级 SDF clearance：

```text
g(s, q) = phi_B(x_A(s, q_A), q_B)
```

区域平均 gap：

```text
G_r(q) =
    (1 / A_r)
    integral_{Gamma_A} psi_r(s) g(s, q) dA
```

离散后：

```text
G_r =
    (1 / A_r)
    sum_{e in patch(r)} sum_{g in quad(e)}
        w_g J_e psi_r(s_g) phi_B(x_A(s_g), q_B)
```

这一步是当前 SFC 最关键的理论替换：

```text
不是 sample gap 决定 active
而是 region gap G_r 决定 active
```

### 3.3 区域接触方向

点级 SDF normal：

```text
n_g = n_B(x_A(s_g), q_B)
```

区域平均 normal：

```text
m_r =
    integral psi_r(s) omega_g n_g dA
```

```text
n_r = m_r / |m_r|
```

其中 `omega_g` 可以取 1，也可以取和当前 active overclosure 相关的权重。但为了避免单点压力峰支配，默认应使用面积/形函数一致权重：

```text
omega_g = 1
```

如果采用 secondary-normal line projection，则接触方向应由 secondary constraint region 的平均法向给出：

```text
n_r = normalize(integral psi_r n_A(s) dA)
```

而 SDF closest-feature normal 只用于：

- 判断 geometric open/closed guard；
- closest payload；
- master sensitivity；
- fallback。

### 3.4 Linear penalty pressure-overclosure

区域 overclosure：

```text
h_r = < -G_r >_+
```

线性罚压力：

```text
p_r = k_r h_r
```

接触势：

```text
Pi_c(q) =
    1/2 sum_r A_r k_r < -G_r(q) >_+^2
```

接触虚功：

```text
delta Pi_c =
    - sum_r A_r p_r delta G_r
```

因此接触力：

```text
f_c = - dPi_c/dq
```

这与 Abaqus linear pressure-overclosure 的正常方向罚函数语义一致。若要模拟 hard contact 的 penalty approximation，可以让 `k_r` 来自 underlying element stiffness 尺度，而不是 case-specific curve fitting。

### 3.5 Contact Jacobian

对一个 region：

```text
G_r =
    (1 / A_r) sum_g W_{rg} phi_B(x_g, q_B)
```

其中：

```text
W_{rg} = w_g J_e psi_r(s_g)
```

冻结 closest-feature 几何时：

```text
delta phi_B =
    n_g^T delta x_A(s_g)
    - n_g^T delta x_B*(s_g)
```

主面最近点：

```text
x_B*(s_g) = sum_a N_a^B(xi_g*) x_a^B
```

于是：

```text
delta G_r =
    sum_b D_{rb}^A delta x_b^A
    + sum_a D_{ra}^B delta x_a^B
```

其中：

```text
D_{rb}^A =
    (1 / A_r)
    sum_g W_{rg} N_b^A(s_g) n_g^T
```

```text
D_{ra}^B =
    -
    (1 / A_r)
    sum_g W_{rg} N_a^B(xi_g*) n_g^T
```

将两者合并为一行：

```text
D_r = dG_r/dq
```

接触力：

```text
f_c =
    sum_{active r} A_r p_r D_r^T
```

罚函数一致材料切线主项：

```text
K_c^{mat} =
    sum_{active r} A_r k_r D_r^T D_r
```

完整几何切线还包括：

```text
K_c^{geo} =
    sum_{active r} A_r p_r dD_r/dq
```

第一阶段可实现：

```text
K_c = K_c^{mat}
```

并在文档中明确为 Abaqus-equivalent normal penalty 的 frozen-geometry Newton tangent。第二阶段加入：

- `dn/dq`；
- `d xi*/dq`；
- `dA_r/dq`；
- region normal smoothing derivative。

### 3.6 Active status

区域 active 判据：

```text
active_r iff G_r < -g_tol
```

接触状态更新应有 hysteresis 或 tolerance：

```text
closed if G_r < -g_close
open if G_r > g_open
hold otherwise
```

其中 `g_close, g_open` 必须来自网格尺度、求解容差或 Abaqus-style penetration tolerance，不允许按单个算例调参。

推荐尺度：

```text
g_tol = c_g h_rms eps_solver
```

或：

```text
g_tol = min(c1 h_surface, c2 characteristic_displacement_tolerance)
```

这里 `h_surface` 是 secondary region 的特征长度。

### 3.7 Force redistribution smoothing

Abaqus 文档明确 surface-to-surface 接触有 sliding 时的 nodal contact force redistribution smoothing，默认平滑阶次与 secondary surface element order 一致。

SFC 中应实现：

```text
F_a^A =
    sum_r A_r p_r
    integral_R psi_r N_a^A n_r dA / A_r
```

而不是把所有压力集中到一个 sample 或一个节点。对于线性 TET4 face：

```text
linear smoothing:
  force weights follow linear face shape functions
```

对二阶面可扩展到 quadratic redistribution。

这一步应替代当前经验性的 pressure smoothing factor。

### 3.8 Output semantics

求解和输出必须同口径：

```text
COPEN_r  = G_r
CPRESS_r = p_r
CAREA_r  = A_r
```

节点输出：

```text
CPRESS_node(a) =
    sum_r M_{ar} A_r p_r / sum_r M_{ar} A_r
```

其中 `M_{ar}` 来自同一套 force redistribution / region participation 权重。

这样 SFC VTK 中的 pressure、penetration、active mask 才能和 Abaqus CPRESS/COPEN 进行有意义的对比。

## 4. HHT implicit dynamics coupling

动力学方程：

```text
M a + f_int(q) - f_ext - f_c(q) = 0
```

HHT 形式可写为：

```text
R_{n+1} =
    M a_{n+1}
    + (1 + alpha) [f_int(q_{n+1}) - f_ext_{n+1} - f_c(q_{n+1})]
    - alpha [f_int(q_n) - f_ext_n - f_c(q_n)]
```

Newmark predictor：

```text
q_{n+1} = q_pred + beta dt^2 a_{n+1}
v_{n+1} = v_pred + gamma dt a_{n+1}
```

Newton tangent：

```text
K_eff =
    M / (beta dt^2)
    + (1 + alpha) [K_int - K_c]
```

这里符号取决于残差定义；实现时必须保证：

```text
delta R = K_eff delta q
```

接触状态变化时：

- 重新构造 region gap `G_r`；
- 重新判断 active set；
- 检查 contact force increment；
- 必要时 cutback。

## 5. Abaqus-equivalent nonlinear solve loop

每个增量：

```text
given q_n, v_n, a_n, active_state_n
predict q_pred, v_pred
for Newton iteration i:
    1. current geometry q_i
    2. update SDF oracle / closest-feature payload
    3. build secondary constraint regions
    4. compute G_r, n_r, A_r, D_r
    5. update active status with tolerance
    6. assemble f_c and K_c
    7. assemble HHT residual R_i
    8. solve K_eff delta q = -R_i
    9. line search if needed
   10. check:
       - force residual
       - displacement correction
       - contact constraint residual
       - contact force increment
       - active status stability
if not converged:
    cutback dt and retry
accept q_{n+1}, v_{n+1}, a_{n+1}, active_state_{n+1}
```

不能再使用：

```text
固定迭代次数后无条件接受
```

否则应力/应变对齐无法稳定。

## 6. SDF 在该框架中的正确角色

SDF oracle 负责：

```text
phi_B(x, q_B)
n_B(x, q_B)
closest face id
barycentric / natural coordinates
master payload
candidate acceleration
```

SDF 不负责：

```text
定义 surface-to-surface constraint region
定义 pressure smoothing
定义 active set tolerance
定义 HHT/Newton/cutback
定义 Abaqus CPRESS/COPEN 输出语义
```

因此论文和代码应把创新点写为：

```text
SDF-guided current-surface geometry oracle
+ Abaqus-equivalent surface-to-surface contact operator
```

而不是：

```text
SDF query alone solves contact
```

## 7. 与当前实现的差距清单

当前实现中已经具备：

- current-space SDF / Lagrangian oracle；
- closest-feature payload；
- secondary-normal projection；
- slave/master Jacobian 主项；
- penalty force；
- HHT 基础框架；
- 部分 active set stability；
- VTK 输出。

仍需补齐：

1. secondary constraint region `R_r` 的正式定义；
2. 区域平均 gap `G_r` 作为唯一 active 判据；
3. 基于 `G_r` 的 pressure-overclosure；
4. force redistribution smoothing，而非 sample pressure aggregation；
5. `D_r = dG_r/dq` 的区域 Jacobian；
6. `K_c = A_r k_r D_r^T D_r` 的 matrix-free 或 sparse tangent；
7. contact force increment / penetration / active status convergence；
8. cutback；
9. CPRESS/COPEN 输出与求解同口径；
10. 平面压入 -> 局部齿面 -> 全齿轮的逐层验证。

## 8. 最小可实现版本

第一阶段不需要完全复刻 Abaqus 内部所有平滑细节。最小 Abaqus-equivalent 版本是：

```text
secondary-node constraint regions
linear face shape function weights
region gap G_r
region pressure p_r = k < -G_r >_+
force redistribution from same region weights
frozen-geometry consistent tangent K_c = A k D^T D
active status tolerance + contact force increment convergence
same CPRESS/COPEN output projection
```

这个版本已经比当前 sample-level contact 更接近 Abaqus 文档和接触力学文献。

## 9. 验证路径

不能直接用全齿轮作为第一验证。建议：

1. **平面双柔性块压入**

   验证：

   ```text
   RF/U
   CPRESS/COPEN
   active region
   stress/strain p95
   ```

2. **曲面齿面局部 patch**

   验证：

   ```text
   finite sliding active region
   pressure smoothing
   contact force redistribution
   stress/strain peak
   ```

3. **全齿轮 10 步**

   验证：

   ```text
   displacement error < 10%
   von Mises p95 error < 10%
   equivalent strain p95 error < 10%
   active mask recall/precision stable
   ```

4. **全齿轮长程**

   验证：

   ```text
   same dt
   same output frame times
   complete solve time
   SFC faster than Abaqus only after accuracy gate passes
   ```

## 10. 理论结论

根据 Abaqus 官方文档和计算接触力学文献，可以推导出一个与 Abaqus surface-to-surface 正常接触等价的 SDF 框架：

```text
current FEM surface
-> SDF / closest-feature geometry oracle
-> secondary constraint regions
-> region-averaged gap G_r
-> pressure-overclosure p_r
-> variational contact force
-> consistent tangent
-> HHT/Newton/cutback
-> CPRESS/COPEN same-semantics output
```

当前 SFC 的关键理论缺口是：

```text
缺少 Abaqus-style secondary constraint-region contact operator。
```

只修 SDF 查询、扩大候选半径或调罚刚度，无法稳定解决应力/应变曲线不一致问题。
