# 开源接触 Benchmark 替代计划执行交接

## 结论

已经开始按替代计划执行：NAFEMS/SimScale rounded-punch 路线不再作为论文证据；
SfePy two-body contact 已迁移到 true dynamic narrow-band SDF field replay；
FuzzyContact 已作为可选 VTU 三维场数据源接入；CalculiX 现有 `contactenergy`
和 C3D8 trajectory replay 保留为工程外部参考；Code_Aster frictionless
Hertz/indentation 暂列为后续解析 benchmark 候选。

## 已完成工作

1. **SfePy 外部接触参考线**

   - 脚本：`validation/run_external_contact_solver_comparison.py`
   - 外部求解器：SfePy `two_bodies_contact.py`
   - SfePy 负责求解 two-body penalty contact 终态。
   - SFC 不再用旧 `dynamic_surface_sdf(...)` replay；现在从 SfePy 终态
     master surface 构造 `DynamicNarrowBandSDF.build_required_points(...)`
     真场，并用 `query_phi(...)` 插值查询 gap。
   - 输出增加 `field_population_mode`、`field_valid_node_count`、
     `field_update_seconds` 和 scalar-gradient norm 范围，用于证明这是 field
     replay 而不是 projection query replay。

2. **FuzzyContact VTU 三维场参考线**

   - 脚本：`validation/run_fuzzycontact_vtu_reference.py`
   - 数据源：Mendeley Data `tdgkkjyr2f`，DOI `10.17632/tdgkkjyr2f.3`
   - 默认模式只抓取公开元数据，不下载大文件。
   - 用户提供 `--source-dir` 或显式 `--download --extract` 后，脚本会解析 VTU
     文件，统计 point-data fields，并生成三维表面云图 PNG/PDF。
   - 该路径只作为 optional validation，不是 core solver dependency。

3. **测试**

   - `tests/test_open_source_contact_benchmark_plan.py` 锁定：
     SfePy runner 必须包含 `DynamicNarrowBandSDF` 和 `build_required_points`，
     且不能重新引入 `dynamic_surface_sdf`。
   - `tests/test_fuzzycontact_vtu_reference.py` 用小型 ASCII VTU fixture 验证：
     VTU 解析、字段统计、三维 surface preview、metadata-only 不下载。

## Claim 边界

这些新增路径可以支持：

- 公开、可复现外部接触终态上的 SDF field gap replay；
- 开源 VTU 数据集的三维位移/应变/应力云图审计；
- CalculiX 既有工程 replay 作为补充视觉/物理一致性证据。

它们不能支持：

- 在没有开放参考数值的情况下宣称通过 NAFEMS 官方 benchmark；
- friction、self-contact、GPU、barrier contact、POD、neural SDF；
- 将 SfePy、CalculiX、FuzzyContact 或 Code_Aster 变成 core solver 依赖；
- 外部求解器源码级等价或所有工程接触场景的泛化正确性。

## 后续建议

下一步可以在 FuzzyContact 数据集下载完成后，将真实 VTU 解场导出为论文图：

```bash
python validation/run_fuzzycontact_vtu_reference.py --source-dir <extracted-dataset> --out-dir results/fuzzycontact_vtu_reference
```

Code_Aster 应优先选择 frictionless Hertz/indentation 案例，因为它更容易形成
解析接触 benchmark，并且不违反当前“不引入 friction 主 claim”的约束。
