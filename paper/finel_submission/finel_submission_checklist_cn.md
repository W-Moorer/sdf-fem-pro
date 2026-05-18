# FINEL 投稿准备清单

目标期刊：*Finite Elements in Analysis and Design*

## 已完成

- 摘要控制在 250 词以内：当前约 178 词。
- 已加入 1-7 个英文 keywords：当前 7 个。
- 已生成独立 highlights 文件：`highlights.txt`。
- 每条 highlight 不超过 85 个字符。
- 正文表格使用可编辑 LaTeX 表格，不是图片。
- 图件已有 PNG/PDF 输出。
- 正文已移除内部 `Phase-*` 叙事。
- 正文已加入 Data Availability、Declaration of Competing Interest、Funding。
- 正文已加入 Elsevier 要求的生成式 AI 辅助写作声明，并生成 `generative_ai_statement.txt`。
- bibliography style 已改为 `unsrt`，参考文献按出现顺序编号。
- 稿件主线已对齐 FINEL：finite-element computational methodology、implementation、numerical demonstration。

## 投稿前仍需人工确认

- 替换 `Anonymous Authors` 为正式作者、单位、通信作者邮箱和完整地址。
- 确认 funding statement 是否真实；如有基金，需要替换当前无基金声明。
- 确认 data availability：如果要提高可复现性，建议把 `results/` 关键 CSV、脚本和版本信息归档到 Zenodo/Mendeley Data 并加入 DOI。
- 完成 Elsevier declaration of competing interests 在线工具并上传 `.doc/.docx` 文件。
- 确认是否需要在投稿系统中披露 AI-assisted writing/editing。
- 如不希望在正文中保留 AI 声明，必须确认没有使用生成式 AI 辅助 manuscript preparation；否则按 Elsevier 要求保留该声明。
- 检查所有 figures 是否按 `Figure_1`, `Figure_2`, ... 命名并作为独立文件上传。
- 确认图像分辨率：halftone 不低于 300 dpi，line art 不低于 1000 dpi，组合图不低于 500 dpi。
- 确认是否提交 supplementary material：建议包含 validation scripts、CSV 表格、VTU 场云图。

## 建议上传文件

- `paper/main.tex`
- `paper/references.bib`
- `paper/main.pdf`
- `paper/finel_submission/highlights.txt`
- `paper/finel_submission/keywords.txt`
- `paper/finel_submission/cover_letter.md`
- `paper/finel_submission/data_availability_statement.txt`
- `paper/finel_submission/declaration_of_competing_interest.txt`
- `paper/finel_submission/funding_statement.txt`
- `paper/finel_submission/generative_ai_statement.txt`
- 独立图件：`results/true_sdf_final/figures/*.pdf`, `results/external_visual/figures/*.pdf`, `results/true_field_solver_timing_formal/figures/*.pdf`, `results/native_contact_backend_ablation_quick/figures/backend_ablation_overview.pdf`

## FINEL 口径

推荐摘要/cover letter 中坚持以下表述：

```text
The paper presents a finite-element computational methodology and implementation for dynamic SDF-based contact. Projection is internalized as a field-construction kernel; repeated contact queries are performed by interpolation of a current-space narrow-band SDF field.
```

避免以下表述：

```text
The method is a general-purpose production contact solver.
The method is always faster than projection or CalculiX.
The method supports friction, self-contact, nonlinear FEM, GPU, or barrier contact.
```
