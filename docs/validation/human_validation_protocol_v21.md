# 情感意象 v2.1 人工验证协议

## 目的

本协议用于验证 `build_affective_imagery_labels_v21.py` 生成的包装情感意象标签是否可靠。验证对象是评论句子中的“包装对象-情感表达-意象维度”关系，以及这些句子关系聚合到 `parent_asin` 商品级标签后的结果。

本协议不重新判定 5,180 个 metadata 商品池，也不把现有 GPT 辅助复核写成独立真人 gold validation。metadata 审查结果只作为已冻结的商品范围来源。

## 执行时点

正式人工验证必须在以下条件全部满足后执行：

- v2.1 已修复已知规则问题并通过回归测试。
- v2.1 已在 5,180 商品池上完成一次全量运行。
- 输出目录为 `data/processed/affective_imagery_labels_v21_5180` 或另一个明确记录的冻结目录。
- 本次抽样使用固定随机种子 `42`，并保存抽样脚本、输入文件哈希和输出样本哈希。

如果 v2.1 在人工验证后再次改动规则、词典、抽样逻辑或输入数据，必须重新生成验证样本并重新审计受影响部分。

## 输入文件

正式验证使用 v2.1 全量输出中的以下文件：

- `37_relation_constrained_sentence_evidence_v21.parquet` 或 `.csv.gz`：句子级关系证据。
- `38_relation_constrained_imagery_dimensions_v21.csv`：维度覆盖统计。
- `39_product_imagery_labels_v21.csv`：商品级 PU 标签。
- `39b_product_dimension_evidence_v21.csv`：商品-维度证据聚合。
- `40_relation_constrained_summary_v21.json`：运行摘要和策略说明。
- `41_relation_constrained_audit_sample_v21.csv`：句子级审计样本。
- `43_uncertain_targeted_recovered_v21.parquet` 或 `.csv.gz`：从 uncertain 中恢复的证据。

真实标注表和 adjudication 结果只保存在本地 `data/manual_validation/affective_imagery_v21_5180/`，不进入 Git。当前 01–08 已生成，A1/A2 尚未开始。

## 可执行工具与当前工作区

本阶段使用三个离线脚本，不访问网络，也不会自动生成或伪造人工判断：

```text
scripts/validation/prepare_affective_imagery_validation_v21.py
scripts/validation/validate_affective_imagery_annotations_v21.py
scripts/validation/summarize_affective_imagery_validation_v21.py
```

正式工作区已经生成，不应再次执行准备命令。若只为复现，必须使用新的临时输出目录，并提供 V2.1 evidence 和 upstream classified 输入：

```powershell
python -B scripts/validation/prepare_affective_imagery_validation_v21.py `
  --input-dir data/processed/affective_imagery_labels_v21_5180 `
  --output-dir data/manual_validation/affective_imagery_v21_reproduction `
  --upstream-classified data/processed/strict_visual_packaging_v11_5180/15_packaging_sentences_rule_classified.parquet `
  --v21-evidence data/processed/affective_imagery_labels_v21_5180/37_relation_constrained_sentence_evidence_v21.parquet
```

正式工作区文件：

- `01_sentence_items.csv`：600 个句子任务。
- `02_product_dimension_items.csv`：240 个商品—维度任务。
- `03_annotations_A1.csv`：A1 的 840 项答案。
- `04_annotations_A2.csv`：A2 的 840 项答案。
- `05_adjudication_template.csv`：分歧裁决模板。
- `06_validation_manifest.json`：输入、输出和抽样 provenance。
- `07_product_dimension_evidence_context.csv`：商品—维度任务完整回查档案。
- `08_product_dimension_reviewer_context.csv`：商品—维度任务日常 reviewer packet。

汇总命令在 A1/A2 和裁决完成后使用：

```powershell
python -B scripts/validation/summarize_affective_imagery_validation_v21.py `
  --items-dir data/manual_validation/affective_imagery_v21_5180 `
  --a1 data/manual_validation/affective_imagery_v21_5180/03_annotations_A1.csv `
  --a2 data/manual_validation/affective_imagery_v21_5180/04_annotations_A2.csv
```

当前功能基线中，`validate_affective_imagery_annotations_v21.py --help` 会因 `parser` 未定义而失败；它不阻塞开始标注，但最终自动验证前必须先修复并补 CLI 回归测试。

### 禁止覆盖

A1/A2 一旦开始，不得在正式目录执行准备脚本的 `--overwrite`。该操作会重新生成空白 03/04，可能清空已经完成的审核结果。任何复跑只能写入新的临时目录。真实标注文件不得提交到 Git。

## 验证单位

主要验证单位是句子级 evidence row，而不是整条评论或整件商品。每一行只审一个预测关系：

```text
sentence / clause_text
+ object_term
+ expression_raw / expression_lemma
+ dimension_code
+ package_level
```

商品级标签只作为第二层检查：若一个商品在某维度上被标为 `1`，必须能追溯到至少一条 adjudicated valid 的 outer package relation evidence。

## 抽样设计

正式句子级样本目标为 600 行。若某组可用行数少于配额，则该组全取，剩余名额按其他组可用量重新分配。

| audit_group | 目标行数 | 目的 |
| --- | ---: | --- |
| `outer_relation_evidence` | 120 | 主图建模可用正例精度 |
| `recovered_v21` | 120 | uncertain 恢复规则精度 |
| `uncertain_not_recovered` | 120 | 恢复规则漏标检查 |
| `inner_relation_evidence` | 80 | 内包装误入主图标签风险 |
| `ambiguous_relation_evidence` | 80 | 包装层级不明风险 |
| `strict_without_relation_evidence` | 80 | strict 句子未形成关系证据的漏标检查 |

每个 `audit_group` 内按 `dimension_code` 分层。低频维度优先抽满可用行；高频维度用固定 seed 随机抽样。抽样程序不能先追加第六组再用全局 `.head(sample_size)` 截断，否则会系统性丢掉 `strict_without_relation_evidence`。

商品级聚合检查目标为 240 个 product-dimension 检查项：

- 对每个 `keep_for_core_model = 1` 的维度，抽 30 个 positive product-dimension pairs。
- 对 pilot-only 或低频维度，若 positive product-dimension pairs 少于 30，则全取；否则抽 30 个。
- 额外抽 60 个高评论曝光但对应维度为 unlabeled 的 product-dimension pairs，用于检查明显漏标信号。
- 若总量超过 240，优先保留 core 维度和低频维度，再用 seed `42` 抽样。

## 标注流程

A1 和 A2 必须分别审核同一套 840 项，而不是各做一半。A1 只填写 03，不读取 04；A2 只填写 04，不读取 03。句子任务优先依据 `clause_text`，必要时参考完整 `sentence`，不得用商品最终标签反推句子判断。

商品—维度任务优先查看 08。出现低置信度、A1/A2 分歧、疑似漏标、`focus_review_flag=1` 或需要完整上下文时，必须回查 07。

如果 A1/A2 由两个 Codex/GPT 对话完成，只能称为“双重独立模型辅助审核”，不能称为两名真人标注员或人工金标准。真实人工裁决完成前，`human_gold_validation_pending = true`。

标注步骤如下：

1. 判断句子是否在描述零售商品包装的视觉外观。
2. 判断包装对象和情感/意象表达之间是否存在明确关系。
3. 判断包装层级是 outer、inner、ambiguous、non_packaging 还是 uncertain。
4. 判断预测维度是否正确；如错误，给出人工维度或 `none`。
5. 记录错误类型、置信度和中文理由。
6. 两名标注者不一致时，由 adjudicator 冻结最终结果。

## 通过标准

人工验证不是为了证明标签没有噪声，而是为了决定当前 v2.1 是否足够进入后续图像建模。建议采用以下预声明阈值：

| 指标 | 阈值 | 未达标处理 |
| --- | ---: | --- |
| `outer_relation_evidence` 中 valid outer relation precision | >= 0.90 | 回到 v2.1 规则修正 |
| `recovered_v21` 中 valid outer relation precision | >= 0.85 | 收紧 recovery 规则 |
| valid outer evidence 的 dimension accuracy | >= 0.85 | 修正维度词典或关系规则 |
| `inner_relation_evidence` 误判为 outer 的比例 | <= 0.10 | 修正 package_level 规则 |
| `ambiguous_relation_evidence` 可安全纳入 outer 的比例 | 不自动纳入 | 仅做敏感性分析 |
| `uncertain_not_recovered` 明确应恢复比例 | <= 0.15 | 扩展 recovery 规则并重跑 |
| `strict_without_relation_evidence` 明确应形成关系比例 | <= 0.15 | 修正关系抽取规则 |
| 双人标注一致率 | >= 0.80 | 澄清指南并重标问题组 |

若任一主指标未达标，不应手工静默修补标签。可接受的处理方式只有两种：

- 修改 v2.1 规则后全量重跑，并重新验证受影响组。
- 使用透明的 frozen changes ledger 应用人工 adjudication，并记录每一条变化的原值、目标值、理由和哈希。

## 指标计算

主要报告以下指标：

- relation precision：预测关系中 adjudicated valid 的比例。
- outer package precision：预测 outer 中人工也判为 outer 的比例。
- dimension accuracy：adjudicated valid relation 中维度一致的比例。
- recovered precision：`recovered_v21` 中 adjudicated valid 的比例。
- false-negative signal：`uncertain_not_recovered` 或 `strict_without_relation_evidence` 中应被纳入的比例。
- product-label traceability：商品级 `1` 标签是否至少有一条 adjudicated valid outer evidence。
- inter-annotator agreement：双人标注一致率；如计算条件满足，也报告 Cohen's kappa。

不要把 `0` 标签解释为“确认不存在该意象”。本项目标签语义是 Positive-Unlabeled：`1` 表示观察到消费者主动提及，`0` 表示未观察到。

## 冻结规则

通过验证后，冻结以下材料：

- v2.1 输出目录路径和所有关键文件 SHA-256。
- 抽样脚本版本、随机种子和抽样样本 SHA-256。
- 双人标注原表和 adjudicated final 表。
- 指标汇总表。
- 若有人工修订，保存 changes ledger，不覆盖原始 v2.1 输出。

冻结后的主图建模只使用 adjudicated valid 的 outer retail package evidence 生成或确认标签。inner 和 ambiguous evidence 可用于补充分析，不进入默认主图模型标签。
