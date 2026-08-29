# 情感意象 v2.1 人工标注 Schema

## 文件用途

本 schema 用于人工验证 `41_relation_constrained_audit_sample_v21.csv` 中的句子级证据，以及抽样出的 product-dimension 聚合检查项。所有真实标注结果保存在本地 `data/manual_validation/affective_imagery_v21_5180/`，不进入 Git。

标注表不得删除行、重排行或改写模型输出列。人工判断只写入 `human_*`、`adjudicated_*` 和记录字段。

### 查看方式

1. **sentence 任务**（600条）：直接查看 03/04 当前行中的句子、子句、对象词等上下文。
2. **product_dimension 任务**（240条）：主要上下文来自 **`08_product_dimension_reviewer_context.csv`**（reviewer packet）。该表通过 `annotation_item_id` 连接到 02 中的每个商品—维度任务，按优先级层级精心采样。
3. **`07_product_dimension_evidence_context.csv`** 是完整审计档案，在以下情况回查：低置信度、A1/A2意见不一致、发现可能漏标信号、`focus_review_flag=1` 或需要全面复核时。

### positive 任务判断 traceable

对于 `model_label_value=1` 的 product-dimension 任务：
1. 主要从08中查找该 `annotation_item_id` 对应的上下文行。
2. `is_target_dimension_evidence=1` 且 `is_outer_eligible_evidence=1` 的行是模型判定为 outer 零售包装的证据。
3. 审核者判断是否有至少一条 outer 证据可以合理追溯到商品外包装的消费者评价。
4. 若无 outer 证据行（仅 non-outer），应结合 non-outer 证据判断是否实际存在 outer 信号被误判。**必须回查07获取完整上下文。**

### unlabeled 任务判断 missed_signal

对于 `model_label_value=0` 的 product-dimension 任务：
1. 主要从08中查找该 `annotation_item_id` 对应的上下文行。08是风险增强、确定性抽取的审核上下文。
2. 检查提供的包装候选上下文中是否存在明显被漏标的包装意象信号。
3. **检查范围仅限08（及回查的07）提供的包装候选上下文**，不代表已检查该商品的全部原始评论。
4. **不得把"08中未发现证据"表述成"全部评论中确定不存在证据"**。
5. 若 `context_status = no_candidate_context`（该商品无任何候选上下文），应优先填写 `uncertain` 并说明"无可用的包装候选上下文"，不能自动判 `no`。
6. **A1/A2出现以下情况时必须回查07完整上下文**：发现可能的漏标证据；无法判断；低置信度；A1/A2意见不一致；`focus_review_flag == 1`。

### 两个 focus review 商品

| annotation_item_id | parent_asin | 风险 |
|---|---|---|
| prod-96d1031a1b179bc5 | B0BWLWY25M | presentation与tea bags并列，inner判断置信度medium |
| prod-4b13f829c881c342 | B0C5ZMZBKS | artwork可能对应outer包装插画，可能存在outer false negative |

对应的 sentence items：`sent-0d82bad8e43882b8`, `sent-228e38f8b6424b47`。

### 标注独立性

A1和A2必须独立，不能查看对方答案。如果使用两个Codex/GPT对话，只能称为独立模型辅助审核，不能称为两名人类标注者。

双人标注采用两个彼此独立的长表：`03_annotations_A1.csv` 和 `04_annotations_A2.csv`。每个文件都包含同一组 `annotation_item_id`，但 `annotator_id` 分别固定为 `A1` 和 `A2`。唯一键为：

```text
(annotation_item_id, annotator_id, annotation_round)
```

adjudication 单独保存到 `05_adjudication_template.csv` 或其填写后的副本。该文件可以包含 `A1_*` 和 `A2_*` 前缀列用于并排复核，但不得覆盖 A1/A2 原始标注文件。

## 模型输出列

以下列来自 v2.1 输出，作为标注上下文保留：

| 列名 | 含义 |
| --- | --- |
| `sample_order` | 抽样顺序，从 1 开始 |
| `parent_asin` | 商品级分析单位 |
| `review_id` | 评论 ID |
| `sentence_id` | 句子 ID |
| `user_id` | 评论用户 ID；仅用于去重和统计，不用于身份分析 |
| `audit_group` | 审计分组 |
| `sentence` | 完整句子 |
| `normalized_sentence` | 规范化句子 |
| `clause_text` | v2.1 识别出的关系所在子句 |
| `clause_index` | 子句序号 |
| `relation_type` | 规则识别出的关系类型 |
| `object_term` | 包装对象词 |
| `package_level` | 预测包装层级 |
| `eligible_for_main_image_model` | 是否默认可进入主图模型标签 |
| `negated` | 规则是否识别为否定 |
| `source_kind` | strict 或 recovered |
| `source_type` | 更细的来源类型 |
| `dimension_code` | 预测意象维度 |
| `dimension_name_cn` | 预测意象维度中文名 |
| `polarity` | 预测极性 |
| `expression_raw` | 原句中的情感/意象表达 |
| `expression_lemma` | 归一化表达 |
| `rating` | 评论星级 |
| `verified_purchase` | 是否 verified purchase |
| `helpful_vote` | helpful vote 数 |
| `pipeline_version` | v2.1 程序版本 |

若某些 audit group 来自未形成 relation evidence 的原始句子，`dimension_code`、`expression_raw` 或 `package_level` 可为空。标注者仍需判断是否存在应被算法捕获的关系。

## 人工标注列

| 列名 | 必填 | 允许值 | 含义 |
| --- | --- | --- | --- |
| `human_packaging_visual` | 是 | `yes` / `no` / `uncertain` | 句子是否描述零售商品包装的视觉外观 |
| `human_relation_valid` | 是 | `yes` / `no` / `uncertain` | 包装对象和情感/意象表达之间是否有明确关系 |
| `human_package_level` | 是 | `outer` / `inner` / `ambiguous` / `non_packaging` / `uncertain` | 人工判断的包装层级 |
| `human_dimension_code` | 是 | 见维度代码表，另可填 `none` / `uncertain` | 人工判断的主维度 |
| `human_additional_dimension_codes` | 否 | 管道符分隔的维度代码，或空 | 同一句明显有多个维度时记录附加维度 |
| `human_polarity` | 是 | `positive` / `negative` / `uncertain` | 人工判断的情感极性 |
| `human_action` | 是 | `keep` / `drop` / `change_dimension` / `change_package_level` / `change_polarity` / `add_missing` | 对模型行的人工处理建议 |
| `human_error_type` | 是 | 见错误类型表 | 若需要修改或删除，记录主要错误类型 |
| `human_product_label_traceable` | 条件必填 | `yes` / `no` / `uncertain` | product-dimension 正例是否能追溯到有效 outer evidence |
| `human_unlabeled_missed_signal` | 条件必填 | `yes` / `no` / `uncertain` | product-dimension unlabeled 检查项是否发现明显漏标信号 |
| `human_confidence` | 是 | `high` / `medium` / `low` | 标注者对判断的置信度 |
| `human_rationale_cn` | 条件必填 | 自由文本 | `human_action != keep`、`uncertain` 或 `low` 时必须填写 |
| `annotator_id` | 是 | 预先分配的匿名 ID | 标注者 ID，例如 `A1`、`A2` |
| `annotation_round` | 是 | `1` / `2` / `adjudication` | 标注轮次 |

`human_action = change_polarity` 只允许与 `human_error_type = negation_error` 配对；与 `other` 或 `none` 配对无效。

`adjudicated_action = change_polarity` 同样只允许与 `adjudicated_error_type = negation_error` 配对。最终裁决还要求 `keep -> none`；`other` 只能由 adjudicator 配合非 `keep` action 显式使用，并必须填写非空 `adjudication_note_cn`。`keep + other` 和 `change_polarity + other` 均无效。

## Adjudication 列

双人标注不一致或低置信度时，由 adjudicator 冻结以下列：

| 列名 | 允许值 | 含义 |
| --- | --- | --- |
| `adjudicated_packaging_visual` | `yes` / `no` / `uncertain` | 最终包装视觉判断 |
| `adjudicated_relation_valid` | `yes` / `no` / `uncertain` | 最终关系有效性 |
| `adjudicated_package_level` | `outer` / `inner` / `ambiguous` / `non_packaging` / `uncertain` | 最终包装层级 |
| `adjudicated_dimension_code` | 见维度代码表，另可填 `none` / `uncertain` | 最终主维度 |
| `adjudicated_additional_dimension_codes` | 管道符分隔的 canonical 维度代码，或空 | 最终附加维度；保持原字符串，不排序、不去重、不 case-fold，逐 token 精确校验 |
| `adjudicated_polarity` | `positive` / `negative` / `uncertain` | 最终极性 |
| `adjudicated_action` | `keep` / `drop` / `change_dimension` / `change_package_level` / `change_polarity` / `add_missing` | 最终处理动作 |
| `adjudicated_error_type` | 见错误类型表 | 最终错误类型 |
| `adjudicated_product_label_traceable` | `yes` / `no` / `uncertain` | 最终 product-dimension 正例可追溯判断 |
| `adjudicated_unlabeled_missed_signal` | `yes` / `no` / `uncertain` | 最终 unlabeled 漏标信号判断 |
| `adjudication_note_cn` | 自由文本 | 冻结理由；mapping unresolved 的人工闭合以及使用 `other` 时必须非空 |

### Final action/error mapping consistency

sentence 的最终 semantic core 由 `adjudicated_packaging_visual`、`adjudicated_relation_valid`、`adjudicated_package_level`、`adjudicated_dimension_code`、`adjudicated_additional_dimension_codes` 和 `adjudicated_polarity` 组成。canonical validator 将这些列结构性适配到 frozen mapping 输入；附加维度原字符串交给 contract 已声明的 `split_union` transform，validator 不另建 dimension-union 逻辑。

frozen mapping 返回 `resolved` 时，`adjudicated_action` 和 `adjudicated_error_type` 必须逐字段严格等于 mapping 结果，不能自由填写。mapping 返回 `unresolved` 时不得自动 fallback；保留并报告原始 `reason_code`，由 adjudicator 显式提供 canonical action/error 和非空 note。复核后判断无需修改时可使用 `keep + none + nonblank note`；需要修改时选择最合适的 canonical action/error，只有 taxonomy 确实无法准确表达错误时才可使用非 `keep` action + `other` + nonblank note。人工闭合不会形成新的自动 mapping 规则。

product-dimension 的 action/error mapping 为 `not_applicable`，继续使用 product 专用裁决字段。

validator 报告 sentence 总数、mapping resolved/unresolved 数、resolved mismatch 行数及 unresolved reason distribution；失败异常保留同一完整报告。

## 维度代码表

| 代码 | 中文名 | 判定口径 |
| --- | --- | --- |
| `general_visual_appeal` | 一般视觉吸引力 | pretty、beautiful、attractive 等泛化视觉好看判断 |
| `cute_friendly` | 可爱亲和感 | cute、sweet、fun、whimsical、friendly 等亲和或可爱感 |
| `premium_refined` | 高级精致感 | elegant、classy、premium、refined、sophisticated 等高级或精致感 |
| `gift_presentation` | 礼赠呈现感 | giftable、nice presentation、suitable as a gift 等礼赠场景 |
| `simple_modern` | 简约现代感 | simple、minimal、clean、modern 等视觉设计风格；环保少包装不算 |
| `natural_botanical` | 自然植物感 | natural、botanical、earthy 等视觉氛围；成分天然或味道天然不算 |
| `calming_soft` | 舒缓柔和感 | calm、soothing、soft、gentle 等视觉氛围；饮用后的功效不算 |
| `cheerful_colorful` | 活力愉悦感 | colorful、bright、cheerful、happy 等颜色或视觉情绪 |
| `traditional_vintage` | 传统复古感 | vintage、classic、old-fashioned、heritage 等传统或复古视觉风格 |
| `negative_appearance` | 负面外观感 | ugly、cheap-looking、unattractive 等包装外观负面评价；运输损坏单独判错 |
| `none` | 无 | 不属于任何情感意象维度 |
| `uncertain` | 不确定 | 证据不足以稳定判断 |

## 错误类型表

| 代码 | 使用场景 |
| --- | --- |
| `none` | 模型判断可保留 |
| `nonvisual_content` | 表达描述茶汤、味道、香气、功效、配方或商品内容，不是包装视觉 |
| `shipping_or_seller` | 表达描述运输箱、破损、配送、卖家包装或履约体验 |
| `inner_packaging` | 证据描述内袋、茶包、filter bag、sachet 等内包装，不应进入主图标签 |
| `ambiguous_package_level` | 不能判断是外包装还是内包装 |
| `wrong_dimension` | 关系有效但维度错 |
| `negation_error` | 否定、转折或对比关系处理错误 |
| `relation_missing` | 模型行没有有效包装对象-表达关系 |
| `missed_relation` | 算法未捕获一条明确应纳入的关系 |
| `duplicate_or_near_duplicate` | 重复或近重复证据影响判断 |
| `context_missing` | 仅凭当前句子无法判断 |
| `other` | 其他错误，必须在 `human_rationale_cn` 中说明 |

## 判定规则

`human_packaging_visual = yes` 只用于零售商品包装的视觉外观，包括盒、罐、袋、标签、外包装图案、颜色、排版和整体呈现。运输箱、Amazon 包装、卖家额外包材、破损配送体验不算。

`human_relation_valid = yes` 要求包装对象和情感/意象表达在同一子句或清楚的局部上下文中相连。句子里同时出现包装词和 beautiful、cute 等词，但该词实际描述茶、味道、香气、礼物内容或品牌感受时，应判为 `no`。

`human_package_level = outer` 只用于可合理对应商品主图的零售外包装。`inner` 用于茶包、内袋、小 sachet、filter bag 等内部单元。无法稳定区分时用 `ambiguous`，不要为了提高样本量强行判 outer。

`human_action = add_missing` 只用于 `uncertain_not_recovered` 或 `strict_without_relation_evidence` 中人工认为应形成有效证据的行。对已有预测正例，若模型关系错误应使用 `drop`、`change_dimension` 或 `change_package_level`。

## 数据质量约束

- 所有枚举值使用小写 snake_case。
- 空值只允许出现在文档明确标为非必填的列。
- `human_rationale_cn` 不应粘贴大段评论原文，只写判断理由。
- 同一行的 `annotator_id` 和 `annotation_round` 组合必须唯一。
- Adjudication 不覆盖双人原始标注，而是新增最终列。
- 标注文件导出前应检查行数、`sample_order` 唯一性和必填列空值。
