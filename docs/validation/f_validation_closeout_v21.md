# F：A1/A2 验证工具链技术返工与安全收口（v2.1）

## 1. 目的与边界

本文记录 F validation closeout 的四项技术修复及正式工作区的安全操作边界。当前状态判断以仓库根目录 `CHATGPT_START_HERE.md` 的最新审计快照为准；旧文档中“A1/A2 尚未开始/尚未填写”等段落属于历史状态。

本任务只处理 F 验证工具链。不得因此重跑 metadata、评论匹配、评论清洗、V2.1 正式流程、E 或 G，也不得开始图片阶段、选择最终曝光阈值，或把 Positive-Unlabeled 标签中的 `0` 解释为 confirmed negative。

正式 F 工作区为：

```text
data/manual_validation/affective_imagery_v21_5180/
```

A1/A2 已经开始并完成首轮双重独立模型辅助审核后，**正式目录绝对禁止运行 `--overwrite`**。准备脚本的普通 `--overwrite` 路径仍仅用于新的临时/复现目录，因为它会重新生成 01–08，包括 03/04。

## 2. 实现结构

公开入口仍为：

```text
scripts/validation/prepare_affective_imagery_validation_v21.py
scripts/validation/validate_affective_imagery_annotations_v21.py
```

原准备实现完整保留在：

```text
scripts/validation/_prepare_affective_imagery_validation_v21_core.py
```

该 core 是返工前 `prepare_affective_imagery_validation_v21.py` 的原始 blob，用于保持既有抽样、01–08 构建、稳定 ID 和 reviewer packet 行选择逻辑不变。公开 prepare 入口只覆盖本次需要修正的 reviewer 统计和安全 CLI；不会重新设计抽样算法。

manifest-only 的独立安全实现位于：

```text
scripts/validation/affective_imagery_validation_manifest_v21.py
```

这样可以把“重建工作区”和“只刷新 06”隔离成不同代码路径，避免 manifest 刷新误入 `--overwrite`/01–08 重建路径。

## 3. 修复一：positive/no-outer validator

### 旧问题

旧 validator 要求所有 positive product-dimension item 都有 outer target evidence，只对两个硬编码 focus `annotation_item_id` 例外。这把数据事实错误地绑定到了固定 ID；正式数据存在更多合法的 positive/no-outer 项，因此真实端到端校验会结构性失败。

### 新规则

对 `model_label_value=1` 的 product-dimension item：

1. 必须至少存在一条 `is_target_dimension_evidence=1` 的 target-dimension evidence。
2. 若其中存在 `is_outer_eligible_evidence=1`，正常通过结构校验。
3. 若 target-dimension evidence 存在，但全部为 non-outer：
   - 这是合法的 `positive/no-outer` 审核情形；
   - validator 不做结构性失败；
   - `annotation_item_id` 被收集到 `positive_no_outer_annotation_item_ids` 并在 CLI summary 中报告，供后续 adjudication 使用。
4. 若 target-dimension evidence 完全为 0，仍立即失败。

新逻辑不包含正式 10 个 item 的 allowlist，也不依赖两个 focus ID。

同时继续检查：unlabeled item 不允许出现 target-dimension outer eligible evidence；01/02/07/08 的 item ID、`parent_asin`、target dimension、label value 和 reviewer context rank 必须一致。

## 4. 修复二：reviewer packet backfill tier 归属

### 旧问题

08 的行选择本身正确，但旧统计把所有 backfill 统一累加到 `other_candidate.backfill_selected_count`。因此 `initial_quota_selected_count`、`backfill_selected_count` 与 `final_selected_count` 的 tier 归属不真实。

### 新规则

本次**不改变 08 选择的任何行，也不改变 08 行顺序**。

返工后的统计从实际已选 08 行的 `review_priority_tier` 得到每个 tier 的真实 `final_selected_count`，然后按：

```text
backfill_selected_count = final_selected_count - initial_quota_selected_count
```

回填到该行自己的真实 reviewer tier。所有 tier 的 backfill 之和必须等于 `total_backfill_rows`，所有 tier 的 final 之和必须等于 unlabeled reviewer rows，否则立即失败。

manifest 同时保留原始 reviewer tier 统计，并提供便于验收的五类聚合 `tier_category_stats`：

```text
formal    = mandatory_focus_or_direct_target + formal_other_outer
strict    = upstream_visual_strict
uncertain = upstream_uncertain
excluded  = upstream_excluded
other     = other_candidate
```

正式审计快照中的 `formal 788 / strict 328 / uncertain 300 / excluded 300 / other 84` 只用于用户在真实本地正式数据上验收，**未写入生产逻辑或测试 fixture 的正式期望值**。

## 5. 修复三：真正只读的 `--validate-only`

入口：

```text
python scripts/validation/validate_affective_imagery_annotations_v21.py --items-dir data/manual_validation/affective_imagery_v21_5180 --validate-only
```

`--validate-only` 的语义是：

- 只读取现有 01–08；
- 校验 07 的 positive target evidence、positive/no-outer、unlabeled 冲突和键一致性；
- 校验 08 覆盖、rank、30 行 cap、01/02 键一致性以及 07/08 full-context count；
- 校验 A1 和 A2 的现有 annotation schema/条件必填规则；
- 读取并检查 06 是有效 JSON；
- 不构建、不写入 05；
- 不修改 06；
- 不修改 03/04；
- 不创建任何目录或文件。

运行前会记录 01–08 的 SHA-256；运行结束（包括 annotation validation 失败）再次读取 SHA-256。只要任何文件变化，就以安全错误退出。成功的零写入检查会打印：

```text
Zero-write SHA-256 check: PASS
```

若 A2 的 173 条条件必填理由尚未修复，A2 annotation validation 仍应失败并返回非零退出码。该模式**不会放宽 `human_rationale_cn` 的 schema 条件**来掩盖 A2 问题。

## 6. 修复四：`--refresh-manifest-only`

准备脚本新增：

```text
--refresh-manifest-only
--dry-run
```

### 6.1 dry-run

```text
python scripts/validation/prepare_affective_imagery_validation_v21.py --output-dir data/manual_validation/affective_imagery_v21_5180 --refresh-manifest-only --dry-run
```

该命令：

- 读取现有 01–08；
- 不写任何文件；
- 在内存中使用现有 07 重新执行确定性 reviewer selection；
- 要求内存重建的 08 与磁盘现有 08 的列、行、值和顺序完全一致；
- 计算 prospective 06、positive/no-outer、tier quota/backfill/final 和 total backfill；
- 打印旧 06 SHA-256 和 prospective 06 SHA-256；
- 对 01–05、07、08 做 SHA-256 不变检查。

### 6.2 正式 manifest-only 写入

```text
python scripts/validation/prepare_affective_imagery_validation_v21.py --output-dir data/manual_validation/affective_imagery_v21_5180 --refresh-manifest-only
```

该命令只允许修改：

```text
06_validation_manifest.json
```

受保护文件为：

```text
01_sentence_items.csv
02_product_dimension_items.csv
03_annotations_A1.csv
04_annotations_A2.csv
05_adjudication_template.csv
07_product_dimension_evidence_context.csv
08_product_dimension_reviewer_context.csv
```

写入前计算这些文件 SHA-256；06 使用同目录临时文件 + `os.replace` 原子替换；写入后再次计算受保护文件 SHA-256。若任一受保护文件变化，工具会把 06 回滚到刷新前字节并失败。

该路径不接受 `--overwrite`，也不会调用普通 preparation/overwrite 路径。

## 7. manifest 刷新内容

manifest-only 会保留现有 manifest 中未被本次重新推导的 provenance，例如 input/source manifest 信息；同时从现有 01/02/03/04/07/08 刷新：

- sentence / product-dimension / annotation item count；
- 07 context row/item count；
- positive item count；
- unlabeled item count；
- positive with outer；
- positive without outer；
- `positive_no_outer_annotation_item_ids`；
- unlabeled candidate-context presence；
- reviewer row/item count；
- positive / unlabeled reviewer distributions；
- raw reviewer tier 的 candidate / initial quota / backfill / final counts；
- `tier_category_stats` 五类聚合；
- total backfill；
- 受保护文件 SHA-256；
- `annotations_started`（根据 03/04 已有 `human_*` 内容判断）。

`0` 标签的解释仍为 `unlabeled_not_observed`，不是 confirmed negative。

## 8. 推荐验证顺序

1. 获取 `chatgpt/f-validation-closeout` 分支。
2. 运行 F targeted tests。
3. 运行全量 `python -m pytest -q`。
4. 查看 prepare/validator `--help`。
5. 对正式 F workspace 运行 validator `--validate-only`。如果 A2 条件理由尚未修复，记录并返回 A2 已知失败；不要通过修改 schema 绕过。
6. 运行 manifest-only `--dry-run`，检查 positive/no-outer、tier final counts、total backfill 和 protected SHA summary。
7. dry-run 与审计事实一致后，运行真正的 `--refresh-manifest-only`，只写 06。
8. 再次运行 manifest-only dry-run/只读统计并检查 01–05、07、08 SHA-256 未变。
9. A2 定向理由修复完成后，再执行完整 annotation `--validate-only`，要求 A1/A2 都通过。

## 9. 必须等待 A2 修复完成的操作

在 A2 173 条条件必填 rationale 尚未修复前：

- 可以执行 F targeted/full tests；
- 可以执行 `--validate-only`，但应接受并记录 A2 rationale 的非零结果；
- 可以执行 manifest-only dry-run 和 06-only refresh，因为该操作不修改 03/04 的判断内容；
- **不能把 A1/A2 完整 annotation validation 宣布为通过**；
- **不能据此生成/冻结新的 05 最终裁决结果**；
- **不能进入最终分歧裁决完成后的标签冻结步骤**；
- **不能开始图片阶段或最终建模**。

A2 修复完成后，再运行完整 `--validate-only`；只有 A1/A2 schema/条件校验都通过，才继续后续 05 裁决与最终验证报告冻结。
