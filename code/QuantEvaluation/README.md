# QuantEvaluation

本目录用于量化评测，采用按 Set 分组的结构（Set-A 到 Set-F）。

## 目录结构

- Shared/
  - unified_input_schema_template.json：统一输入模板
- Set-A_SingleHop_Factoid/
  - input_template_set_a.json
- Set-B_MultiHop_GlobalSummary/
  - input_template_set_b.json
- Set-C_HardNegative_Discrimination/
  - input_template_set_c.json
- Set-D_Stateful_Trajectories/
  - input_template_set_d.json
- Set-E_Noise_Stress/
  - input_template_set_e.json
- Set-F_Efficiency_Benchmark/
  - input_template_set_f.json

## 统一字段规范表（中文）

说明：

- 必填级别：Y=所有 Set 必填，C=条件必填（仅某些 Set 必填），O=选填。
- 类型采用 JSON 语义：string、number、integer、array、object、null。
- 指标名称为建议映射，可按你的论文指标名替换。

| 字段 | 必填级别 | 类型 | 示例 | 主要用于指标 | Set 要求说明 |
|---|---|---|---|---|---|
| id | Y | string | SA001 | 样本追踪、错误分析 | 所有 Set 必填，且全局唯一 |
| eval_set | Y | string | Set-A | 分组统计 | 所有 Set 必填，值为 Set-A~Set-F |
| type | Y | string | micro/macro/mixed | 分层统计（题型） | 所有 Set 必填 |
| question | Y | string | 二叉树的中序遍历是什么？ | 全部指标的输入 | 所有 Set 必填 |
| ground_truth | Y | string | 标准答案文本 | 答案正确性、事实一致性 | 所有 Set 建议作为金标必填 |
| golden_sources | C | array | ["doc_12#chunk_3"] | 证据命中率、引用准确率 | Set-A/B/C/D/E 强烈建议；Set-F 可选 |
| gold_answer_points | C | array | ["定义", "核心性质", "例子"] | 覆盖率、完整性 | Set-B/C/D/E 推荐必填；Set-A/F 可选 |
| required_concepts | C | array | ["DFS", "回溯"] | 概念召回率、概念覆盖率 | Set-B/C 推荐；其余可选 |
| supporting_facts | C | array | ["事实1", "事实2"] | 多跳证据链完整度 | Set-B 推荐必填；Set-A/C/D/E/F 可选 |
| hard_negative_pairs | C | array<object> | [{"concept_a":"栈","concept_b":"队列","expected_key_difference":"先进后出 vs 先进先出"}] | 易混概念区分准确率 | Set-C 必填；其他 Set 通常不需要 |
| trajectory_id | C | string | TRJ001 | 轨迹级指标（多轮一致性） | Set-D 必填；其他 Set 通常不需要 |
| turn_id | C | integer | 1 | 轮次级指标 | Set-D 必填；其他 Set 通常不需要 |
| user_id | C | string | U001 | 用户级状态分析 | Set-D 推荐必填；其他 Set 可空 |
| current_turn | C | integer/null | 3 | 状态阶段分析 | Set-D 推荐；其他 Set 可空 |
| qa_score | O | number/null | 0.62 | 状态提升、前后对比 | Set-D 推荐；其他 Set 可空 |
| current_struggle | O | string/null | 指针边界错误 | 状态感知命中率 | Set-D 推荐；其他 Set 可空 |
| persona | O | string | forgetting | 用户画像鲁棒性 | Set-D 推荐；其他 Set 可空 |
| expected_route | O | string | 先讲先修，再讲目标概念 | 推理路径一致性 | Set-D/C 可选，其余通常不需要 |
| noise_profile | C | string | none/light/hard | 噪声鲁棒性跌落率 | Set-E 必填；其他 Set 默认 none |
| max_hop | C | integer/null | 1/2/3 | 跳数分层性能 | Set-A/B 推荐必填；Set-F 可空 |
| tags | O | array | ["latency", "hard"] | 分桶统计、效率分组 | Set-F 推荐必填；其他 Set 可选 |
| golden_parent_syllabus_ids | O | array | ["1.2.3"] | 大纲级命中率、教学路径对齐 | Set-D/C 推荐；其他 Set 可选 |
| candidate_reference_materials | O | array | ["doc_8#chunk_1"] | 候选证据分析、重排分析 | 全部 Set 可选 |
| split | O | string | train/dev/test | 数据划分稳定性 | 全部 Set 推荐 |
| difficulty | O | string | easy/medium/hard | 难度分层统计 | 全部 Set 推荐 |
| notes | O | string | 人工备注 | 误差分析 | 全部 Set 可选 |

## Set 需求矩阵（你最关心的“哪些需要、哪些不需要”）

通用最小必填（建议所有 Set）：

- id、eval_set、type、question、ground_truth

| Set | 该 Set 额外必填 | 强烈建议 | 通常不需要（可空） |
|---|---|---|---|
| Set-A 单跳事实 | max_hop(=1) | golden_sources、split、difficulty | hard_negative_pairs、trajectory_id、turn_id、noise_profile(light/hard) |
| Set-B 多跳总结 | max_hop(>=2) | supporting_facts、required_concepts、gold_answer_points、golden_sources | trajectory_id、turn_id、hard_negative_pairs |
| Set-C 易混辨析 | hard_negative_pairs | required_concepts、gold_answer_points、golden_sources | trajectory_id、turn_id、noise_profile(light/hard) |
| Set-D 状态轨迹 | trajectory_id、turn_id | user_id、current_turn、persona、current_struggle、qa_score、expected_route、golden_parent_syllabus_ids | hard_negative_pairs（除非你做“状态+辨析”联合题） |
| Set-E 噪声压力 | noise_profile(light/hard) | clean/noisy 成对样本标记（可放 notes/tags）、golden_sources | trajectory_id、turn_id、hard_negative_pairs |
| Set-F 效率基准 | 无额外硬必填（除通用） | tags（标注负载桶）、split（固定评测批次） | supporting_facts、hard_negative_pairs、trajectory_id、turn_id |

## 字段填写建议（避免后续返工）

1. 先填通用最小必填，再填 Set 专属必填。
2. Set-D 若做状态实验，user_id/trajectory_id/turn_id 尽量完整，不要只填一部分。
3. Set-E 建议同题做 clean 与 noisy 两个样本，便于直接计算鲁棒性跌落。
4. Set-F 优先保证问题分布稳定（tags + split），不要把内容质量指标和效率指标混在同一结论里。

## 与现有代码兼容性

- Group A 现有读取器仅依赖 id/type/question，多余字段会被忽略。
- Group B 现有读取器仅依赖 id/type/question，多余字段会被忽略。
- Group C 读取器会使用状态相关字段，其余未知字段通常可忽略。

##镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "E:\hf_cache"
$env:HF_HUB_CACHE = "E:\hf_cache\hub"
$env:SENTENCE_TRANSFORMERS_HOME = "E:\hf_cache\sentence_transformers"

$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "E:\hf_cache"
$env:HF_HUB_CACHE = "E:\hf_cache\hub"
$env:SENTENCE_TRANSFORMERS_HOME = "E:\hf_cache\sentence_transformers"

$env:SET_A_ONLY_QID = ""
$env:SET_A_LIMIT = "0"
$env:SET_A_RANDOM_SAMPLE_N = "120"
$env:SET_A_RANDOM_SEED = "42"
