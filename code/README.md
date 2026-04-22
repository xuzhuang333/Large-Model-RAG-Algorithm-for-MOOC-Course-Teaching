# Stateful-EduRAG 毕业设计项目说明（中文）

本项目是一个面向教学场景的 RAG（Retrieval-Augmented Generation）毕业设计工程，核心目标是验证**状态感知检索与生成**是否优于传统方案。  
当前实验主结论方向为：`Group C > Group B > Group A`。

---

## 1. 项目目标

1. 构建三个可对比的检索增强问答系统：
- `Group A`：Baseline 平铺检索方案。
- `Group B`：树结构/聚类摘要增强方案（RAPTOR 思路）。
- `Group C`：状态感知 + 双路检索 + 教学策略增强的最终方案。

2. 建立量化评测体系（Set-A ~ Set-D，后续可扩展 Set-E/F）：
- `Set-A`：单跳事实能力。
- `Set-B`：多跳与全局总结能力。
- `Set-C`：硬负样本辨析能力。
- `Set-D`：多轮状态轨迹能力（遗忘/挣扎画像）。

3. 在相同题集下比较不同组性能，分析 Group C 的增益来源与稳定性。

---

## 2. 目录结构

```text
code/
├─ Retrieval Ablation Experiment/      # A/B/C 三组模型实现
│  ├─ Group A/
│  ├─ Group B/
│  └─ Group C/
├─ QuantEvaluation/                    # 量化评测工程（Set-A ~ Set-D）
│  ├─ Set-A_SingleHop_Factoid/
│  ├─ Set-B_MultiHop_GlobalSummary/
│  ├─ Set-C_HardNegative_Discrimination/
│  ├─ Set-D_Stateful_Trajectories/
│  ├─ Shared/
│  └─ 方案总览.md
├─ QAdata/                             # 问答与实验相关数据
└─ agent requirement/                  # 过程文档/需求记录（如有）
```

---

## 3. 关键脚本说明

### 3.1 模型侧（Retrieval Ablation Experiment）

- Group A
  - `group_a_baseline_indexing.py`
  - `group_a_pipeline.py`
- Group B
  - `group_b_leaf_indexing.py`
  - `group_b_recursive_tree_builder.py`
  - `group_b_tree_relay_resume.py`
- Group C
  - `group_c_static_foundation_builder.py`
  - `group_c_backfill_syllabus_embeddings.py`
  - `group_c_dual_retriever.py`

### 3.2 评测侧（QuantEvaluation）

- Set-A
  - `build_set_a_dataset_with_llm.py`
  - `run_set_a_quantitative_evaluation.py`
- Set-B
  - `build_set_b_dataset_with_llm.py`
  - `run_set_b_quantitative_evaluation.py`
- Set-C
  - `build_set_c_dataset_with_group_c_hardneg.py`
  - `run_set_c_quantitative_evaluation.py`
- Set-D
  - `build_set_d_trajectories.py`
  - `run_set_d_quantitative_evaluation.py`
  - `rewrite_set_d_summary_from_samples.py`（离线重写 summary，无需重复调用 LLM）

---

## 4. 环境与依赖

推荐使用 Conda 环境（项目当前常用环境名）：

```powershell
conda activate rag_py310
```

如需使用 HuggingFace 镜像与缓存，可设置：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "E:\hf_cache"
$env:HF_HUB_CACHE = "E:\hf_cache\hub"
$env:SENTENCE_TRANSFORMERS_HOME = "E:\hf_cache\sentence_transformers"
```

---

## 5. 快速开始（推荐流程）

1. 先确认数据库与索引（A/B/C 各组）构建完成。  
2. 在 `QuantEvaluation` 中按 Set 生成数据集并运行评测。  
3. 对 Set-D 可优先使用离线重写脚本调整指标口径，避免重复花费 LLM 成本。  

示例（Set-D 离线重写）：

```powershell
conda run -n rag_py310 python "E:\graduate_project\code\QuantEvaluation\Set-D_Stateful_Trajectories\rewrite_set_d_summary_from_samples.py"
```

---

## 6. 输出文件约定

- 每个 Set 通常输出：
  - `*_quant_eval_samples.json`：逐题细粒度记录
  - `*_quant_eval_summary.json`：聚合统计结果
- Set-D 重点关注状态相关指标：
  - `Forgetting Recovery`
  - `Struggle Responsiveness`
  - `State Uplift`
  - `Trajectory Empowerment`
  - 以及扩展状态指标（如 state activation / hardmatch 等）

---

## 7. 实验注意事项

1. 评测前固定随机种子与采样参数，确保复现实验。  
2. 评测配置建议显式写环境变量，避免继承历史 session 配置。  
3. 对比实验必须保证同题比较（配对比较），避免样本分布偏差。  
4. Set-D 建议同时保留“原始口径”和“状态增强口径”以便论文论证。

---

## 8. 当前状态（简述）

- Set-A / Set-B：实现较稳定。  
- Set-C / Set-D：已完成流程实现，实现较稳定。
---

## 9. 维护建议

1. 优先维护 `QuantEvaluation/方案总览.md` 作为指标与实验矩阵总规范。  
2. 新增指标时优先做“离线重算脚本”，避免反复调用在线模型。  
3. 保持 A/B/C 输入输出协议一致，便于公平对比与自动聚合。

