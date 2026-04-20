#这个文件定义了Group B的Recursive Tree Builder类，负责从Neo4j数据库中读取当前层的节点，基于它们的文本内容和嵌入进行递归式的聚类和摘要生成，最终构建出一个多层次的树状结构。这个类实现了RAPTOR算法中的核心思想，通过不断地对节点进行聚类和生成摘要来逐层构建树，同时在聚类过程中引入了内部递归机制，以更细粒度地控制每个聚类的大小和质量。整个流程中还包含了多种停止条件，以确保树的生长过程既高效又符合预设的限制。
from __future__ import annotations

import hashlib
import importlib
import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture

from chunking_utils import TokenCounter
from shared_retrieval_utils import (
    DEFAULT_ARK_API_BASE,
    DEFAULT_DEEPSEEK_ENDPOINT,
    ark_chat_completion,
    get_adaptive_generation_params,
)

logger = logging.getLogger("GroupB_Recursive_Tree")


@dataclass
class GroupBTreeGrowthConfig:
    reduction_dimension: int = 10
    umap_metric: str = "cosine"
    gmm_prob_threshold: float = 0.1
    gmm_max_clusters: int = 50
    cluster_max_tokens: int = 3500
    tree_max_layers: int | None = None
    random_seed: int = 224
    summarizer_model: str = DEFAULT_DEEPSEEK_ENDPOINT
    summarizer_api_base: str = DEFAULT_ARK_API_BASE
    summarizer_api_key: str | None = None
    summary_max_tokens: int = 220
    relationship_type: str = "GROUP_B_PARENT_OF"
    internal_split_max_depth: int = 12


class GroupBRecursiveTreeBuilder:
    def __init__(
        self,
        repository,
        embedding_model: SentenceTransformer,
        token_counter: TokenCounter,
        config: GroupBTreeGrowthConfig,
    ) -> None:
        self.repository = repository
        self.embedding_model = embedding_model
        self.token_counter = token_counter
        self.config = config

    def grow_tree(self, start_layer: int = 0) -> dict[str, int]:
        stats = {
            "layers_grown": 0,
            "summary_nodes_created": 0,
            "stopped_by_condition": 0,
            "stopped_by_max_layers": 0,
        }

        current_layer = start_layer
        growth_steps = 0
        max_growth_steps = self.config.tree_max_layers

        while True:
            if max_growth_steps is not None and growth_steps >= max_growth_steps:
                logger.info(
                    "Stop growth at layer=%d, reached max_growth_steps=%d",
                    current_layer,
                    max_growth_steps,
                )
                stats["stopped_by_max_layers"] = 1
                break

            layer_nodes = self.repository.fetch_nodes_by_layer(current_layer)
            node_count = len(layer_nodes)

            logger.info("Growing layer=%d, node_count=%d", current_layer, node_count)

            # Strict RAPTOR stop criterion: N <= d + 1 means clustering is infeasible.
            if node_count <= self.config.reduction_dimension + 1:
                logger.info(
                    "Stop growth at layer=%d, infeasible clustering: N=%d <= d+1=%d",
                    current_layer,
                    node_count,
                    self.config.reduction_dimension + 1,
                )
                stats["stopped_by_condition"] = 1
                break

            logger.info("Stage[clustering] start: layer=%d", current_layer)
            clusters = self._cluster_with_internal_recursion(layer_nodes)
            logger.info(
                "Stage[clustering] done: layer=%d, clusters=%d",
                current_layer,
                len(clusters),
            )
            if not clusters:
                logger.info("Stop growth at layer=%d, no valid clusters produced", current_layer)
                stats["stopped_by_condition"] = 1
                break

            summary_rows = []
            next_layer = current_layer + 1
            total_clusters = len(clusters)
            logger.info(
                "Stage[summarization] start: layer=%d, total_clusters=%d",
                current_layer,
                total_clusters,
            )
            for cluster_order, cluster_nodes in enumerate(clusters, start=1):
                summary_text = self._summarize_cluster(cluster_nodes)
                if not summary_text.strip():
                    continue

                emb = self.embedding_model.encode(
                    [summary_text],
                    normalize_embeddings=True,
                    batch_size=1,
                )[0]
                summary_rows.append(
                    {
                        "node_id": self._build_summary_node_id(
                            next_layer,
                            cluster_order,
                            summary_text,
                            [node["node_id"] for node in cluster_nodes],
                        ),
                        "text": summary_text,
                        "layer": next_layer,
                        "cluster_order": cluster_order,
                        "token_count": self.token_counter.count_tokens(summary_text),
                        "child_count": len(cluster_nodes),
                        "child_node_ids": [node["node_id"] for node in cluster_nodes],
                        "embedding": [float(x) for x in emb],
                    }
                )

                if cluster_order % 5 == 0 or cluster_order == total_clusters:
                    logger.info(
                        "Stage[summarization] progress: layer=%d, completed=%d/%d",
                        current_layer,
                        cluster_order,
                        total_clusters,
                    )

            if not summary_rows:
                logger.info("Stop growth at layer=%d, no summary nodes generated", current_layer)
                stats["stopped_by_condition"] = 1
                break

            inserted = self.repository.insert_summary_nodes(
                summary_rows,
                relationship_type=self.config.relationship_type,
            )
            logger.info(
                "Layer growth done: source_layer=%d, next_layer=%d, clusters=%d, inserted=%d",
                current_layer,
                next_layer,
                len(summary_rows),
                inserted,
            )

            stats["layers_grown"] += 1
            stats["summary_nodes_created"] += inserted
            current_layer = next_layer
            growth_steps += 1

        return stats

    def _build_summary_node_id(
        self,
        layer: int,
        cluster_order: int,
        summary_text: str,
        child_node_ids: list[str],
    ) -> str:
        # Keep node_id stable across reruns by binding identity to structure (layer + children),
        # not to LLM text variation.
        seed = f"{layer}::{','.join(sorted(child_node_ids))}".encode(
            "utf-8",
            errors="ignore",
        )
        digest = hashlib.sha1(seed).hexdigest()[:20]
        return f"B{layer}_{digest}"

    def _summarize_cluster(self, cluster_nodes: list[dict]) -> str:
        context = "\n\n".join(
            f"[Node:{node['node_id']}]\n{node['text']}" for node in cluster_nodes
        )
        prompt = f"""
你是课程资料整理助手。请阅读以下同一语义簇中的多个文本节点，生成一个简洁且信息完整的中文摘要。
要求：
1. 只基于给定文本，不要引入外部知识。
2. 保留关键概念、定义、步骤和约束。
3. 摘要应具备检索友好性，优先保留术语。
4. 输出为单段文本，不要使用标题或序号。

[Cluster Context]
{context}

请输出摘要：
"""
        try:
            temperature, adaptive_max_tokens = get_adaptive_generation_params(
                prompt,
                task="generate",
            )
            max_tokens = min(self.config.summary_max_tokens, adaptive_max_tokens)
            summary = ark_chat_completion(
                model=self.config.summarizer_model,
                prompt=prompt,
                api_base=self.config.summarizer_api_base,
                api_key=self.config.summarizer_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=180,
            )
            return summary.strip()
        except Exception as exc:
            # Fallback keeps the pipeline robust even when the summarizer endpoint is unavailable.
            logger.warning("Cluster summarization failed, using fallback text: %s", exc)
            truncated = context[:600].strip()
            return truncated if truncated else "根据现有资料无法生成摘要"

    def _cluster_with_internal_recursion(self, nodes: list[dict]) -> list[list[dict]]:
        labels = self._perform_clustering(np.array([node["embedding"] for node in nodes]))
        clusters = self._labels_to_clusters(nodes, labels)

        final_clusters: list[list[dict]] = []
        for cluster_nodes in clusters:
            final_clusters.extend(self._split_cluster_recursively(cluster_nodes, depth=0))
        return [cluster for cluster in final_clusters if cluster]

    def _split_cluster_recursively(self, cluster_nodes: list[dict], depth: int) -> list[list[dict]]:
        if len(cluster_nodes) <= 1:
            return [cluster_nodes]

        if depth >= self.config.internal_split_max_depth:
            logger.warning(
                "Internal split reached max depth=%d, fallback to greedy split (size=%d)",
                self.config.internal_split_max_depth,
                len(cluster_nodes),
            )
            return self._greedy_split_by_budget(cluster_nodes)

        total_tokens = sum(int(node.get("token_count", 0)) for node in cluster_nodes)
        if total_tokens <= self.config.cluster_max_tokens:
            return [cluster_nodes]

        if len(cluster_nodes) <= self.config.reduction_dimension + 1:
            return self._greedy_split_by_budget(cluster_nodes)

        labels = self._perform_clustering(
            np.array([node["embedding"] for node in cluster_nodes])
        )
        sub_clusters = self._labels_to_clusters(cluster_nodes, labels)

        # Prevent infinite recursion when clustering collapses into a single unchanged cluster.
        if len(sub_clusters) <= 1:
            return self._greedy_split_by_budget(cluster_nodes)

        parent_set = set(node["node_id"] for node in cluster_nodes)
        unique_sub_clusters: list[list[dict]] = []
        seen_signatures: set[tuple[str, ...]] = set()
        has_identical_child = False
        for sub in sub_clusters:
            signature = tuple(sorted(node["node_id"] for node in sub))
            if not signature:
                continue
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_sub_clusters.append(sub)
            if set(signature) == parent_set:
                has_identical_child = True

        if len(unique_sub_clusters) <= 1 or has_identical_child:
            logger.warning(
                "Internal split made no structural progress at depth=%d (size=%d), fallback to greedy split",
                depth,
                len(cluster_nodes),
            )
            return self._greedy_split_by_budget(cluster_nodes)

        flattened = [node["node_id"] for cluster in sub_clusters for node in cluster]
        if len(set(flattened)) <= 1:
            return self._greedy_split_by_budget(cluster_nodes)

        result: list[list[dict]] = []
        for sub_cluster in unique_sub_clusters:
            result.extend(self._split_cluster_recursively(sub_cluster, depth=depth + 1))
        return result

    def _greedy_split_by_budget(self, nodes: list[dict]) -> list[list[dict]]:
        buckets: list[list[dict]] = []
        current_bucket: list[dict] = []
        current_tokens = 0

        for node in nodes:
            node_tokens = int(node.get("token_count", 0))
            if current_bucket and current_tokens + node_tokens > self.config.cluster_max_tokens:
                buckets.append(current_bucket)
                current_bucket = [node]
                current_tokens = node_tokens
                continue

            current_bucket.append(node)
            current_tokens += node_tokens

        if current_bucket:
            buckets.append(current_bucket)
        return buckets

    def _labels_to_clusters(self, nodes: list[dict], labels: list[np.ndarray]) -> list[list[dict]]:
        if not labels:
            return []

        all_label_values: list[int] = []
        for label_array in labels:
            all_label_values.extend(label_array.tolist())

        if not all_label_values:
            return [[node] for node in nodes]

        unique_labels = sorted(set(all_label_values))
        clusters: list[list[dict]] = []
        for label in unique_labels:
            members = [nodes[idx] for idx, label_arr in enumerate(labels) if label in label_arr]
            if members:
                clusters.append(members)
        return clusters

    def _perform_clustering(self, embeddings: np.ndarray) -> list[np.ndarray]:
        n_samples = len(embeddings)
        logger.info("Stage[perform_clustering] start: n_samples=%d", n_samples)
        if n_samples == 0:
            return []
        if n_samples == 1:
            return [np.array([0])]
        if n_samples <= 2:
            return [np.array([0]) for _ in range(n_samples)]

        dim = min(self.config.reduction_dimension, n_samples - 2)
        dim = max(dim, 1)

        reduced_global = self._reduce_global_embeddings(embeddings, dim)
        global_labels, n_global_clusters = self._gmm_soft_cluster(reduced_global)

        all_local_clusters = [np.array([], dtype=int) for _ in range(n_samples)]
        total_clusters = 0

        for global_cluster_idx in range(n_global_clusters):
            global_indices = [
                idx for idx, label_arr in enumerate(global_labels) if global_cluster_idx in label_arr
            ]
            if not global_indices:
                continue

            global_cluster_embeddings = embeddings[global_indices]
            if len(global_cluster_embeddings) <= dim + 1:
                local_labels = [np.array([0]) for _ in global_cluster_embeddings]
                n_local_clusters = 1
            else:
                reduced_local = self._reduce_local_embeddings(global_cluster_embeddings, dim)
                local_labels, n_local_clusters = self._gmm_soft_cluster(reduced_local)

            for local_cluster_idx in range(n_local_clusters):
                member_indices = [
                    global_indices[j]
                    for j, label_arr in enumerate(local_labels)
                    if local_cluster_idx in label_arr
                ]
                for member_idx in member_indices:
                    all_local_clusters[member_idx] = np.append(
                        all_local_clusters[member_idx],
                        local_cluster_idx + total_clusters,
                    )

            total_clusters += n_local_clusters

        # Guarantee each node belongs to at least one cluster.
        for idx in range(n_samples):
            if all_local_clusters[idx].size == 0:
                all_local_clusters[idx] = np.array([idx], dtype=int)

        logger.info(
            "Stage[perform_clustering] done: n_samples=%d, total_clusters=%d",
            n_samples,
            total_clusters,
        )
        return all_local_clusters

    def _reduce_global_embeddings(self, embeddings: np.ndarray, dim: int) -> np.ndarray:
        if len(embeddings) <= 2:
            return embeddings

        umap_module = self._get_umap_module()
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
        n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))

        logger.info(
            "Stage[UMAP-global] start: n_samples=%d, dim=%d, n_neighbors=%d, metric=%s",
            len(embeddings),
            dim,
            n_neighbors,
            self.config.umap_metric,
        )

        reducer = umap_module.UMAP(
            n_neighbors=n_neighbors,
            n_components=dim,
            metric=self.config.umap_metric,
            random_state=self.config.random_seed,
        )
        reduced = reducer.fit_transform(embeddings)
        logger.info(
            "Stage[UMAP-global] done: input_shape=%s, output_shape=%s",
            tuple(embeddings.shape),
            tuple(reduced.shape),
        )
        return reduced

    def _reduce_local_embeddings(self, embeddings: np.ndarray, dim: int) -> np.ndarray:
        if len(embeddings) <= 2:
            return embeddings

        umap_module = self._get_umap_module()
        n_neighbors = max(2, min(10, len(embeddings) - 1))
        logger.info(
            "Stage[UMAP-local] start: n_samples=%d, dim=%d, n_neighbors=%d, metric=%s",
            len(embeddings),
            dim,
            n_neighbors,
            self.config.umap_metric,
        )
        reducer = umap_module.UMAP(
            n_neighbors=n_neighbors,
            n_components=dim,
            metric=self.config.umap_metric,
            random_state=self.config.random_seed,
        )
        reduced = reducer.fit_transform(embeddings)
        logger.info(
            "Stage[UMAP-local] done: input_shape=%s, output_shape=%s",
            tuple(embeddings.shape),
            tuple(reduced.shape),
        )
        return reduced

    @staticmethod
    def _get_umap_module():
        try:
            return importlib.import_module("umap")
        except ImportError as exc:
            raise ImportError(
                "UMAP dependency missing. Please install package 'umap-learn'."
            ) from exc

    def _get_optimal_clusters(self, embeddings: np.ndarray) -> int:
        max_clusters = min(self.config.gmm_max_clusters, len(embeddings))
        if max_clusters <= 1:
            return 1

        candidates = np.arange(1, max_clusters)
        if candidates.size == 0:
            return 1

        logger.info(
            "Stage[BIC-search] start: n_samples=%d, candidates=%d, k_range=[%d,%d)",
            len(embeddings),
            len(candidates),
            1,
            max_clusters,
        )

        bics: list[float] = []
        valid_candidates: list[int] = []
        for idx, n_clusters in enumerate(candidates, start=1):
            try:
                gm = GaussianMixture(
                    n_components=n_clusters,
                    random_state=self.config.random_seed,
                )
                gm.fit(embeddings)
                bic_value = float(gm.bic(embeddings))
                bics.append(bic_value)
                valid_candidates.append(int(n_clusters))

                if idx % 5 == 0 or idx == len(candidates):
                    current_best = valid_candidates[int(np.argmin(np.array(bics)))]
                    logger.info(
                        "Stage[BIC-search] progress: %d/%d, last_k=%d, last_bic=%.4f, current_best_k=%d",
                        idx,
                        len(candidates),
                        int(n_clusters),
                        bic_value,
                        current_best,
                    )
            except Exception:
                continue

        if not bics:
            return 1

        best_index = int(np.argmin(np.array(bics)))
        best_k = valid_candidates[best_index]
        logger.info("Stage[BIC-search] done: best_k=%d", best_k)
        return best_k

    def _gmm_soft_cluster(self, embeddings: np.ndarray) -> tuple[list[np.ndarray], int]:
        n_clusters = self._get_optimal_clusters(embeddings)
        if n_clusters <= 1:
            return [np.array([0]) for _ in range(len(embeddings))], 1

        gm = GaussianMixture(
            n_components=n_clusters,
            random_state=self.config.random_seed,
        )
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)

        labels: list[np.ndarray] = []
        for row in probs:
            selected = np.where(row > self.config.gmm_prob_threshold)[0]
            if selected.size == 0:
                selected = np.array([int(np.argmax(row))], dtype=int)
            labels.append(selected)

        return labels, n_clusters
