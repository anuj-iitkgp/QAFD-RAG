"""
LightRAG Baseline (Dual-Level Graph Retrieval).
Faithfully models LightRAG's dual-level retrieval paradigm:
1. Low-level retrieval: Specific entity-level keyword and local 1-hop neighborhood matching.
2. High-level retrieval: Broad relational/theme matching across broader graph connectivity.
Combines both levels to rank relevant candidate passages.
"""

import time
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Any
from data.loaders.base_loader import Passage
from embeddings.base import BaseEmbeddingModel
from retrieval.base_retriever import BaseRetriever, RetrievalResult


class LightRAGRetriever(BaseRetriever):
    """
    LightRAG baseline implementing dual-level retrieval.
    Balances low-level (entity/fact focus) and high-level (thematic/relation focus).
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        passages: Dict[str, Passage],
        embedding_model: BaseEmbeddingModel,
        low_level_weight: float = 0.5
    ):
        self.graph = graph
        self.passages = passages
        self.embedding_model = embedding_model
        self.low_level_weight = low_level_weight

        self.pids = list(passages.keys())
        self.passage_texts = [f"{p.title} {p.text}" for p in passages.values()]
        if self.passage_texts:
            self.passage_embeddings = self.embedding_model.embed_texts(self.passage_texts)
        else:
            self.passage_embeddings = np.empty((0, embedding_model.dimension), dtype=np.float32)

        # Collect entities
        self.entity_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get("node_type") == "entity"]
        self.entity_names = [self.graph.nodes[n].get("entity_name", str(n)) for n in self.entity_nodes]
        if self.entity_names:
            self.entity_embeddings = self.embedding_model.embed_texts(self.entity_names)
        else:
            self.entity_embeddings = np.empty((0, embedding_model.dimension), dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        start_time = time.perf_counter()

        if len(self.pids) == 0:
            return RetrievalResult(
                query=query,
                retrieved_passages=[],
                retrieved_nodes=[],
                passage_scores={},
                node_scores={},
                latency_ms=0.0
            )

        q_vec = self.embedding_model.embed_query(query)

        # 1. High-level retrieval: direct semantic passage-level matching
        high_level_sims = self.embedding_model.batch_cosine_similarity(q_vec, self.passage_embeddings)
        high_level_scores = {pid: float(high_level_sims[i]) for i, pid in enumerate(self.pids)}

        # 2. Low-level retrieval: entity-level matching and 1-hop propagation to passages
        low_level_scores = {pid: 0.0 for pid in self.pids}
        if len(self.entity_embeddings) > 0:
            ent_sims = self.embedding_model.batch_cosine_similarity(q_vec, self.entity_embeddings)
            # Propagate entity scores to neighboring passages
            for i, ent_node in enumerate(self.entity_nodes):
                ent_score = float(ent_sims[i])
                if ent_score > 0.2:
                    for neighbor in self.graph.neighbors(ent_node):
                        if self.graph.nodes[neighbor].get("node_type") == "passage":
                            pid = self.graph.nodes[neighbor].get("passage_id")
                            if pid and pid in low_level_scores:
                                low_level_scores[pid] += ent_score

        # Normalize low-level scores
        max_low = max(low_level_scores.values()) if low_level_scores else 1.0
        if max_low > 0:
            for pid in low_level_scores:
                low_level_scores[pid] /= max_low

        # Combine dual levels: Score = w * Low + (1 - w) * High
        combined_scores: Dict[str, float] = {}
        for pid in self.pids:
            score = (
                self.low_level_weight * low_level_scores.get(pid, 0.0) +
                (1.0 - self.low_level_weight) * high_level_scores.get(pid, 0.0)
            )
            combined_scores[pid] = float(score)

        ranked_pids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_pids = [pid for pid, _ in ranked_pids[:top_k]]
        retrieved_passages = [self.passages[pid] for pid in top_pids if pid in self.passages]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=query,
            retrieved_passages=retrieved_passages,
            retrieved_nodes=top_pids,
            passage_scores=combined_scores,
            node_scores=combined_scores,
            paths=[f"Query -> [DualLevel: High={high_level_scores[pid]:.2f}, Low={low_level_scores[pid]:.2f}] -> ({self.passages[pid].title})" for pid in top_pids],
            latency_ms=latency_ms,
            diffusion_steps=1,
            metadata={"retriever": "LightRAG", "low_level_weight": self.low_level_weight}
        )
