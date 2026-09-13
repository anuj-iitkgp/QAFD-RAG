"""
Standard Dense Vector Retrieval (Vector RAG) baseline.
Retrieves top-k passages based purely on embedding cosine similarity, without graph diffusion.
"""

import time
import numpy as np
from typing import List, Dict
from data.loaders.base_loader import Passage
from embeddings.base import BaseEmbeddingModel
from retrieval.base_retriever import BaseRetriever, RetrievalResult


class VectorRAGRetriever(BaseRetriever):
    """Dense semantic vector retrieval baseline."""

    def __init__(self, passages: Dict[str, Passage], embedding_model: BaseEmbeddingModel):
        self.passages = passages
        self.embedding_model = embedding_model
        self.pids = list(passages.keys())
        self.texts = [f"{p.title} {p.text}" for p in passages.values()]
        if self.texts:
            self.passage_embeddings = self.embedding_model.embed_texts(self.texts)
        else:
            self.passage_embeddings = np.empty((0, self.embedding_model.dimension), dtype=np.float32)

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
        sims = self.embedding_model.batch_cosine_similarity(q_vec, self.passage_embeddings)

        passage_scores = {pid: float(sims[i]) for i, pid in enumerate(self.pids)}
        ranked_pids = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)

        top_pids = [pid for pid, _ in ranked_pids[:top_k]]
        retrieved_passages = [self.passages[pid] for pid in top_pids if pid in self.passages]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=query,
            retrieved_passages=retrieved_passages,
            retrieved_nodes=top_pids,
            passage_scores=passage_scores,
            node_scores=passage_scores,
            paths=[f"Query -> [Vector Similarity={passage_scores[pid]:.3f}] -> ({self.passages[pid].title})" for pid in top_pids],
            latency_ms=latency_ms,
            diffusion_steps=0,
            metadata={"retriever": "VectorRAG"}
        )
