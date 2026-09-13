"""
GraphRAG Baseline (Community-Based Hierarchical Graph Retrieval).
Faithfully models Microsoft GraphRAG's community detection and hierarchical retrieval strategy.
Partitions the knowledge graph into communities, computes community centroid representations,
and retrieves passages from top query-matching communities.
"""

import time
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Any
from data.loaders.base_loader import Passage
from embeddings.base import BaseEmbeddingModel
from retrieval.base_retriever import BaseRetriever, RetrievalResult


class GraphRAGRetriever(BaseRetriever):
    """
    GraphRAG baseline implementing community-based hierarchical retrieval.
    1. Detects communities using modularity-based community detection.
    2. Builds community summary representations.
    3. Matches query to communities, then retrieves candidate passages within top communities.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        passages: Dict[str, Passage],
        embedding_model: BaseEmbeddingModel
    ):
        self.graph = graph
        self.passages = passages
        self.embedding_model = embedding_model

        self.communities: List[Set[str]] = []
        self.community_embeddings: np.ndarray = np.empty((0, embedding_model.dimension), dtype=np.float32)
        self.community_passages: List[List[str]] = []

        self._build_communities()

    def _build_communities(self) -> None:
        """Detects graph communities and computes centroid representations."""
        # Convert to undirected graph for community detection
        undirected = self.graph.to_undirected()
        if undirected.number_of_nodes() == 0:
            return

        try:
            # Modularity-based greedy community detection
            import networkx.algorithms.community as nx_comm
            comms = list(nx_comm.greedy_modularity_communities(undirected))
            self.communities = [set(c) for c in comms]
        except Exception:
            # Fallback to connected components
            self.communities = [set(c) for c in nx.connected_components(undirected)]

        # Collect passages in each community and build centroid embedding
        comm_texts = []
        for comm in self.communities:
            p_in_comm = []
            node_texts = []
            for n in comm:
                data = self.graph.nodes[n]
                if data.get("node_type") == "passage":
                    pid = data.get("passage_id")
                    if pid and pid in self.passages:
                        p_in_comm.append(pid)
                        p = self.passages[pid]
                        node_texts.append(f"{p.title} {p.text[:150]}")
                else:
                    name = data.get("entity_name", str(n))
                    node_texts.append(name)

            self.community_passages.append(p_in_comm)
            comm_texts.append(" ".join(node_texts))

        if comm_texts:
            self.community_embeddings = self.embedding_model.embed_texts(comm_texts)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        start_time = time.perf_counter()

        if not self.communities or len(self.passages) == 0:
            return RetrievalResult(
                query=query,
                retrieved_passages=[],
                retrieved_nodes=[],
                passage_scores={},
                node_scores={},
                latency_ms=0.0
            )

        # 1. Match query against community representations
        q_vec = self.embedding_model.embed_query(query)
        comm_sims = self.embedding_model.batch_cosine_similarity(q_vec, self.community_embeddings)

        # Rank communities
        ranked_comm_idx = np.argsort(-comm_sims)

        # 2. Score passages within top communities
        passage_scores: Dict[str, float] = {}
        for c_idx in ranked_comm_idx:
            c_score = float(comm_sims[c_idx])
            p_list = self.community_passages[c_idx]

            for pid in p_list:
                if pid not in passage_scores:
                    p = self.passages[pid]
                    p_vec = self.embedding_model.embed_query(f"{p.title} {p.text}")
                    direct_sim = self.embedding_model.cosine_similarity(q_vec, p_vec)
                    # Hierarchical score = community score * 0.4 + direct similarity * 0.6
                    passage_scores[pid] = 0.4 * c_score + 0.6 * direct_sim

        # If some passages were not in communities, score them directly
        for pid, p in self.passages.items():
            if pid not in passage_scores:
                p_vec = self.embedding_model.embed_query(f"{p.title} {p.text}")
                passage_scores[pid] = 0.5 * self.embedding_model.cosine_similarity(q_vec, p_vec)

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
            paths=[f"Query -> Community #{c} -> ({self.passages[pid].title})" for c, pid in enumerate(top_pids)],
            latency_ms=latency_ms,
            diffusion_steps=1,
            metadata={"retriever": "GraphRAG", "num_communities": len(self.communities)}
        )
