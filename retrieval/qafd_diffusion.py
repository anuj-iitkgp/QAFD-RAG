"""
Query-Aware Flow Diffusion (QAFD) Retrieval Engine with Push-Relabel Propagation.

Mathematical Intuition & Design Rationale:
------------------------------------------
In traditional graph RAG, retrieval relies either on independent semantic similarity
(dense embeddings) or heuristic graph traversals (random walks, PageRank, BFS).
However, multi-hop reasoning requires identifying connected chains of evidence where
the information from a question's seed entities 'flows' through relational constraints
to find the destination passage containing the final answer.

QAFD-RAG models this process as a continuous flow diffusion network on a heterogeneous
knowledge graph G = (V, E) with capacities c(u, v):
1. Potential Field (Height) h(u):
   Nodes with high query relevance start at a high potential: h(u) = alpha * s_0(u).
   This potential acts as a gradient driving flow downhill along admissible edges.
2. Flow Conservation & Excess e(u):
   Each node maintains an excess flow e(u). A node is active if e(u) > epsilon.
3. Push Operation:
   Flow moves along admissible edges (u, v) where residual capacity r(u, v) > 0 and h(u) > h(v):
       Delta_f = min(e(u), step_size * (h(u) - h(v)) * r(u, v))
   Excess and flows are updated anti-symmetrically.
4. Relabel Operation:
   If an active node cannot push flow downhill because all residual neighbors have
   equal or higher height, its potential is lifted:
       h(u) <- min_{v: r(u, v) > 0} h(v) + step_size
   This allows trapped flow to escape and discover alternative relational paths.
5. Convergence:
   The algorithm terminates when no node has active excess: max_u e(u) <= epsilon,
   or when max_iterations is reached.

Configurable Paper Defaults:
----------------------------
- alpha = 2.0       (Potential gradient scaling / height parameter)
- epsilon = 0.01    (Convergence stopping threshold for active excess)
- step_size = 0.2   (Continuous flow push rate along admissible edges)
"""

import time
import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
from data.loaders.base_loader import Passage
from embeddings.base import BaseEmbeddingModel
from retrieval.base_retriever import BaseRetriever, RetrievalResult
from retrieval.path_extractor import MultiHopPathExtractor

logger = logging.getLogger(__name__)


class QAFDFlowDiffusionRetriever(BaseRetriever):
    """
    Query-Aware Flow Diffusion Retriever using Push-Relabel Propagation.
    Faithfully implements the push-relabel flow diffusion equations.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        passages: Dict[str, Passage],
        embedding_model: BaseEmbeddingModel,
        alpha: float = 2.0,
        epsilon: float = 0.01,
        step_size: float = 0.2,
        max_iterations: int = 100,
        damping_factor: float = 0.85,
        seed_threshold: float = 0.20,
        max_seeds: int = 10,
        query_aware: bool = True  # Used for ablation: query-aware vs uniform initialization
    ):
        self.graph = graph
        self.passages = passages
        self.embedding_model = embedding_model

        # Core paper parameters
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.damping_factor = float(damping_factor)
        self.seed_threshold = float(seed_threshold)
        self.max_seeds = int(max_seeds)
        self.query_aware = query_aware

        self.path_extractor = MultiHopPathExtractor(graph)
        self._precompute_node_embeddings()

    def _precompute_node_embeddings(self) -> None:
        """Precomputes text representations and embeddings for all nodes in the graph."""
        self.node_list = list(self.graph.nodes())
        self.node_texts = []
        for n in self.node_list:
            data = self.graph.nodes[n]
            # Use title and text preview for passages; entity name for entities
            if data.get("node_type") == "passage":
                text = f"{data.get('title', '')} {data.get('text', '')[:200]}"
            else:
                text = data.get("entity_name", data.get("display_name", str(n)))
            self.node_texts.append(text)

        if self.node_texts:
            self.node_embeddings = self.embedding_model.embed_texts(self.node_texts)
        else:
            self.node_embeddings = np.empty((0, self.embedding_model.dimension), dtype=np.float32)

    def _initialize_query_flow(self, query: str) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
        """
        Computes initial semantic similarity and sets up initial excess e(u) and height h(u).
        Mathematical intuition:
        - s_0(u) = max(0, cos(e_q, e_u))
        - Active seeds S = top-k nodes where s_0(u) >= seed_threshold
        - Query-aware excess: e(u) = s_0(u) / sum_{w in S} s_0(w)
        - Query-aware potential: h(u) = alpha * s_0(u)
        """
        q_vec = self.embedding_model.embed_query(query)
        sims = self.embedding_model.batch_cosine_similarity(q_vec, self.node_embeddings)

        s_0: Dict[str, float] = {}
        for idx, node in enumerate(self.node_list):
            s_0[node] = max(0.0, float(sims[idx]))

        # Select candidate seeds
        scored_nodes = sorted(s_0.items(), key=lambda x: x[1], reverse=True)
        seeds = [n for n, score in scored_nodes[:self.max_seeds] if score >= self.seed_threshold]
        if not seeds:
            # Fallback: take top 2 scoring nodes
            seeds = [n for n, _ in scored_nodes[:2]]

        e: Dict[str, float] = {n: 0.0 for n in self.node_list}
        h: Dict[str, float] = {n: 0.0 for n in self.node_list}

        if self.query_aware:
            # Paper-faithful query-aware initialization
            seed_sim_sum = sum(s_0[s] for s in seeds)
            if seed_sim_sum == 0:
                seed_sim_sum = 1.0

            for s in seeds:
                e[s] = s_0[s] / seed_sim_sum
                h[s] = self.alpha * s_0[s]
        else:
            # Ablation: uniform seed initialization without query-aware weighting
            uniform_val = 1.0 / len(seeds) if seeds else 1.0
            for s in seeds:
                e[s] = uniform_val
                h[s] = self.alpha * 1.0

        return e, h, seeds

    def _run_push_relabel_diffusion(
        self,
        e: Dict[str, float],
        h: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], int]:
        """
        Executes push-relabel flow diffusion propagation.
        Maintains edge flows f(u, v) and residual capacities r(u, v) = c(u, v) - f(u, v).
        Terminates when max_u e(u) <= epsilon or max_iterations reached.
        """
        # Initialize edge capacities and flows
        capacities: Dict[Tuple[str, str], float] = {}
        flows: Dict[Tuple[str, str], float] = {}

        for u, v, data in self.graph.edges(data=True):
            cap = float(data.get("capacity", data.get("weight", 1.0)))
            capacities[(u, v)] = cap
            flows[(u, v)] = 0.0
            # Ensure reverse edge entry exists in residual tracking
            if (v, u) not in capacities:
                capacities[(v, u)] = 0.0
                flows[(v, u)] = 0.0

        absorbed_flow: Dict[str, float] = {n: 0.0 for n in self.node_list}
        step = 0

        while step < self.max_iterations:
            # 1. Identify active nodes where e(u) > epsilon
            active_nodes = [u for u in self.node_list if e[u] > self.epsilon]
            if not active_nodes:
                # Convergence reached! All excess <= epsilon
                break

            # Process active nodes
            progress_made = False
            for u in active_nodes:
                if e[u] <= self.epsilon:
                    continue

                # Inspect neighbors in residual graph
                pushed_any = False
                # Outgoing edges from u
                neighbors = list(self.graph.successors(u))

                # Check admissible edges: r(u, v) > 0 and h(u) > h(v)
                for v in neighbors:
                    cap = capacities.get((u, v), 1.0)
                    cur_f = flows.get((u, v), 0.0)
                    res_cap = cap - cur_f

                    if res_cap > 1e-6 and h[u] > h[v]:
                        # Admissible edge found -> PUSH OPERATION
                        # Delta_f = min(e(u), step_size * (h(u) - h(v)) * res_cap)
                        height_diff = h[u] - h[v]
                        delta_f = min(e[u], self.step_size * height_diff * res_cap)

                        if delta_f > 1e-7:
                            flows[(u, v)] = cur_f + delta_f
                            flows[(v, u)] = flows.get((v, u), 0.0) - delta_f
                            e[u] -= delta_f
                            e[v] += delta_f
                            pushed_any = True
                            progress_made = True

                        if e[u] <= self.epsilon:
                            break

                # If node still has excess > epsilon and could not push -> RELABEL OPERATION
                if e[u] > self.epsilon and not pushed_any:
                    # Find min height among residual neighbors: min_{v: r(u, v) > 0} h(v)
                    min_neighbor_h = float("inf")
                    for v in neighbors:
                        cap = capacities.get((u, v), 1.0)
                        cur_f = flows.get((u, v), 0.0)
                        if (cap - cur_f) > 1e-6:
                            if h[v] < min_neighbor_h:
                                min_neighbor_h = h[v]

                    if min_neighbor_h < float("inf"):
                        # Lift height by step_size
                        h[u] = min_neighbor_h + self.step_size
                        progress_made = True
                    else:
                        # Dead-end node: absorb excess locally
                        absorbed_flow[u] += e[u]
                        e[u] = 0.0

                # Information damping / local retention
                # Absorbs (1 - damping_factor) fraction to ensure localized community clustering
                damp_absorbed = e[u] * (1.0 - self.damping_factor)
                absorbed_flow[u] += damp_absorbed
                e[u] -= damp_absorbed

            step += 1
            if not progress_made:
                # No further pushes or relabels possible
                break

        # Flush any remaining excess into absorbed flow
        for n in self.node_list:
            absorbed_flow[n] += e[n]

        return absorbed_flow, flows, step

    def _score_nodes_and_passages(
        self,
        absorbed_flow: Dict[str, float],
        flows: Dict[Tuple[str, str], float],
        initial_sims: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculates final relevance score for all nodes and passages.
        Passage scoring:
            Score(p) = Flow(p) + sum_{e in Entities(p)} w(p, e) * Flow(e)
        Rewarding connected multi-hop reasoning over isolated keyword matches.
        """
        # 1. Node flow score = absorbed flow + positive incoming flows
        node_scores: Dict[str, float] = {}
        for n in self.node_list:
            incoming = sum(max(0.0, flows.get((u, n), 0.0)) for u in self.graph.predecessors(n))
            node_scores[n] = float(absorbed_flow.get(n, 0.0) + 0.5 * incoming + 0.2 * initial_sims.get(n, 0.0))

        # 2. Passage scoring
        passage_scores: Dict[str, float] = {}
        for pid, passage in self.passages.items():
            p_node = f"p::{pid}"
            p_score = node_scores.get(p_node, 0.0)

            # Aggregate connected entity flow
            entity_flow_sum = 0.0
            if self.graph.has_node(p_node):
                for neighbor in self.graph.neighbors(p_node):
                    if self.graph.nodes[neighbor].get("node_type") == "entity":
                        edge_w = float(self.graph[p_node][neighbor].get("weight", 1.0))
                        entity_flow_sum += edge_w * node_scores.get(neighbor, 0.0)

            total_passage_score = p_score + 0.6 * entity_flow_sum
            passage_scores[pid] = float(total_passage_score)

        return node_scores, passage_scores

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Executes the end-to-end QAFD retrieval pipeline:
        1. Embed query
        2. Initialize query-aware flow and potential field
        3. Propagate flow using push-relabel operations
        4. Score and rank passages
        5. Trace multi-hop reasoning paths
        """
        start_time = time.perf_counter()

        # Step 1 & 2: Initialization
        e, h, seeds = self._initialize_query_flow(query)
        initial_sims = {n: h[n] / self.alpha if self.alpha > 0 else 0.0 for n in self.node_list}

        # Step 3: Push-relabel diffusion
        absorbed_flow, flows, steps = self._run_push_relabel_diffusion(e, h)

        # Step 4: Scoring
        node_scores, passage_scores = self._score_nodes_and_passages(absorbed_flow, flows, initial_sims)

        # Step 5: Top-k ranking
        ranked_pids = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
        top_pids = [pid for pid, _ in ranked_pids[:top_k]]
        retrieved_passages = [self.passages[pid] for pid in top_pids if pid in self.passages]

        # Step 6: Multi-hop path tracing
        target_nodes = [f"p::{pid}" for pid in top_pids]
        paths = self.path_extractor.trace_paths(
            seeds=seeds,
            target_passages=target_nodes,
            edge_flows=flows,
            max_depth=3
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=query,
            retrieved_passages=retrieved_passages,
            retrieved_nodes=top_pids,
            passage_scores=passage_scores,
            node_scores=node_scores,
            paths=paths,
            latency_ms=latency_ms,
            diffusion_steps=steps,
            metadata={
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "step_size": self.step_size,
                "num_seeds": len(seeds),
                "seeds": seeds,
                "query_aware": self.query_aware
            }
        )
