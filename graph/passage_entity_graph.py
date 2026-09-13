"""
Heterogeneous Passage-Entity Graph representation.
Contains passage nodes, entity nodes, fact edges, passage-entity association edges, and synonymy edges.
Fully supports configurable ablations (without synonymy edges, without fact edges).
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from data.loaders.base_loader import Passage
from graph.openie_extractor import OpenIEExtractor
from embeddings.base import BaseEmbeddingModel


class PassageEntityGraph:
    """
    Heterogeneous Passage-Entity Graph for QAFD-RAG.
    Nodes:
      - Passage nodes ('passage::{passage_id}')
      - Entity nodes ('entity::{entity_name}')
    Edges:
      - Fact edges (Entity -> Entity)
      - Association edges (Passage <-> Entity)
      - Synonymy edges (Entity <-> Entity)
    """

    def __init__(
        self,
        extractor: Optional[OpenIEExtractor] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        include_fact_edges: bool = True,
        include_synonymy_edges: bool = True,
        include_association_edges: bool = True,
        synonymy_threshold: float = 0.80
    ):
        self.graph = nx.DiGraph()
        self.extractor = extractor or OpenIEExtractor()
        self.embedding_model = embedding_model
        self.include_fact_edges = include_fact_edges
        self.include_synonymy_edges = include_synonymy_edges
        self.include_association_edges = include_association_edges
        self.synonymy_threshold = synonymy_threshold

        self.passages: Dict[str, Passage] = {}
        self.entity_nodes: Set[str] = set()
        self.passage_nodes: Set[str] = set()

    @staticmethod
    def passage_node_id(pid: str) -> str:
        return f"p::{pid}"

    @staticmethod
    def entity_node_id(name: str) -> str:
        return f"e::{name.strip().lower()}"

    def add_passage_node(self, passage: Passage) -> str:
        node_id = self.passage_node_id(passage.id)
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                node_type="passage",
                passage_id=passage.id,
                title=passage.title,
                text=passage.text,
                is_supporting=passage.is_supporting,
                display_name=passage.title or f"Passage {passage.id}"
            )
            self.passages[passage.id] = passage
            self.passage_nodes.add(node_id)
        return node_id

    def add_entity_node(self, entity_name: str) -> str:
        clean_name = entity_name.strip()
        if not clean_name:
            return ""
        node_id = self.entity_node_id(clean_name)
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                node_type="entity",
                entity_name=clean_name,
                display_name=clean_name
            )
            self.entity_nodes.add(node_id)
        return node_id

    def add_fact_edge(self, sub_node: str, ob_node: str, relation: str, weight: float = 1.0) -> None:
        if not self.include_fact_edges or sub_node == ob_node or not sub_node or not ob_node:
            return

        # Directed fact edge + reciprocal residual edge
        if self.graph.has_edge(sub_node, ob_node):
            self.graph[sub_node][ob_node]["capacity"] = self.graph[sub_node][ob_node].get("capacity", 1.0) + weight
        else:
            self.graph.add_edge(
                sub_node,
                ob_node,
                edge_type="fact",
                relation=relation,
                capacity=weight,
                weight=weight
            )

        # Back-edge with lower residual capacity to allow multi-hop flow balancing
        if not self.graph.has_edge(ob_node, sub_node):
            self.graph.add_edge(
                ob_node,
                sub_node,
                edge_type="fact_reciprocal",
                relation=f"inv_{relation}",
                capacity=weight * 0.5,
                weight=weight * 0.5
            )

    def add_association_edge(self, p_node: str, e_node: str, weight: float = 1.0) -> None:
        if not self.include_association_edges or not p_node or not e_node:
            return

        # Bidirectional association between passage and entity
        for (u, v) in [(p_node, e_node), (e_node, p_node)]:
            if self.graph.has_edge(u, v):
                self.graph[u][v]["capacity"] = self.graph[u][v].get("capacity", 1.0) + weight
            else:
                self.graph.add_edge(
                    u,
                    v,
                    edge_type="association",
                    relation="mentions",
                    capacity=weight,
                    weight=weight
                )

    def add_synonymy_edges(self, entities: List[str]) -> int:
        """Connects semantically similar entity nodes based on embedding cosine similarity."""
        if not self.include_synonymy_edges or not self.embedding_model or len(entities) < 2:
            return 0

        unique_entities = sorted(list({e.strip() for e in entities if e.strip()}))
        if len(unique_entities) < 2:
            return 0

        embeddings = self.embedding_model.embed_texts(unique_entities)
        added_count = 0

        # Pairwise cosine similarities
        sim_matrix = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        norm_matrix = np.dot(norms, norms.T)
        sim_matrix = sim_matrix / norm_matrix

        for i in range(len(unique_entities)):
            for j in range(i + 1, len(unique_entities)):
                sim = float(sim_matrix[i, j])
                if sim >= self.synonymy_threshold:
                    u_node = self.entity_node_id(unique_entities[i])
                    v_node = self.entity_node_id(unique_entities[j])
                    if self.graph.has_node(u_node) and self.graph.has_node(v_node):
                        for u, v in [(u_node, v_node), (v_node, u_node)]:
                            self.graph.add_edge(
                                u,
                                v,
                                edge_type="synonymy",
                                relation="synonym",
                                capacity=sim,
                                weight=sim
                            )
                        added_count += 1

        return added_count

    def build_from_passages(self, passages: List[Passage]) -> None:
        """Constructs the complete heterogeneous passage-entity graph from candidate passages."""
        all_entities = []

        for p in passages:
            p_node = self.add_passage_node(p)
            extraction = self.extractor.extract(text=p.text, passage_id=p.id, title=p.title)

            # 1. Association edges for passage -> entities
            for ent_name in extraction.entities:
                e_node = self.add_entity_node(ent_name)
                if e_node:
                    self.add_association_edge(p_node, e_node, weight=1.0)
                    all_entities.append(ent_name)

            # 2. Fact edges for extracted triples
            for t in extraction.triples:
                sub_node = self.add_entity_node(t.subject)
                ob_node = self.add_entity_node(t.object)
                if sub_node and ob_node:
                    self.add_fact_edge(sub_node, ob_node, relation=t.relation, weight=t.confidence)
                    # Also associate passage with both subject and object
                    self.add_association_edge(p_node, sub_node, weight=0.8)
                    self.add_association_edge(p_node, ob_node, weight=0.8)
                    all_entities.append(t.subject)
                    all_entities.append(t.object)

        # 3. Synonymy edges
        if self.include_synonymy_edges and self.embedding_model:
            self.add_synonymy_edges(all_entities)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def num_passages(self) -> int:
        return len(self.passage_nodes)

    @property
    def num_entities(self) -> int:
        return len(self.entity_nodes)
