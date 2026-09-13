"""
Entity Graph representation.
Builds and maintains a directed graph where nodes are entities and edges are semantic relationships.
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from data.loaders.base_loader import Passage
from graph.openie_extractor import OpenIEExtractor, Triple


class EntityGraph:
    """
    Entity Graph representation (Entity A -> relationship -> Entity B).
    Paper-faithful: Models pure entity-relation multi-hop topology.
    """

    def __init__(self, extractor: Optional[OpenIEExtractor] = None):
        self.graph = nx.DiGraph()
        self.extractor = extractor or OpenIEExtractor()
        self.node_passages: Dict[str, Set[str]] = {}

    def add_entity(self, entity_name: str, passage_id: Optional[str] = None, **attrs) -> None:
        """Adds an entity node to the graph."""
        norm_name = entity_name.strip()
        if not norm_name:
            return

        if not self.graph.has_node(norm_name):
            self.graph.add_node(
                norm_name,
                name=norm_name,
                node_type="entity",
                passages=set(),
                **attrs
            )
            self.node_passages[norm_name] = set()

        if passage_id:
            self.graph.nodes[norm_name]["passages"].add(passage_id)
            self.node_passages[norm_name].add(passage_id)

    def add_relation(
        self,
        subject: str,
        relation: str,
        obj: str,
        weight: float = 1.0,
        passage_id: Optional[str] = None
    ) -> None:
        """Adds a directed relation edge between subject and object entities."""
        sub = subject.strip()
        ob = obj.strip()
        rel = relation.strip()
        if not sub or not ob or sub.lower() == ob.lower():
            return

        self.add_entity(sub, passage_id=passage_id)
        self.add_entity(ob, passage_id=passage_id)

        if self.graph.has_edge(sub, ob):
            # Accumulate weight and keep relation history
            edge_data = self.graph[sub][ob]
            edge_data["weight"] = edge_data.get("weight", 1.0) + weight
            if rel and rel not in edge_data.get("relations", []):
                edge_data.setdefault("relations", []).append(rel)
        else:
            self.graph.add_edge(
                sub,
                ob,
                relation=rel,
                relations=[rel] if rel else [],
                weight=weight,
                edge_type="fact"
            )

    def build_from_passages(self, passages: List[Passage]) -> None:
        """Extracts entities and relations from candidate passages and builds the graph."""
        for p in passages:
            extraction = self.extractor.extract(text=p.text, passage_id=p.id, title=p.title)

            # Register entities from passage
            for ent in extraction.entities:
                self.add_entity(ent, passage_id=p.id)

            # Register fact triples
            for t in extraction.triples:
                self.add_relation(
                    subject=t.subject,
                    relation=t.relation,
                    obj=t.object,
                    weight=t.confidence,
                    passage_id=p.id
                )

    def get_passages_for_entity(self, entity_name: str) -> Set[str]:
        """Returns set of passage IDs associated with this entity."""
        return self.node_passages.get(entity_name.strip(), set())

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()
