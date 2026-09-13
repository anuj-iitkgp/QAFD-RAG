"""Unit tests for graph construction and persistence."""

import os
import pytest
from data.loaders.base_loader import Passage
from graph.entity_graph import EntityGraph
from graph.passage_entity_graph import PassageEntityGraph
from graph.graph_store import GraphStore
from embeddings.local_embed import LocalHashEmbeddings


@pytest.fixture
def sample_passages():
    return [
        Passage(
            id="p1",
            title="Alexander Graham Bell",
            text="Alexander Graham Bell was born in Edinburgh. He invented the telephone.",
            is_supporting=True,
            entities=["Alexander Graham Bell", "Edinburgh"]
        ),
        Passage(
            id="p2",
            title="Scotland",
            text="Scotland is a country in the United Kingdom. Its capital is Edinburgh.",
            is_supporting=True,
            entities=["Scotland", "Edinburgh"]
        )
    ]


def test_entity_graph_construction(sample_passages):
    eg = EntityGraph()
    eg.build_from_passages(sample_passages)

    assert eg.num_nodes > 0
    assert eg.num_edges > 0
    assert eg.graph.has_node("Edinburgh")
    # Edinburgh should be associated with passages
    assert len(eg.get_passages_for_entity("Edinburgh")) >= 1


def test_passage_entity_graph_construction(sample_passages):
    emb = LocalHashEmbeddings(dimension=64)
    peg = PassageEntityGraph(embedding_model=emb, synonymy_threshold=0.8)
    peg.build_from_passages(sample_passages)

    assert peg.num_passages == 2
    assert peg.num_entities > 0
    assert peg.num_edges > 0

    # Verify passage nodes have correct attributes
    assert peg.graph.has_node("p::p1")
    assert peg.graph.nodes["p::p1"]["node_type"] == "passage"


def test_graph_store_save_load(tmp_path, sample_passages):
    store = GraphStore(cache_dir=str(tmp_path))
    eg = EntityGraph()
    eg.build_from_passages(sample_passages)

    path = store.save(eg, "test_eg")
    assert os.path.exists(path)

    loaded = store.load("test_eg")
    assert loaded is not None
    assert loaded.num_nodes == eg.num_nodes
