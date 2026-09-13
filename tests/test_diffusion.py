"""Unit tests for Query-Aware Flow Diffusion with Push-Relabel Propagation."""

import pytest
import networkx as nx
import numpy as np
from retrieval.qafd_diffusion import QAFDFlowDiffusionRetriever
from data.loaders.base_loader import Passage
from embeddings.local_embed import LocalHashEmbeddings


@pytest.fixture
def sample_graph_and_passages():
    """Creates a controlled synthetic heterogeneous graph for flow diffusion testing."""
    g = nx.DiGraph()
    # Add passage nodes
    g.add_node("p::p1", node_type="passage", passage_id="p1", title="Title 1", text="Text about entity Alpha.")
    g.add_node("p::p2", node_type="passage", passage_id="p2", title="Title 2", text="Text about entity Beta.")

    # Add entity nodes
    g.add_node("e::alpha", node_type="entity", entity_name="Alpha", display_name="Alpha")
    g.add_node("e::beta", node_type="entity", entity_name="Beta", display_name="Beta")

    # Add edges with capacities
    # p1 <-> alpha
    g.add_edge("p::p1", "e::alpha", edge_type="association", capacity=1.0, weight=1.0)
    g.add_edge("e::alpha", "p::p1", edge_type="association", capacity=1.0, weight=1.0)

    # alpha -> beta (fact edge)
    g.add_edge("e::alpha", "e::beta", edge_type="fact", capacity=1.0, weight=1.0, relation="relates_to")
    g.add_edge("e::beta", "e::alpha", edge_type="fact_reciprocal", capacity=0.5, weight=0.5, relation="inv_relates_to")

    # beta <-> p2
    g.add_edge("e::beta", "p::p2", edge_type="association", capacity=1.0, weight=1.0)
    g.add_edge("p::p2", "e::beta", edge_type="association", capacity=1.0, weight=1.0)

    passages = {
        "p1": Passage(id="p1", title="Title 1", text="Text about entity Alpha.", is_supporting=False),
        "p2": Passage(id="p2", title="Title 2", text="Text about entity Beta.", is_supporting=True)
    }

    emb = LocalHashEmbeddings(dimension=64, seed=42)
    return g, passages, emb


def test_qafd_initialization(sample_graph_and_passages):
    g, passages, emb = sample_graph_and_passages
    retriever = QAFDFlowDiffusionRetriever(
        graph=g,
        passages=passages,
        embedding_model=emb,
        alpha=2.0,
        epsilon=0.01,
        step_size=0.2
    )

    e, h, seeds = retriever._initialize_query_flow("Alpha")
    assert len(seeds) > 0
    # Seed nodes must receive non-zero initial excess and potential
    assert sum(e.values()) > 0.99  # Normalized to ~1.0
    for s in seeds:
        assert h[s] > 0.0


def test_push_relabel_convergence(sample_graph_and_passages):
    g, passages, emb = sample_graph_and_passages
    retriever = QAFDFlowDiffusionRetriever(
        graph=g,
        passages=passages,
        embedding_model=emb,
        alpha=2.0,
        epsilon=0.01,
        step_size=0.2,
        max_iterations=50
    )

    e, h, seeds = retriever._initialize_query_flow("Tell me about Alpha and Beta")
    absorbed_flow, flows, steps = retriever._run_push_relabel_diffusion(e, h)

    # Must converge within max_iterations
    assert steps <= 50
    # Total flow absorbed must match initial flow conservation
    total_absorbed = sum(absorbed_flow.values())
    assert total_absorbed > 0.5


def test_flow_parameter_sensitivity(sample_graph_and_passages):
    g, passages, emb = sample_graph_and_passages

    # High step size vs low step size
    ret_fast = QAFDFlowDiffusionRetriever(g, passages, emb, step_size=0.5, epsilon=0.01)
    ret_slow = QAFDFlowDiffusionRetriever(g, passages, emb, step_size=0.05, epsilon=0.01)

    res_fast = ret_fast.retrieve("Alpha", top_k=2)
    res_slow = ret_slow.retrieve("Alpha", top_k=2)

    assert len(res_fast.retrieved_passages) > 0
    assert len(res_slow.retrieved_passages) > 0
    # Both must converge within max iterations
    assert res_fast.diffusion_steps > 0
    assert res_slow.diffusion_steps > 0
    # Different step sizes yield different flow distributions
    assert res_fast.passage_scores != res_slow.passage_scores
