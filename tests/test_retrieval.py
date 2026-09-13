"""Unit tests for retrieval and multi-hop path extraction."""

import pytest
from data.loaders.base_loader import Passage
from graph.passage_entity_graph import PassageEntityGraph
from embeddings.local_embed import LocalHashEmbeddings
from retrieval.qafd_diffusion import QAFDFlowDiffusionRetriever
from baselines.vector_rag import VectorRAGRetriever
from baselines.graphrag import GraphRAGRetriever
from baselines.lightrag import LightRAGRetriever


@pytest.fixture
def test_corpus():
    passages = [
        Passage(
            id="musique_p0",
            title="Alexander Graham Bell",
            text="Alexander Graham Bell was born in Edinburgh. He is credited with the telephone.",
            is_supporting=True
        ),
        Passage(
            id="musique_p1",
            title="Scotland",
            text="Scotland is part of the United Kingdom. Its capital is Edinburgh.",
            is_supporting=True
        ),
        Passage(
            id="musique_p2",
            title="Thomas Edison",
            text="Thomas Edison was an American inventor who developed electric power generation.",
            is_supporting=False
        )
    ]
    p_dict = {p.id: p for p in passages}
    emb = LocalHashEmbeddings(dimension=128)
    peg = PassageEntityGraph(embedding_model=emb)
    peg.build_from_passages(passages)
    return peg, p_dict, emb


def test_qafd_multi_hop_retrieval(test_corpus):
    peg, p_dict, emb = test_corpus
    qafd = QAFDFlowDiffusionRetriever(peg.graph, p_dict, emb)

    res = qafd.retrieve("What is the capital of the country where the inventor of the telephone was born?", top_k=2)
    assert len(res.retrieved_passages) == 2
    assert len(res.paths) > 0
    # The path should mention entities or query transitions
    assert any("Query" in path for path in res.paths)


def test_baseline_retrievers(test_corpus):
    peg, p_dict, emb = test_corpus

    vrag = VectorRAGRetriever(p_dict, emb)
    v_res = vrag.retrieve("telephone inventor", top_k=2)
    assert len(v_res.retrieved_passages) == 2

    grag = GraphRAGRetriever(peg.graph, p_dict, emb)
    g_res = grag.retrieve("telephone inventor", top_k=2)
    assert len(g_res.retrieved_passages) == 2

    lrag = LightRAGRetriever(peg.graph, p_dict, emb)
    l_res = lrag.retrieve("telephone inventor", top_k=2)
    assert len(l_res.retrieved_passages) == 2
