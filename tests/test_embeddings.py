"""Unit tests for embedding providers and persistent caching."""

import os
import pytest
import numpy as np
from embeddings.local_embed import LocalHashEmbeddings
from embeddings.cache import EmbeddingCache
from embeddings.factory import get_embedding_model


def test_local_embeddings():
    emb = LocalHashEmbeddings(dimension=128, seed=42)
    vec1 = emb.embed_query("quantum mechanics")
    vec2 = emb.embed_query("quantum mechanics")
    vec3 = emb.embed_query("gardening tools")

    assert vec1.shape == (128,)
    # Determinism
    np.testing.assert_allclose(vec1, vec2, rtol=1e-5)

    # Cosine similarity should be higher for identical than unrelated
    sim_same = emb.cosine_similarity(vec1, vec2)
    sim_diff = emb.cosine_similarity(vec1, vec3)
    assert sim_same > 0.99
    assert sim_same > sim_diff


def test_embedding_cache(tmp_path):
    cache = EmbeddingCache(cache_dir=str(tmp_path))
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    cache.put("test-model", "test query", vec)
    cached_vec = cache.get("test-model", "test query")

    assert cached_vec is not None
    np.testing.assert_allclose(cached_vec, vec)
    assert cache.get("test-model", "nonexistent") is None


def test_embedding_factory():
    m_local = get_embedding_model("local-hash-embed", dimension=64)
    assert m_local.dimension == 64

    m_openai = get_embedding_model("text-embedding-3-small", dimension=1536)
    assert m_openai.model_name == "text-embedding-3-small"

    m_nvidia = get_embedding_model("nvidia/nv-embed-v2", dimension=4096)
    assert m_nvidia.model_name == "nvidia/nv-embed-v2"
