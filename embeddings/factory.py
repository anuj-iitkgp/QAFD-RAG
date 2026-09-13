"""
Embedding model factory.
Dynamically instantiates and configures the requested embedding provider.
"""

from typing import Dict, Any, Optional
from embeddings.base import BaseEmbeddingModel
from embeddings.openai_embed import OpenAIEmbeddings
from embeddings.nvidia_embed import NvidiaEmbeddings
from embeddings.local_embed import LocalHashEmbeddings
from embeddings.cache import EmbeddingCache


def get_embedding_model(
    model_name: str,
    dimension: Optional[int] = None,
    cache_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> BaseEmbeddingModel:
    """
    Factory function to instantiate pluggable embedding models.
    Supports:
    - 'nvidia-nv-embed-v2' / 'nvidia/nv-embed-v2'
    - 'text-embedding-3-small' / 'openai'
    - 'local-hash-embed' / 'local'
    """
    cache = EmbeddingCache(cache_dir=cache_dir or ".cache/embeddings")
    name_lower = model_name.lower().strip()

    if "nvidia" in name_lower or "nv-embed" in name_lower:
        dim = dimension or 4096
        return NvidiaEmbeddings(model_name="nvidia/nv-embed-v2", dimension=dim, cache=cache)
    elif "openai" in name_lower or "text-embedding-3" in name_lower:
        dim = dimension or 1536
        return OpenAIEmbeddings(model_name="text-embedding-3-small", dimension=dim, cache=cache)
    elif "local" in name_lower or "hash" in name_lower:
        dim = dimension or 256
        return LocalHashEmbeddings(model_name="local-hash-embed", dimension=dim, cache=cache)
    else:
        # Default fallback
        dim = dimension or 256
        return LocalHashEmbeddings(model_name=model_name, dimension=dim, cache=cache)
