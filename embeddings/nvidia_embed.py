"""
NVIDIA embedding provider (nvidia-nv-embed-v2).
Connects to NVIDIA NIM OpenAI-compatible endpoint with persistent disk caching and graceful offline fallback.
"""

import os
import logging
import numpy as np
from typing import List, Optional
from embeddings.base import BaseEmbeddingModel
from embeddings.cache import EmbeddingCache
from embeddings.local_embed import LocalHashEmbeddings

logger = logging.getLogger(__name__)


class NvidiaEmbeddings(BaseEmbeddingModel):
    """NVIDIA nv-embed-v2 provider with persistent disk caching."""

    def __init__(
        self,
        model_name: str = "nvidia/nv-embed-v2",
        dimension: int = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None
    ):
        # Default dimension for nv-embed-v2 is 4096 (can be mapped/truncated if desired)
        super().__init__(model_name=model_name, dimension=dimension)
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.cache = cache or EmbeddingCache()
        self._client = None
        self._fallback = None

        if self.api_key and self.api_key not in ("your_nvidia_api_key_here", "your_openai_api_key_here"):
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                logger.warning(f"Could not initialize NVIDIA client: {e}. Enabling fallback.")
                self._fallback = LocalHashEmbeddings(model_name=f"fallback-{model_name}", dimension=self.dimension)
        else:
            logger.info("NVIDIA_API_KEY not set. Operating in offline/reproducibility fallback mode.")
            self._fallback = LocalHashEmbeddings(model_name=f"fallback-{model_name}", dimension=self.dimension)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        cached_vecs, missing_indices = self.cache.get_batch(self.model_name, texts)
        if not missing_indices:
            return np.vstack([v for v in cached_vecs if v is not None])

        missing_texts = [texts[i] for i in missing_indices]
        if self._client:
            try:
                # NVIDIA NIM OpenAI-compatible endpoint
                response = self._client.embeddings.create(
                    input=missing_texts,
                    model=self.model_name,
                    extra_body={"input_type": "passage", "truncate": "END"}
                )
                computed = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            except Exception as e:
                logger.warning(f"NVIDIA API call failed: {e}. Falling back to deterministic local embeddings.")
                if not self._fallback:
                    self._fallback = LocalHashEmbeddings(model_name=f"fallback-{self.model_name}", dimension=self.dimension)
                computed = [self._fallback.embed_query(t) for t in missing_texts]
        else:
            if not self._fallback:
                self._fallback = LocalHashEmbeddings(model_name=f"fallback-{self.model_name}", dimension=self.dimension)
            computed = [self._fallback.embed_query(t) for t in missing_texts]

        for idx, vec in zip(missing_indices, computed):
            self.cache.put(self.model_name, texts[idx], vec)
            cached_vecs[idx] = vec

        return np.vstack([v for v in cached_vecs if v is not None])

    def embed_query(self, query: str) -> np.ndarray:
        cached = self.cache.get(self.model_name, query)
        if cached is not None:
            return cached
        if self._client:
            try:
                response = self._client.embeddings.create(
                    input=[query],
                    model=self.model_name,
                    extra_body={"input_type": "query", "truncate": "END"}
                )
                vec = np.array(response.data[0].embedding, dtype=np.float32)
                self.cache.put(self.model_name, query, vec)
                return vec
            except Exception as e:
                logger.warning(f"NVIDIA API query call failed: {e}. Falling back.")
                if not self._fallback:
                    self._fallback = LocalHashEmbeddings(model_name=f"fallback-{self.model_name}", dimension=self.dimension)
                return self._fallback.embed_query(query)
        else:
            if not self._fallback:
                self._fallback = LocalHashEmbeddings(model_name=f"fallback-{self.model_name}", dimension=self.dimension)
            return self._fallback.embed_query(query)
