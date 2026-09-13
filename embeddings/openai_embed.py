"""
OpenAI embedding provider (text-embedding-3-small).
Supports disk caching, batch processing, and graceful fallback when API key is unset.
"""

import os
import logging
import numpy as np
from typing import List, Optional
from embeddings.base import BaseEmbeddingModel
from embeddings.cache import EmbeddingCache
from embeddings.local_embed import LocalHashEmbeddings

logger = logging.getLogger(__name__)


class OpenAIEmbeddings(BaseEmbeddingModel):
    """OpenAI text-embedding-3-small provider with persistent disk caching."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None
    ):
        super().__init__(model_name=model_name, dimension=dimension)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.cache = cache or EmbeddingCache()
        self._client = None
        self._fallback = None

        if self.api_key and self.api_key != "your_openai_api_key_here":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}. Enabling fallback.")
                self._fallback = LocalHashEmbeddings(model_name=f"fallback-{model_name}", dimension=self.dimension)
        else:
            logger.info("OPENAI_API_KEY not set or placeholder. Operating in offline/reproducibility fallback mode.")
            self._fallback = LocalHashEmbeddings(model_name=f"fallback-{model_name}", dimension=self.dimension)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # 1. Check disk cache
        cached_vecs, missing_indices = self.cache.get_batch(self.model_name, texts)
        if not missing_indices:
            return np.vstack([v for v in cached_vecs if v is not None])

        # 2. Compute missing vectors
        missing_texts = [texts[i] for i in missing_indices]
        if self._client:
            try:
                response = self._client.embeddings.create(
                    input=missing_texts,
                    model=self.model_name
                )
                computed = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            except Exception as e:
                logger.warning(f"OpenAI API call failed: {e}. Falling back to deterministic local embeddings.")
                if not self._fallback:
                    self._fallback = LocalHashEmbeddings(model_name=f"fallback-{self.model_name}", dimension=self.dimension)
                computed = [self._fallback.embed_query(t) for t in missing_texts]
        else:
            if not self._fallback:
                self._fallback = LocalHashEmbeddings(model_name=f"fallback-{self.model_name}", dimension=self.dimension)
            computed = [self._fallback.embed_query(t) for t in missing_texts]

        # 3. Update cache and reconstruct full array
        for idx, vec in zip(missing_indices, computed):
            self.cache.put(self.model_name, texts[idx], vec)
            cached_vecs[idx] = vec

        return np.vstack([v for v in cached_vecs if v is not None])

    def embed_query(self, query: str) -> np.ndarray:
        cached = self.cache.get(self.model_name, query)
        if cached is not None:
            return cached
        res = self.embed_texts([query])[0]
        return res
