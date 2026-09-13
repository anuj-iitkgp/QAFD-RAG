"""
Deterministic local semantic hash-projection embedding model.
Provides zero-dependency, reproducible dense vector embeddings for offline execution,
testing, and benchmark validation without paid API keys.
"""

import hashlib
import numpy as np
from typing import List
from embeddings.base import BaseEmbeddingModel
from embeddings.cache import EmbeddingCache


class LocalHashEmbeddings(BaseEmbeddingModel):
    """
    Paper-faithful / reproducible local embedding baseline.
    Generates deterministic, dense semantic representations using character and subword
    n-gram hashing projected onto a reproducible pseudo-random orthogonal basis.
    Preserves lexical, morphological, and entity overlap properties.
    """

    def __init__(
        self,
        model_name: str = "local-hash-embed",
        dimension: int = 256,
        seed: int = 42,
        cache: EmbeddingCache = None
    ):
        super().__init__(model_name=model_name, dimension=dimension)
        self.seed = seed
        self.cache = cache or EmbeddingCache()
        
        # Build deterministic projection matrix
        rng = np.random.RandomState(seed)
        # Random Gaussian projection matrix
        self.projection_matrix = rng.randn(dimension, 4096).astype(np.float32)
        # Orthogonalize columns using QR decomposition
        q, _ = np.linalg.qr(self.projection_matrix.T)
        self.projection_matrix = q.T[:dimension, :].astype(np.float32)

    def _text_to_feature_vector(self, text: str) -> np.ndarray:
        """Extracts hashed n-gram bag-of-features vector in 4096-dim space."""
        text = text.lower().strip()
        features = np.zeros(4096, dtype=np.float32)
        if not text:
            return features

        words = text.split()
        # Word features & n-grams
        for w in words:
            # Word token hash
            h_val = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % 4096
            features[h_val] += 1.0

            # Subword 3-grams
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    sub = w[i:i+3]
                    h_sub = int(hashlib.md5(sub.encode("utf-8")).hexdigest(), 16) % 4096
                    features[h_sub] += 0.5

        # Word bigrams for word-order awareness
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h_bi = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16) % 4096
            features[h_bi] += 1.5

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features

    def _embed_single(self, text: str) -> np.ndarray:
        cache_id = f"{self.model_name}_d{self.dimension}"
        cached = self.cache.get(cache_id, text)
        if cached is not None and cached.shape == (self.dimension,):
            return cached

        feat = self._text_to_feature_vector(text)
        vec = np.dot(self.projection_matrix, feat)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec = np.zeros(self.dimension, dtype=np.float32)

        self.cache.put(cache_id, text, vec)
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = [self._embed_single(t) for t in texts]
        return np.vstack(embeddings) if embeddings else np.empty((0, self.dimension), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_single(query)
