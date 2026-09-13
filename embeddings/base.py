"""
Pluggable embedding model abstract interface.
Defines contracts for dense vector generation and cosine similarity calculation.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, model_name: str, dimension: int = 256):
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into a 2D numpy array of shape (len(texts), dimension)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D numpy array of shape (dimension,)."""
        pass

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two 1D vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def batch_cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Computes cosine similarity between a 1D query vector and a 2D matrix of doc vectors."""
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0 or len(doc_vecs) == 0:
            return np.zeros(len(doc_vecs), dtype=np.float32)
        
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        doc_norms[doc_norms == 0] = 1e-9
        
        sims = np.dot(doc_vecs, query_vec) / (doc_norms * q_norm)
        return np.nan_to_num(sims, nan=0.0)
