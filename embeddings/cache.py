"""
Persistent disk cache for embedding vectors.
Keyed by SHA256(model_name + "::" + text), saves and loads vectors as compressed numpy arrays.
"""

import os
import hashlib
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Disk cache for text embeddings to prevent redundant API calls."""

    def __init__(self, cache_dir: str = ".cache/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index_path = os.path.join(self.cache_dir, "index.json")
        self._index: Dict[str, str] = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding cache index: {e}")
        return {}

    def _save_index(self) -> None:
        try:
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache index: {e}")

    def _hash_key(self, model_name: str, text: str) -> str:
        content = f"{model_name}::{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, model_name: str, text: str) -> Optional[np.ndarray]:
        """Retrieves cached embedding for text if available."""
        key = self._hash_key(model_name, text)
        if key in self._index:
            file_path = os.path.join(self.cache_dir, self._index[key])
            if os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    return data["embedding"]
                except Exception:
                    pass
        return None

    def put(self, model_name: str, text: str, embedding: np.ndarray) -> None:
        """Saves embedding vector to disk cache."""
        key = self._hash_key(model_name, text)
        filename = f"{key}.npz"
        file_path = os.path.join(self.cache_dir, filename)
        try:
            np.savez_compressed(file_path, embedding=embedding)
            self._index[key] = filename
            self._save_index()
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def get_batch(self, model_name: str, texts: List[str]) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Retrieves cached embeddings for a batch of texts.
        Returns:
            cached_vectors: list of embeddings (or None if missing)
            missing_indices: list of indices for texts that need computation
        """
        results: List[Optional[np.ndarray]] = []
        missing_indices: List[int] = []
        for idx, text in enumerate(texts):
            vec = self.get(model_name, text)
            results.append(vec)
            if vec is None:
                missing_indices.append(idx)
        return results, missing_indices
