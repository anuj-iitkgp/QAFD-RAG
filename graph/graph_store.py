"""
Graph serialization and disk storage.
Saves and loads constructed graphs (EntityGraph and PassageEntityGraph) to disk.
"""

import os
import pickle
import json
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class GraphStore:
    """Manages disk persistence and caching for constructed graphs."""

    def __init__(self, cache_dir: str = ".cache/graphs"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_path(self, graph_name: str) -> str:
        return os.path.join(self.cache_dir, f"{graph_name}.pkl")

    def save(self, graph_obj: Any, graph_name: str) -> str:
        """Serializes graph object to disk using pickle."""
        file_path = self._get_path(graph_name)
        try:
            with open(file_path, "wb") as f:
                pickle.dump(graph_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved graph '{graph_name}' to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save graph '{graph_name}': {e}")
            raise

    def load(self, graph_name: str) -> Optional[Any]:
        """Loads serialized graph object from disk if it exists."""
        file_path = self._get_path(graph_name)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "rb") as f:
                graph_obj = pickle.load(f)
            logger.info(f"Loaded cached graph '{graph_name}' from {file_path}")
            return graph_obj
        except Exception as e:
            logger.warning(f"Failed to load cached graph '{graph_name}' from {file_path}: {e}")
            return None

    def exists(self, graph_name: str) -> bool:
        return os.path.exists(self._get_path(graph_name))
