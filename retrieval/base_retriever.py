"""
Base retriever interface and unified retrieval result container.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from data.loaders.base_loader import Passage


class RetrievalResult(BaseModel):
    """Encapsulates all outputs from a retrieval step."""
    query: str
    retrieved_passages: List[Passage]
    retrieved_nodes: List[str]
    passage_scores: Dict[str, float]
    node_scores: Dict[str, float]
    paths: List[str] = Field(default_factory=list, description="Reasoning paths traced through the graph")
    latency_ms: float = Field(default=0.0, description="Retrieval latency in milliseconds")
    diffusion_steps: int = Field(default=0, description="Number of diffusion / propagation steps")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract interface for all retrievers (QAFD-RAG, GraphRAG, LightRAG, Vector RAG)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieves top-k passages given a query."""
        pass
