"""
Base dataset loader and unified data models for multi-hop QA datasets.
Preserves question, supporting evidence, answer, passage information, and entity metadata.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field


class SupportingFact(BaseModel):
    """Represents a supporting fact or sentence identified as gold evidence."""
    title: str = Field(..., description="Title of the article or passage")
    sent_id: int = Field(default=-1, description="0-indexed sentence ID if applicable, or -1")
    text: Optional[str] = Field(default="", description="The exact sentence or fact text")


class Passage(BaseModel):
    """Represents a text passage or document chunk in the corpus."""
    id: str = Field(..., description="Unique identifier for the passage")
    title: str = Field(..., description="Title or source heading of the passage")
    text: str = Field(..., description="Full text content of the passage")
    is_supporting: bool = Field(default=False, description="True if this passage contains gold evidence")
    entities: List[str] = Field(default_factory=list, description="Entities mentioned in or extracted from the passage")


class QAItem(BaseModel):
    """Unified schema for multi-hop QA instances across MuSiQue, HotpotQA, and 2WikiMultiHopQA."""
    id: str = Field(..., description="Unique identifier of the sample")
    dataset: str = Field(..., description="Source dataset name (musique, hotpotqa, 2wikimultihopqa)")
    question: str = Field(..., description="Multi-hop question text")
    answer: str = Field(..., description="Gold answer text")
    answer_aliases: List[str] = Field(default_factory=list, description="Alternative acceptable answers")
    passages: List[Passage] = Field(default_factory=list, description="Full candidate passage pool")
    supporting_facts: List[SupportingFact] = Field(default_factory=list, description="Gold supporting evidence")
    entities: List[str] = Field(default_factory=list, description="Key entities associated with question/reasoning")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset-specific metadata (hops, type, etc.)")

    def get_gold_passage_ids(self) -> Set[str]:
        """Returns the set of passage IDs that contain supporting evidence."""
        # Check passages marked supporting or matching supporting fact titles
        gold_ids = set()
        for p in self.passages:
            if p.is_supporting:
                gold_ids.add(p.id)
        if not gold_ids and self.supporting_facts:
            fact_titles = {f.title.lower().strip() for f in self.supporting_facts}
            for p in self.passages:
                if p.title.lower().strip() in fact_titles:
                    gold_ids.add(p.id)
        return gold_ids


class BaseDatasetLoader(ABC):
    """Abstract dataset loader interface."""

    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        self.data_path = data_path
        self.max_samples = max_samples

    @abstractmethod
    def load(self) -> List[QAItem]:
        """Loads and parses the dataset into a list of unified QAItem instances."""
        pass
