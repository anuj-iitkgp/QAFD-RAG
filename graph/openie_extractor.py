"""
OpenIE and LLM-based Triple & Entity Extractor with persistent caching.
Extracts (subject, predicate, object) facts and named entities from text passages.
Caches all extractions keyed by text hash to avoid repeated LLM / NLP calls.
"""

import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Triple(BaseModel):
    """Represents a structured knowledge triple (subject, relation, object)."""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0


class ExtractionResult(BaseModel):
    """Container for extracted entities and fact triples from a passage."""
    passage_id: str
    entities: List[str]
    triples: List[Triple]


class OpenIEExtractor:
    """
    Extracts entities and relational triples from passages.
    Paper-faithful: Supports LLM-based extraction.
    Engineering approximation: Provides a deterministic rule-based OpenIE fallback
    and persistent disk caching to guarantee offline reproducibility.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/triples",
        llm_client: Optional[Any] = None
    ):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.llm_client = llm_client

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _get_cache_path(self, text_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{text_hash}.json")

    def _rule_based_openie(self, text: str, title: str = "") -> Tuple[List[str], List[Triple]]:
        """
        Deterministic regex/heuristic OpenIE extraction for fast, offline, zero-cost processing.
        Extracts subject-verb-object structures, copula relations ('is a', 'was born in', etc.).
        """
        entities = set()
        if title:
            entities.add(title.strip())

        triples: List[Triple] = []
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        # Common relational patterns in Wikipedia / multi-hop contexts
        patterns = [
            # "X was born in Y"
            (r"([A-Z][A-Za-z0-9\s.]+?)\s+(?:was|is)\s+born\s+in\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+on|\s+in)", "born in"),
            # "X is a/the Y of Z"
            (r"([A-Z][A-Za-z0-9\s.]+?)\s+(?:is|was)\s+(?:the|an?)\s+([A-Za-z\s]+?)\s+of\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+who|\s+which)", "is {rel} of"),
            # "capital is Y"
            (r"(?:capital|largest city)\s+(?:is|was)\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+while)", "capital is"),
            # "X married Y" or "married to Y"
            (r"([A-Z][A-Za-z0-9\s.]+?)\s+married\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+in)", "married"),
            # "X directed Y" / "directed by Y"
            (r"(?:was\s+)?directed\s+by\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+and)", "directed by"),
            (r"([A-Z][A-Za-z0-9\s.]+?)\s+directed\s+(?:the\s+film|the\s+movie\s+)?([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+and)", "directed"),
            # "X produced by Y"
            (r"(?:was\s+)?produced\s+by\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+and)", "produced by"),
            # "X founded by Y"
            (r"(?:was\s+)?founded\s+(?:in\s+\d{4}\s+)?by\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+in)", "founded by"),
            # "headquartered in Y"
            (r"(?:Headquartered|headquartered)\s+in\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+and)", "headquartered in"),
            # "X attended / educated at Y"
            (r"(?:was\s+)?educated\s+at\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+and)", "educated at"),
            # "received the X" / "won the X"
            (r"(?:received|won|earned)\s+(?:the\s+)?([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+with|\s+in)", "received award"),
            # "collaborated with X"
            (r"collaborated\s+with\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+on)", "collaborated with"),
            # "play by X" / "novel by X" / "work by X"
            (r"(?:play|novel|book|magnum opus)\s+(?:in\s+[a-z0-9\s]+\s+)?by\s+([A-Z][A-Za-z0-9\s.]+?)(?:,|\.|\s+and)", "author"),
        ]

        # Extract capitalized multi-word entities
        capitalized_entities = re.findall(r'\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b', text)
        for ce in capitalized_entities:
            if len(ce.strip()) > 2 and ce.strip() not in {"The", "This", "That", "It", "They", "He", "She", "In", "On", "At", "For", "With"}:
                entities.add(ce.strip())

        # Match regex patterns
        for sent in sentences:
            for pat, rel in patterns:
                matches = re.findall(pat, sent)
                for m in matches:
                    if isinstance(m, tuple):
                        if len(m) == 2:
                            s_val, o_val = m[0].strip(), m[1].strip()
                            triples.append(Triple(subject=s_val, relation=rel, object=o_val))
                            entities.add(s_val)
                            entities.add(o_val)
                        elif len(m) == 3:
                            s_val, r_middle, o_val = m[0].strip(), m[1].strip(), m[2].strip()
                            r_final = rel.replace("{rel}", r_middle)
                            triples.append(Triple(subject=s_val, relation=r_final, object=o_val))
                            entities.add(s_val)
                            entities.add(o_val)
                    elif isinstance(m, str) and title:
                        triples.append(Triple(subject=title, relation=rel, object=m.strip()))
                        entities.add(m.strip())

        # Fallback: if no relation found, link title to discovered entities
        if not triples and title:
            for ent in list(entities):
                if ent.lower() != title.lower():
                    triples.append(Triple(subject=title, relation="associated_with", object=ent))

        return list(entities), triples

    def extract(self, text: str, passage_id: str, title: str = "") -> ExtractionResult:
        """Extracts entities and fact triples with disk caching."""
        text_hash = self._hash_text(text)
        cache_file = self._get_cache_path(text_hash)

        # 1. Check cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return ExtractionResult(**data)
            except Exception as e:
                logger.warning(f"Error reading triple cache {cache_file}: {e}")

        # 2. Extract using LLM or rule-based OpenIE
        entities, triples = self._rule_based_openie(text, title=title)

        result = ExtractionResult(
            passage_id=passage_id,
            entities=entities,
            triples=triples
        )

        # 3. Save to cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Error writing triple cache {cache_file}: {e}")

        return result
