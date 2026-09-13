"""
MuSiQue dataset loader.
Parses multi-hop questions, paragraph contexts with gold supporting indicators, and entity hops.
"""

import json
import os
import re
from typing import List, Optional
from data.loaders.base_loader import BaseDatasetLoader, QAItem, Passage, SupportingFact


class MuSiQueLoader(BaseDatasetLoader):
    """Loader for MuSiQue multi-hop dataset."""

    def load(self) -> List[QAItem]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"MuSiQue data file not found at: {self.data_path}")

        items: List[QAItem] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sample_id = str(raw.get("id", f"musique_{line_idx}"))
                question = raw.get("question", "")
                answer = raw.get("answer", "")
                answer_aliases = raw.get("answer_aliases", [])

                # Parse candidate passages
                raw_paragraphs = raw.get("paragraphs", [])
                passages: List[Passage] = []
                supporting_facts: List[SupportingFact] = []
                entities = set()

                for p in raw_paragraphs:
                    pid = f"{sample_id}_p{p.get('idx', len(passages))}"
                    title = p.get("title", "")
                    text = p.get("paragraph_text", "")
                    is_supp = bool(p.get("is_supporting", False))

                    # Basic entity heuristic: title is usually an entity
                    p_entities = [title] if title else []
                    entities.add(title)

                    passages.append(Passage(
                        id=pid,
                        title=title,
                        text=text,
                        is_supporting=is_supp,
                        entities=p_entities
                    ))

                    if is_supp:
                        supporting_facts.append(SupportingFact(
                            title=title,
                            sent_id=-1,
                            text=text
                        ))

                # Question decomposition metadata
                metadata = {
                    "question_decomposition": raw.get("question_decomposition", []),
                    "num_hops": len(raw.get("question_decomposition", []))
                }

                item = QAItem(
                    id=sample_id,
                    dataset="musique",
                    question=question,
                    answer=answer,
                    answer_aliases=answer_aliases,
                    passages=passages,
                    supporting_facts=supporting_facts,
                    entities=list(entities),
                    metadata=metadata
                )
                items.append(item)

                if self.max_samples and len(items) >= self.max_samples:
                    break

        return items
