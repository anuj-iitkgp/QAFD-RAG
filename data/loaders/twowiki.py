"""
2WikiMultiHopQA dataset loader.
Parses multi-hop questions with structured evidence triples and supporting passages.
"""

import json
import os
from typing import List, Optional
from data.loaders.base_loader import BaseDatasetLoader, QAItem, Passage, SupportingFact


class TwoWikiLoader(BaseDatasetLoader):
    """Loader for 2WikiMultiHopQA dataset."""

    def load(self) -> List[QAItem]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"2WikiMultiHopQA data file not found at: {self.data_path}")

        items: List[QAItem] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                raw_list = json.loads(content)
            else:
                raw_list = [json.loads(line) for line in content.split("\n") if line.strip()]

        for raw in raw_list:
            sample_id = str(raw.get("_id", raw.get("id", f"2wiki_{len(items)}")))
            question = raw.get("question", "")
            answer = raw.get("answer", "")
            answer_aliases = raw.get("answer_aliases", [])

            # Supporting facts
            gold_facts_raw = raw.get("supporting_facts", [])
            supporting_titles = {f[0] for f in gold_facts_raw if isinstance(f, (list, tuple)) and len(f) >= 1}
            supporting_facts: List[SupportingFact] = []
            for f in gold_facts_raw:
                if isinstance(f, (list, tuple)) and len(f) >= 2:
                    supporting_facts.append(SupportingFact(
                        title=str(f[0]),
                        sent_id=int(f[1]),
                        text=""
                    ))

            # Context
            raw_context = raw.get("context", [])
            passages: List[Passage] = []
            entities = set()

            for p_idx, ctx in enumerate(raw_context):
                if not isinstance(ctx, (list, tuple)) or len(ctx) < 2:
                    continue
                title = str(ctx[0])
                sentences = ctx[1]
                if isinstance(sentences, list):
                    text = " ".join(sentences)
                else:
                    text = str(sentences)

                is_supp = title in supporting_titles
                pid = f"{sample_id}_p{p_idx}"

                entities.add(title)
                passages.append(Passage(
                    id=pid,
                    title=title,
                    text=text,
                    is_supporting=is_supp,
                    entities=[title]
                ))

            # Evidence triples (2WikiMultiHopQA provides gold relation triples)
            evidences = raw.get("evidences", [])

            metadata = {
                "type": raw.get("type", "compositional"),
                "evidences": evidences
            }

            item = QAItem(
                id=sample_id,
                dataset="2wikimultihopqa",
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
