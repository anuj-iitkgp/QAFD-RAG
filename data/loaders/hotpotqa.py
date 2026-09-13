"""
HotpotQA dataset loader.
Parses multi-hop questions, paragraph contexts with sentence-level gold supporting facts, and bridges/comparisons.
"""

import json
import os
from typing import List, Optional
from data.loaders.base_loader import BaseDatasetLoader, QAItem, Passage, SupportingFact


class HotpotQALoader(BaseDatasetLoader):
    """Loader for HotpotQA multi-hop benchmark."""

    def load(self) -> List[QAItem]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"HotpotQA data file not found at: {self.data_path}")

        items: List[QAItem] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            # HotpotQA can be a single JSON array or JSON lines
            content = f.read().strip()
            if content.startswith("["):
                raw_list = json.loads(content)
            else:
                raw_list = [json.loads(line) for line in content.split("\n") if line.strip()]

        for raw in raw_list:
            sample_id = str(raw.get("_id", raw.get("id", f"hotpot_{len(items)}")))
            question = raw.get("question", "")
            answer = raw.get("answer", "")

            # Gold supporting facts: list of [title, sent_id]
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

            # Context: list of [title, [sent0, sent1, ...]]
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

            # Attach sentence text to supporting facts if available
            for sf in supporting_facts:
                for ctx in raw_context:
                    if str(ctx[0]) == sf.title and isinstance(ctx[1], list):
                        if 0 <= sf.sent_id < len(ctx[1]):
                            sf.text = ctx[1][sf.sent_id]

            metadata = {
                "type": raw.get("type", "bridge"),
                "level": raw.get("level", "medium"),
            }

            item = QAItem(
                id=sample_id,
                dataset="hotpotqa",
                question=question,
                answer=answer,
                passages=passages,
                supporting_facts=supporting_facts,
                entities=list(entities),
                metadata=metadata
            )
            items.append(item)

            if self.max_samples and len(items) >= self.max_samples:
                break

        return items
