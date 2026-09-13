"""
Standard Question Answering evaluation metrics (Exact Match and token-level F1).
Faithful to official SQuAD and HotpotQA evaluation standards.
Normalizes text by lowercasing, removing punctuation, and stripping articles.
"""

import re
import string
from typing import List, Union


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(s: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', s)

    def white_space_fix(s: str) -> str:
        return ' '.join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Computes exact match between prediction and gold answer after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return float(f1)


def compute_qa_metrics(prediction: str, ground_truth: str, aliases: List[str] = None) -> dict:
    """Computes max EM and F1 across ground truth and valid answer aliases."""
    candidates = [ground_truth]
    if aliases:
        candidates.extend(aliases)

    em = max(exact_match_score(prediction, c) for c in candidates)
    f1 = max(f1_score(prediction, c) for c in candidates)

    return {
        "exact_match": em,
        "f1": f1
    }
