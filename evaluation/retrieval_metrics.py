"""
Retrieval evaluation metrics for multi-hop question answering.
Computes Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and Hit@K against gold supporting evidence.
"""

from typing import List, Set, Dict, Any


def recall_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Computes the fraction of gold passages retrieved within top-k."""
    if not gold:
        return 1.0
    top_k_retrieved = set(retrieved[:k])
    hits = len(top_k_retrieved & gold)
    return float(hits / len(gold))


def precision_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Computes the fraction of top-k retrieved passages that are gold."""
    if k <= 0 or not retrieved:
        return 0.0
    top_k_retrieved = set(retrieved[:k])
    hits = len(top_k_retrieved & gold)
    return float(hits / min(k, len(retrieved)))


def mean_reciprocal_rank(retrieved: List[str], gold: Set[str]) -> float:
    """Computes reciprocal rank of the first relevant gold passage retrieved."""
    if not gold:
        return 1.0
    for rank_idx, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / rank_idx
    return 0.0


def hit_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
    """Returns 1.0 if at least one gold passage appears in top-k, else 0.0."""
    if not gold:
        return 1.0
    top_k_retrieved = set(retrieved[:k])
    return 1.0 if len(top_k_retrieved & gold) > 0 else 0.0


def compute_retrieval_metrics(retrieved_pids: List[str], gold_pids: Set[str], k_values: List[int] = None) -> Dict[str, float]:
    """Computes comprehensive retrieval metrics across multiple k thresholds."""
    if k_values is None:
        k_values = [1, 2, 3, 5]

    metrics = {
        "mrr": mean_reciprocal_rank(retrieved_pids, gold_pids)
    }

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_pids, gold_pids, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_pids, gold_pids, k)
        metrics[f"hit@{k}"] = hit_at_k(retrieved_pids, gold_pids, k)

    return metrics
