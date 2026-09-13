"""
Comprehensive end-to-end evaluation pipeline for QAFD-RAG and baselines.
Profiles retrieval latency, generation latency, EM, F1, Recall@K, Precision@K, and MRR.
Exports structured result summaries to CSV and JSON.
"""

import os
import time
import json
import csv
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from data.loaders.base_loader import QAItem, Passage
from retrieval.base_retriever import BaseRetriever, RetrievalResult
from models.llm_client import LLMClient
from evaluation.qa_metrics import compute_qa_metrics
from evaluation.retrieval_metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Runs end-to-end multi-hop QA evaluation across dataset items."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        top_k: int = 5,
        results_dir: str = "results/tables"
    ):
        self.llm_client = llm_client or LLMClient()
        self.top_k = top_k
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def format_context(self, passages: List[Passage]) -> str:
        """Formats retrieved passages into a structured multi-hop context string."""
        context_blocks = []
        for i, p in enumerate(passages, 1):
            context_blocks.append(f"[Passage {i}: {p.title}]\n{p.text}")
        return "\n\n".join(context_blocks)

    def evaluate_item(
        self,
        item: QAItem,
        retriever: BaseRetriever
    ) -> Dict[str, Any]:
        """Evaluates a single multi-hop QA sample."""
        # 1. Retrieval
        ret_start = time.perf_counter()
        ret_result: RetrievalResult = retriever.retrieve(item.question, top_k=self.top_k)
        ret_latency = (time.perf_counter() - ret_start) * 1000.0

        # 2. Retrieval Metrics
        gold_pids = item.get_gold_passage_ids()
        retrieved_pids = [p.id for p in ret_result.retrieved_passages]
        ret_metrics = compute_retrieval_metrics(retrieved_pids, gold_pids, k_values=[1, 2, 3, self.top_k])

        # 3. Context Preparation & Generation
        context_str = self.format_context(ret_result.retrieved_passages)
        gen_start = time.perf_counter()
        prediction = self.llm_client.answer_question(item.question, context_str)
        gen_latency = (time.perf_counter() - gen_start) * 1000.0

        # 4. QA Metrics
        qa_res = compute_qa_metrics(prediction, item.answer, item.answer_aliases)

        total_latency = ret_latency + gen_latency

        return {
            "id": item.id,
            "dataset": item.dataset,
            "question": item.question,
            "gold_answer": item.answer,
            "predicted_answer": prediction,
            "exact_match": qa_res["exact_match"],
            "f1": qa_res["f1"],
            "mrr": ret_metrics["mrr"],
            "recall@1": ret_metrics.get("recall@1", 0.0),
            "recall@2": ret_metrics.get("recall@2", 0.0),
            "recall@3": ret_metrics.get("recall@3", 0.0),
            f"recall@{self.top_k}": ret_metrics.get(f"recall@{self.top_k}", 0.0),
            "retrieval_latency_ms": ret_latency,
            "generation_latency_ms": gen_latency,
            "total_latency_ms": total_latency,
            "num_retrieved_passages": len(ret_result.retrieved_passages),
            "diffusion_steps": ret_result.diffusion_steps,
            "retrieved_titles": [p.title for p in ret_result.retrieved_passages],
            "sample_path": ret_result.paths[0] if ret_result.paths else ""
        }

    def evaluate_pipeline(
        self,
        items: List[QAItem],
        retriever_builder_func: Any,
        pipeline_name: str = "QAFD-RAG"
    ) -> Dict[str, Any]:
        """
        Runs evaluation over all items, instantiating per-item graph retriever if needed,
        and saves aggregated results.
        """
        logger.info(f"Starting evaluation of {pipeline_name} on {len(items)} items...")
        sample_results: List[Dict[str, Any]] = []

        for item in items:
            # Build retriever for this item's passage corpus
            retriever = retriever_builder_func(item)
            res = self.evaluate_item(item, retriever)
            sample_results.append(res)

        # Aggregate metrics
        agg_metrics = {
            "pipeline": pipeline_name,
            "num_samples": len(sample_results),
            "exact_match": float(np.mean([r["exact_match"] for r in sample_results])),
            "f1": float(np.mean([r["f1"] for r in sample_results])),
            "mrr": float(np.mean([r["mrr"] for r in sample_results])),
            "recall@1": float(np.mean([r["recall@1"] for r in sample_results])),
            "recall@2": float(np.mean([r["recall@2"] for r in sample_results])),
            "recall@3": float(np.mean([r["recall@3"] for r in sample_results])),
            f"recall@{self.top_k}": float(np.mean([r[f"recall@{self.top_k}"] for r in sample_results])),
            "retrieval_latency_ms": float(np.mean([r["retrieval_latency_ms"] for r in sample_results])),
            "total_latency_ms": float(np.mean([r["total_latency_ms"] for r in sample_results])),
            "avg_diffusion_steps": float(np.mean([r["diffusion_steps"] for r in sample_results])),
        }

        # Save results to JSON and CSV
        clean_name = pipeline_name.lower().replace(" ", "_").replace("-", "_")
        json_file = os.path.join(self.results_dir, f"{clean_name}_results.json")
        csv_file = os.path.join(self.results_dir, f"{clean_name}_samples.csv")

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"summary": agg_metrics, "samples": sample_results}, f, indent=2)

        if sample_results:
            keys = list(sample_results[0].keys())
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(sample_results)

        logger.info(f"Completed {pipeline_name}: EM={agg_metrics['exact_match']:.3f}, F1={agg_metrics['f1']:.3f}, Recall@{self.top_k}={agg_metrics[f'recall@{self.top_k}']:.3f}")
        return agg_metrics
