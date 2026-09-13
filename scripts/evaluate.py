"""
Unified evaluation script for QAFD-RAG and baselines across all benchmark datasets.
Usage:
    python scripts/evaluate.py [--config configs/qafd_rag.yaml] [--datasets musique,hotpotqa,2wikimultihopqa]
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import logging
import csv
import json
from data.loaders import MuSiQueLoader, HotpotQALoader, TwoWikiLoader
from graph import PassageEntityGraph, GraphStore
from embeddings import get_embedding_model
from retrieval import QAFDFlowDiffusionRetriever
from baselines import VectorRAGRetriever, GraphRAGRetriever, LightRAGRetriever
from models import LLMClient
from evaluation import PipelineEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def load_config(config_path: str = "configs/qafd_rag.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_dataset_items(dataset_name: str, data_dir: str = "data/raw"):
    ds = dataset_name.lower().strip()
    if ds == "musique":
        return MuSiQueLoader(os.path.join(data_dir, "musique/sample.jsonl")).load()
    elif ds == "hotpotqa":
        return HotpotQALoader(os.path.join(data_dir, "hotpotqa/sample.json")).load()
    elif ds in ("2wikimultihopqa", "2wiki"):
        return TwoWikiLoader(os.path.join(data_dir, "2wikimultihopqa/sample.json")).load()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate QAFD-RAG and baselines")
    parser.add_argument("--config", default="configs/qafd_rag.yaml", help="Path to config file")
    parser.add_argument("--datasets", default="musique,hotpotqa,2wikimultihopqa", help="Comma-separated dataset list")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    data_dir = cfg.get("data_dir", "data/raw")
    tables_dir = cfg.get("output", {}).get("tables_dir", "results/tables")
    os.makedirs(tables_dir, exist_ok=True)

    embedding_model = get_embedding_model(cfg.get("embedding_model", "local-hash-embed"), dimension=cfg.get("embedding_dimension", 256))
    llm_client = LLMClient(model_name=cfg.get("llm_model", "local-extractive"))
    top_k = cfg.get("top_k", 5)

    evaluator = PipelineEvaluator(llm_client=llm_client, top_k=top_k, results_dir=tables_dir)

    all_benchmark_results = []

    for ds in datasets:
        items = get_dataset_items(ds, data_dir)
        store = GraphStore(cache_dir=cfg.get("cache", {}).get("graphs_cache", ".cache/graphs"))

        # Pre-build graphs
        graphs = {}
        passages_map = {}
        for item in items:
            peg_name = f"{ds}_item_{item.id}_peg"
            peg = store.load(peg_name)
            if peg is None:
                peg = PassageEntityGraph(embedding_model=embedding_model)
                peg.build_from_passages(item.passages)
                store.save(peg, peg_name)
            graphs[item.id] = peg.graph
            passages_map[item.id] = {p.id: p for p in item.passages}

        methods = [
            ("VectorRAG", lambda it: VectorRAGRetriever(passages_map[it.id], embedding_model)),
            ("GraphRAG", lambda it: GraphRAGRetriever(graphs[it.id], passages_map[it.id], embedding_model)),
            ("LightRAG", lambda it: LightRAGRetriever(graphs[it.id], passages_map[it.id], embedding_model)),
            ("QAFD-RAG", lambda it: QAFDFlowDiffusionRetriever(
                graphs[it.id], passages_map[it.id], embedding_model,
                alpha=cfg.get("alpha", 2.0), epsilon=cfg.get("epsilon", 0.01), step_size=cfg.get("step_size", 0.2)
            ))
        ]

        for m_name, m_builder in methods:
            summary = evaluator.evaluate_pipeline(items, m_builder, f"{m_name}_{ds.upper()}")
            summary["dataset"] = ds.upper()
            summary["method"] = m_name
            all_benchmark_results.append(summary)

    # Print summary table
    print("\n" + "=" * 95)
    print("CROSS-DATASET BENCHMARK EVALUATION SUMMARY")
    print("=" * 95)
    header = f"{'Dataset':<16} | {'Method':<12} | {'EM':<6} | {'F1':<6} | {'MRR':<6} | {'R@1':<6} | {'R@2':<6} | {'R@5':<6} | {'Latency(ms)':<12}"
    print(header)
    print("-" * len(header))
    for r in all_benchmark_results:
        print(f"{r['dataset']:<16} | {r['method']:<12} | {r['exact_match']:<6.3f} | {r['f1']:<6.3f} | {r['mrr']:<6.3f} | {r['recall@1']:<6.3f} | {r['recall@2']:<6.3f} | {r.get('recall@5', 0.0):<6.3f} | {r['retrieval_latency_ms']:<12.2f}")
    print("=" * 95)

    # Save to consolidated CSV
    summary_csv = os.path.join(tables_dir, "consolidated_benchmark_evaluation.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_benchmark_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_benchmark_results)
    logger.info(f"Saved consolidated benchmark evaluation to {summary_csv}")


if __name__ == "__main__":
    main()
