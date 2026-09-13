"""
Fair comparative baseline evaluation script.
Executes QAFD-RAG, GraphRAG, LightRAG, and Vector RAG under the identical experimental protocol:
same question set, same top-k budget, same embedding model, same answer generator, and same evaluation metrics.
Usage:
    python scripts/run_baselines.py [--config configs/qafd_rag.yaml] [--dataset musique] [--top_k 5]
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import logging
import json
import csv
from data.loaders import MuSiQueLoader, HotpotQALoader, TwoWikiLoader
from graph import PassageEntityGraph, GraphStore
from embeddings import get_embedding_model
from retrieval import QAFDFlowDiffusionRetriever
from baselines import VectorRAGRetriever, GraphRAGRetriever, LightRAGRetriever
from models import LLMClient
from evaluation import PipelineEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_baselines")


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
    parser = argparse.ArgumentParser(description="Run fair comparative baselines")
    parser.add_argument("--config", default="configs/qafd_rag.yaml", help="Path to config file")
    parser.add_argument("--dataset", default=None, help="Dataset name")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k budget")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = args.dataset or cfg.get("dataset", "musique")
    data_dir = cfg.get("data_dir", "data/raw")
    top_k = args.top_k or cfg.get("top_k", 5)
    embed_name = cfg.get("embedding_model", "local-hash-embed")
    embed_dim = cfg.get("embedding_dimension", 256)
    llm_name = cfg.get("llm_model", "local-extractive")

    logger.info("=" * 60)
    logger.info("FAIR BASELINE COMPARISON EXPERIMENT")
    logger.info(f"Dataset: {dataset_name} | Top-K: {top_k} | Embedding: {embed_name} | LLM: {llm_name}")
    logger.info("=" * 60)

    items = get_dataset_items(dataset_name, data_dir)
    embedding_model = get_embedding_model(embed_name, dimension=embed_dim)
    llm_client = LLMClient(model_name=llm_name)
    store = GraphStore(cache_dir=cfg.get("cache", {}).get("graphs_cache", ".cache/graphs"))

    # Pre-build / load graphs
    graphs = {}
    passages_map = {}
    for item in items:
        peg_name = f"{dataset_name}_item_{item.id}_peg"
        peg = store.load(peg_name)
        if peg is None:
            peg = PassageEntityGraph(embedding_model=embedding_model)
            peg.build_from_passages(item.passages)
            store.save(peg, peg_name)
        graphs[item.id] = peg.graph
        passages_map[item.id] = {p.id: p for p in item.passages}

    evaluator = PipelineEvaluator(
        llm_client=llm_client,
        top_k=top_k,
        results_dir=cfg.get("output", {}).get("tables_dir", "results/tables")
    )

    baseline_configs = [
        ("VectorRAG", lambda it: VectorRAGRetriever(passages_map[it.id], embedding_model)),
        ("GraphRAG", lambda it: GraphRAGRetriever(graphs[it.id], passages_map[it.id], embedding_model)),
        ("LightRAG", lambda it: LightRAGRetriever(graphs[it.id], passages_map[it.id], embedding_model)),
        ("QAFD-RAG", lambda it: QAFDFlowDiffusionRetriever(
            graphs[it.id],
            passages_map[it.id],
            embedding_model,
            alpha=cfg.get("alpha", 2.0),
            epsilon=cfg.get("epsilon", 0.01),
            step_size=cfg.get("step_size", 0.2)
        ))
    ]

    all_summaries = []

    for name, builder in baseline_configs:
        summary = evaluator.evaluate_pipeline(
            items=items,
            retriever_builder_func=builder,
            pipeline_name=f"{name}_{dataset_name.upper()}"
        )
        all_summaries.append(summary)

    # Output comparison table
    print("\n" + "=" * 80)
    print(f"FAIR COMPARISON SUMMARY ON {dataset_name.upper()} (Top-K={top_k})")
    print("=" * 80)
    header = f"{'Method':<12} | {'EM':<6} | {'F1':<6} | {'MRR':<6} | {'R@1':<6} | {'R@2':<6} | {'R@K':<6} | {'Ret. Lat (ms)':<14} | {'Steps':<6}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        p_name = s["pipeline"].split("_")[0]
        print(f"{p_name:<12} | {s['exact_match']:<6.3f} | {s['f1']:<6.3f} | {s['mrr']:<6.3f} | {s['recall@1']:<6.3f} | {s['recall@2']:<6.3f} | {s[f'recall@{top_k}']:<6.3f} | {s['retrieval_latency_ms']:<14.2f} | {s['avg_diffusion_steps']:<6.1f}")
    print("=" * 80)

    # Save comparative table CSV
    tables_dir = cfg.get("output", {}).get("tables_dir", "results/tables")
    comp_file = os.path.join(tables_dir, f"baseline_comparison_{dataset_name}.csv")
    with open(comp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)
    logger.info(f"Saved baseline comparison to {comp_file}")


if __name__ == "__main__":
    main()
