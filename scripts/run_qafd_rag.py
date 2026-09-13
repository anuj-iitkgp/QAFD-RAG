"""
Main execution script for QAFD-RAG.
Runs Query-Aware Flow Diffusion on the selected dataset and generates answers.
Usage:
    python scripts/run_qafd_rag.py [--config configs/qafd_rag.yaml] [--dataset musique] [--alpha 2.0] [--epsilon 0.01] [--step_size 0.2] [--top_k 5]
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import logging
from data.loaders import MuSiQueLoader, HotpotQALoader, TwoWikiLoader
from graph import PassageEntityGraph, EntityGraph, GraphStore
from embeddings import get_embedding_model
from retrieval import QAFDFlowDiffusionRetriever
from models import LLMClient
from evaluation import PipelineEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_qafd_rag")


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
    parser = argparse.ArgumentParser(description="Run QAFD-RAG pipeline")
    parser.add_argument("--config", default="configs/qafd_rag.yaml", help="Path to config file")
    parser.add_argument("--dataset", default=None, help="Dataset name")
    parser.add_argument("--alpha", type=float, default=None, help="Potential scaling alpha")
    parser.add_argument("--epsilon", type=float, default=None, help="Convergence tolerance epsilon")
    parser.add_argument("--step_size", type=float, default=None, help="Flow push step size")
    parser.add_argument("--top_k", type=int, default=None, help="Retrieval budget top-k")
    parser.add_argument("--embedding_model", default=None, help="Embedding provider")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = args.dataset or cfg.get("dataset", "musique")
    data_dir = cfg.get("data_dir", "data/raw")
    alpha = args.alpha if args.alpha is not None else cfg.get("alpha", 2.0)
    epsilon = args.epsilon if args.epsilon is not None else cfg.get("epsilon", 0.01)
    step_size = args.step_size if args.step_size is not None else cfg.get("step_size", 0.2)
    top_k = args.top_k if args.top_k is not None else cfg.get("top_k", 5)
    embed_name = args.embedding_model or cfg.get("embedding_model", "local-hash-embed")
    embed_dim = cfg.get("embedding_dimension", 256)
    llm_name = cfg.get("llm_model", "local-extractive")

    logger.info("=" * 60)
    logger.info("RUNNING QAFD-RAG EVALUATION")
    logger.info(f"Dataset: {dataset_name} | Top-K: {top_k} | Alpha: {alpha} | Epsilon: {epsilon} | Step Size: {step_size}")
    logger.info(f"Embedding: {embed_name} | LLM: {llm_name}")
    logger.info("=" * 60)

    embedding_model = get_embedding_model(embed_name, dimension=embed_dim)
    llm_client = LLMClient(model_name=llm_name)
    store = GraphStore(cache_dir=cfg.get("cache", {}).get("graphs_cache", ".cache/graphs"))

    datasets_to_run = ["musique", "hotpotqa", "2wikimultihopqa"] if dataset_name == "all" else [dataset_name]

    for ds in datasets_to_run:
        items = get_dataset_items(ds, data_dir)
        p_dict = {}

        def build_retriever_for_item(item):
            # Try to load cached graph or build
            peg_name = f"{ds}_item_{item.id}_peg"
            peg = store.load(peg_name)
            if peg is None:
                peg = PassageEntityGraph(
                    embedding_model=embedding_model,
                    include_fact_edges=cfg.get("include_fact_edges", True),
                    include_synonymy_edges=cfg.get("include_synonymy_edges", True),
                    synonymy_threshold=cfg.get("synonymy_threshold", 0.80)
                )
                peg.build_from_passages(item.passages)
                store.save(peg, peg_name)

            p_map = {p.id: p for p in item.passages}
            return QAFDFlowDiffusionRetriever(
                graph=peg.graph,
                passages=p_map,
                embedding_model=embedding_model,
                alpha=alpha,
                epsilon=epsilon,
                step_size=step_size,
                max_iterations=cfg.get("max_diffusion_iterations", 100),
                damping_factor=cfg.get("damping_factor", 0.85)
            )

        evaluator = PipelineEvaluator(
            llm_client=llm_client,
            top_k=top_k,
            results_dir=cfg.get("output", {}).get("tables_dir", "results/tables")
        )

        summary = evaluator.evaluate_pipeline(
            items=items,
            retriever_builder_func=build_retriever_for_item,
            pipeline_name=f"QAFD_RAG_{ds.upper()}"
        )

        print(f"\n--- {ds.upper()} Results ---")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k:22s}: {v:.4f}")
            else:
                print(f"  {k:22s}: {v}")


if __name__ == "__main__":
    main()
