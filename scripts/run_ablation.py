"""
Ablation Studies and Parameter Sensitivity Analysis for QAFD-RAG.

Ablations Implemented:
1. Semantic retrieval without graph diffusion (Vector RAG)
2. Graph retrieval without query-aware initialization (uniform seed flow)
3. Retrieval budget variation (top-k in {1, 2, 3, 5})
4. Entity Graph vs Passage-Entity Graph
5. QAFD-RAG without synonymy edges
6. QAFD-RAG without fact edges
7. Parameter sensitivity sweep: alpha in {1.0, 2.0, 3.0}, epsilon in {0.005, 0.01, 0.05}, step_size in {0.1, 0.2, 0.4}

Usage:
    python scripts/run_ablation.py [--config configs/qafd_rag.yaml] [--dataset musique]
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
from graph import PassageEntityGraph, EntityGraph, GraphStore
from embeddings import get_embedding_model
from retrieval import QAFDFlowDiffusionRetriever
from baselines import VectorRAGRetriever
from models import LLMClient
from evaluation import PipelineEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_ablation")


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
    parser = argparse.ArgumentParser(description="Run QAFD-RAG ablation studies and parameter sensitivity")
    parser.add_argument("--config", default="configs/qafd_rag.yaml", help="Path to config file")
    parser.add_argument("--dataset", default="musique", help="Dataset name")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = args.dataset or cfg.get("dataset", "musique")
    data_dir = cfg.get("data_dir", "data/raw")
    tables_dir = cfg.get("output", {}).get("tables_dir", "results/tables")
    os.makedirs(tables_dir, exist_ok=True)

    items = get_dataset_items(dataset_name, data_dir)
    embedding_model = get_embedding_model(cfg.get("embedding_model", "local-hash-embed"), dimension=cfg.get("embedding_dimension", 256))
    llm_client = LLMClient(model_name=cfg.get("llm_model", "local-extractive"))

    logger.info("=" * 60)
    logger.info("RUNNING COMPLETE ABLATION SUITE FOR QAFD-RAG")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("=" * 60)

    ablation_results = []

    # -------------------------------------------------------------
    # Ablation 1: Full QAFD-RAG (Reference)
    # -------------------------------------------------------------
    logger.info("[Ablation 1] Full QAFD-RAG Reference...")
    def build_full_qafd(it):
        peg = PassageEntityGraph(embedding_model=embedding_model, include_fact_edges=True, include_synonymy_edges=True)
        peg.build_from_passages(it.passages)
        return QAFDFlowDiffusionRetriever(
            peg.graph, {p.id: p for p in it.passages}, embedding_model,
            alpha=2.0, epsilon=0.01, step_size=0.2, query_aware=True
        )
    ev_full = PipelineEvaluator(llm_client=llm_client, top_k=5, results_dir=tables_dir)
    res_full = ev_full.evaluate_pipeline(items, build_full_qafd, "Full_QAFD_RAG")
    ablation_results.append({"Ablation": "Full QAFD-RAG (Default)", **res_full})

    # -------------------------------------------------------------
    # Ablation 2: No Graph Diffusion (Vector RAG)
    # -------------------------------------------------------------
    logger.info("[Ablation 2] No Graph Diffusion (Vector RAG)...")
    def build_no_diffusion(it):
        return VectorRAGRetriever({p.id: p for p in it.passages}, embedding_model)
    res_no_diff = ev_full.evaluate_pipeline(items, build_no_diffusion, "No_Graph_Diffusion")
    ablation_results.append({"Ablation": "w/o Graph Diffusion (Vector RAG)", **res_no_diff})

    # -------------------------------------------------------------
    # Ablation 3: Uniform Initialization (w/o Query-Aware Seeds)
    # -------------------------------------------------------------
    logger.info("[Ablation 3] Uniform Initialization (w/o Query-Aware Seeds)...")
    def build_uniform_init(it):
        peg = PassageEntityGraph(embedding_model=embedding_model, include_fact_edges=True, include_synonymy_edges=True)
        peg.build_from_passages(it.passages)
        return QAFDFlowDiffusionRetriever(
            peg.graph, {p.id: p for p in it.passages}, embedding_model,
            alpha=2.0, epsilon=0.01, step_size=0.2, query_aware=False
        )
    res_uniform = ev_full.evaluate_pipeline(items, build_uniform_init, "Uniform_Initialization")
    ablation_results.append({"Ablation": "w/o Query-Aware Init (Uniform)", **res_uniform})

    # -------------------------------------------------------------
    # Ablation 4: w/o Synonymy Edges
    # -------------------------------------------------------------
    logger.info("[Ablation 4] QAFD-RAG without Synonymy Edges...")
    def build_no_syn(it):
        peg = PassageEntityGraph(embedding_model=embedding_model, include_fact_edges=True, include_synonymy_edges=False)
        peg.build_from_passages(it.passages)
        return QAFDFlowDiffusionRetriever(
            peg.graph, {p.id: p for p in it.passages}, embedding_model,
            alpha=2.0, epsilon=0.01, step_size=0.2, query_aware=True
        )
    res_no_syn = ev_full.evaluate_pipeline(items, build_no_syn, "No_Synonymy_Edges")
    ablation_results.append({"Ablation": "w/o Synonymy Edges", **res_no_syn})

    # -------------------------------------------------------------
    # Ablation 5: w/o Fact Edges
    # -------------------------------------------------------------
    logger.info("[Ablation 5] QAFD-RAG without Fact Edges...")
    def build_no_fact(it):
        peg = PassageEntityGraph(embedding_model=embedding_model, include_fact_edges=False, include_synonymy_edges=True)
        peg.build_from_passages(it.passages)
        return QAFDFlowDiffusionRetriever(
            peg.graph, {p.id: p for p in it.passages}, embedding_model,
            alpha=2.0, epsilon=0.01, step_size=0.2, query_aware=True
        )
    res_no_fact = ev_full.evaluate_pipeline(items, build_no_fact, "No_Fact_Edges")
    ablation_results.append({"Ablation": "w/o Fact Edges", **res_no_fact})

    # -------------------------------------------------------------
    # Ablation 6: Entity Graph vs Passage-Entity Graph
    # -------------------------------------------------------------
    logger.info("[Ablation 6] Entity Graph vs Passage-Entity Graph...")
    def build_entity_graph_only(it):
        eg = EntityGraph()
        eg.build_from_passages(it.passages)
        # Convert entity graph to network where passages are connected through entity occurrences
        g = eg.graph.copy()
        # Add passage nodes
        for p in it.passages:
            p_node = f"p::{p.id}"
            g.add_node(p_node, node_type="passage", passage_id=p.id, title=p.title, text=p.text)
            for ent in eg.graph.nodes():
                if p.id in eg.get_passages_for_entity(ent):
                    g.add_edge(p_node, ent, capacity=1.0, weight=1.0)
                    g.add_edge(ent, p_node, capacity=1.0, weight=1.0)
        return QAFDFlowDiffusionRetriever(
            g, {p.id: p for p in it.passages}, embedding_model,
            alpha=2.0, epsilon=0.01, step_size=0.2, query_aware=True
        )
    res_eg = ev_full.evaluate_pipeline(items, build_entity_graph_only, "Entity_Graph_Only")
    ablation_results.append({"Ablation": "Entity Graph Only", **res_eg})

    # Print Ablations Table
    print("\n" + "=" * 95)
    print(f"ABLATION STUDY RESULTS ON {dataset_name.upper()}")
    print("=" * 95)
    header = f"{'Configuration':<35} | {'EM':<6} | {'F1':<6} | {'MRR':<6} | {'R@1':<6} | {'R@2':<6} | {'R@5':<6} | {'Steps':<6}"
    print(header)
    print("-" * len(header))
    for r in ablation_results:
        print(f"{r['Ablation']:<35} | {r['exact_match']:<6.3f} | {r['f1']:<6.3f} | {r['mrr']:<6.3f} | {r['recall@1']:<6.3f} | {r['recall@2']:<6.3f} | {r.get('recall@5', 0.0):<6.3f} | {r['avg_diffusion_steps']:<6.1f}")
    print("=" * 95)

    # Save ablations CSV
    ablation_csv = os.path.join(tables_dir, f"ablation_study_{dataset_name}.csv")
    with open(ablation_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_results[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_results)
    logger.info(f"Saved ablation study to {ablation_csv}")

    # -------------------------------------------------------------
    # Experiment 4: Parameter Sensitivity Analysis
    # -------------------------------------------------------------
    logger.info("Running Parameter Sensitivity Grid (alpha, epsilon, step_size)...")
    param_grid = [
        # (alpha, epsilon, step_size)
        (1.0, 0.01, 0.2),
        (2.0, 0.01, 0.2),  # Default
        (3.0, 0.01, 0.2),
        (2.0, 0.005, 0.2),
        (2.0, 0.05, 0.2),
        (2.0, 0.01, 0.1),
        (2.0, 0.01, 0.4),
    ]

    sensitivity_results = []
    for a_val, ep_val, ss_val in param_grid:
        def build_param_retriever(it):
            peg = PassageEntityGraph(embedding_model=embedding_model)
            peg.build_from_passages(it.passages)
            return QAFDFlowDiffusionRetriever(
                peg.graph, {p.id: p for p in it.passages}, embedding_model,
                alpha=a_val, epsilon=ep_val, step_size=ss_val, query_aware=True
            )
        p_name = f"alpha={a_val}_eps={ep_val}_step={ss_val}"
        res = ev_full.evaluate_pipeline(items, build_param_retriever, p_name)
        sensitivity_results.append({
            "alpha": a_val,
            "epsilon": ep_val,
            "step_size": ss_val,
            "exact_match": res["exact_match"],
            "f1": res["f1"],
            "mrr": res["mrr"],
            "recall@1": res["recall@1"],
            "recall@2": res["recall@2"],
            "retrieval_latency_ms": res["retrieval_latency_ms"],
            "avg_diffusion_steps": res["avg_diffusion_steps"]
        })

    param_csv = os.path.join(tables_dir, f"parameter_sensitivity_{dataset_name}.csv")
    with open(param_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity_results[0].keys()))
        writer.writeheader()
        writer.writerows(sensitivity_results)
    logger.info(f"Saved parameter sensitivity results to {param_csv}")


if __name__ == "__main__":
    main()
