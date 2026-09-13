"""
Script to build and cache Entity Graphs and Passage-Entity Graphs.
Usage:
    python scripts/build_graph.py [--config configs/qafd_rag.yaml] [--dataset musique]
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import logging
from data.loaders import MuSiQueLoader, HotpotQALoader, TwoWikiLoader
from graph import EntityGraph, PassageEntityGraph, GraphStore
from embeddings import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_graph")


def load_config(config_path: str = "configs/qafd_rag.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_loader(dataset_name: str, data_dir: str = "data/raw"):
    ds = dataset_name.lower().strip()
    if ds == "musique":
        return MuSiQueLoader(os.path.join(data_dir, "musique/sample.jsonl"))
    elif ds == "hotpotqa":
        return HotpotQALoader(os.path.join(data_dir, "hotpotqa/sample.json"))
    elif ds == "2wikimultihopqa" or ds == "2wiki":
        return TwoWikiLoader(os.path.join(data_dir, "2wikimultihopqa/sample.json"))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Build and cache knowledge graphs")
    parser.add_argument("--config", default="configs/qafd_rag.yaml", help="Path to config file")
    parser.add_argument("--dataset", default=None, help="Dataset name (musique, hotpotqa, 2wikimultihopqa, all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or cfg.get("dataset", "musique")
    data_dir = cfg.get("data_dir", "data/raw")
    embed_model_name = cfg.get("embedding_model", "local-hash-embed")
    embed_dim = cfg.get("embedding_dimension", 256)

    store = GraphStore(cache_dir=cfg.get("cache", {}).get("graphs_cache", ".cache/graphs"))
    embedding_model = get_embedding_model(embed_model_name, dimension=embed_dim)

    datasets_to_build = ["musique", "hotpotqa", "2wikimultihopqa"] if dataset == "all" else [dataset]

    for ds_name in datasets_to_build:
        logger.info(f"Loading dataset: {ds_name}...")
        loader = get_loader(ds_name, data_dir)
        items = loader.load()
        logger.info(f"Loaded {len(items)} samples from {ds_name}.")

        for idx, item in enumerate(items):
            # 1. Entity Graph
            eg_name = f"{ds_name}_item_{item.id}_entity_graph"
            if not store.exists(eg_name):
                eg = EntityGraph()
                eg.build_from_passages(item.passages)
                store.save(eg, eg_name)
                logger.info(f"Built EntityGraph for {item.id}: {eg.num_nodes} nodes, {eg.num_edges} edges")

            # 2. Passage-Entity Graph
            peg_name = f"{ds_name}_item_{item.id}_peg"
            if not store.exists(peg_name):
                peg = PassageEntityGraph(
                    embedding_model=embedding_model,
                    include_fact_edges=cfg.get("include_fact_edges", True),
                    include_synonymy_edges=cfg.get("include_synonymy_edges", True),
                    synonymy_threshold=cfg.get("synonymy_threshold", 0.80)
                )
                peg.build_from_passages(item.passages)
                store.save(peg, peg_name)
                logger.info(f"Built PassageEntityGraph for {item.id}: {peg.num_nodes} nodes, {peg.num_edges} edges")

    logger.info("Graph construction and caching completed successfully.")


if __name__ == "__main__":
    main()
