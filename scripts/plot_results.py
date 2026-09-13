"""
Publication-Quality Visualization Generator for QAFD-RAG.

Generates publication-ready figures for:
1. Baseline Comparison (F1, Exact Match across datasets)
2. Retrieval Metrics (MRR, Recall@1, Recall@2, Recall@5)
3. Latency Comparison (Retrieval Latency vs Methods)
4. Ablation Study Breakdown
5. Flow-Diffusion Parameter Sensitivity (alpha, epsilon, step_size)
6. Example Multi-Hop Graph Path Visualization (Flow intensities and node types)

Saved to results/figures/ as high-resolution (300 DPI) PNG and PDF files.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from data.loaders import MuSiQueLoader
from graph import PassageEntityGraph
from embeddings import get_embedding_model
from retrieval import QAFDFlowDiffusionRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plot_results")

# Set publication-quality style
sns.set_theme(style="whitegrid", font="sans-serif", font_scale=1.1)
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})


def plot_baseline_comparison(tables_dir: str, figures_dir: str):
    csv_file = os.path.join(tables_dir, "consolidated_benchmark_evaluation.csv")
    if not os.path.exists(csv_file):
        logger.warning(f"{csv_file} not found. Skipping baseline comparison plot.")
        return

    data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    datasets = sorted(list({r["dataset"] for r in data}))
    methods = ["VectorRAG", "GraphRAG", "LightRAG", "QAFD-RAG"]
    
    # Plot 1: F1 and EM across datasets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    x = np.arange(len(datasets))
    width = 0.2
    palette = ["#4A90E2", "#50E3C2", "#F5A623", "#E94E77"]

    for i, m in enumerate(methods):
        em_vals = [float(next(r["exact_match"] for r in data if r["dataset"] == ds and r["method"] == m)) for ds in datasets]
        f1_vals = [float(next(r["f1"] for r in data if r["dataset"] == ds and r["method"] == m)) for ds in datasets]
        ax1.bar(x + i * width - 0.3, em_vals, width, label=m, color=palette[i], edgecolor="black", linewidth=0.5)
        ax2.bar(x + i * width - 0.3, f1_vals, width, label=m, color=palette[i], edgecolor="black", linewidth=0.5)

    ax1.set_title("Exact Match (EM) by Dataset", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylabel("Exact Match Score")
    ax1.set_ylim(0, 1.15)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none")

    ax2.set_title("F1 Score by Dataset", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_ylabel("Token F1 Score")
    ax2.set_ylim(0, 1.15)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none")

    fig_path = os.path.join(figures_dir, "baseline_qa_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig_path}")

    # Plot 2: Retrieval MRR and Recall
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    for i, m in enumerate(methods):
        mrr_vals = [float(next(r["mrr"] for r in data if r["dataset"] == ds and r["method"] == m)) for ds in datasets]
        r2_vals = [float(next(r["recall@2"] for r in data if r["dataset"] == ds and r["method"] == m)) for ds in datasets]
        ax1.bar(x + i * width - 0.3, mrr_vals, width, label=m, color=palette[i], edgecolor="black", linewidth=0.5)
        ax2.bar(x + i * width - 0.3, r2_vals, width, label=m, color=palette[i], edgecolor="black", linewidth=0.5)

    ax1.set_title("Mean Reciprocal Rank (MRR)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylabel("MRR")
    ax1.set_ylim(0, 1.15)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none")

    ax2.set_title("Multi-Hop Recall@2", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_ylabel("Recall@2")
    ax2.set_ylim(0, 1.15)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none")

    fig_path = os.path.join(figures_dir, "retrieval_metrics_mrr_recall.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig_path}")

    # Plot 3: Latency comparison
    plt.figure(figsize=(8, 5), dpi=300)
    method_latencies = {}
    for m in methods:
        lats = [float(r["retrieval_latency_ms"]) for r in data if r["method"] == m]
        method_latencies[m] = np.mean(lats)

    bars = plt.bar(method_latencies.keys(), method_latencies.values(), color=palette, edgecolor="black", linewidth=0.8, width=0.5)
    plt.title("Retrieval Latency Comparison", fontsize=13, fontweight="bold")
    plt.ylabel("Mean Latency (ms)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} ms", ha="center", va="bottom", fontsize=10)
    plt.ylim(0, max(method_latencies.values()) * 1.3)
    fig_path = os.path.join(figures_dir, "latency_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig_path}")


def plot_ablation_study(tables_dir: str, figures_dir: str):
    csv_file = os.path.join(tables_dir, "ablation_study_musique.csv")
    if not os.path.exists(csv_file):
        return

    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    labels = [r["Ablation"].replace(" (Default)", "") for r in rows]
    mrrs = [float(r["mrr"]) for r in rows]
    r2s = [float(r["recall@2"]) for r in rows]
    steps = [float(r["avg_diffusion_steps"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=300)
    y_pos = np.arange(len(labels))

    ax1.barh(y_pos - 0.15, mrrs, height=0.3, label="MRR", color="#4A90E2", edgecolor="black", linewidth=0.5)
    ax1.barh(y_pos + 0.15, r2s, height=0.3, label="Recall@2", color="#50E3C2", edgecolor="black", linewidth=0.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Metric Value")
    ax1.set_xlim(0, 1.2)
    ax1.set_title("QAFD-RAG Architectural Ablation Breakdown", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right")

    fig_path = os.path.join(figures_dir, "ablation_study_chart.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig_path}")


def plot_parameter_sensitivity(tables_dir: str, figures_dir: str):
    csv_file = os.path.join(tables_dir, "parameter_sensitivity_musique.csv")
    if not os.path.exists(csv_file):
        return

    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Plot impact of step_size and epsilon
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Filter alpha variations
    alpha_rows = [r for r in rows if float(r["step_size"]) == 0.2 and float(r["epsilon"]) == 0.01]
    alphas = [float(r["alpha"]) for r in alpha_rows]
    alpha_steps = [float(r["avg_diffusion_steps"]) for r in alpha_rows]
    alpha_r1 = [float(r["recall@1"]) for r in alpha_rows]

    ax1.plot(alphas, alpha_steps, marker="o", color="#E94E77", linewidth=2, label="Diffusion Steps")
    ax1.set_xlabel("Alpha (Potential Gradient Multiplier)")
    ax1.set_ylabel("Average Diffusion Steps")
    ax1.set_title("Effect of Alpha on Convergence", fontsize=12, fontweight="bold")
    ax1.grid(True)

    # Step size variations
    step_rows = [r for r in rows if float(r["alpha"]) == 2.0 and float(r["epsilon"]) == 0.01]
    step_sizes = [float(r["step_size"]) for r in step_rows]
    step_lats = [float(r["retrieval_latency_ms"]) for r in step_rows]

    ax2.plot(step_sizes, step_lats, marker="s", color="#4A90E2", linewidth=2, label="Latency (ms)")
    ax2.set_xlabel("Step Size (Push Scaling eta)")
    ax2.set_ylabel("Retrieval Latency (ms)")
    ax2.set_title("Effect of Step Size on Latency", fontsize=12, fontweight="bold")
    ax2.grid(True)

    fig_path = os.path.join(figures_dir, "parameter_sensitivity_analysis.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig_path}")


def plot_multihop_graph_flow_path(figures_dir: str):
    """Generates an illustrative graph diagram of QAFD flow diffusion on an authentic multi-hop question."""
    items = MuSiQueLoader("data/raw/musique/sample.jsonl").load()
    item = items[0]  # "What is the capital of the country where the inventor of the telephone was born?"

    emb = get_embedding_model("local-hash-embed", dimension=128)
    peg = PassageEntityGraph(embedding_model=emb)
    peg.build_from_passages(item.passages)
    p_map = {p.id: p for p in item.passages}

    retriever = QAFDFlowDiffusionRetriever(peg.graph, p_map, emb, alpha=2.0, epsilon=0.01, step_size=0.2)
    e, h, seeds = retriever._initialize_query_flow(item.question)
    absorbed_flow, flows, _ = retriever._run_push_relabel_diffusion(e, h)

    # Create visualization subgraph of top active nodes
    active_nodes = [n for n in peg.graph.nodes() if absorbed_flow.get(n, 0.0) > 0.01 or n in seeds or peg.graph.nodes[n].get("node_type") == "passage"]
    subg = peg.graph.subgraph(active_nodes[:12])

    plt.figure(figsize=(12, 8), dpi=300)
    pos = nx.spring_layout(subg, seed=42, k=1.2)

    # Node color by type
    node_colors = []
    labels = {}
    for n in subg.nodes():
        nd = subg.nodes[n]
        labels[n] = nd.get("display_name", str(n))
        if n in seeds:
            node_colors.append("#FF6B6B")  # Red for query seed
        elif nd.get("node_type") == "passage":
            if nd.get("is_supporting"):
                node_colors.append("#51CF66")  # Green for supporting passage
            else:
                node_colors.append("#CED4DA")  # Gray for distractor passage
        else:
            node_colors.append("#339AF0")  # Blue for entity

    # Edge widths by flow
    edge_widths = []
    for u, v in subg.edges():
        f_val = flows.get((u, v), 0.0)
        edge_widths.append(max(0.8, min(f_val * 8.0, 5.0)))

    nx.draw_networkx_nodes(subg, pos, node_color=node_colors, node_size=1800, alpha=0.9, edgecolors="#222222", linewidths=1.5)
    nx.draw_networkx_edges(subg, pos, width=edge_widths, edge_color="#495057", arrows=True, arrowsize=18, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(subg, pos, labels=labels, font_size=8, font_weight="bold")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Query Seed Entity', markerfacecolor='#FF6B6B', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Intermediate Entity Node', markerfacecolor='#339AF0', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Gold Supporting Passage', markerfacecolor='#51CF66', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Distractor Passage', markerfacecolor='#CED4DA', markersize=12),
        plt.Line2D([0], [0], color='#495057', lw=3, label='Flow Diffusion Channel')
    ]
    plt.legend(handles=legend_elements, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.title(f"QAFD-RAG Flow Diffusion Reasoning Path\nQuestion: '{item.question}'", fontsize=12, fontweight="bold", pad=15)
    plt.axis('off')

    fig_path = os.path.join(figures_dir, "multihop_graph_flow_path.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig_path}")


def main():
    tables_dir = "results/tables"
    figures_dir = "results/figures"
    os.makedirs(figures_dir, exist_ok=True)

    logger.info("Generating publication-quality visualization figures...")
    plot_baseline_comparison(tables_dir, figures_dir)
    plot_ablation_study(tables_dir, figures_dir)
    plot_parameter_sensitivity(tables_dir, figures_dir)
    plot_multihop_graph_flow_path(figures_dir)
    logger.info("All visualization figures successfully generated.")


if __name__ == "__main__":
    main()
