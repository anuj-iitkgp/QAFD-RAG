"""
FastAPI Web Application Server for QAFD-RAG.
Provides interactive APIs for multi-hop question answering, flow diffusion visualization,
comparative baselines, and benchmark analytics.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from data.loaders import MuSiQueLoader, HotpotQALoader, TwoWikiLoader, QAItem, Passage
from graph import PassageEntityGraph, EntityGraph, GraphStore
from embeddings import get_embedding_model
from retrieval import QAFDFlowDiffusionRetriever
from baselines import VectorRAGRetriever, GraphRAGRetriever, LightRAGRetriever
from models import LLMClient
from evaluation import compute_qa_metrics, compute_retrieval_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qafd_web_app")

app = FastAPI(
    title="QAFD-RAG Interactive Web Application",
    description="Query-Aware Flow Diffusion for Graph-Based Retrieval-Augmented Generation",
    version="1.0.0"
)

# Global cache of datasets and precomputed graphs
DATASETS: Dict[str, List[QAItem]] = {}
GRAPHS: Dict[str, Any] = {}
EMBEDDING_MODEL = None
LLM_CLIENT = None
STORE = None


def initialize_app_resources():
    global DATASETS, EMBEDDING_MODEL, LLM_CLIENT, STORE
    data_dir = "data/raw"
    try:
        DATASETS["musique"] = MuSiQueLoader(os.path.join(data_dir, "musique/sample.jsonl")).load()
    except Exception:
        DATASETS["musique"] = []
    try:
        DATASETS["hotpotqa"] = HotpotQALoader(os.path.join(data_dir, "hotpotqa/sample.json")).load()
    except Exception:
        DATASETS["hotpotqa"] = []
    try:
        DATASETS["2wikimultihopqa"] = TwoWikiLoader(os.path.join(data_dir, "2wikimultihopqa/sample.json")).load()
    except Exception:
        DATASETS["2wikimultihopqa"] = []

    EMBEDDING_MODEL = get_embedding_model("local-hash-embed", dimension=128)
    LLM_CLIENT = LLMClient(model_name="local-extractive")
    STORE = GraphStore(".cache/graphs")
    logger.info("Web application resources initialized successfully.")


# Initialize immediately
initialize_app_resources()


@app.on_event("startup")
def on_startup():
    initialize_app_resources()


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class QueryRequest(BaseModel):
    dataset: str = "musique"
    sample_id: Optional[str] = None
    question: Optional[str] = None
    custom_passages: Optional[List[Dict[str, str]]] = None
    method: str = "QAFD-RAG"  # QAFD-RAG, VectorRAG, GraphRAG, LightRAG
    alpha: float = 2.0
    epsilon: float = 0.01
    step_size: float = 0.2
    top_k: int = 3
    query_aware: bool = True


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/api/samples")
def get_samples():
    """Returns list of preloaded benchmark questions for quick user selection."""
    samples = []
    for ds_name, items in DATASETS.items():
        for it in items:
            samples.append({
                "dataset": ds_name,
                "id": it.id,
                "question": it.question,
                "gold_answer": it.answer,
                "num_passages": len(it.passages),
                "supporting_count": len(it.get_gold_passage_ids())
            })
    return {"samples": samples}


@app.post("/api/run")
def run_query(req: QueryRequest):
    """Executes retrieval and question answering with flow graph extraction."""
    start_total = time.perf_counter()

    # 1. Resolve Question and Passages
    item = None
    passages = []
    gold_answer = ""
    gold_pids = set()

    if req.sample_id:
        # Find preset sample
        for ds_name, items in DATASETS.items():
            for it in items:
                if it.id == req.sample_id:
                    item = it
                    passages = it.passages
                    gold_answer = it.answer
                    gold_pids = it.get_gold_passage_ids()
                    question = it.question
                    break
            if item:
                break
        if not item:
            raise HTTPException(status_code=404, detail="Sample ID not found")
    elif req.question and req.custom_passages:
        question = req.question
        passages = [
            Passage(
                id=f"cust_{i}",
                title=p.get("title", f"Passage {i}"),
                text=p.get("text", ""),
                is_supporting=bool(p.get("is_supporting", False))
            )
            for i, p in enumerate(req.custom_passages)
        ]
        gold_answer = "N/A (Custom Query)"
    else:
        # Default to first sample of dataset
        items = DATASETS.get(req.dataset, DATASETS.get("musique", []))
        if not items:
            raise HTTPException(status_code=400, detail="No samples available in dataset")
        item = items[0]
        passages = item.passages
        gold_answer = item.answer
        gold_pids = item.get_gold_passage_ids()
        question = item.question

    p_map = {p.id: p for p in passages}

    # 2. Build or Load Heterogeneous Graph
    peg_name = f"{req.dataset}_item_{item.id}_peg" if item else f"custom_{int(time.time())}"
    peg = STORE.load(peg_name) if item else None
    if peg is None:
        peg = PassageEntityGraph(embedding_model=EMBEDDING_MODEL, synonymy_threshold=0.80)
        peg.build_from_passages(passages)
        if item:
            STORE.save(peg, peg_name)

    # 3. Instantiate Retriever
    ret_start = time.perf_counter()
    edge_flows = {}
    node_flows = {}

    if req.method == "VectorRAG":
        retriever = VectorRAGRetriever(p_map, EMBEDDING_MODEL)
        ret_res = retriever.retrieve(question, top_k=req.top_k)
    elif req.method == "GraphRAG":
        retriever = GraphRAGRetriever(peg.graph, p_map, EMBEDDING_MODEL)
        ret_res = retriever.retrieve(question, top_k=req.top_k)
    elif req.method == "LightRAG":
        retriever = LightRAGRetriever(peg.graph, p_map, EMBEDDING_MODEL)
        ret_res = retriever.retrieve(question, top_k=req.top_k)
    else:
        # Default: QAFD-RAG Flow Diffusion
        retriever = QAFDFlowDiffusionRetriever(
            graph=peg.graph,
            passages=p_map,
            embedding_model=EMBEDDING_MODEL,
            alpha=req.alpha,
            epsilon=req.epsilon,
            step_size=req.step_size,
            query_aware=req.query_aware
        )
        ret_res = retriever.retrieve(question, top_k=req.top_k)
        # Capture internal flow values for visualization
        e, h, seeds = retriever._initialize_query_flow(question)
        absorbed, flows, _ = retriever._run_push_relabel_diffusion(e, h)
        edge_flows = {f"{u}->{v}": float(val) for (u, v), val in flows.items() if val > 0.001}
        node_flows = {n: float(absorbed.get(n, 0.0)) for n in peg.graph.nodes()}

    ret_latency = (time.perf_counter() - ret_start) * 1000.0

    # 4. Generate Answer via LLM
    context_str = "\n\n".join([f"[{p.title}]\n{p.text}" for p in ret_res.retrieved_passages])
    gen_start = time.perf_counter()
    prediction = LLM_CLIENT.answer_question(question, context_str)
    gen_latency = (time.perf_counter() - gen_start) * 1000.0

    # 5. Evaluate Metrics
    qa_eval = compute_qa_metrics(prediction, gold_answer, item.answer_aliases if item else None)
    retrieved_ids = [p.id for p in ret_res.retrieved_passages]
    ret_eval = compute_retrieval_metrics(retrieved_ids, gold_pids, k_values=[1, 2, req.top_k])

    # 6. Build Graph Visualization Data (Nodes & Edges for Vis.js)
    vis_nodes = []
    vis_edges = []
    retrieved_p_nodes = {f"p::{p.id}" for p in ret_res.retrieved_passages}

    for n in peg.graph.nodes():
        data = peg.graph.nodes[n]
        n_type = data.get("node_type", "entity")
        label = data.get("display_name", str(n))
        flow_val = node_flows.get(n, 0.0)

        # Visual styling based on role
        if n in retrieved_p_nodes:
            color = "#51CF66" if data.get("is_supporting") else "#FCC419"  # Green for supporting, yellow for distractor
            shape = "box"
            size = 25
        elif n_type == "passage":
            color = "#CED4DA"
            shape = "box"
            size = 18
        else:
            color = "#339AF0"  # Blue for entity
            shape = "dot"
            size = 14 + min(int(flow_val * 30), 20)

        vis_nodes.append({
            "id": n,
            "label": label[:25] + "..." if len(label) > 25 else label,
            "title": f"Type: {n_type}<br>Name: {label}<br>Flow Score: {flow_val:.3f}",
            "color": color,
            "shape": shape,
            "size": size
        })

    for u, v, edata in peg.graph.edges(data=True):
        rel = edata.get("relation", edata.get("edge_type", ""))
        flow_key = f"{u}->{v}"
        f_val = edge_flows.get(flow_key, 0.0)
        is_high_flow = f_val > 0.02

        vis_edges.append({
            "from": u,
            "to": v,
            "label": rel if is_high_flow else "",
            "title": f"Relation: {rel}<br>Flow: {f_val:.4f}",
            "arrows": "to",
            "width": max(1.0, min(f_val * 8.0, 6.0)),
            "color": {"color": "#E94E77" if is_high_flow else "#ADB5BD", "opacity": 0.8 if is_high_flow else 0.3}
        })

    total_latency = (time.perf_counter() - start_total) * 1000.0

    return {
        "question": question,
        "method": req.method,
        "prediction": prediction,
        "gold_answer": gold_answer,
        "exact_match": qa_eval["exact_match"],
        "f1": qa_eval["f1"],
        "mrr": ret_eval["mrr"],
        "recall_at_k": ret_eval.get(f"recall@{req.top_k}", 0.0),
        "diffusion_steps": ret_res.diffusion_steps,
        "retrieval_latency_ms": round(ret_latency, 2),
        "generation_latency_ms": round(gen_latency, 2),
        "total_latency_ms": round(total_latency, 2),
        "retrieved_passages": [
            {
                "id": p.id,
                "title": p.title,
                "text": p.text,
                "is_supporting": p.is_supporting,
                "score": round(ret_res.passage_scores.get(p.id, 0.0), 4)
            }
            for p in ret_res.retrieved_passages
        ],
        "paths": ret_res.paths,
        "graph_data": {
            "nodes": vis_nodes,
            "edges": vis_edges
        }
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns cross-dataset evaluation summary tables."""
    csv_file = "results/tables/consolidated_benchmark_evaluation.csv"
    if os.path.exists(csv_file):
        import csv
        with open(csv_file, "r", encoding="utf-8") as f:
            return {"benchmarks": list(csv.DictReader(f))}
    return {"benchmarks": []}


@app.get("/api/ablations")
def get_ablations():
    """Returns ablation study results."""
    csv_file = "results/tables/ablation_study_musique.csv"
    if os.path.exists(csv_file):
        import csv
        with open(csv_file, "r", encoding="utf-8") as f:
            return {"ablations": list(csv.DictReader(f))}
    return {"ablations": []}


# Serve static files and frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>QAFD-RAG Web Server Running. Please build static/index.html</h1>"
