"""
Multi-hop reasoning path extractor.
Traces maximal flow paths through the knowledge graph from query seeds to retrieved passages.
Supports paths like: Query -> Entity A -> Relation -> Entity B -> Passage -> Entity C -> Supporting Passage.
"""

import networkx as nx
from typing import List, Dict, Tuple, Set, Optional


class MultiHopPathExtractor:
    """Extracts interpretable multi-hop paths guided by flow diffusion intensities."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def trace_paths(
        self,
        seeds: List[str],
        target_passages: List[str],
        edge_flows: Dict[Tuple[str, str], float],
        max_depth: int = 3
    ) -> List[str]:
        """
        Traces paths from seed nodes to target passage nodes prioritizing high-flow edges.
        Returns formatted strings representing the reasoning trajectories.
        """
        formatted_paths: List[str] = []
        seed_set = set(seeds)

        for p_node in target_passages:
            if not self.graph.has_node(p_node):
                continue

            # Greedy backward or forward search along maximum flow
            # Let's search backward from target passage to find best reaching seed
            best_path = None
            best_flow = -1.0

            # Find candidate paths using BFS up to max_depth
            queue = [([p_node], 1.0)]
            visited = {p_node}

            while queue:
                curr_path, curr_bottleneck = queue.pop(0)
                curr_node = curr_path[-1]

                if curr_node in seed_set and len(curr_path) > 1:
                    if curr_bottleneck > best_flow:
                        best_flow = curr_bottleneck
                        best_path = list(reversed(curr_path))
                    continue

                if len(curr_path) > max_depth + 1:
                    continue

                # Inspect incoming neighbors (predecessors)
                preds = list(self.graph.predecessors(curr_node))
                # Sort by flow pushed from pred to curr_node
                scored_preds = []
                for pred in preds:
                    if pred not in visited:
                        flow_val = edge_flows.get((pred, curr_node), 0.0)
                        scored_preds.append((pred, flow_val))
                scored_preds.sort(key=lambda x: x[1], reverse=True)

                for pred, f_val in scored_preds[:3]:
                    visited.add(pred)
                    queue.append((curr_path + [pred], min(curr_bottleneck, max(f_val, 0.01))))

            if best_path:
                path_str = self._format_path(best_path, edge_flows)
                formatted_paths.append(path_str)
            else:
                # Fallback: direct association
                display = self.graph.nodes[p_node].get("display_name", p_node)
                formatted_paths.append(f"Query -> [Direct Retrieval] -> {display}")

        return formatted_paths

    def _format_path(self, path_nodes: List[str], edge_flows: Dict[Tuple[str, str], float]) -> str:
        """Formats node sequence into a readable multi-hop trace."""
        segments = ["Query"]
        for i in range(len(path_nodes)):
            u = path_nodes[i]
            u_name = self.graph.nodes[u].get("display_name", u)
            u_type = self.graph.nodes[u].get("node_type", "node")
            
            if i > 0:
                prev = path_nodes[i - 1]
                edge_data = self.graph.get_edge_data(prev, u, default={})
                rel = edge_data.get("relation", edge_data.get("edge_type", "connected_to"))
                flow_val = edge_flows.get((prev, u), 0.0)
                segments.append(f"--[{rel} | flow={flow_val:.3f}]-->")

            segments.append(f"({u_type}: {u_name})")

        return " ".join(segments)
