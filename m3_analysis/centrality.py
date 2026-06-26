#!/usr/bin/env python3
"""
Module 7 — Betweenness Centrality
Computes shortest-path Brandes betweenness centrality for infrastructure nodes and edges.
Identifies gatekeeper intersections, critical chokepoints, and Single Points of Failure (articulation cut vertices).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import pickle
import json
import numpy as np
import networkx as nx

def compute_betweenness_centrality(graph: nx.Graph, weight: str = "weight") -> dict:
    """
    Calculates node and edge betweenness centrality and identifies structural vulnerabilities.
    """
    if graph.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "gatekeepers": [],
            "spofs": [],
            "nodes": {},
            "edges": {}
        }
        
    # Shortest path node betweenness
    node_bc = nx.betweenness_centrality(graph, weight=weight, normalized=True)
    # Shortest path edge betweenness (Bonus feature)
    edge_bc = nx.edge_betweenness_centrality(graph, weight=weight, normalized=True)
    
    # Articulation points (cut vertices) -> Single Points of Failure
    spof_nodes = list(nx.articulation_points(graph))
    
    # Threshold for gatekeepers (top 15% or score > 0.05)
    bc_values = list(node_bc.values())
    thresh = float(np.percentile(bc_values, 85)) if bc_values else 0.0
    gatekeepers = [n for n, c in node_bc.items() if c >= max(thresh, 0.05)]
    
    # Enrich node details
    enriched_nodes = {}
    for n, score in node_bc.items():
        n_data = graph.nodes[n]
        enriched_nodes[str(n)] = {
            "node_id": n,
            "x": n_data.get("x", 0),
            "y": n_data.get("y", 0),
            "degree": n_data.get("degree", graph.degree[n]),
            "betweenness": round(score, 6),
            "is_spof": n in spof_nodes,
            "is_gatekeeper": n in gatekeepers
        }
        
    enriched_edges = {}
    for (u, v), score in edge_bc.items():
        edge_key = f"{u}-{v}"
        e_data = graph[u][v]
        enriched_edges[edge_key] = {
            "source": u,
            "target": v,
            "length": round(e_data.get("length", 0.0), 2),
            "edge_betweenness": round(score, 6)
        }
        
    report = {
        "num_nodes": len(enriched_nodes),
        "num_spofs": len(spof_nodes),
        "num_gatekeepers": len(gatekeepers),
        "spof_node_ids": spof_nodes,
        "gatekeeper_node_ids": gatekeepers,
        "nodes": enriched_nodes,
        "edges": enriched_edges
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="Standalone Member 3 Network Analysis Engine")
    parser.add_argument('--source', type=str, default="m2", help="Source pipeline indicator")
    parser.add_argument('--m2_file', type=str, default=None, help="Input healed M2 graph pickle file")
    parser.add_argument('--input_graph', type=str, default=None, help="Input graph pickle file")
    parser.add_argument('--output_dir', type=str, default="dashboard_exports/", help="Output directory")
    args = parser.parse_args()
    
    graph_path = args.m2_file or args.input_graph or "dashboard_exports/healed_road_graph.gpickle"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Cannot find input graph file: {graph_path}")
        
    from m3_analysis import run_standalone_m3
    run_standalone_m3(graph_path, args.output_dir)

if __name__ == "__main__":
    main()
