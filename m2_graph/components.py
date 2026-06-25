#!/usr/bin/env python3
"""
Module 3 — Connected Component Analysis
Evaluates topological fragmentation in road networks caused by occlusions.
Calculates component counts, largest connected component (LCC) metrics, and statistical distributions.
"""

import os
import argparse
import pickle
import json
import numpy as np
import networkx as nx

def analyze_connected_components(graph: nx.Graph) -> dict:
    """
    Analyzes connected components of a NetworkX road graph.
    
    Returns dictionary matching dashboard JSON contracts.
    """
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    
    if total_nodes == 0:
        return {
            "num_connected_components": 0,
            "largest_component_nodes": 0,
            "largest_component_edges": 0,
            "largest_component_ratio": 0.0,
            "component_node_sizes": [],
            "stats": {"mean": 0, "min": 0, "max": 0, "median": 0}
        }
        
    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)
    
    comp_sizes = [len(c) for c in components]
    lcc_nodes = set(components[0])
    lcc_subgraph = graph.subgraph(lcc_nodes)
    
    report = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "num_connected_components": len(components),
        "largest_component_nodes": len(lcc_nodes),
        "largest_component_edges": lcc_subgraph.number_of_edges(),
        "largest_component_ratio": round(len(lcc_nodes) / total_nodes, 4),
        "component_node_sizes": comp_sizes,
        "stats": {
            "mean": round(float(np.mean(comp_sizes)), 2),
            "min": int(np.min(comp_sizes)),
            "max": int(np.max(comp_sizes)),
            "median": round(float(np.median(comp_sizes)), 2)
        }
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="M2 Module 3: Connected Component Analysis")
    parser.add_argument('--input_graph', type=str, required=True, help="Input graph pickle")
    parser.add_argument('--output_report', type=str, default="connectivity_report.json", help="Output JSON report")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_graph):
        raise FileNotFoundError(f"Input graph not found: {args.input_graph}")
        
    with open(args.input_graph, 'rb') as f:
        graph = pickle.load(f)
        
    report = analyze_connected_components(graph)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    with open(args.output_report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Connectivity report saved to: {os.path.abspath(args.output_report)}")

if __name__ == "__main__":
    main()
