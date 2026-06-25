#!/usr/bin/env python3
"""
Module 6 — Connectivity Evaluation
Quantifies topological improvement achieved by road network healing.
Calculates before vs after component statistics and the macro ConnectivityRatio.
"""

import os
import argparse
import pickle
import json
import networkx as nx
from .components import analyze_connected_components

def evaluate_healing_improvement(before_graph: nx.Graph, after_graph: nx.Graph) -> dict:
    """
    Computes connectivity improvement report between fragmented and healed road graphs.
    """
    before_stats = analyze_connected_components(before_graph)
    after_stats = analyze_connected_components(after_graph)
    
    lcc_before = before_stats["largest_component_nodes"]
    lcc_after = after_stats["largest_component_nodes"]
    
    if lcc_before == 0:
        ratio = 0.0
    else:
        ratio = round(lcc_after / lcc_before, 4)
        
    healed_edges = sum(1 for _, _, d in after_graph.edges(data=True) if d.get("is_healed", False))
    
    report = {
        "before_healing": {
            "num_connected_components": before_stats["num_connected_components"],
            "largest_component_nodes": lcc_before,
            "largest_component_ratio": before_stats["largest_component_ratio"]
        },
        "after_healing": {
            "num_connected_components": after_stats["num_connected_components"],
            "largest_component_nodes": lcc_after,
            "largest_component_ratio": after_stats["largest_component_ratio"]
        },
        "healing_metrics": {
            "healed_edges_added": healed_edges,
            "components_merged": before_stats["num_connected_components"] - after_stats["num_connected_components"],
            "connectivity_ratio": ratio
        }
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="M2 Module 6: Connectivity Evaluation")
    parser.add_argument('--before_graph', type=str, required=True, help="Input before graph pickle")
    parser.add_argument('--after_graph', type=str, required=True, help="Input after graph pickle")
    parser.add_argument('--output_eval', type=str, default="connectivity_improvement_report.json", help="Output JSON report")
    args = parser.parse_args()
    
    with open(args.before_graph, 'rb') as f:
        g_before = pickle.load(f)
    with open(args.after_graph, 'rb') as f:
        g_after = pickle.load(f)
        
    report = evaluate_healing_improvement(g_before, g_after)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_eval)), exist_ok=True)
    with open(args.output_eval, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Connectivity improvement report saved to: {os.path.abspath(args.output_eval)}")

if __name__ == "__main__":
    main()
