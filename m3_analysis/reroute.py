#!/usr/bin/env python3
"""
Bonus Feature — Route Rerouting Engine
Calculates original shortest path vs alternate detour paths after simulated junction failures.
Quantifies detour distance delay and travel time penalty.
"""

import os
import argparse
import pickle
import json
import networkx as nx

def compute_rerouting_impact(graph: nx.Graph, source: int, target: int, failed_nodes: list = [], weight: str = "weight") -> dict:
    """
    Evaluates detour route trajectory and travel delay caused by infrastructure knockouts.
    """
    if source not in graph or target not in graph:
        return {"error": "Source or target node not in graph"}
        
    try:
        orig_path = nx.shortest_path(graph, source, target, weight=weight)
        orig_dist = round(float(nx.shortest_path_length(graph, source, target, weight=weight)), 2)
    except nx.NetworkXNoPath:
        return {"error": "No baseline path exists between source and target"}
        
    damaged_graph = graph.copy()
    damaged_graph.remove_nodes_from([n for n in failed_nodes if n in damaged_graph])
    
    try:
        alt_path = nx.shortest_path(damaged_graph, source, target, weight=weight)
        alt_dist = round(float(nx.shortest_path_length(damaged_graph, source, target, weight=weight)), 2)
        is_reachable = True
        delay = round(alt_dist - orig_dist, 2)
        delay_percent = round((delay / orig_dist) * 100.0, 2) if orig_dist > 0 else 0.0
    except nx.NetworkXNoPath:
        alt_path = []
        alt_dist = None
        is_reachable = False
        delay = None
        delay_percent = None
        
    # Convert node paths to spatial coordinate sequences for dashboard plotting
    orig_coords = [(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in orig_path]
    alt_coords = [(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in alt_path] if is_reachable else []
    
    return {
        "source": source,
        "target": target,
        "failed_nodes": failed_nodes,
        "is_reachable_after_failure": is_reachable,
        "original_route": {
            "node_path": orig_path,
            "spatial_path": orig_coords,
            "distance": orig_dist
        },
        "alternate_route": {
            "node_path": alt_path,
            "spatial_path": alt_coords,
            "distance": alt_dist
        },
        "delay_metrics": {
            "absolute_delay": delay,
            "delay_percentage": delay_percent
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Bonus Feature: Route Rerouting Engine")
    parser.add_argument('--input_graph', type=str, required=True, help="Input graph pickle")
    parser.add_argument('--source', type=int, required=True, help="Source node ID")
    parser.add_argument('--target', type=int, required=True, help="Target node ID")
    parser.add_argument('--failed_nodes', type=int, nargs='*', default=[], help="Failed node IDs")
    args = parser.parse_args()
    
    with open(args.input_graph, 'rb') as f:
        graph = pickle.load(f)
        
    result = compute_rerouting_impact(graph, args.source, args.target, args.failed_nodes)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
