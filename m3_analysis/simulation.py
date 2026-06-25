#!/usr/bin/env python3
"""
Module 9 — Network Stress Testing
Simulates urban infrastructure failures (floods, bridge collapses, accidents).
Benchmarks network degradation under single-node and multi-node (top-k) knockouts.
Recomputes connectivity ratios, largest connected component reachability, average shortest paths,
and global network routing efficiency.
"""

import os
import argparse
import pickle
import json
import networkx as nx
from m2_graph.components import analyze_connected_components

def compute_graph_performance_metrics(graph: nx.Graph, weight: str = "weight") -> dict:
    """
    Computes macro routing performance metrics for a graph.
    Handles disconnected topologies by evaluating component-level statistics.
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return {
            "num_nodes": n,
            "num_components": 0 if n == 0 else 1,
            "lcc_nodes": n,
            "reachability_ratio": 1.0 if n == 1 else 0.0,
            "lcc_average_shortest_path": 0.0,
            "global_efficiency": 0.0
        }
        
    comp_report = analyze_connected_components(graph)
    lcc_size = comp_report["largest_component_nodes"]
    reachability = comp_report["largest_component_ratio"]
    
    # LCC Average Shortest Path Length
    if lcc_size > 1:
        # Get LCC subgraph
        components = list(nx.connected_components(graph))
        components.sort(key=len, reverse=True)
        lcc_subgraph = graph.subgraph(components[0])
        try:
            lcc_asp = round(nx.average_shortest_path_length(lcc_subgraph, weight=weight), 4)
        except nx.NetworkXError:
            lcc_asp = 0.0
    else:
        lcc_asp = 0.0
        
    # Global Efficiency (unweighted shortest path inverse average)
    efficiency = round(nx.global_efficiency(graph), 4)
    
    return {
        "num_nodes": n,
        "num_components": comp_report["num_connected_components"],
        "lcc_nodes": lcc_size,
        "reachability_ratio": reachability,
        "lcc_average_shortest_path": lcc_asp,
        "global_efficiency": efficiency
    }

def simulate_network_failures(graph: nx.Graph, ranked_nodes: list, k_values: list = [1, 3, 5, 10], weight: str = "weight") -> dict:
    """
    Simulates single-node and simultaneous top-k node disaster knockouts.
    """
    baseline_metrics = compute_graph_performance_metrics(graph, weight=weight)
    
    top_node_ids = [item["node_id"] for item in ranked_nodes]
    
    # 1. Single Node Failure Benchmarks (top 5 critical nodes)
    single_node_sims = {}
    for node_id in top_node_ids[:5]:
        if node_id not in graph:
            continue
        damaged_graph = graph.copy()
        damaged_graph.remove_node(node_id)
        metrics = compute_graph_performance_metrics(damaged_graph, weight=weight)
        
        degrad = round(baseline_metrics["global_efficiency"] - metrics["global_efficiency"], 4)
        single_node_sims[str(node_id)] = {
            "knockout_node_id": node_id,
            "metrics_after_failure": metrics,
            "efficiency_degradation": degrad
        }
        
    # 2. Multi-Node Simultaneous Failures (top-k)
    multi_node_sims = {}
    for k in k_values:
        knockout_set = top_node_ids[:k]
        damaged_graph = graph.copy()
        damaged_graph.remove_nodes_from([n for n in knockout_set if n in damaged_graph])
        metrics = compute_graph_performance_metrics(damaged_graph, weight=weight)
        
        degrad = round(baseline_metrics["global_efficiency"] - metrics["global_efficiency"], 4)
        multi_node_sims[f"top_{k}_knockouts"] = {
            "k": k,
            "knockout_nodes": knockout_set,
            "metrics_after_failure": metrics,
            "efficiency_degradation": degrad
        }
        
    report = {
        "baseline": baseline_metrics,
        "single_node_failures": single_node_sims,
        "multi_node_failures": multi_node_sims
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="M3 Module 9: Network Stress Testing")
    parser.add_argument('--input_graph', type=str, required=True, help="Input graph pickle")
    parser.add_argument('--input_ranking', type=str, required=True, help="Input ranked_nodes JSON")
    parser.add_argument('--output_sim', type=str, default="simulation_results.json", help="Output simulation JSON")
    args = parser.parse_args()
    
    with open(args.input_graph, 'rb') as f:
        graph = pickle.load(f)
    with open(args.input_ranking, 'r') as f:
        ranking = json.load(f)
        full_list = ranking.get("full_ranking", [])
        
    sim_results = simulate_network_failures(graph, full_list)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_sim)), exist_ok=True)
    with open(args.output_sim, 'w') as f:
        json.dump(sim_results, f, indent=2)
    print(f"Simulation results saved to: {os.path.abspath(args.output_sim)}")

if __name__ == "__main__":
    main()
