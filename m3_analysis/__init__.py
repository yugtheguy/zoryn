"""
Route Resilience: M3 Network Analysis & Criticality Simulation Package
Implements Modules 7 through 11 plus bonus features of the ISRO Hackathon Pipeline.
Standalone Member 3 owns all network analysis, criticality ranking, simulation, resilience, and dashboard exports.
"""

import os
import pickle
from .centrality import compute_betweenness_centrality
from .ranking import rank_critical_nodes
from .simulation import simulate_network_failures
from .resilience import compute_resilience_index
from .reroute import compute_rerouting_impact
from .dashboard_export import export_dashboard_data

def run_standalone_m3(
    graph_path: str,
    output_dir: str = "dashboard_exports/",
    simulate_k_values: list = [1, 3, 5, 10]
) -> dict:
    """
    Master standalone Member 3 entry point performing ALL network analysis:
    Centrality, Ranking, Disaster Simulation, Resilience, Rerouting, Dashboard Exports.
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Input graph artifact not found: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    print("\n===========================================================")
    print("ISRO Route Resilience: Standalone Member 3 Analysis Engine")
    print("===========================================================")
    print(f"[Ingested Graph]: {os.path.abspath(graph_path)}")
    print(f"[Dashboard Output]: {os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)

    # MODULE 7 — BETWEENNESS CENTRALITY
    print("\n[M3 Module 7] Computing Shortest-Path Betweenness Centrality & SPOFs...")
    centrality_report = compute_betweenness_centrality(graph)
    print(f"  -> Identified {centrality_report['num_gatekeepers']} gatekeepers and {centrality_report['num_spofs']} Single Points of Failure.")

    # MODULE 8 — CRITICALITY RANKING
    print("\n[M3 Module 8] Ranking Critical Infrastructure Intersections...")
    ranking_report = rank_critical_nodes(centrality_report)
    if ranking_report["top_10"]:
        top1 = ranking_report["top_10"][0]
        print(f"  -> #1 Critical Intersection ID: {top1['node_id']} (Betweenness: {top1['betweenness']:.4f}).")

    # MODULE 9 — DISASTER SIMULATION
    print("\n[M3 Module 9] Simulating Disaster Knockout Stress Scenarios...")
    sim_report = simulate_network_failures(graph, ranking_report["full_ranking"], k_values=simulate_k_values)
    print(f"  -> Simulated {len(simulate_k_values)} disaster knockout tiers.")

    # MODULE 10 — RESILIENCE INDEX
    print("\n[M3 Module 10] Calculating Macro Network Resilience Index...")
    resilience_report = compute_resilience_index(sim_report)
    print(f"  -> Overall Resilience Index: {resilience_report['overall_resilience_index']:.4f} ({resilience_report['overall_classification']}).")

    # MODULE 11 — DASHBOARD EXPORTS
    print("\n[M3 Module 11] Exporting Consolidated Dashboard JSON & GeoJSON Schema...")
    exported_paths = export_dashboard_data(graph, centrality_report, ranking_report, sim_report, resilience_report, output_dir)
    print("  -> Exported deliverables:")
    for name, p in exported_paths.items():
        print(f"      [{name}]: {os.path.basename(p)}")

    print("===========================================================\n")
    return exported_paths

__all__ = [
    "compute_betweenness_centrality",
    "rank_critical_nodes",
    "simulate_network_failures",
    "compute_resilience_index",
    "compute_rerouting_impact",
    "export_dashboard_data",
    "run_standalone_m3",
]
