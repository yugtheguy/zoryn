#!/usr/bin/env python3
"""
Route Resilience: Master Graph Pipeline Orchestrator (Member 2 Lead)
Orchestrates Modules 1 through 11: Skeletonization -> Graph Extraction -> Component Analysis
-> Gap Detection -> Topological Healing -> Centrality Analysis -> Stress Testing -> Dashboard Export.
"""

import os
import argparse
import time
import json
import cv2
import numpy as np

from m2_graph import (
    skeletonize_road_mask,
    extract_road_graph,
    analyze_connected_components,
    detect_occlusion_gaps,
    heal_road_topology,
    evaluate_healing_improvement,
    export_graph_artifacts
)
from m3_analysis import (
    compute_betweenness_centrality,
    rank_critical_nodes,
    simulate_network_failures,
    compute_resilience_index,
    export_dashboard_data
)

def run_master_pipeline(
    input_mask_path: str,
    output_dir: str,
    max_gap_dist: float = 65.0,
    min_angle_score: float = 0.3,
    min_component_size: int = 20,
    simulate_k_values: list = [1, 3, 5, 10]
):
    """Executes the complete Member 2 graph engineering workflow."""
    start_time = time.time()
    print(f"===========================================================")
    print(f"ISRO Route Resilience: Member 2 Graph Pipeline Execution")
    print(f"===========================================================")
    print(f"Input Mask: {os.path.abspath(input_mask_path)}")
    print(f"Output Directory: {os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    if not os.path.exists(input_mask_path):
        raise FileNotFoundError(f"Input mask file not found: {input_mask_path}")
    mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)

    # MODULE 1 — SKELETONIZATION
    print("\n[Module 1] Running Morphological Skeletonization...")
    t0 = time.time()
    skel = skeletonize_road_mask(mask, min_component_size=min_component_size)
    skel_path = os.path.join(output_dir, "skeleton.png")
    cv2.imwrite(skel_path, skel)
    print(f"  -> Skeleton generated ({time.time() - t0:.2f}s). Saved to: {skel_path}")

    # MODULE 2 — GRAPH EXTRACTION
    print("\n[Module 2] Extracting Compact Mathematical Graph...")
    t0 = time.time()
    raw_graph = extract_road_graph(skel)
    export_graph_artifacts(raw_graph, output_dir, basename="raw_road_graph")
    print(f"  -> Raw Graph extracted ({raw_graph.number_of_nodes()} nodes, {raw_graph.number_of_edges()} edges) ({time.time() - t0:.2f}s).")

    # MODULE 3 — CONNECTED COMPONENT ANALYSIS
    print("\n[Module 3] Analyzing Fragmented Connected Components...")
    raw_comp_report = analyze_connected_components(raw_graph)
    comp_path = os.path.join(output_dir, "raw_connectivity_report.json")
    with open(comp_path, 'w') as f:
        json.dump(raw_comp_report, f, indent=2)
    print(f"  -> Found {raw_comp_report['num_connected_components']} components. Largest Component Ratio: {raw_comp_report['largest_component_ratio']*100:.1f}%.")

    # MODULE 4 — GAP DETECTION
    print("\n[Module 4] Detecting Occlusion Gaps...")
    t0 = time.time()
    gaps_list = detect_occlusion_gaps(raw_graph, max_distance=max_gap_dist, min_angle_score=min_angle_score)
    gaps_path = os.path.join(output_dir, "candidate_gaps.json")
    with open(gaps_path, 'w') as f:
        json.dump({"num_candidate_gaps": len(gaps_list), "candidates": gaps_list}, f, indent=2)
    print(f"  -> Detected {len(gaps_list)} candidate gap pairs ({time.time() - t0:.2f}s).")

    # MODULE 5 — TOPOLOGICAL HEALING
    print("\n[Module 5] Executing Union-Find & MST Topological Healing...")
    t0 = time.time()
    healed_graph = heal_road_topology(raw_graph, gaps_list)
    export_graph_artifacts(healed_graph, output_dir, basename="healed_road_graph")
    print(f"  -> Topology healed ({healed_graph.number_of_nodes()} nodes, {healed_graph.number_of_edges()} edges) ({time.time() - t0:.2f}s).")

    # MODULE 6 — CONNECTIVITY EVALUATION
    print("\n[Module 6] Evaluating Healing Improvement...")
    eval_report = evaluate_healing_improvement(raw_graph, healed_graph)
    eval_path = os.path.join(output_dir, "connectivity_improvement_report.json")
    with open(eval_path, 'w') as f:
        json.dump(eval_report, f, indent=2)
    hm = eval_report["healing_metrics"]
    print(f"  -> Healed Edges Added: {hm['healed_edges_added']}. Components Merged: {hm['components_merged']}. Connectivity Ratio: {hm['connectivity_ratio']:.3f}.")

    # MODULE 7 — BETWEENNESS CENTRALITY
    print("\n[Module 7] Computing Shortest-Path Betweenness Centrality...")
    t0 = time.time()
    centrality_report = compute_betweenness_centrality(healed_graph)
    print(f"  -> Identified {centrality_report['num_gatekeepers']} gatekeepers and {centrality_report['num_spofs']} Single Points of Failure ({time.time() - t0:.2f}s).")

    # MODULE 8 — CRITICALITY RANKING
    print("\n[Module 8] Ranking Infrastructure Nodes...")
    ranking_report = rank_critical_nodes(centrality_report)
    if ranking_report["top_10"]:
        top1 = ranking_report["top_10"][0]
        print(f"  -> #1 Critical Node ID: {top1['node_id']} (Betweenness: {top1['betweenness']:.4f}).")

    # MODULE 9 — NETWORK STRESS TESTING
    print("\n[Module 9] Simulating Disaster Stress Testing...")
    t0 = time.time()
    sim_report = simulate_network_failures(healed_graph, ranking_report["full_ranking"], k_values=simulate_k_values)
    print(f"  -> Stress simulation complete across {len(simulate_k_values)} multi-node knockout scenarios ({time.time() - t0:.2f}s).")

    # MODULE 10 — RESILIENCE INDEX
    print("\n[Module 10] Calculating Network Resilience Index...")
    resilience_report = compute_resilience_index(sim_report)
    print(f"  -> Macro Resilience Index: {resilience_report['overall_resilience_index']:.4f} ({resilience_report['overall_classification']}).")

    # MODULE 11 — DASHBOARD JSON & GeoJSON EXPORTS
    print("\n[Module 11] Exporting Visualization Artifacts for Dashboarding...")
    exported_paths = export_dashboard_data(healed_graph, centrality_report, ranking_report, sim_report, resilience_report, output_dir)
    print("  -> Exported deliverables:")
    for name, p in exported_paths.items():
        print(f"      [{name}]: {os.path.basename(p)}")

    elapsed = time.time() - start_time
    print(f"\n===========================================================")
    print(f"Pipeline Completed Successfully in {elapsed:.2f} seconds!")
    print(f"Dashboard Artifacts Directory: {os.path.abspath(output_dir)}")
    print(f"===========================================================")
    return exported_paths

def main():
    parser = argparse.ArgumentParser(description="ISRO Route Resilience Master Graph Pipeline (Member 2)")
    parser.add_argument('--input_mask', type=str, required=True, help="Path to input binary road mask image")
    parser.add_argument('--output_dir', type=str, default="dashboard_exports/", help="Path to output artifacts directory")
    parser.add_argument('--max_gap_dist', type=float, default=65.0, help="Max gap reconnection Euclidean distance")
    parser.add_argument('--min_angle_score', type=float, default=0.3, help="Min angular cosine alignment score")
    parser.add_argument('--min_size', type=int, default=20, help="Min connected component size noise filter")
    parser.add_argument('--simulate_k', type=int, nargs='*', default=[1, 3, 5, 10], help="List of K knockout counts")
    args = parser.parse_args()

    run_master_pipeline(
        input_mask_path=args.input_mask,
        output_dir=args.output_dir,
        max_gap_dist=args.max_gap_dist,
        min_angle_score=args.min_angle_score,
        min_component_size=args.min_size,
        simulate_k_values=args.simulate_k
    )

if __name__ == "__main__":
    main()
