#!/usr/bin/env python3
"""
Route Resilience: Master Pipeline Orchestrator (Member 2 -> Member 3 Handoff)
Orchestrates the refactored flow:
Input Mask -> Member 2 (Skeletonization, Graph Extraction, Gap Detection, Topological Healing)
-> Final Healed Graph (.gpickle) -> Standalone Member 3 -> Dashboard JSON Deliverables.
"""

import os
import argparse
import time
import json
import cv2

from m2_graph import (
    skeletonize_road_mask,
    extract_road_graph,
    analyze_connected_components,
    detect_occlusion_gaps,
    heal_road_topology,
    evaluate_healing_improvement,
    export_graph_artifacts
)
# Redirected import invoking Standalone Member 3 instead of duplicate M3 ownership
from m3_analysis import run_standalone_m3

def run_master_pipeline(
    input_mask_path: str,
    output_dir: str,
    max_gap_dist: float = 65.0,
    min_angle_score: float = 0.3,
    min_component_size: int = 20,
    simulate_k_values: list = [1, 3, 5, 10]
):
    """Executes Member 2 graph generation & topological healing, then hands off to Standalone Member 3."""
    start_time = time.time()
    print(f"===========================================================")
    print(f"ISRO Route Resilience: Member 2 Graph Generation Pipeline")
    print(f"===========================================================")
    print(f"Input Mask: {os.path.abspath(input_mask_path)}")
    print(f"Output Directory: {os.path.abspath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)

    # Ingest binary road mask
    if not os.path.exists(input_mask_path):
        raise FileNotFoundError(f"Input mask file not found: {input_mask_path}")
    mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)

    # MODULE 1 — SKELETONIZATION
    print("\n[M2 Module 1] Running Morphological Centerline Thinning...")
    t0 = time.time()
    skel = skeletonize_road_mask(mask, min_component_size=min_component_size)
    skel_path = os.path.join(output_dir, "skeleton.png")
    cv2.imwrite(skel_path, skel)
    print(f"  -> Skeleton generated ({time.time() - t0:.2f}s). Saved to: {os.path.basename(skel_path)}")

    # MODULE 2 — GRAPH EXTRACTION
    print("\n[M2 Module 2] Extracting Compact Mathematical Graph G=(V,E)...")
    t0 = time.time()
    raw_graph = extract_road_graph(skel)
    export_graph_artifacts(raw_graph, output_dir, basename="raw_road_graph")
    print(f"  -> Raw Graph extracted ({raw_graph.number_of_nodes()} nodes, {raw_graph.number_of_edges()} edges) ({time.time() - t0:.2f}s).")

    # MODULE 3 — CONNECTED COMPONENT ANALYSIS
    print("\n[M2 Module 3] Analyzing Fragmented Connected Components...")
    raw_comp_report = analyze_connected_components(raw_graph)
    comp_path = os.path.join(output_dir, "raw_connectivity_report.json")
    with open(comp_path, 'w') as f:
        json.dump(raw_comp_report, f, indent=2)
    print(f"  -> Found {raw_comp_report['num_connected_components']} components. LCC Reachability: {raw_comp_report['largest_component_ratio']*100:.1f}%.")

    # MODULE 4 — GAP DETECTION
    print("\n[M2 Module 4] Detecting Occlusion Gaps Across Cloud/Tree Clutter...")
    t0 = time.time()
    gaps_list = detect_occlusion_gaps(raw_graph, max_distance=max_gap_dist, min_angle_score=min_angle_score)
    gaps_path = os.path.join(output_dir, "candidate_gaps.json")
    with open(gaps_path, 'w') as f:
        json.dump({"num_candidate_gaps": len(gaps_list), "candidates": gaps_list}, f, indent=2)
    print(f"  -> Detected {len(gaps_list)} candidate gap pairs ({time.time() - t0:.2f}s).")

    # MODULE 5 — TOPOLOGICAL HEALING
    print("\n[M2 Module 5] Executing Union-Find DSU & Kruskal MST Healing...")
    t0 = time.time()
    healed_graph = heal_road_topology(raw_graph, gaps_list)
    export_graph_artifacts(healed_graph, output_dir, basename="healed_road_graph")
    healed_pkl_path = os.path.join(output_dir, "healed_road_graph.gpickle")
    print(f"  -> Topology healed ({healed_graph.number_of_nodes()} nodes, {healed_graph.number_of_edges()} edges) ({time.time() - t0:.2f}s).")

    # MODULE 6 — CONNECTIVITY IMPROVEMENT EVALUATION
    print("\n[M2 Module 6] Evaluating Topological Healing Improvement...")
    eval_report = evaluate_healing_improvement(raw_graph, healed_graph)
    eval_path = os.path.join(output_dir, "connectivity_improvement_report.json")
    with open(eval_path, 'w') as f:
        json.dump(eval_report, f, indent=2)
    hm = eval_report["healing_metrics"]
    print(f"  -> Healed Edges Added: {hm['healed_edges_added']}. Components Merged: {hm['components_merged']}. Connectivity Ratio: {hm['connectivity_ratio']:.3f}.")

    print(f"\n[Member 2 Complete] Graph generated and exported to: {healed_pkl_path}")
    
    # HANDOFF TO STANDALONE MEMBER 3
    # Member 2 does not own centrality, ranking, disaster simulation, resilience, routing, or dashboard exports.
    print("\n[Handoff to Standalone Member 3] Invoking Final M3 Analysis Engine...")
    exported_paths = run_standalone_m3(healed_pkl_path, output_dir=output_dir, simulate_k_values=simulate_k_values)

    elapsed = time.time() - start_time
    print(f"===========================================================")
    print(f"Integrated Pipeline Completed Successfully in {elapsed:.2f} seconds!")
    print(f"Dashboard JSON Deliverables Directory: {os.path.abspath(output_dir)}")
    print(f"===========================================================")
    return exported_paths

def main():
    parser = argparse.ArgumentParser(description="ISRO Route Resilience Master Graph Pipeline (M2 -> M3 Integration)")
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
