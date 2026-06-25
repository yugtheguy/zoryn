#!/usr/bin/env python3
"""
Module 8 — Criticality Ranking
Ranks urban mobility infrastructure nodes by Betweenness Centrality.
Generates top-10, top-20, and full sorted rankings for disaster prioritization dashboards.
"""

import os
import argparse
import json

def rank_critical_nodes(centrality_report: dict) -> dict:
    """
    Produces sorted criticality ranking tables from Module 7 centrality output.
    """
    nodes_dict = centrality_report.get("nodes", {})
    all_nodes = list(nodes_dict.values())
    
    # Sort descending by betweenness score
    all_nodes.sort(key=lambda x: x.get("betweenness", 0.0), reverse=True)
    
    # Assign rank index (1-indexed)
    for idx, item in enumerate(all_nodes):
        item["rank"] = idx + 1
        
    top_10 = all_nodes[:10]
    top_20 = all_nodes[:20]
    
    report = {
        "num_ranked_nodes": len(all_nodes),
        "top_10": top_10,
        "top_20": top_20,
        "full_ranking": all_nodes
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="M3 Module 8: Criticality Ranking")
    parser.add_argument('--input_centrality', type=str, required=True, help="Input critical_nodes JSON")
    parser.add_argument('--output_ranking', type=str, default="ranked_nodes.json", help="Output ranked JSON")
    args = parser.parse_args()
    
    with open(args.input_centrality, 'r') as f:
        centrality = json.load(f)
        
    ranking = rank_critical_nodes(centrality)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_ranking)), exist_ok=True)
    with open(args.output_ranking, 'w') as f:
        json.dump(ranking, f, indent=2)
    print(f"Ranked nodes report saved to: {os.path.abspath(args.output_ranking)}")

if __name__ == "__main__":
    main()
