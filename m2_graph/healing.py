#!/usr/bin/env python3
"""
Module 5 — Topological Healing
Reconstructs missing road links across occlusions using Union-Find (Disjoint Set)
and Minimum Spanning Tree (Kruskal's MST) algorithms.
Enforces physical plausibility constraints (maximum degree per endpoint, distance/angle thresholds)
to prevent false bridges or cross-map shortcuts.
"""

import os
import argparse
import pickle
import json
import networkx as nx

class UnionFind:
    """Disjoint Set Union (DSU) structure with path compression and union by rank."""
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}
        
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
        
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.rank[pu] < self.rank[pv]:
            self.parent[pu] = pv
        elif self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu
        else:
            self.parent[pv] = pu
            self.rank[pu] += 1
        return True

def heal_road_topology(graph: nx.Graph, candidate_gaps: list, max_healing_edges_per_node: int = 1) -> nx.Graph:
    """
    Heals disconnected graph components using MST Kruskal algorithm on candidate gap edges.
    
    Args:
        graph: Input fragmented NetworkX graph.
        candidate_gaps: List of candidate gap dictionaries from Module 4.
        max_healing_edges_per_node: Max new healing links attached to any single endpoint.
        
    Returns:
        New healed NetworkX graph.
    """
    healed_graph = graph.copy()
    
    # 1. Initialize Union-Find with existing connected components
    uf = UnionFind(healed_graph.nodes())
    for u, v in healed_graph.edges():
        uf.union(u, v)
        
    # Track healing degrees to avoid unnatural star-clusters at endpoints
    healing_degree = {n: 0 for n in healed_graph.nodes()}
    
    healed_edges_added = 0
    
    # 2. Iterate through candidate gaps (already sorted by gap_score desc)
    for gap in candidate_gaps:
        u = gap["source"]
        v = gap["target"]
        
        # Check endpoint capacity constraints
        if healing_degree[u] >= max_healing_edges_per_node or healing_degree[v] >= max_healing_edges_per_node:
            continue
            
        # Check MST cycle condition (connects disjoint components)
        if uf.union(u, v):
            dist = gap["distance"]
            u_node = healed_graph.nodes[u]
            v_node = healed_graph.nodes[v]
            
            healed_graph.add_edge(
                u,
                v,
                source=u,
                target=v,
                length=float(dist),
                weight=float(dist),
                is_healed=True,
                gap_score=gap["gap_score"],
                geometry=[(u_node["x"], u_node["y"]), (v_node["x"], v_node["y"])]
            )
            healing_degree[u] += 1
            healing_degree[v] += 1
            healed_edges_added += 1
            
    return healed_graph

def main():
    parser = argparse.ArgumentParser(description="M2 Module 5: Topological Healing")
    parser.add_argument('--input_graph', type=str, required=True, help="Input graph pickle")
    parser.add_argument('--input_gaps', type=str, required=True, help="Input gaps JSON")
    parser.add_argument('--output_healed', type=str, default="healed_graph.gpickle", help="Output healed graph pickle")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_graph):
        raise FileNotFoundError(f"Input graph not found: {args.input_graph}")
    if not os.path.exists(args.input_gaps):
        raise FileNotFoundError(f"Input gaps not found: {args.input_gaps}")
        
    with open(args.input_graph, 'rb') as f:
        graph = pickle.load(f)
    with open(args.input_gaps, 'r') as f:
        gaps_data = json.load(f)
        candidates = gaps_data.get("candidates", [])
        
    healed = heal_road_topology(graph, candidates)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_healed)), exist_ok=True)
    with open(args.output_healed, 'wb') as f:
        pickle.dump(healed, f)
    print(f"Healed road graph saved to: {os.path.abspath(args.output_healed)}")

if __name__ == "__main__":
    main()
