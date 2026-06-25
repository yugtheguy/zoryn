#!/usr/bin/env python3
"""
Module 4 — Gap Detection
Identifies broken road centerlines across occlusions (tree canopy, shadows).
Computes Euclidean distances, endpoint orientation vectors, angular alignment scores,
and composite weighted gap scores for all candidate reconnection pairs.
"""

import os
import argparse
import pickle
import json
import math
import numpy as np
import networkx as nx

def compute_endpoint_heading(graph: nx.Graph, node_id: int) -> tuple:
    """
    Computes unit heading vector pointing outward from an endpoint into potential gaps.
    """
    node_data = graph.nodes[node_id]
    neighbors = list(graph.neighbors(node_id))
    if not neighbors:
        return (0.0, 0.0) # Isolated
        
    adj_id = neighbors[0]
    edge_data = graph[node_id][adj_id]
    geom = edge_data.get("geometry", [])
    
    nx_coord, ny_coord = node_data["x"], node_data["y"]
    
    # Find a point slightly inside the road edge to compute direction
    if len(geom) >= 2:
        # Check orientation of geom list
        if geom[0] == (nx_coord, ny_coord):
            # geom starts at node, look ahead
            idx = min(5, len(geom) - 1)
            inner_pt = geom[idx]
        else:
            # geom ends at node, look backward
            idx = max(0, len(geom) - 6)
            inner_pt = geom[idx]
    else:
        adj_data = graph.nodes[adj_id]
        inner_pt = (adj_data["x"], adj_data["y"])
        
    dx = nx_coord - inner_pt[0]
    dy = ny_coord - inner_pt[1]
    length = math.hypot(dx, dy)
    
    if length == 0:
        return (0.0, 0.0)
    return (dx / length, dy / length)

def detect_occlusion_gaps(graph: nx.Graph, max_distance: float = 60.0, min_angle_score: float = 0.3) -> list:
    """
    Detects candidate gaps between topological endpoints.
    
    Returns list of candidate dictionaries sorted by gap score (descending).
    """
    # 1. Identify endpoints (degree 1)
    endpoints = [n for n, d in graph.degree() if d == 1]
    if len(endpoints) < 2:
        return []
        
    # Precompute headings and coordinates
    ep_info = {}
    for ep in endpoints:
        data = graph.nodes[ep]
        ep_info[ep] = {
            "x": data["x"],
            "y": data["y"],
            "heading": compute_endpoint_heading(graph, ep),
            "comp": nx.node_connected_component(graph, ep)
        }
        
    candidates = []
    
    # 2. Evaluate pairwise candidates
    for i in range(len(endpoints)):
        u = endpoints[i]
        u_data = ep_info[u]
        
        for j in range(i + 1, len(endpoints)):
            v = endpoints[j]
            v_data = ep_info[v]
            
            # Avoid reconnecting endpoints within the same small connected component
            # unless it's closing a large loop (optional, but standard healing bridges disjoint components)
            if u_data["comp"] is v_data["comp"]:
                continue
                
            dx = v_data["x"] - u_data["x"]
            dy = v_data["y"] - u_data["y"]
            dist = math.hypot(dx, dy)
            
            if dist == 0 or dist > max_distance:
                continue
                
            # Unit gap vector from u to v
            gx, gy = dx / dist, dy / dist
            
            hx_u, hy_u = u_data["heading"]
            hx_v, hy_v = v_data["heading"]
            
            # Alignment of u's outward heading towards v
            align_u = max(0.0, hx_u * gx + hy_u * gy)
            # Alignment of v's outward heading towards u (-gx, -gy)
            align_v = max(0.0, hx_v * (-gx) + hy_v * (-gy))
            # Opposition between u and v headings
            opp = max(0.0, -(hx_u * hx_v + hy_u * hy_v))
            
            # If headings are (0,0) due to degenerate geometry, fallback to dist only
            if (hx_u, hy_u) == (0.0, 0.0) or (hx_v, hy_v) == (0.0, 0.0):
                angle_score = 0.5
            else:
                angle_score = (align_u + align_v + opp) / 3.0
                
            if angle_score < min_angle_score:
                continue
                
            dist_score = 1.0 - (dist / max_distance)
            
            # Weighted composite score
            composite_score = round(0.4 * dist_score + 0.6 * angle_score, 4)
            
            candidates.append({
                "source": u,
                "target": v,
                "source_coords": [u_data["x"], u_data["y"]],
                "target_coords": [v_data["x"], v_data["y"]],
                "distance": round(dist, 2),
                "orientation_score": round(angle_score, 4),
                "gap_score": composite_score
            })
            
    candidates.sort(key=lambda x: x["gap_score"], reverse=True)
    return candidates

def main():
    parser = argparse.ArgumentParser(description="M2 Module 4: Gap Detection")
    parser.add_argument('--input_graph', type=str, required=True, help="Input graph pickle")
    parser.add_argument('--output_gaps', type=str, default="candidate_gaps.json", help="Output JSON gaps")
    parser.add_argument('--max_dist', type=float, default=60.0, help="Max distance gap threshold")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_graph):
        raise FileNotFoundError(f"Input graph not found: {args.input_graph}")
        
    with open(args.input_graph, 'rb') as f:
        graph = pickle.load(f)
        
    gaps = detect_occlusion_gaps(graph, max_distance=args.max_dist)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_gaps)), exist_ok=True)
    with open(args.output_gaps, 'w') as f:
        json.dump({"num_candidate_gaps": len(gaps), "candidates": gaps}, f, indent=2)
    print(f"Detected {len(gaps)} candidate gaps saved to: {os.path.abspath(args.output_gaps)}")

if __name__ == "__main__":
    main()
