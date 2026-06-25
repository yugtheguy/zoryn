#!/usr/bin/env python3
"""
Module 2 — Graph Extraction
Converts binary skeleton centerlines into a compact mathematical NetworkX graph G=(V, E).
Identifies topological endpoints (degree=1) and junctions/intersections (degree>=3),
compacting degree-2 paths into weighted edges with exact geometric path lengths.
"""

import os
import argparse
import pickle
import numpy as np
import networkx as nx
import cv2

def get_8_neighbours(y: int, x: int, height: int, width: int):
    """Yields valid 8-connected neighbour coordinates (ny, nx)."""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                yield ny, nx

def extract_road_graph(skeleton_image: np.ndarray) -> nx.Graph:
    """
    Extracts a compacted NetworkX road graph from a skeleton image.
    
    Nodes are Endpoints (deg 1) and Junctions (deg >= 3).
    Edges represent continuous road segments between nodes, storing length and geometry.
    """
    if skeleton_image.ndim > 2:
        skeleton_image = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2GRAY)
        
    skel = skeleton_image > 0
    h, w = skel.shape
    
    # 1. Compute pixel neighbour counts
    pixel_graph = nx.Graph()
    road_pixels = set(zip(*np.where(skel)))  # set of (y, x)
    
    for y, x in road_pixels:
        pixel_graph.add_node((x, y))  # Note: storing spatial coordinates as (x, y)
        for ny, nx_coord in get_8_neighbours(y, x, h, w):
            if (ny, nx_coord) in road_pixels:
                # Add edge with Euclidean distance
                dist = np.hypot(nx_coord - x, ny - y)
                pixel_graph.add_edge((x, y), (nx_coord, ny), weight=dist)
                
    # 2. Identify key topological nodes (degree != 2)
    key_nodes = {}
    node_id_counter = 1
    
    for node, deg in pixel_graph.degree():
        if deg != 2:
            node_type = "endpoint" if deg == 1 else "junction" if deg >= 3 else "isolated"
            key_nodes[node] = {
                "id": node_id_counter,
                "x": int(node[0]),
                "y": int(node[1]),
                "degree": int(deg),
                "type": node_type
            }
            node_id_counter += 1
            
    # Handle closed rings (all nodes degree 2)
    # If there are connected components with no key nodes, pick one arbitrary node per component
    for comp in nx.connected_components(pixel_graph):
        if not any(n in key_nodes for n in comp):
            arbitrary_node = next(iter(comp))
            key_nodes[arbitrary_node] = {
                "id": node_id_counter,
                "x": int(arbitrary_node[0]),
                "y": int(arbitrary_node[1]),
                "degree": 2,
                "type": "ring_anchor"
            }
            node_id_counter += 1

    # 3. Build compacted graph
    compact_graph = nx.Graph()
    for pt, attr in key_nodes.items():
        compact_graph.add_node(attr["id"], **attr)
        
    # Mapping coordinate -> node_id
    coord_to_id = {pt: attr["id"] for pt, attr in key_nodes.items()}
    
    # Trace paths between key nodes
    visited_edges = set()
    
    for start_pt, start_id in coord_to_id.items():
        for nxt_pt in pixel_graph.neighbors(start_pt):
            edge_key = tuple(sorted((start_pt, nxt_pt)))
            if edge_key in visited_edges:
                continue
                
            visited_edges.add(edge_key)
            current_pt = nxt_pt
            prev_pt = start_pt
            path_coords = [start_pt, current_pt]
            accumulated_length = pixel_graph[start_pt][nxt_pt]["weight"]
            
            # Walk along degree-2 nodes until reaching another key node
            while current_pt not in coord_to_id:
                neighbors = list(pixel_graph.neighbors(current_pt))
                # Next node is the one that is not prev_pt
                next_step = neighbors[0] if neighbors[0] != prev_pt else neighbors[1]
                edge_key = tuple(sorted((current_pt, next_step)))
                visited_edges.add(edge_key)
                
                accumulated_length += pixel_graph[current_pt][next_step]["weight"]
                prev_pt = current_pt
                current_pt = next_step
                path_coords.append(current_pt)
                
            end_id = coord_to_id[current_pt]
            
            # Add compacted edge
            compact_graph.add_edge(
                start_id,
                end_id,
                source=start_id,
                target=end_id,
                length=float(accumulated_length),
                weight=float(accumulated_length),
                geometry=[(int(p[0]), int(p[1])) for p in path_coords]
            )
            
    return compact_graph

def main():
    parser = argparse.ArgumentParser(description="M2 Module 2: Graph Extraction Pipeline")
    parser.add_argument('--input_skel', type=str, required=True, help="Input skeleton image")
    parser.add_argument('--output_graph', type=str, default="road_graph.gpickle", help="Output graph pickle")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_skel):
        raise FileNotFoundError(f"Skeleton image not found: {args.input_skel}")
        
    skel = cv2.imread(args.input_skel, cv2.IMREAD_GRAYSCALE)
    graph = extract_road_graph(skel)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_graph)), exist_ok=True)
    with open(args.output_graph, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Extracted road graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges) saved to: {os.path.abspath(args.output_graph)}")

if __name__ == "__main__":
    main()
