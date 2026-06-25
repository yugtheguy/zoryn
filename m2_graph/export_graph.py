#!/usr/bin/env python3
"""
M2 Graph Exporter
Serializes NetworkX graph objects into standard format artifacts (.gpickle / .pkl, .gml).
"""

import os
import argparse
import pickle
import networkx as nx

def export_graph_artifacts(graph: nx.Graph, output_dir: str, basename: str = "road_graph"):
    """
    Exports graph to pickle and GML.
    """
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f"{basename}.gpickle")
    gml_path = os.path.join(output_dir, f"{basename}.gml")
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(graph, f)
        
    # Prepare graph for GML export (list attributes must be stringified or omitted)
    gml_graph = nx.Graph()
    for n, data in graph.nodes(data=True):
        clean_data = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
        gml_graph.add_node(n, **clean_data)
        
    for u, v, data in graph.edges(data=True):
        clean_data = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
        gml_graph.add_edge(u, v, **clean_data)
        
    nx.write_gml(gml_graph, gml_path)
    return pkl_path, gml_path

def main():
    parser = argparse.ArgumentParser(description="M2 Graph Export Utility")
    parser.add_argument('--input_graph', type=str, required=True, help="Input graph pickle")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/graphs/", help="Output directory")
    args = parser.parse_args()
    
    with open(args.input_graph, 'rb') as f:
        graph = pickle.load(f)
        
    pkl, gml = export_graph_artifacts(graph, args.output_dir)
    print(f"Graph artifacts exported:\n  Pickle: {os.path.abspath(pkl)}\n  GML: {os.path.abspath(gml)}")

if __name__ == "__main__":
    main()
