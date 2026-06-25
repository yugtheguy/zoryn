#!/usr/bin/env python3
"""
Module 11 — Dashboard Exports
Consolidates and formats graph topologies, centrality heatmaps, stress test simulations,
and macro resilience scores into dashboard-ready JSON and GeoJSON artifacts required
by Streamlit, Leaflet, and GIS (QGIS) visualization systems.
"""

import os
import argparse
import pickle
import json
import networkx as nx

def export_dashboard_data(
    graph: nx.Graph,
    centrality_report: dict,
    ranking_report: dict,
    simulation_report: dict,
    resilience_report: dict,
    output_dir: str
) -> dict:
    """
    Serializes all pipeline deliverables to dashboard contract JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. graph_data.json
    nodes_list = []
    for n, data in graph.nodes(data=True):
        nodes_list.append({
            "id": n,
            "x": data.get("x", 0),
            "y": data.get("y", 0),
            "degree": data.get("degree", graph.degree[n]),
            "type": data.get("type", "standard")
        })
        
    edges_list = []
    for u, v, data in graph.edges(data=True):
        edges_list.append({
            "source": u,
            "target": v,
            "length": round(data.get("length", 0.0), 2),
            "weight": round(data.get("weight", 0.0), 2),
            "is_healed": data.get("is_healed", False),
            "geometry": data.get("geometry", [])
        })
        
    graph_data = {
        "num_nodes": len(nodes_list),
        "num_edges": len(edges_list),
        "nodes": nodes_list,
        "edges": edges_list
    }
    graph_path = os.path.join(output_dir, "graph_data.json")
    with open(graph_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
        
    # 2. critical_nodes.json
    crit_path = os.path.join(output_dir, "critical_nodes.json")
    with open(crit_path, 'w') as f:
        json.dump(centrality_report, f, indent=2)
        
    # 3. heatmap_edges.json
    edges_centrality = centrality_report.get("edges", {})
    heatmap_edges = []
    for e_key, e_info in edges_centrality.items():
        u, v = e_info["source"], e_info["target"]
        geom = graph[u][v].get("geometry", [])
        heatmap_edges.append({
            "edge_id": e_key,
            "source": u,
            "target": v,
            "edge_betweenness": e_info["edge_betweenness"],
            "geometry": geom
        })
    heatmap_path = os.path.join(output_dir, "heatmap_edges.json")
    with open(heatmap_path, 'w') as f:
        json.dump({"num_edges": len(heatmap_edges), "heatmap": heatmap_edges}, f, indent=2)
        
    # 4. ranked_nodes.json
    ranked_path = os.path.join(output_dir, "ranked_nodes.json")
    with open(ranked_path, 'w') as f:
        json.dump(ranking_report, f, indent=2)

    # 5. simulation_results.json
    sim_path = os.path.join(output_dir, "simulation_results.json")
    with open(sim_path, 'w') as f:
        json.dump(simulation_report, f, indent=2)
        
    # 6. resilience_report.json
    res_path = os.path.join(output_dir, "resilience_report.json")
    with open(res_path, 'w') as f:
        json.dump(resilience_report, f, indent=2)
        
    # 7. Bonus Feature: GeoJSON Export (network_export.geojson)
    features = []
    for n, data in graph.nodes(data=True):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [data.get("x", 0), data.get("y", 0)]
            },
            "properties": {
                "id": n,
                "node_type": data.get("type", "node"),
                "degree": data.get("degree", graph.degree[n])
            }
        })
        
    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry", [])
        if not geom:
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            geom = [(u_data.get("x", 0), u_data.get("y", 0)), (v_data.get("x", 0), v_data.get("y", 0))]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[pt[0], pt[1]] for pt in geom]
            },
            "properties": {
                "source": u,
                "target": v,
                "length": data.get("length", 0.0),
                "is_healed": data.get("is_healed", False)
            }
        })
        
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    geojson_path = os.path.join(output_dir, "network_export.geojson")
    with open(geojson_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)
        
    return {
        "graph_data": graph_path,
        "critical_nodes": crit_path,
        "heatmap_edges": heatmap_path,
        "ranked_nodes": ranked_path,
        "simulation_results": sim_path,
        "resilience_report": res_path,
        "geojson_export": geojson_path
    }

def main():
    parser = argparse.ArgumentParser(description="M3 Module 11: Dashboard Exports")
    parser.add_argument('--input_graph', type=str, required=True, help="Input graph pickle")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/analysis/", help="Output directory")
    args = parser.parse_args()
    
    # Standalone demo stub
    print("Run via unified pipeline.py for complete multi-report consolidation.")

if __name__ == "__main__":
    main()
