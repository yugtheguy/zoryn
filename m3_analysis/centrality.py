import argparse
import sys
import os
from typing import Tuple, List, Dict, Any

from loaders import OSMnxGraphLoader, M2GraphLoader
from analysis import calculate_node_centralities, calculate_edge_betweenness, calculate_network_metrics
from simulation import DisasterSimulator
from routing import RouteRouter
from ai import AIAdvisoryGenerator
from exporters import JSONExporter
from utils.helpers import get_logger, timer_decorator
from config import DEFAULT_OUTPUT_DIR, DEFAULT_K_CENTRALITY, DEFAULT_DISASTER_FRACTION

logger = get_logger("M3_Main")

def parse_route_arg(route_str: str) -> List[Dict[str, Any]]:
    """
    Parses semicolon-separated routes of format 'name:lat1,lon1:lat2,lon2'.
    """
    routes = []
    for item in route_str.split(';'):
        if not item.strip():
            continue
        try:
            parts = item.split(':')
            name = parts[0]
            lat1, lon1 = map(float, parts[1].split(','))
            lat2, lon2 = map(float, parts[2].split(','))
            routes.append({
                "name": name,
                "origin": (lat1, lon1),
                "destination": (lat2, lon2)
            })
        except Exception as e:
            logger.error(f"Failed to parse route spec '{item}': {e}. Format must be 'name:lat1,lon1:lat2,lon2'")
    return routes

@timer_decorator
def main():
    parser = argparse.ArgumentParser(description="M3 Network Analysis & Resilience Pipeline")
    parser.add_argument('--source', type=str, choices=['osmnx', 'm2'], default='osmnx',
                        help="Graph source: 'osmnx' (development) or 'm2' (production)")
    parser.add_argument('--place', type=str, default="Manhattan, New York, USA",
                        help="OSMnx place query string (used if source=osmnx)")
    
    # Support contract compatibility with other members
    parser.add_argument('--graph_dir', type=str, default="/kaggle/working/graphs/",
                        help="Directory where Member 2 graph resides (used to resolve default m2 file)")
    parser.add_argument('--m2_file', type=str, default=None,
                        help="Direct path to Member 2 graph file. If not set, resolves in graph_dir.")
    
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for serialized JSON contracts")
    
    # Simulation settings
    parser.add_argument('--disaster_type', type=str, 
                        choices=['flood', 'bridge_collapse', 'road_block', 'major_junction_failure', 'multiple_failure', 'random_failure'], 
                        default='flood',
                        help="Type of disaster scenario to simulate")
    parser.add_argument('--flood_center', type=str, default="40.7128,-74.0060",
                        help="Latitude,longitude coordinate of the center of flood")
    parser.add_argument('--flood_radius', type=float, default=0.5,
                        help="Radius of the flood impact in kilometers")
    parser.add_argument('--attack_fraction', type=float, default=DEFAULT_DISASTER_FRACTION,
                        help="Fraction of nodes/edges removed during simulated attacks")
    
    # Performance speed-up configuration
    parser.add_argument('--k_centrality', type=int, default=DEFAULT_K_CENTRALITY,
                        help="Sample size node subset count for centrality profiling")
    
    # Route profiling definition
    parser.add_argument('--routes', type=str,
                        default="Hospital Corridor:40.7110,-74.0080:40.7190,-74.0010;Emergency Access:40.7200,-74.0100:40.7150,-74.0050",
                        help="Semicolon-separated routing specifications")

    args = parser.parse_args()

    logger.info("Initializing M3 Network Resilience Pipeline...")

    # Resolve graph input paths
    if args.source == 'osmnx':
        loader = OSMnxGraphLoader(place_query=args.place)
    else:
        m2_filepath = args.m2_file
        if not m2_filepath:
            # Check for standard file names inside the graph_dir
            pickle_path = os.path.join(args.graph_dir, "road_network.gpickle")
            geojson_path = os.path.join(args.graph_dir, "road_network.geojson")
            if os.path.exists(pickle_path):
                m2_filepath = pickle_path
            elif os.path.exists(geojson_path):
                m2_filepath = geojson_path
            else:
                m2_filepath = pickle_path # Default fallback to trigger error
        loader = M2GraphLoader(filepath=m2_filepath)

    # 1. Load network graph
    try:
        graph = loader.load_graph()
    except Exception as e:
        logger.error(f"Critical error loading road network: {e}")
        sys.exit(1)

    # 2. Compute Baseline Indicators
    logger.info("Running baseline network profiling...")
    pre_metrics = calculate_network_metrics(graph, sample_size=args.k_centrality)
    node_centralities = calculate_node_centralities(graph, k=args.k_centrality)
    edge_centralities = calculate_edge_betweenness(graph, k=args.k_centrality)

    # 3. Simulate Disaster State
    logger.info(f"Simulating network degradation ({args.disaster_type})...")
    simulator = DisasterSimulator(graph)
    disaster_params = {}
    disaster_res = {}

    if args.disaster_type == 'flood':
        try:
            lat, lon = map(float, args.flood_center.split(','))
        except ValueError:
            logger.error("Failed to parse flood_center. Coordinates must be comma separated: lat,lon")
            sys.exit(1)
        disaster_params = {
            "center": [lat, lon],
            "radius_meters": args.flood_radius * 1000.0
        }
        disaster_res = simulator.simulate_flood(center=(lat, lon), radius_km=args.flood_radius)
    elif args.disaster_type == 'bridge_collapse':
        disaster_params = {
            "target_type": "edge",
            "fraction": args.attack_fraction
        }
        disaster_res = simulator.simulate_bridge_collapse(
            edge_centrality=edge_centralities,
            fraction=args.attack_fraction
        )
    elif args.disaster_type == 'road_block':
        # Default roadblock simulates removing top 1% centrality edges if list is not provided
        disaster_params = {
            "target_type": "edge",
            "fraction": 0.01
        }
        disaster_res = simulator.simulate_bridge_collapse(
            edge_centrality=edge_centralities,
            fraction=0.01
        )
        disaster_res["simulation_type"] = "Road Block"
    elif args.disaster_type == 'major_junction_failure':
        disaster_params = {
            "target_type": "node",
            "fraction": args.attack_fraction
        }
        disaster_res = simulator.simulate_major_junction_failure(
            node_centrality=node_centralities["betweenness"],
            fraction=args.attack_fraction
        )
    elif args.disaster_type == 'random_failure':
        disaster_params = {
            "target_type": "node",
            "fraction": args.attack_fraction
        }
        disaster_res = simulator.simulate_random_infrastructure_failure(target_type="node", fraction=args.attack_fraction)
    else: # multiple failure
        disaster_params = {
            "description": "Combination of flood and highest centrality failures"
        }
        # Simulate flood and also remove top 1 centrality node
        sorted_nodes = sorted(node_centralities["betweenness"].items(), key=lambda item: item[1], reverse=True)
        top_node = [sorted_nodes[0][0]] if sorted_nodes else []
        try:
            lat, lon = map(float, args.flood_center.split(','))
            flood_params = ((lat, lon), args.flood_radius)
        except ValueError:
            flood_params = (None, None)

        disaster_res = simulator.simulate_multiple_failure(
            node_list=top_node,
            flood_center=flood_params[0],
            flood_radius_km=flood_params[1]
        )

    post_graph = disaster_res["graph"]

    # 4. Compute Degraded Indicators
    logger.info("Running degraded network profiling...")
    post_metrics = calculate_network_metrics(post_graph, sample_size=args.k_centrality)

    # 5. Calculate Route detour options
    logger.info("Calculating emergency route alternatives...")
    route_specs = parse_route_arg(args.routes)
    router = RouteRouter(graph, post_graph)
    route_results = []
    for spec in route_specs:
        res = router.analyze_route(spec["origin"], spec["destination"], spec["name"])
        route_results.append(res)

    # 6. Generate AI Decision Support
    logger.info("Generating AI emergency advisory...")
    dummy_disaster_metrics = {
        "scenario_type": args.disaster_type,
        "metrics": {
            "pre_disaster": pre_metrics,
            "post_disaster": post_metrics,
            "delta": {
                "efficiency_loss_percent": ((pre_metrics["efficiency"] - post_metrics["efficiency"]) / pre_metrics["efficiency"] * 100.0) if pre_metrics["efficiency"] > 0 else 0.0,
                "newly_isolated_nodes": max(0, post_metrics["isolated_nodes_count"] - pre_metrics["isolated_nodes_count"])
            }
        }
    }
    ai_gen = AIAdvisoryGenerator()
    advisory = ai_gen.generate_advisory(dummy_disaster_metrics, route_results)

    # 7. Output serialized JSON contracts
    logger.info("Serializing dashboard outputs...")
    exporter = JSONExporter(output_dir=args.output_dir)
    exporter.export_centrality(node_centralities, edge_centralities)
    exporter.export_disaster_metrics(
        args.disaster_type, 
        disaster_params, 
        pre_metrics, 
        post_metrics,
        removed_nodes=disaster_res.get("removed_nodes"),
        removed_edges=disaster_res.get("removed_edges")
    )
    exporter.export_routing_detours(route_results)
    exporter.export_ai_advisory(advisory)

    logger.info("M3 Network Analysis & Resilience Pipeline completed successfully.")

if __name__ == "__main__":
    main()
