from typing import Tuple, List, Dict, Any
import networkx as nx
from utils.helpers import get_logger, haversine_distance, timer_decorator

logger = get_logger("RouteIntelligence")

class RouteRouter:
    def __init__(self, pre_graph: nx.MultiDiGraph, post_graph: nx.MultiDiGraph):
        """
        Router initialized with both intact and degraded graphs.
        """
        self.pre_graph = pre_graph
        self.post_graph = post_graph

    def find_nearest_node(self, graph: nx.MultiDiGraph, point: Tuple[float, float]) -> int:
        """
        Locates the closest node ID in the graph to a (lat, lon) coordinates point.
        """
        lat, lon = point
        closest_node = None
        min_dist = float('inf')

        for node, data in graph.nodes(data=True):
            n_lat = data.get('lat')
            n_lon = data.get('lon')
            if n_lat is not None and n_lon is not None:
                dist = haversine_distance((lat, lon), (n_lat, n_lon))
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node

        if closest_node is None:
            raise ValueError("No valid nodes with coordinates found in the graph.")
        return closest_node

    def get_path_coords(self, graph: nx.MultiDiGraph, path_nodes: List[int]) -> List[Tuple[float, float]]:
        """
        Converts list of node IDs into coordinate pairs [(lat, lon), ...].
        """
        coords = []
        for n in path_nodes:
            lat = graph.nodes[n].get('lat', 0.0)
            lon = graph.nodes[n].get('lon', 0.0)
            coords.append((lat, lon))
        return coords

    @timer_decorator
    def analyze_route(self, origin: Tuple[float, float], destination: Tuple[float, float], route_name: str = "Route") -> Dict[str, Any]:
        """
        Compares route shortest-paths pre- and post-disaster, calculating detour ratios.
        """
        try:
            u_pre = self.find_nearest_node(self.pre_graph, origin)
            v_pre = self.find_nearest_node(self.pre_graph, destination)
        except Exception as e:
            logger.error(f"Failed to resolve nearest nodes for {route_name}: {e}")
            return {
                "route_name": route_name,
                "origin": origin,
                "destination": destination,
                "status": "NODE_NOT_FOUND",
                "detour_ratio": -1.0
            }

        # Pre-disaster route calculation
        try:
            pre_length, pre_path = nx.single_source_dijkstra(self.pre_graph, u_pre, v_pre, weight='length')
            pre_coords = self.get_path_coords(self.pre_graph, pre_path)
        except nx.NetworkXNoPath:
            logger.warning(f"No pre-disaster path exists for {route_name}.")
            return {
                "route_name": route_name,
                "origin": origin,
                "destination": destination,
                "status": "NO_PRE_PATH",
                "detour_ratio": -1.0
            }

        # Verify endpoints still exist in the post-disaster network
        if not (self.post_graph.has_node(u_pre) and self.post_graph.has_node(v_pre)):
            return {
                "route_name": route_name,
                "origin": origin,
                "destination": destination,
                "pre_disaster_path": pre_coords,
                "pre_disaster_distance": pre_length,
                "post_disaster_path": [],
                "post_disaster_distance": -1.0,
                "detour_ratio": -1.0,
                "status": "DISRUPTED_ISOLATED"
            }

        # Post-disaster route calculation
        try:
            post_length, post_path = nx.single_source_dijkstra(self.post_graph, u_pre, v_pre, weight='length')
            post_coords = self.get_path_coords(self.post_graph, post_path)
            detour_ratio = post_length / pre_length if pre_length > 0 else 1.0
            status = "RE_ROUTED" if post_path != pre_path else "UNHARMED"
        except nx.NetworkXNoPath:
            post_coords = []
            post_length = -1.0
            detour_ratio = -1.0
            status = "DISRUPTED_ISOLATED"

        return {
            "route_name": route_name,
            "origin": origin,
            "destination": destination,
            "pre_disaster_path": pre_coords,
            "pre_disaster_distance": pre_length,
            "post_disaster_path": post_coords,
            "post_disaster_distance": post_length,
            "detour_ratio": detour_ratio,
            "status": status
        }
