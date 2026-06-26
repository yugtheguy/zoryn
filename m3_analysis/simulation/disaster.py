from typing import Tuple, Dict, Any, Optional, List
import random
from enum import Enum
import networkx as nx
from utils.helpers import get_logger, haversine_distance, timer_decorator

logger = get_logger("DisasterSimulation")

class SimulationType(str, Enum):
    FLOOD = "Flood"
    BRIDGE_COLLAPSE = "Bridge Collapse"
    ROAD_BLOCK = "Road Block"
    MAJOR_JUNCTION_FAILURE = "Major Junction Failure"
    MULTIPLE_FAILURE = "Multiple Failure"
    RANDOM_INFRASTRUCTURE_FAILURE = "Random Infrastructure Failure"

class DisasterSimulator:
    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initializes simulator with a road network graph.
        """
        self.original_graph = graph

    def _collect_removed_edges_for_nodes(self, graph: nx.MultiDiGraph, nodes: List[int]) -> List[Tuple]:
        """
        Collects all edges connected to nodes that are being removed.
        """
        edges = []
        for node in nodes:
            if graph.has_node(node):
                # Add outgoing edges
                for u, v, k in graph.out_edges(node, keys=True):
                    edges.append((u, v, k))
                # Add incoming edges
                for u, v, k in graph.in_edges(node, keys=True):
                    edges.append((u, v, k))
        return list(set(edges))

    def _resolve_edges(self, graph: nx.MultiDiGraph, edge_list: List[Tuple]) -> List[Tuple[int, int, Any]]:
        """
        Resolves edge list (which may contain 2-tuples or 3-tuples) to standard 3-tuples.
        """
        resolved = []
        for edge in edge_list:
            if len(edge) == 3:
                u, v, k = edge
                if graph.has_edge(u, v, k):
                    resolved.append((u, v, k))
            elif len(edge) == 2:
                u, v = edge
                if graph.has_edge(u, v):
                    for k in graph[u][v].keys():
                        resolved.append((u, v, k))
        return list(set(resolved))

    @timer_decorator
    def simulate_flood(self, center: Tuple[float, float], radius_km: float) -> Dict[str, Any]:
        """
        Removes nodes (and connected edges) within a given radius of a coordinate center.
        """
        degraded_graph = self.original_graph.copy()
        nodes_to_remove = []

        for node, data in degraded_graph.nodes(data=True):
            lat = data.get('lat')
            lon = data.get('lon')
            if lat is not None and lon is not None:
                dist = haversine_distance((lat, lon), center)
                if dist <= radius_km:
                    nodes_to_remove.append(node)

        removed_edges = self._collect_removed_edges_for_nodes(degraded_graph, nodes_to_remove)
        degraded_graph.remove_nodes_from(nodes_to_remove)

        logger.info(f"Flood Simulation: Removed {len(nodes_to_remove)} nodes and {len(removed_edges)} edges.")
        return {
            "graph": degraded_graph,
            "removed_nodes": nodes_to_remove,
            "removed_edges": removed_edges,
            "simulation_type": SimulationType.FLOOD.value
        }

    @timer_decorator
    def simulate_bridge_collapse(self, edge_list: Optional[List[Tuple]] = None, edge_centrality: Optional[Dict[Tuple, float]] = None, fraction: float = 0.05) -> Dict[str, Any]:
        """
        Simulates bridge collapses by removing specific edges.
        Does not calculate centrality; relies on pre-computed values.
        """
        degraded_graph = self.original_graph.copy()
        edges_to_remove = []

        if edge_list is not None:
            edges_to_remove = self._resolve_edges(degraded_graph, edge_list)
        elif edge_centrality is not None:
            sorted_edges = sorted(edge_centrality.items(), key=lambda item: item[1], reverse=True)
            num_to_remove = int(len(sorted_edges) * fraction)
            raw_edges = [edge for edge, val in sorted_edges[:num_to_remove]]
            edges_to_remove = self._resolve_edges(degraded_graph, raw_edges)
        else:
            raise ValueError("Either edge_list or edge_centrality ranking must be provided for bridge collapse simulation.")

        for u, v, k in edges_to_remove:
            if degraded_graph.has_edge(u, v, k):
                degraded_graph.remove_edge(u, v, k)

        logger.info(f"Bridge Collapse Simulation: Removed {len(edges_to_remove)} edges.")
        return {
            "graph": degraded_graph,
            "removed_nodes": [],
            "removed_edges": edges_to_remove,
            "simulation_type": SimulationType.BRIDGE_COLLAPSE.value
        }

    @timer_decorator
    def simulate_road_block(self, edge_list: List[Tuple]) -> Dict[str, Any]:
        """
        Simulates road blocks by removing specific edges.
        """
        degraded_graph = self.original_graph.copy()
        edges_to_remove = self._resolve_edges(degraded_graph, edge_list)

        for u, v, k in edges_to_remove:
            if degraded_graph.has_edge(u, v, k):
                degraded_graph.remove_edge(u, v, k)

        logger.info(f"Road Block Simulation: Removed {len(edges_to_remove)} edges.")
        return {
            "graph": degraded_graph,
            "removed_nodes": [],
            "removed_edges": edges_to_remove,
            "simulation_type": SimulationType.ROAD_BLOCK.value
        }

    @timer_decorator
    def simulate_major_junction_failure(self, node_list: Optional[List[int]] = None, node_centrality: Optional[Dict[int, float]] = None, fraction: float = 0.05) -> Dict[str, Any]:
        """
        Simulates major junction failures by removing specific intersection nodes.
        Does not calculate centrality; relies on pre-computed values.
        """
        degraded_graph = self.original_graph.copy()
        nodes_to_remove = []

        if node_list is not None:
            nodes_to_remove = [n for n in node_list if degraded_graph.has_node(n)]
        elif node_centrality is not None:
            sorted_nodes = sorted(node_centrality.items(), key=lambda item: item[1], reverse=True)
            num_to_remove = int(len(sorted_nodes) * fraction)
            nodes_to_remove = [node for node, val in sorted_nodes[:num_to_remove]]
        else:
            raise ValueError("Either node_list or node_centrality ranking must be provided for major junction failure simulation.")

        removed_edges = self._collect_removed_edges_for_nodes(degraded_graph, nodes_to_remove)
        degraded_graph.remove_nodes_from(nodes_to_remove)

        logger.info(f"Major Junction Failure Simulation: Removed {len(nodes_to_remove)} nodes and {len(removed_edges)} edges.")
        return {
            "graph": degraded_graph,
            "removed_nodes": nodes_to_remove,
            "removed_edges": removed_edges,
            "simulation_type": SimulationType.MAJOR_JUNCTION_FAILURE.value
        }

    @timer_decorator
    def simulate_random_infrastructure_failure(self, target_type: str = "node", fraction: float = 0.05) -> Dict[str, Any]:
        """
        Simulates random failures of network nodes or edges.
        """
        degraded_graph = self.original_graph.copy()
        removed_nodes = []
        removed_edges = []

        if target_type == "node":
            nodes = list(degraded_graph.nodes())
            num_to_remove = int(len(nodes) * fraction)
            removed_nodes = random.sample(nodes, num_to_remove)
            removed_edges = self._collect_removed_edges_for_nodes(degraded_graph, removed_nodes)
            degraded_graph.remove_nodes_from(removed_nodes)
        elif target_type == "edge":
            edges = list(degraded_graph.edges(keys=True))
            num_to_remove = int(len(edges) * fraction)
            removed_edges = random.sample(edges, num_to_remove)
            for u, v, k in removed_edges:
                if degraded_graph.has_edge(u, v, k):
                    degraded_graph.remove_edge(u, v, k)
        else:
            raise ValueError(f"Invalid target_type: {target_type}. Must be 'node' or 'edge'.")

        logger.info(f"Random Infrastructure Failure: Removed {len(removed_nodes)} nodes and {len(removed_edges)} edges.")
        return {
            "graph": degraded_graph,
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges,
            "simulation_type": SimulationType.RANDOM_INFRASTRUCTURE_FAILURE.value
        }

    @timer_decorator
    def simulate_multiple_failure(self, node_list: Optional[List[int]] = None, edge_list: Optional[List[Tuple]] = None, flood_center: Optional[Tuple[float, float]] = None, flood_radius_km: Optional[float] = None) -> Dict[str, Any]:
        """
        Simulates multiple concurrent failures (e.g. flood zone + specific road blocks + bridge collapses).
        """
        degraded_graph = self.original_graph.copy()
        total_nodes_removed = []
        total_edges_removed = []

        # 1. Apply flood if defined
        if flood_center is not None and flood_radius_km is not None:
            flood_res = self.simulate_flood(flood_center, flood_radius_km)
            total_nodes_removed.extend(flood_res["removed_nodes"])
            total_edges_removed.extend(flood_res["removed_edges"])
            degraded_graph = flood_res["graph"]

        # 2. Apply node failures
        if node_list is not None:
            nodes_to_remove = [n for n in node_list if degraded_graph.has_node(n)]
            node_edges = self._collect_removed_edges_for_nodes(degraded_graph, nodes_to_remove)
            total_nodes_removed.extend(nodes_to_remove)
            total_edges_removed.extend(node_edges)
            degraded_graph.remove_nodes_from(nodes_to_remove)

        # 3. Apply edge failures
        if edge_list is not None:
            edges_to_remove = self._resolve_edges(degraded_graph, edge_list)
            total_edges_removed.extend(edges_to_remove)
            for u, v, k in edges_to_remove:
                if degraded_graph.has_edge(u, v, k):
                    degraded_graph.remove_edge(u, v, k)

        # Deduplicate
        total_nodes_removed = list(set(total_nodes_removed))
        total_edges_removed = list(set(total_edges_removed))

        logger.info(f"Multiple Failure Simulation: Total removed nodes: {len(total_nodes_removed)}, total removed edges: {len(total_edges_removed)}")
        return {
            "graph": degraded_graph,
            "removed_nodes": total_nodes_removed,
            "removed_edges": total_edges_removed,
            "simulation_type": SimulationType.MULTIPLE_FAILURE.value
        }
