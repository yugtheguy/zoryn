import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from utils.helpers import get_logger
from config import DEFAULT_OUTPUT_DIR, METADATA_MODULE, METADATA_VERSION

logger = get_logger("JSONExporter")

class JSONExporter:
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        """
        Initializes exporter with an output directory.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _inject_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Injects a standard metadata block at the top level of the JSON payload.
        """
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "module": METADATA_MODULE,
            "version": METADATA_VERSION
        }
        # Place metadata first while preserving the rest of the keys
        return {"metadata": metadata, **data}

    def export_centrality(self, node_centrality: Dict[str, Dict[int, float]], edge_centrality: Dict[Tuple[int, int], float]) -> str:
        """
        Serializes and exports centrality rankings as road_centrality.json.
        """
        nodes_list = []
        bet = node_centrality.get("betweenness", {})
        cls = node_centrality.get("closeness", {})
        deg = node_centrality.get("degree", {})

        for n in deg.keys():
            nodes_list.append({
                "id": n,
                "betweenness": bet.get(n, 0.0),
                "closeness": cls.get(n, 0.0),
                "degree": deg.get(n, 0.0)
            })

        edges_list = []
        for edge_tuple, val in edge_centrality.items():
            if len(edge_tuple) == 3:
                u, v, _ = edge_tuple
            else:
                u, v = edge_tuple
            edges_list.append({
                "u": u,
                "v": v,
                "edge_betweenness": val
            })

        data = {
            "nodes": nodes_list,
            "edges": edges_list
        }

        final_data = self._inject_metadata(data)
        filepath = os.path.join(self.output_dir, "road_centrality.json")
        with open(filepath, 'w') as f:
            json.dump(final_data, f, indent=2)

        logger.info(f"Exported centrality scores to {filepath}")
        return filepath

    def export_disaster_metrics(self, scenario_type: str, disaster_params: Dict[str, Any], pre_metrics: Dict[str, Any], post_metrics: Dict[str, Any], removed_nodes: Optional[List[int]] = None, removed_edges: Optional[List[Tuple]] = None) -> str:
        """
        Serializes and exports delta network health metrics as disaster_metrics.json.
        """
        pre_eff = pre_metrics.get("efficiency", 0.0)
        post_eff = post_metrics.get("efficiency", 0.0)
        eff_loss = ((pre_eff - post_eff) / pre_eff * 100.0) if pre_eff > 0 else 0.0

        pre_path = pre_metrics.get("average_shortest_path", 0.0)
        post_path = post_metrics.get("average_shortest_path", 0.0)
        path_inc = ((post_path - pre_path) / pre_path * 100.0) if pre_path > 0 else 0.0

        pre_isolated = pre_metrics.get("isolated_nodes_count", 0)
        post_isolated = post_metrics.get("isolated_nodes_count", 0)
        newly_isolated = max(0, post_isolated - pre_isolated)

        # Standardize removed edges to [[u, v, key], ...] for JSON compatibility
        formatted_edges = []
        if removed_edges:
            for edge in removed_edges:
                if len(edge) == 3:
                    formatted_edges.append([edge[0], edge[1], str(edge[2])])
                elif len(edge) == 2:
                    formatted_edges.append([edge[0], edge[1], "0"])

        data = {
            "scenario_type": scenario_type,
            "disaster_params": disaster_params,
            "removed_nodes": removed_nodes or [],
            "removed_edges": formatted_edges,
            "metrics": {
                "pre_disaster": pre_metrics,
                "post_disaster": post_metrics,
                "delta": {
                    "efficiency_loss_percent": eff_loss,
                    "path_increase_percent": path_inc,
                    "newly_isolated_nodes": newly_isolated
                }
            }
        }

        final_data = self._inject_metadata(data)
        filepath = os.path.join(self.output_dir, "disaster_metrics.json")
        with open(filepath, 'w') as f:
            json.dump(final_data, f, indent=2)

        logger.info(f"Exported disaster metrics to {filepath}")
        return filepath

    def export_routing_detours(self, route_analyses: List[Dict[str, Any]]) -> str:
        """
        Serializes and exports detour route details as routing_detours.json.
        """
        data = {
            "routes": route_analyses
        }

        final_data = self._inject_metadata(data)
        filepath = os.path.join(self.output_dir, "routing_detours.json")
        with open(filepath, 'w') as f:
            json.dump(final_data, f, indent=2)

        logger.info(f"Exported routing detours to {filepath}")
        return filepath

    def export_ai_advisory(self, advisory: Dict[str, Any]) -> str:
        """
        Serializes and exports LLM advisory reports as ai_advisory.json.
        """
        final_data = self._inject_metadata(advisory)
        filepath = os.path.join(self.output_dir, "ai_advisory.json")
        with open(filepath, 'w') as f:
            json.dump(final_data, f, indent=2)

        logger.info(f"Exported AI advisory to {filepath}")
        return filepath
