import os
import pickle
import json
import networkx as nx
from loaders.base import BaseGraphLoader
from utils.helpers import get_logger, haversine_distance

logger = get_logger("M2Loader")

class M2GraphLoader(BaseGraphLoader):
    def __init__(self, filepath: str):
        """
        Args:
            filepath: Path to the gpickle or geojson file.
        """
        self.filepath = filepath

    def load_graph(self) -> nx.MultiDiGraph:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Member 2 graph file not found at {self.filepath}")

        _, ext = os.path.splitext(self.filepath.lower())

        if ext == '.gpickle':
            logger.info(f"Loading Member 2 networkx graph from pickle: {self.filepath}")
            with open(self.filepath, 'rb') as f:
                graph = pickle.load(f)
            if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                raise TypeError(f"Loaded object is not a networkx Graph: {type(graph)}")
            # Enforce MultiDiGraph representation
            if not isinstance(graph, nx.MultiDiGraph):
                graph = nx.MultiDiGraph(graph)
        elif ext in ['.geojson', '.json']:
            logger.info(f"Constructing networkx graph from GeoJSON: {self.filepath}")
            graph = self._load_from_geojson()
        else:
            raise ValueError(f"Unsupported file format: {ext}. Must be .gpickle or .geojson")

        # Ensure consistent node attributes for coordinates
        for node, data in graph.nodes(data=True):
            if 'lat' not in data and 'y' in data:
                data['lat'] = data['y']
            if 'lon' not in data and 'x' in data:
                data['lon'] = data['x']

        logger.info(f"Successfully loaded Member 2 graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        return graph

    def _load_from_geojson(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        with open(self.filepath, 'r') as f:
            data = json.load(f)

        features = data.get('features', [])
        coord_to_node_id = {}
        node_counter = 0

        def get_node_id(lon: float, lat: float) -> int:
            nonlocal node_counter
            # Standardize coordinates slightly to merge extremely close vertices
            key = (round(lon, 6), round(lat, 6))
            if key not in coord_to_node_id:
                coord_to_node_id[key] = node_counter
                graph.add_node(node_counter, lat=lat, lon=lon, x=lon, y=lat)
                node_counter += 1
            return coord_to_node_id[key]

        for feature in features:
            geom = feature.get('geometry', {})
            props = feature.get('properties', {})
            geom_type = geom.get('type')

            if geom_type == 'LineString':
                coords = geom.get('coordinates', [])
                if len(coords) < 2:
                    continue
                for i in range(len(coords) - 1):
                    lon1, lat1 = coords[i]
                    lon2, lat2 = coords[i+1]

                    u = get_node_id(lon1, lat1)
                    v = get_node_id(lon2, lat2)

                    # Length in meters
                    length = haversine_distance((lat1, lon1), (lat2, lon2)) * 1000.0

                    edge_attrs = {
                        'length': props.get('length', length),
                        'name': props.get('name', f"segment_{u}_{v}"),
                        'oneway': props.get('oneway', False),
                        **props
                    }
                    graph.add_edge(u, v, **edge_attrs)

            elif geom_type == 'MultiLineString':
                lines = geom.get('coordinates', [])
                for coords in lines:
                    if len(coords) < 2:
                        continue
                    for i in range(len(coords) - 1):
                        lon1, lat1 = coords[i]
                        lon2, lat2 = coords[i+1]

                        u = get_node_id(lon1, lat1)
                        v = get_node_id(lon2, lat2)

                        length = haversine_distance((lat1, lon1), (lat2, lon2)) * 1000.0

                        edge_attrs = {
                            'length': props.get('length', length),
                            'name': props.get('name', f"segment_{u}_{v}"),
                            'oneway': props.get('oneway', False),
                            **props
                        }
                        graph.add_edge(u, v, **edge_attrs)

        return graph
