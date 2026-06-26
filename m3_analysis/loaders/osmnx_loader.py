from typing import Optional, Tuple
import osmnx as ox
import networkx as nx
from loaders.base import BaseGraphLoader
from utils.helpers import get_logger

logger = get_logger("OSMnxLoader")

class OSMnxGraphLoader(BaseGraphLoader):
    def __init__(
        self,
        place_query: Optional[str] = "Manhattan, New York, USA",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        network_type: str = "drive"
    ):
        """
        Args:
            place_query: Place query string.
            bbox: Bounding box coordinates (north, south, east, west).
            network_type: Type of road network.
        """
        self.place_query = place_query
        self.bbox = bbox
        self.network_type = network_type

    def load_graph(self) -> nx.MultiDiGraph:
        if self.bbox is not None:
            north, south, east, west = self.bbox
            logger.info(f"Downloading street network from OSMnx for bbox: north={north}, south={south}, east={east}, west={west}")
            graph = ox.graph_from_bbox(bbox=(north, south, east, west), network_type=self.network_type)
        elif self.place_query is not None:
            logger.info(f"Downloading street network from OSMnx for place: {self.place_query}")
            graph = ox.graph_from_place(self.place_query, network_type=self.network_type)
        else:
            raise ValueError("Either place_query or bbox must be provided to OSMnxGraphLoader.")

        # Ensure consistent node attributes for lat/lon coordinates
        for node, data in graph.nodes(data=True):
            if 'lat' not in data and 'y' in data:
                data['lat'] = data['y']
            if 'lon' not in data and 'x' in data:
                data['lon'] = data['x']

        logger.info(f"Loaded OSMnx graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        return graph
