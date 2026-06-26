from abc import ABC, abstractmethod
import networkx as nx

class BaseGraphLoader(ABC):
    """Abstract base class for network graph loaders."""

    @abstractmethod
    def load_graph(self) -> nx.MultiDiGraph:
        """Loads and returns a networkx MultiDiGraph."""
        pass
