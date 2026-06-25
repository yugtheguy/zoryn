import unittest
import numpy as np
import networkx as nx
from m2_graph import extract_road_graph, analyze_connected_components

class TestGraphExtraction(unittest.TestCase):
    def test_extract_line_endpoints(self):
        skel = np.zeros((30, 30), dtype=np.uint8)
        skel[15, 5:25] = 255 # Straight horizontal line length 20
        
        graph = extract_road_graph(skel)
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        
        # Check edge length ~ 19.0
        edge = next(iter(graph.edges(data=True)))[2]
        self.assertAlmostEqual(edge["length"], 19.0, delta=1.0)

    def test_components_analysis(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_node(3) # Isolated
        
        report = analyze_connected_components(g)
        self.assertEqual(report["num_connected_components"], 2)
        self.assertEqual(report["largest_component_nodes"], 2)

if __name__ == '__main__':
    unittest.main()
