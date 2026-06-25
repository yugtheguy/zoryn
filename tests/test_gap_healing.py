import unittest
import networkx as nx
from m2_graph import detect_occlusion_gaps, heal_road_topology, evaluate_healing_improvement

class TestGapHealing(unittest.TestCase):
    def test_gap_detection_and_healing(self):
        g = nx.Graph()
        # Segment 1: (10,10) -> (20,10)
        g.add_node(1, x=10, y=10, degree=1, type="endpoint")
        g.add_node(2, x=20, y=10, degree=1, type="endpoint")
        g.add_edge(1, 2, length=10.0, geometry=[(10,10), (15,10), (20,10)])
        
        # Segment 2: (30,10) -> (40,10) across gap of 10px
        g.add_node(3, x=30, y=10, degree=1, type="endpoint")
        g.add_node(4, x=40, y=10, degree=1, type="endpoint")
        g.add_edge(3, 4, length=10.0, geometry=[(30,10), (35,10), (40,10)])
        
        gaps = detect_occlusion_gaps(g, max_distance=20.0, min_angle_score=0.2)
        self.assertTrue(len(gaps) >= 1)
        
        # Check that top candidate connects 2 and 3
        best = gaps[0]
        self.assertEqual({best["source"], best["target"]}, {2, 3})
        
        healed = heal_road_topology(g, gaps)
        self.assertTrue(healed.has_edge(2, 3) or healed.has_edge(3, 2))
        
        eval_report = evaluate_healing_improvement(g, healed)
        self.assertEqual(eval_report["healing_metrics"]["components_merged"], 1)

if __name__ == '__main__':
    unittest.main()
