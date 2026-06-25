import unittest
import networkx as nx
from m3_analysis import (
    compute_betweenness_centrality,
    rank_critical_nodes,
    simulate_network_failures,
    compute_resilience_index,
    compute_rerouting_impact
)

class TestCentralitySimulation(unittest.TestCase):
    def setUp(self):
        # Create a bowtie graph where node 3 is the bridge gatekeeper
        self.g = nx.Graph()
        self.g.add_edges_from([(1,2), (2,3), (1,3), (3,4), (4,5), (3,5)])
        for n in self.g.nodes():
            self.g.nodes[n]["x"] = n*10
            self.g.nodes[n]["y"] = 10
            self.g.nodes[n]["degree"] = self.g.degree[n]
        for u, v in self.g.edges():
            self.g[u][v]["weight"] = 1.0

    def test_centrality_and_ranking(self):
        bc_report = compute_betweenness_centrality(self.g)
        self.assertIn(3, bc_report["spof_node_ids"]) # Cut vertex
        
        ranking = rank_critical_nodes(bc_report)
        self.assertEqual(ranking["top_10"][0]["node_id"], 3)

    def test_simulation_and_resilience(self):
        bc_report = compute_betweenness_centrality(self.g)
        ranking = rank_critical_nodes(bc_report)["full_ranking"]
        
        sims = simulate_network_failures(self.g, ranking, k_values=[1])
        self.assertIn("top_1_knockouts", sims["multi_node_failures"])
        
        res = compute_resilience_index(sims)
        self.assertTrue(0.0 <= res["overall_resilience_index"] <= 1.0)

    def test_rerouting(self):
        # In bowtie, path from 1 to 5 goes through 3. If 3 fails, no path exists.
        reroute = compute_rerouting_impact(self.g, 1, 5, failed_nodes=[3])
        self.assertFalse(reroute["is_reachable_after_failure"])

if __name__ == '__main__':
    unittest.main()
