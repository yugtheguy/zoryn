# ISRO Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis

**Official Hackathon Repository — Member 2 Lead (Graph Pipeline & Computational Geometry Lead)**

---

## Project Overview

In post-disaster urban environments or heavily occluded satellite imagery (tree canopies, building shadows, dense traffic), road segmentation masks frequently suffer from fragmentation and disconnectivity. This repository implements the complete mathematical graph architecture that bridges computer vision segmentation and urban infrastructure resilience dashboarding:

```text
Binary Road Mask &rarr; Morphological Thinning &rarr; Graph Extraction &rarr; Gap Detection &rarr; Union-Find MST Healing &rarr; Brandes Betweenness Centrality &rarr; Disaster Stress Testing &rarr; Dashboard JSON & GeoJSON Exports
```

---

## Architecture & Responsibilities

This repository strictly implements **Member 2 Responsibilities (Modules 1 through 11 plus Bonus Features)**:
*   **Module 1 (Skeletonization)**: Morphological 2D thinning and salt-and-pepper noise filtering (`m2_graph/skeletonize.py`).
*   **Module 2 (Graph Extraction)**: Endpoint (degree 1) and Junction (degree $\ge 3$) detection, compacting continuous road paths into weighted geometric edges (`m2_graph/extract.py`).
*   **Module 3 (Component Analysis)**: Connected component fragmentation statistics (`m2_graph/components.py`).
*   **Module 4 (Gap Detection)**: Pairwise Euclidean distance filtering and cosine angular alignment scoring (`m2_graph/gaps.py`).
*   **Module 5 (Topological Healing)**: Disjoint Set Union (Union-Find) and Kruskal Minimum Spanning Tree reconnection (`m2_graph/healing.py`).
*   **Module 6 (Connectivity Evaluation)**: Quantifying network improvement ratios (`m2_graph/eval_connectivity.py`).
*   **Module 7 (Betweenness Centrality)**: Shortest-path node & edge centrality ranking and Single Points of Failure (articulation cut vertices) (`m3_analysis/centrality.py`).
*   **Module 8 (Criticality Ranking)**: Sorted infrastructure prioritization tables (`m3_analysis/ranking.py`).
*   **Module 9 (Network Stress Testing)**: Simulating single-node and multi-node disaster knockouts (`m3_analysis/simulation.py`).
*   **Module 10 (Resilience Index)**: Macro robustness metric $BaselineASP / DamagedASP$ (`m3_analysis/resilience.py`).
*   **Module 11 & Bonus**: Streamlit/Leaflet/QGIS dashboard JSON consolidation and standard GeoJSON export (`m3_analysis/dashboard_export.py`, `m3_analysis/reroute.py`).

---

## Installation & Setup

1. **Clone the repository and install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify environment via Automated Unit Tests**:
   ```bash
   python -m unittest discover tests/ -v
   ```

---

## CLI Pipeline Usage

### Quickstart Standalone Demo
Run the complete pipeline end-to-end out of the box using synthetic occluded urban test data:

```bash
# Step 1: Generate realistic synthetic binary road mask with occlusions
python generate_sample_data.py --output sample_mask.png --width 512 --height 512

# Step 2: Run master graph pipeline orchestrator
python pipeline.py --input_mask sample_mask.png --output_dir dashboard_exports/ --max_gap_dist 65.0 --simulate_k 1 3 5
```

### CLI Command Options (`pipeline.py`)
*   `--input_mask`: Absolute or relative path to input binary segmentation mask image (required).
*   `--output_dir`: Directory where dashboard JSON and GeoJSON artifacts will be written (default: `dashboard_exports/`).
*   `--max_gap_dist`: Maximum Euclidean distance threshold for healing road gaps (default: `65.0`).
*   `--min_angle_score`: Minimum cosine angular similarity alignment score $[0, 1]$ (default: `0.3`).
*   `--min_size`: Minimum connected component pixel area to keep before thinning (default: `20`).
*   `--simulate_k`: Space-separated integer list representing multi-node knockout stress test counts (default: `1 3 5 10`).

---

## Deliverable Artifacts Contract

Upon execution, `pipeline.py` writes the following dashboard-ready contracts to `--output_dir`:
1. `skeleton.png`: 1-pixel thick thinned road image.
2. `raw_road_graph.gpickle` / `.gml`: Fragmented pre-healing graph object.
3. `healed_road_graph.gpickle` / `.gml`: Continuous reconnected road network.
4. `graph_data.json`: Full graph geometry containing node coordinates and edge pixel line paths.
5. `critical_nodes.json`: Betweenness centrality scores, gatekeeper flags, and Single Points of Failure.
6. `heatmap_edges.json`: Road segment edge betweenness values for GIS polyline coloring.
7. `ranked_nodes.json`: Top-10, Top-20, and complete sorted criticality rankings.
8. `simulation_results.json`: Network efficiency degradation benchmarks under simulated disasters.
9. `resilience_report.json`: Macro network resilience index and qualitative vulnerability classification.
10. `network_export.geojson`: Standard GIS FeatureCollection for QGIS and Leaflet web mapping.

---

## Documentation References
*   [System Architecture Diagram & Data Contracts](docs/ARCHITECTURE.md)
*   [Algorithmic Complexity & Scalability Proofs](docs/COMPLEXITY_ANALYSIS.md)
