# Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility

[![ISRO Hackathon](https://img.shields.io/badge/ISRO%20Hackathon-Route%20Resilience-ff9933?style=for-the-badge&logo=space-exploration-masters)](https://github.com/yugtheguy/zoryn)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Theory-00599C?style=for-the-badge&logo=scipy&logoColor=white)](https://networkx.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## Executive Summary & Mission Objective

During natural disasters (floods, earthquakes, cyclones) or in dense urban clutter, satellite remote sensing imagery is frequently compromised by severe occlusions: tree canopies, cloud cover, building shadows, smoke, and structural debris. Standard deep learning segmentation models output fragmented, disconnected road binary blobs that cannot be directly used by emergency responders or routing algorithms.

**Route Resilience** solves this urban mobility challenge by establishing an end-to-end mathematical architecture bridging computational geometry, graph theory, and geographic information systems (GIS):
1. **Deep Learning Segmentation**: Extracts raw binary road pixel masks from high-resolution satellite imagery.
2. **Topological Healing**: Thins blobs into 1-pixel centerlines and autonomously reconstructs missing road links across occlusion gaps using Minimum Spanning Trees (MST).
3. **Graph-Theoretic Criticality**: Evaluates infrastructure bottlenecks using Brandes' Betweenness Centrality and Hopcroft-Tarjan articulation point cut-vertices (Single Points of Failure).
4. **Disaster Stress Testing**: Simulates simultaneous multi-junction failure knockouts to compute an executive macro **Resilience Index**.
5. **Interactive Simulation Dashboard**: Renders real-time vector GeoJSON overlays, chokepoint heatmaps, and alternate detour routes for urban planners.

---

## Master System Architecture

```text
+-------------------------------------------------------------------------------------------------------------------+
|                                            RAW SATELLITE IMAGERY (.tif/.png)                                      |
+-------------------------------------------------------------------------------------------------------------------+
                                                          |
                                                          v
+-------------------------------------------------------------------------------------------------------------------+
|     M1: SEGMENTATION LEAD (DeepLabV3+ / U-Net / SegFormer PyTorch Models)                                         |
|     -> Outputs: Binary Road Mask (road=255, background=0)                                                         |
+-------------------------------------------------------------------------------------------------------------------+
                                                          |
                                                          v
+-------------------------------------------------------------------------------------------------------------------+
|     M2: GRAPH PIPELINE LEAD (Computational Geometry & Topological Healing)                                        |
|     -> Module 1: Morphological Centerline Thinning (skimage Lee's algorithm + salt-noise removal)                 |
|     -> Module 2: Compact Mathematical Graph Extraction G=(V,E) (Endpoint deg=1 & Junction deg>=3 detection)       |
|     -> Module 3: Connected Component Fragmentation Analysis (LCC reachability metrics)                            |
|     -> Module 4: Occlusion Gap Detection (Pairwise Euclidean distance & outward heading cosine scoring)           |
|     -> Module 5: Topological Healing (Disjoint Set Union DSU + Kruskal MST bridge reconstruction)                 |
|     -> Module 6: Connectivity Improvement Evaluation (ConnectivityRatio = LCC_after / LCC_before)                 |
+-------------------------------------------------------------------------------------------------------------------+
                                                          |
                                                          v
+-------------------------------------------------------------------------------------------------------------------+
|     M3: NETWORK ANALYSIS LEAD (Graph-Theoretic Criticality & Stress Benchmarking)                                 |
|     -> Module 7: Betweenness Centrality & SPOF Detection (Brandes shortest-paths & articulation cut-vertices)     |
|     -> Module 8: Critical Node Ranking Tables (Top-10 / Top-20 Gatekeeper intersection chokepoints)               |
|     -> Module 9: Network Stress Testing (Simulating simultaneous Top-K knockout disaster knockouts)               |
|     -> Module 10: Macro Resilience Index Calculation (ResilienceIndex = BaselineASP / DamagedASP)                 |
|     -> Module 11 & Bonus: Route Detour Rerouting Engine & Consolidated Dashboard/GIS Deliverable Serializer       |
+-------------------------------------------------------------------------------------------------------------------+
                                                          |
                                                          v
+-------------------------------------------------------------------------------------------------------------------+
|     M4: DASHBOARD & VISUALIZATION LEAD (Streamlit / Leaflet / QGIS UI Overlays)                                   |
|     -> Ingests: graph_data.json, critical_nodes.json, heatmap_edges.json, simulation_results.json, .geojson       |
+-------------------------------------------------------------------------------------------------------------------+
```

---

## Modular Division of Team Responsibilities

```text
├── m1_segmentation/       # [Member 1 Lead] Deep learning satellite road extraction (PyTorch)
│   ├── data_loader.py     # Satellite imagery ingestion & Albumentations augmentation pipelines
│   ├── model.py           # U-Net / SegFormer model architectures
│   ├── losses.py          # Focal, Dice, and BCE loss formulations
│   ├── train.py           # Kaggle GPU training loop orchestrator
│   └── inference.py       # Sliding-window prediction generator
│
├── m2_graph/              # [Member 2 Lead] Centerline extraction & topological gap healing
│   ├── skeletonize.py     # [Module 1] Morphological thinning & noise suppression
│   ├── extract.py         # [Module 2] Compact NetworkX graph extraction G=(V,E)
│   ├── components.py      # [Module 3] Fragmentation & component size statistics
│   ├── gaps.py            # [Module 4] Angular vector alignment & gap scoring
│   ├── healing.py         # [Module 5] Union-Find DSU & Kruskal MST healing
│   ├── eval_connectivity.py # [Module 6] ConnectivityRatio calculation
│   └── export_graph.py    # Serializers (.gpickle, .gml)
│
├── m3_analysis/           # [Member 3 Lead] Graph theory criticality & stress testing
│   ├── centrality.py      # [Module 7] Shortest-path betweenness centrality & SPOF cut-vertices
│   ├── ranking.py         # [Module 8] Sorted chokepoint ranking tables
│   ├── simulation.py      # [Module 9] Top-K multi-node disaster knockout stress simulations
│   ├── resilience.py      # [Module 10] Macro network robustness index & classification
│   ├── reroute.py         # [Bonus] Detour route delay computation engine
│   └── dashboard_export.py # [Module 11] Dashboard JSON & GeoJSON formatters
│
├── m4_dashboard/          # [Member 4 Lead] Interactive web mapping & disaster dashboard
│   └── app.py             # Streamlit visual simulation application
│
├── pipeline.py            # Master CLI orchestrator connecting M1 -> M2 -> M3 -> M4
├── generate_sample_data.py # Standalone realistic synthetic binary road mask generator
├── run_kaggle.sh          # Kaggle environment setup script
└── tests/                 # Automated algorithmic test suite (unittest / pytest)
```

---

## Quickstart & Installation

### 1. Local Setup
Clone the repository and install required core computer vision, graph theory, and dashboard libraries:
```bash
git clone https://github.com/yugtheguy/zoryn.git
cd zoryn
pip install -r requirements.txt
```

### 2. Verify Algorithmic Integrity via Unit Tests
Execute the built-in automated verification suite to validate mathematical geometry and graph proofs:
```bash
python -m unittest discover tests/ -v
```

---

## End-to-End Execution Pipeline

You can run the entire 11-module computational pipeline out of the box using synthetic occluded urban test data:

```bash
# Step 1: Generate realistic synthetic binary road mask with occlusions (tree canopy/building shadows)
python generate_sample_data.py --output sample_mask.png --width 512 --height 512

# Step 2: Execute master graph extraction, topological healing, and disaster simulation pipeline
python pipeline.py --input_mask sample_mask.png --output_dir dashboard_exports/ --max_gap_dist 65.0 --simulate_k 1 3 5 10
```

### CLI Command Arguments (`pipeline.py`)
*   `--input_mask`: Path to input binary road mask image generated by M1 models (required).
*   `--output_dir`: Target directory for dashboard JSON and GeoJSON deliverables (default: `dashboard_exports/`).
*   `--max_gap_dist`: Maximum Euclidean distance allowed when candidate matching road gaps (default: `65.0`).
*   `--min_angle_score`: Minimum cosine angular similarity score $[0, 1]$ required to prevent false cross-roads (default: `0.3`).
*   `--min_size`: Minimum connected component pixel area to retain before thinning (filters speckle noise) (default: `20`).
*   `--simulate_k`: Space-separated list of integers representing simultaneous knockout counts for disaster simulation (default: `1 3 5 10`).

---

## Downstream Dashboard JSON & GIS Contracts

Downstream visualization dashboards (`m4_dashboard/app.py`, Leaflet, QGIS) ingest the following standardized contracts produced inside `--output_dir`:

| Artifact Filename | Contract Description | Downstream UI Application |
| :--- | :--- | :--- |
| `skeleton.png` | 1-pixel thinned binary road centerlines. | Image overlay comparison. |
| `raw_road_graph.gpickle` | Fragmented NetworkX graph pre-healing. | NetworkX debugging tools. |
| `healed_road_graph.gpickle`| Fully connected routable NetworkX graph. | Shortest-path routing engines. |
| `graph_data.json` | Complete graph topology (node $(x,y)$ coords & edge pixel polylines). | Streamlit canvas / Leaflet polylines. |
| `critical_nodes.json` | Node betweenness centrality values, gatekeeper flags, and SPOFs. | Map markers / Red alert beacons. |
| `heatmap_edges.json` | Road segment edge betweenness values. | GIS polyline color gradient heatmaps. |
| `ranked_nodes.json` | Top-10 and Top-20 prioritized infrastructure chokepoints. | UI leaderboard tables. |
| `simulation_results.json` | Network global efficiency drop benchmarks across Top-$K$ knockouts. | Streamlit line charts / stress curves. |
| `resilience_report.json` | Executive macro score ($BaselineASP / DamagedASP$) & classification. | Executive KPI scorecard metrics. |
| `network_export.geojson` | RFC 7946 standard GeoJSON FeatureCollection (Points & LineStrings). | Direct QGIS / Web GIS drag-and-drop. |

---

## Kaggle Training vs. GitHub Deployment Strategy

1. **GitHub**: Acts as the single source of truth for deterministic computational algorithms, graph heuristics, unit tests, and dashboard code.
2. **Kaggle**: Used exclusively for GPU training of M1 deep learning segmentation models. Inside a Kaggle Notebook, attach datasets via *Add Input* and execute:
   ```bash
   !bash run_kaggle.sh
   ```
   Checkpoints and predicted probability masks are saved locally on the instance at `/kaggle/working/predictions/`.

---

## Documentation & Algorithmic Proofs

For deep mathematical derivations, space/time asymptotic bounds, and system interaction sequence diagrams, refer to:
*   [Detailed System Architecture & Data Dictionary](docs/ARCHITECTURE.md)
*   [Computational Complexity & Scalability Proofs ($O(V+E)$, Brandes $O(VE)$)](docs/COMPLEXITY_ANALYSIS.md)
