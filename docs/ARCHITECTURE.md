# ISRO Route Resilience: System Architecture (Member 2 Lead)

This document describes the computational geometry and graph-theoretic engineering architecture for **Member 2 (Graph Pipeline Lead)**.

## Architecture & Data Flow Diagram

```mermaid
graph TD
    classDef seg fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef m2 fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#fff;
    classDef m3 fill:#18181b,stroke:#a855f7,stroke-width:2px,color:#fff;
    classDef dash fill:#334155,stroke:#f59e0b,stroke-width:2px,color:#fff;

    Mask["Binary Road Mask<br/>(mask.png)"] :::seg
    
    subgraph M2 Package [M2: Graph Extraction & Topological Healing]
        Mod1["Module 1: Skeletonization<br/>(skimage morphological thinning)"] :::m2
        Mod2["Module 2: Graph Extraction<br/>(Endpoint/Junction detection)"] :::m2
        Mod3["Module 3: Component Analysis<br/>(nx.connected_components)"] :::m2
        Mod4["Module 4: Gap Detection<br/>(Euclidean & Angular Cosine Scoring)"] :::m2
        Mod5["Module 5: Topological Healing<br/>(Union-Find DSU & Kruskal MST)"] :::m2
        Mod6["Module 6: Connectivity Eval<br/>(LCC Ratio & Improvement)"] :::m2
    end

    subgraph M3 Package [M3: Criticality & Disaster Simulation]
        Mod7["Module 7: Betweenness Centrality<br/>(Brandes Shortest Path)"] :::m3
        Mod8["Module 8: Critical Node Ranking<br/>(Top-10 / Top-20 Gatekeepers)"] :::m3
        Mod9["Module 9: Network Stress Testing<br/>(Single & Top-K Knockouts)"] :::m3
        Mod10["Module 10: Resilience Index<br/>(ASP Ratio & Classification)"] :::m3
        Bonus["Bonus: Route Rerouting Engine<br/>(Detour Delay & GeoJSON)"] :::m3
        Mod11["Module 11: Dashboard Exports<br/>(Consolidated JSON/GeoJSON)"] :::m3
    end

    Dash["Visualization Dashboard<br/>(Streamlit / Leaflet / QGIS)"] :::dash

    Mask --> Mod1
    Mod1 -->|skeleton.png| Mod2
    Mod2 -->|raw_graph.gpickle| Mod3
    Mod3 -->|raw_report.json| Mod4
    Mod4 -->|candidate_gaps.json| Mod5
    Mod5 -->|healed_graph.gpickle| Mod6
    Mod6 --> Mod7
    Mod7 -->|centrality.json| Mod8
    Mod8 -->|ranked.json| Mod9
    Mod9 -->|sim_results.json| Mod10
    Mod10 --> Mod11
    Bonus --> Mod11
    Mod11 -->|graph_data.json<br/>critical_nodes.json<br/>network_export.geojson| Dash
```

## Module Contracts & Deliverable Specifications

| Module | Responsible Script | Input Contract | Output Deliverable | Key Algorithmic Mechanics |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1** | `m2_graph/skeletonize.py` | `binary_mask.png` (uint8) | `skeleton.png` | Lee's 2D morphological thinning, removing isolated salt noise ($<20$px). |
| **Module 2** | `m2_graph/extract.py` | `skeleton.png` | `road_graph.gpickle` | 8-connected neighbour traversal. Nodes=deg 1 (Endpoint) & deg $\ge 3$ (Junction). Edges accumulate exact Euclidean curve length. |
| **Module 3** | `m2_graph/components.py` | `road_graph.gpickle` | `connectivity_report.json` | BFS/DFS component partitioning, computing Largest Connected Component (LCC). |
| **Module 4** | `m2_graph/gaps.py` | `road_graph.gpickle` | `candidate_gaps.json` | Pairwise endpoint inward road vectors $\vec{h}$. Angular score $S_{angle}$ + Distance score $S_{dist}$. |
| **Module 5** | `m2_graph/healing.py` | `candidate_gaps.json` | `healed_graph.gpickle` | Disjoint Set Union (DSU) tracking existing components + Kruskal MST selection. |
| **Module 6** | `m2_graph/eval_connectivity.py`| Before/After graphs | `connectivity_improvement_report.json` | $ConnectivityRatio = LCC_{after} / LCC_{before}$. |
| **Module 7** | `m3_analysis/centrality.py` | `healed_graph.gpickle` | `critical_nodes.json` | Brandes algorithm ($O(VE)$). Articulation cut vertices $\rightarrow$ SPOFs. |
| **Module 8** | `m3_analysis/ranking.py` | `critical_nodes.json` | `ranked_nodes.json` | Sorting descending by betweenness centrality score. |
| **Module 9** | `m3_analysis/simulation.py` | `healed_graph.gpickle` | `simulation_results.json` | Knockout simulation removing Top-$K$ nodes. Recomputing global routing efficiency. |
| **Module 10**| `m3_analysis/resilience.py`| `simulation_results.json`| `resilience_report.json` | Macro robustness index $BaselineASP / DamagedASP$. |
| **Module 11**| `m3_analysis/dashboard_export.py`| All reports | JSON / GeoJSON deliverables | Dashboard JSON contracts and standard GeoJSON FeatureCollection. |
