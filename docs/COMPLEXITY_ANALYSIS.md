# Computational Complexity Analysis

This document provides formal time and space complexity bounds for the core algorithms implemented in Member 2's Graph & Resilience Pipeline.

Let:
- $W, H$: Width and height of the input binary segmentation mask image (pixels).
- $N = W \times H$: Total pixel count.
- $V$: Number of vertices (nodes) in the extracted mathematical graph $G=(V,E)$.
- $E$: Number of edges in $G$.
- $P$: Number of topological endpoints (degree 1 nodes in $G$), where $P \le V$.
- $G_{cand}$: Number of candidate gap pairs detected between endpoints, where $G_{cand} \le \frac{P(P-1)}{2}$.

---

## 1. Skeletonization & Noise Filtering (Module 1)
- **Algorithm**: Scikit-Image morphological small object removal followed by Lee's 2D thinning algorithm.
- **Time Complexity**: $O(N)$
  - Small object removal performs two raster scans and Union-Find / BFS labelling: $O(N)$.
  - Morphological thinning iteratively inspects $3\times3$ pixel neighbourhoods across foreground pixels until convergence (road width bounded by constant $w_{road}$): $O(N \cdot w_{road}) = O(N)$.
- **Space Complexity**: $O(N)$ to store the boolean mask and working thinning grids.

---

## 2. Graph Extraction & Path Compaction (Module 2)
- **Algorithm**: 8-connected neighbour adjacency graph construction followed by degree-2 chain compaction.
- **Time Complexity**: $O(N_f)$ where $N_f \le N$ is the number of foreground skeleton pixels.
  - Pixel graph build: $8 \times N_f$ inspections $\rightarrow O(N_f)$.
  - Chain compaction walks each degree-2 pixel exactly once $\rightarrow O(N_f)$.
- **Space Complexity**: $O(V + E + N_f)$ to hold coordinate mappings and compacted graph geometries.

---

## 3. Gap Detection & Angular Scoring (Module 4)
- **Algorithm**: Pairwise Euclidean distance filtering and cosine vector similarity evaluation across endpoints.
- **Time Complexity**: $O(P^2 + P \cdot L_{road})$
  - Endpoint outward heading vector calculation inspects inner edge geometry up to constant depth: $O(P)$.
  - Pairwise distance and dot-product evaluation: $\frac{P(P-1)}{2}$ pairs $\rightarrow O(P^2)$.
- **Space Complexity**: $O(G_{cand})$ to store the filtered candidate gaps list.

---

## 4. Topological Healing via DSU & Kruskal MST (Module 5)
- **Algorithm**: Disjoint Set Union (Union-Find with path compression & union by rank) running Kruskal's Minimum Spanning Tree selection.
- **Time Complexity**: $O(G_{cand} \log G_{cand} + (V + E + G_{cand}) \alpha(V))$
  - Sorting candidate gaps by composite score: $O(G_{cand} \log G_{cand})$.
  - Initializing DSU with existing graph edges: $E$ unions $\rightarrow O(E \alpha(V))$, where $\alpha$ is the inverse Ackermann function ($\alpha(V) \le 4$).
  - Evaluating candidate gaps: $G_{cand}$ find/union operations $\rightarrow O(G_{cand} \alpha(V))$.
- **Space Complexity**: $O(V)$ for DSU parent and rank hash tables.

---

## 5. Betweenness Centrality & SPOF Detection (Module 7)
- **Algorithm**: Brandes' shortest-path betweenness centrality and Hopcroft-Tarjan articulation point detection.
- **Time Complexity**: $O(V \cdot E + V^2 \log V)$
  - Hopcroft-Tarjan DFS for cut vertices (Single Points of Failure): $O(V + E)$.
  - Brandes algorithm for weighted graphs runs Dijkstra's algorithm from each vertex: $V \times O(E + V \log V) = O(VE + V^2 \log V)$.
- **Space Complexity**: $O(V + E)$ for dependency stacks and shortest path trees.

---

## 6. Network Stress Testing Simulation (Module 9)
- **Algorithm**: Node knockout simulation across $|K_{scenarios}|$ knockout batches, recomputing global network routing efficiency.
- **Time Complexity**: $|K_{scenarios}| \times O(V \cdot E + V^2 \log V)$
  - Global efficiency computes shortest path lengths between all node pairs $O(VE + V^2 \log V)$ per knockout graph.
- **Space Complexity**: $O(V + E)$ for damaged graph clones.

---

## Summary Table
| Pipeline Phase | Dominant Time Complexity | Space Complexity | Practical Scalability ($512\times512$ image) |
| :--- | :--- | :--- | :--- |
| **Mask $\rightarrow$ Graph** | $O(N)$ | $O(N)$ | $\sim 0.1$ seconds |
| **Topological Healing**| $O(P^2 + G_{cand} \log G_{cand})$| $O(V + G_{cand})$| $\sim 0.05$ seconds |
| **Centrality & Stress**| $O(|K| \cdot (VE + V^2 \log V))$| $O(V + E)$ | $\sim 0.2$ seconds |
