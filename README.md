# Route Resilience

Occlusion-Robust Road Extraction & Urban Resilience Analysis from Satellite Imagery

## Project Overview

The pipeline transforms raw satellite imagery into a dashboard simulation:

`Satellite Image` &rarr; `Road Segmentation` &rarr; `Binary Mask` &rarr; `Graph Extraction` &rarr; `Centrality Analysis` &rarr; `Disaster Simulation Dashboard`

## Team Structure

*   **M1 - Segmentation**: Road extraction from satellite imagery.
*   **M2 - Graph Extraction**: Skeletonization and graph generation.
*   **M3 - Network Analysis**: Betweenness centrality and resilience analysis.
*   **M4 - Dashboard**: Interactive disaster simulation dashboard.

## Kaggle Workflow

*   Code lives in **GitHub**.
*   Training occurs on **Kaggle**.
*   Datasets are attached using **Kaggle Add Input** (available at `/kaggle/input/...`).
*   Outputs and model checkpoints are saved locally on the instance at `/kaggle/working/...`.

## Dataset Strategy

*   **Phase 1**: DeepGlobe Road Extraction Dataset.
*   **Phase 2**: SpaceNet 5 Road Network Dataset (used for fine-tuning only after the pipeline is stable).
