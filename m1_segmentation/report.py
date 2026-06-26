import csv
import json
import os
import numpy as np

def save_csv_report(metrics_list: list, output_path: str):
    """Saves per-image raw metrics into a CSV."""
    if not metrics_list:
        return
        
    keys = metrics_list[0].keys()
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics_list)

def compute_aggregates(metrics_list: list) -> dict:
    """Computes Mean and StdDev across the dataset."""
    if not metrics_list:
        return {}
        
    keys = metrics_list[0].keys()
    aggregates = {}
    
    for key in keys:
        if key == "image_name":
            continue
        values = [m[key] for m in metrics_list]
        aggregates[f"{key}_mean"] = float(np.mean(values))
        aggregates[f"{key}_std"] = float(np.std(values))
        
    return aggregates

def save_json_report(aggregates: dict, output_path: str):
    """Saves dataset-wide statistics to JSON."""
    with open(output_path, 'w') as f:
        json.dump(aggregates, f, indent=4)

def save_markdown_report(aggregates: dict, output_path: str):
    """Generates a human-readable markdown evaluation summary."""
    md_content = [
        "# M1 Segmentation Evaluation Report",
        "",
        "This report aggregates strict binary mathematical metrics over the validation holdout set.",
        "The metrics guarantee parity with the strict `uint8 {0, 255}` M2 Graph Extraction input contract.",
        "",
        "## Topological Integrity (Primary Objective)",
        f"- **clDice Score:** {aggregates.get('cldice_mean', 0):.4f} ± {aggregates.get('cldice_std', 0):.4f}",
        "",
        "## Standard Spatial Metrics",
        f"- **Validation Loss (Continuous TopologyLoss):** {aggregates.get('validation_loss_mean', 0):.4f}",
        f"- **IoU (Jaccard Index):** {aggregates.get('iou_mean', 0):.4f} ± {aggregates.get('iou_std', 0):.4f}",
        f"- **Dice (F1 Score):** {aggregates.get('dice_f1_mean', 0):.4f} ± {aggregates.get('dice_f1_std', 0):.4f}",
        f"- **Precision:** {aggregates.get('precision_mean', 0):.4f} ± {aggregates.get('precision_std', 0):.4f}",
        f"- **Recall (Sensitivity):** {aggregates.get('recall_mean', 0):.4f} ± {aggregates.get('recall_std', 0):.4f}",
        f"- **Specificity (TNR):** {aggregates.get('specificity_mean', 0):.4f} ± {aggregates.get('specificity_std', 0):.4f}",
        f"- **Balanced Accuracy:** {aggregates.get('balanced_accuracy_mean', 0):.4f} ± {aggregates.get('balanced_accuracy_std', 0):.4f}",
        "",
        "## Secondary Metrics",
        f"- **Pixel Accuracy:** {aggregates.get('pixel_accuracy_mean', 0):.4f} ± {aggregates.get('pixel_accuracy_std', 0):.4f}",
        "> Note: Pixel accuracy is heavily skewed by the class imbalance of road extraction (mostly background) and is not a reliable indicator of success."
    ]
    
    with open(output_path, 'w') as f:
        f.write("\n".join(md_content))
