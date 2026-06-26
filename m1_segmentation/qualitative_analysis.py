import os
import cv2
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from model import get_model
from data_loader import get_dataloaders, MEAN, STD
from metrics import Evaluator

# -----------------------------------------------------------------------------
# Qualitative Analysis Module
# -----------------------------------------------------------------------------

def analyze_connectivity(mask: np.ndarray) -> dict:
    """
    Evaluates topological connectivity using Connected Components.
    Expects mask as discrete binary [0, 1].
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    # Exclude background (label 0)
    if num_labels <= 1:
        return {"components": 0, "largest_ratio": 0.0}
        
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component = np.max(component_areas)
    total_area = np.sum(component_areas)
    
    return {
        "components": num_labels - 1,
        "largest_ratio": float(largest_component / total_area) if total_area > 0 else 0.0
    }

def save_qualitative_figure(img_tensor, true_mask, pred_mask, metrics, output_path):
    """
    Generates the requested 2x2 mathematical qualitative analysis grid.
    """
    # Denormalize Image
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = np.array(STD) * img + np.array(MEAN)
    img = np.clip(img, 0, 1)

    truth = true_mask.squeeze().cpu().numpy()
    pred = pred_mask.squeeze().cpu().numpy()

    # Confusion Map
    diff_map = np.zeros((*truth.shape, 3), dtype=np.float32)
    diff_map[(truth == 1) & (pred == 1)] = [1.0, 1.0, 1.0] # TP: White
    diff_map[(truth == 0) & (pred == 1)] = [1.0, 0.0, 0.0] # FP: Red
    diff_map[(truth == 1) & (pred == 0)] = [0.0, 0.0, 1.0] # FN: Blue
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(
        f"File: {metrics['image_name']}\n"
        f"IoU: {metrics['iou']:.3f} | Dice: {metrics['dice_f1']:.3f} | clDice: {metrics['cldice']:.3f}\n"
        f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}",
        fontsize=14, fontweight='bold', y=0.98
    )

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(truth, cmap="gray")
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pred, cmap="gray")
    axes[1, 0].set_title("Predicted Mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(diff_map)
    axes[1, 1].set_title("Difference Map\n(White=TP, Red=FP, Blue=FN)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def generate_histograms(all_metrics, out_dir):
    """Generates distribution plots for core metrics."""
    metrics_to_plot = ["iou", "dice_f1", "precision", "recall", "cldice"]
    
    for m in metrics_to_plot:
        data = [x[m] for x in all_metrics]
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=30, color='royalblue', edgecolor='black', alpha=0.7)
        plt.title(f"{m.upper()} Distribution across Validation Set")
        plt.xlabel(m.upper())
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(out_dir / f"hist_{m}.png")
        plt.close()

def compute_percentiles(data):
    if not data:
        return {"min": 0, "25th": 0, "50th": 0, "75th": 0, "max": 0, "mean": 0, "std": 0}
    return {
        "min": np.min(data),
        "25th": np.percentile(data, 25),
        "50th": np.percentile(data, 50),
        "75th": np.percentile(data, 75),
        "max": np.max(data),
        "mean": np.mean(data),
        "std": np.std(data)
    }

def generate_reports(all_metrics, out_dir):
    """Generates statistical report.md and research_observations.md"""
    iou_data = [x["iou"] for x in all_metrics]
    dice_data = [x["dice_f1"] for x in all_metrics]
    prec_data = [x["precision"] for x in all_metrics]
    rec_data = [x["recall"] for x in all_metrics]
    cldice_data = [x["cldice"] for x in all_metrics]
    conn_comps = [x["components"] for x in all_metrics]
    
    iou_stats = compute_percentiles(iou_data)
    
    # 1. Statistical Report
    report = [
        "# Statistical Evaluation Report",
        f"**Total Validation Images:** {len(all_metrics)}\n",
        "## IoU Distribution",
        f"- Mean: {iou_stats['mean']:.4f} ± {iou_stats['std']:.4f}",
        f"- Median: {iou_stats['50th']:.4f}",
        f"- Worst: {iou_stats['min']:.4f}",
        f"- Best: {iou_stats['max']:.4f}\n",
        "## Core Metrics (Means)",
        f"- Mean Dice: {np.mean(dice_data):.4f}",
        f"- Mean Precision: {np.mean(prec_data):.4f}",
        f"- Mean Recall: {np.mean(rec_data):.4f}",
        f"- Mean clDice: {np.mean(cldice_data):.4f}",
        f"- Mean Components per Image: {np.mean(conn_comps):.1f}"
    ]
    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(report))
        
    # 2. Research Observations
    obs = [
        "# Auto-Generated Research Observations",
        "## Metric Variance Breakdown",
        f"Precision ({np.mean(prec_data):.3f}) vs Recall ({np.mean(rec_data):.3f}):",
    ]
    if np.mean(rec_data) > np.mean(prec_data) + 0.05:
        obs.append("- **Observation:** Recall is significantly higher than Precision. The model aggressively predicts roads, leading to False Positives (over-segmentation).")
        obs.append("- **Possible Improvement:** Increase the BCE weight or introduce an active negative mining loss.")
    elif np.mean(prec_data) > np.mean(rec_data) + 0.05:
        obs.append("- **Observation:** Precision is significantly higher than Recall. The model is too conservative and misses valid roads (False Negatives).")
        obs.append("- **Possible Improvement:** Decrease BCE weight or decrease threshold below 0.5.")
    else:
        obs.append("- **Observation:** Precision and Recall are balanced. The threshold is optimal.")
        
    obs.append(f"\nIoU ({iou_stats['mean']:.3f}) vs clDice ({np.mean(cldice_data):.3f}):")
    if np.mean(cldice_data) > iou_stats['mean'] + 0.05:
        obs.append("- **Observation:** clDice is higher than IoU. The model captures road centerlines well but struggles with road width/thickness.")
    elif iou_stats['mean'] > np.mean(cldice_data) + 0.05:
        obs.append("- **Observation:** IoU is higher than clDice. The model predicts bulk pixel mass well but suffers from topological disconnections and broken segments.")
        obs.append("- **Possible Improvement:** Increase `cldice_w` in `TopologyLoss` or apply morphological closing post-processing.")
        
    obs.append("\n## Connectivity Findings")
    obs.append(f"- The model averages {np.mean(conn_comps):.1f} disconnected components per image.")
    if np.mean(conn_comps) > 5.0:
        obs.append("- **CRITICAL WARNING:** High component count indicates severe fragmentation. M2 graph extraction will fail to create contiguous routing.")
        
    obs.append("\n## Manual Inspection Required")
    obs.append("Due to the mathematical limitations of unannotated RGB data, categorization of Tree Occlusions, Shadows, and Building Overhangs cannot be performed automatically. **You must visually inspect the `worst_predictions/` folder to manually categorize these physical phenomena.**")

    with open(out_dir / "research_observations.md", "w") as f:
        f.write("\n".join(obs))

@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Deep Qualitative Analysis on {device}")

    # Directories
    base_dir = Path(args.output_dir)
    dirs = {
        "best": base_dir / "best_predictions",
        "avg": base_dir / "average_predictions",
        "worst": base_dir / "worst_predictions",
        "hist": base_dir / "histograms"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Data & Model
    _, val_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size, val_split=0.2)
    model = get_model(architecture="linknet", encoder_name="resnet34").to(device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    evaluator = Evaluator(device)
    all_results = []
    
    # Store images/masks temporarily in memory for the top/bottom sorting
    # Note: For massive datasets, this might OOM. 1200 images of 512x512 is ~1GB RAM.
    # In Kaggle (32GB RAM), this is perfectly safe.
    saved_tensors = {}

    print("Running Inference & Metric Computation...")
    pbar = tqdm(val_loader)
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
        for i in range(images.size(0)):
            img_name = f"batch{batch_idx}_img{i}"
            
            # Metrics
            metrics = evaluator.compute_batch_metrics(logits[i:i+1], masks[i:i+1])
            metrics["image_name"] = img_name
            
            # Connectivity
            conn_stats = analyze_connectivity(preds[i:i+1].squeeze().cpu().numpy())
            metrics.update(conn_stats)
            
            all_results.append(metrics)
            
            # Keep in CPU memory for later visualization sorting
            saved_tensors[img_name] = {
                "img": images[i:i+1].cpu(),
                "true": masks[i:i+1].cpu(),
                "pred": preds[i:i+1].cpu()
            }

    # Sort by IoU
    all_results.sort(key=lambda x: x["iou"], reverse=True)
    
    num_samples = 20
    best_20 = all_results[:num_samples]
    worst_20 = all_results[-num_samples:]
    
    mid_idx = len(all_results) // 2
    avg_20 = all_results[mid_idx - (num_samples//2) : mid_idx + (num_samples//2)]
    
    print("\nGenerating Top 20 2x2 Visualizations...")
    for idx, r in enumerate(tqdm(best_20)):
        t = saved_tensors[r["image_name"]]
        save_qualitative_figure(t["img"][0], t["true"], t["pred"], r, dirs["best"] / f"{idx:02d}_iou{r['iou']:.3f}.png")

    print("Generating Bottom 20 2x2 Visualizations...")
    for idx, r in enumerate(tqdm(worst_20)):
        t = saved_tensors[r["image_name"]]
        save_qualitative_figure(t["img"][0], t["true"], t["pred"], r, dirs["worst"] / f"{idx:02d}_iou{r['iou']:.3f}.png")

    print("Generating Median 20 2x2 Visualizations...")
    for idx, r in enumerate(tqdm(avg_20)):
        t = saved_tensors[r["image_name"]]
        save_qualitative_figure(t["img"][0], t["true"], t["pred"], r, dirs["avg"] / f"{idx:02d}_iou{r['iou']:.3f}.png")

    print("\nGenerating Reports & Histograms...")
    generate_histograms(all_results, dirs["hist"])
    generate_reports(all_results, base_dir)
    
    print(f"\nSUCCESS: All qualitative deliverables saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 Qualitative Research Analysis")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to DeepGlobe /train directory")
    parser.add_argument('--weights', type=str, required=True, help="Path to best_model.pt")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/analysis", help="Output directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    main(args)
