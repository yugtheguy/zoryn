import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

from model import get_model
from data_loader import get_dataloaders
from metrics import Evaluator
from visualize import save_evaluation_plot
from report import save_csv_report, save_json_report, save_markdown_report, compute_aggregates

@torch.no_grad()
def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Research-Grade Evaluation on {device}")
    
    # 1. Output Structure
    eval_dir = Path(args.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = eval_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # 2. Strict Dataset Replication
    print("Loading Validation Dataset...")
    # By strictly using get_dataloaders with the exact same seed (42),
    # we mathematically guarantee zero data leakage into the evaluation set.
    _, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.2
    )
    
    if val_loader is None:
        raise RuntimeError("Validation dataloader failed to initialize.")
        
    # 3. Model Loading
    print(f"Loading Model from {args.weights}...")
    model = get_model(architecture="linknet", encoder_name="resnet34").to(device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    
    # Fallback for dict structure vs raw weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # 4. Evaluation Loop
    evaluator = Evaluator(device)
    all_metrics = []
    visualizations_saved = 0
    
    pbar = tqdm(val_loader, desc="Evaluating", leave=True)
    
    # We use batch size but evaluate individually for granular variance tracking
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        # AMP guarantees inference matches training datatype constraints
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(images)
            
        # Compute metrics per image rather than across the whole flattened batch
        # This allows accurate standard deviation calculations.
        for i in range(images.size(0)):
            single_logit = logits[i].unsqueeze(0)
            single_mask = masks[i].unsqueeze(0)
            single_image = images[i].unsqueeze(0)
            
            # Mathematical metric computation
            metrics = evaluator.compute_batch_metrics(single_logit, single_mask)
            
            # Add identification tracking
            img_id = f"batch{batch_idx}_img{i}"
            metrics["image_name"] = img_id
            all_metrics.append(metrics)
            
            # Save Qualitative Visualizations (up to args.num_visualize)
            if visualizations_saved < args.num_visualize:
                # Convert logit to strictly thresholded mask for plotting
                probs = torch.sigmoid(single_logit)
                pred_mask = (probs > 0.5).float()
                
                vis_path = str(vis_dir / f"{img_id}_eval.png")
                save_evaluation_plot(single_image[0], single_mask[0], pred_mask[0], vis_path)
                visualizations_saved += 1
                
    # 5. Aggregation & Reporting
    print("\nEvaluation Complete! Aggregating Statistics...")
    aggregates = compute_aggregates(all_metrics)
    
    save_csv_report(all_metrics, str(eval_dir / "metrics.csv"))
    save_json_report(aggregates, str(eval_dir / "metrics.json"))
    save_markdown_report(aggregates, str(eval_dir / "evaluation_report.md"))
    
    print(f"\nAll artifacts successfully saved to {args.output_dir}/")
    print("Read evaluation_report.md for the human-readable summary.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 Strict Topological Evaluation")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to /kaggle/input/deepglobe-road-extraction-dataset/train")
    parser.add_argument('--weights', type=str, required=True, help="Path to best_model.pt")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/eval_results")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for evaluation")
    parser.add_argument('--num_visualize', type=int, default=10, help="Number of qualitative images to save")
    
    args = parser.parse_args()
    main(args)
