import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

from data_loader import MEAN, STD

def save_evaluation_plot(
    tensor_image: torch.Tensor, 
    true_mask: torch.Tensor, 
    pred_mask: torch.Tensor, 
    output_path: str
):
    """
    Generates a 1x4 matplotlib figure showing: Original | Truth | Prediction | Difference Map
    Saves to output_path.
    
    Args:
        tensor_image: (3, H, W) normalized tensor
        true_mask: (1, H, W) or (H, W) discrete binary tensor [0, 1]
        pred_mask: (H, W) discrete binary tensor [0, 1]
    """
    # 1. Denormalize Image for visualization
    img = tensor_image.cpu().permute(1, 2, 0).numpy()
    mean = np.array(MEAN)
    std = np.array(STD)
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # 2. Extract masks to 2D numpy arrays
    truth = true_mask.squeeze().cpu().numpy()
    pred = pred_mask.squeeze().cpu().numpy()

    # 3. Create Difference Map (False Positives=Red, False Negatives=Blue, True Positives=White)
    diff_map = np.zeros((*truth.shape, 3), dtype=np.float32)
    
    # True Positives: Road in truth AND pred -> White
    tp = (truth == 1) & (pred == 1)
    diff_map[tp] = [1.0, 1.0, 1.0]
    
    # False Positives: Road in pred, but NOT in truth -> Red (Over-prediction)
    fp = (truth == 0) & (pred == 1)
    diff_map[fp] = [1.0, 0.0, 0.0]
    
    # False Negatives: Road in truth, but NOT in pred -> Blue (Missed road)
    fn = (truth == 1) & (pred == 0)
    diff_map[fn] = [0.0, 0.0, 1.0]

    # 4. Plotting
    plt.figure(figsize=(24, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Satellite Input", fontsize=14)
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(truth, cmap="gray")
    plt.title("Ground Truth (Target)", fontsize=14)
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("M1 Prediction (Binary)", fontsize=14)
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(diff_map)
    plt.title("Diff Map (Red=FP, Blue=FN, White=TP)", fontsize=14)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
