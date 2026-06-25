import torch
import numpy as np
from sklearn.metrics import (
    jaccard_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score, 
    confusion_matrix,
    balanced_accuracy_score
)

from losses import clDiceLoss, TopologyLoss

class Evaluator:
    """
    Computes strict mathematical classification metrics for semantic segmentation.
    Ensures calculations are performed on boolean tensors/arrays to guarantee M2 graph parity.
    """
    def __init__(self, device: torch.device):
        self.device = device
        # TopologyLoss is used to measure divergence exactly as in training
        self.validation_loss_fn = TopologyLoss(bce_w=0.5, dice_w=0.5, cldice_w=0.0).to(device)
        
        # Soft clDice from training is repurposed to score mathematical topological overlap
        self.cldice_fn = clDiceLoss(iterations=5).to(device)

    def compute_batch_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Computes metrics for a single batch.
        Args:
            logits: Raw model predictions (B, 1, H, W)
            targets: Strict ground truth masks (B, 1, H, W)
        """
        # 1. Validation Loss (Continuous)
        val_loss = self.validation_loss_fn(logits, targets).item()

        # 2. Strict Thresholding (Discrete)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # 3. Topological Score (clDice)
        # We pass the discrete preds and targets into the soft_skeletonization to evaluate topology.
        # clDiceLoss outputs (1 - clDice). We invert it to get the raw score.
        cldice_loss = self.cldice_fn(preds, targets).item()
        cldice_score = 1.0 - cldice_loss

        # 4. Scikit-learn Classification Metrics
        y_true = targets.cpu().numpy().flatten().astype(int)
        y_pred = preds.cpu().numpy().flatten().astype(int)

        iou = jaccard_score(y_true, y_pred, average='binary', zero_division=1.0)
        dice = f1_score(y_true, y_pred, average='binary', zero_division=1.0) # Dice is identical to F1
        precision = precision_score(y_true, y_pred, average='binary', zero_division=1.0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=1.0)
        pixel_accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Specificity calculation (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

        return {
            "validation_loss": val_loss,
            "iou": float(iou),
            "dice_f1": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "pixel_accuracy": float(pixel_accuracy),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_acc),
            "cldice": float(cldice_score)
        }
