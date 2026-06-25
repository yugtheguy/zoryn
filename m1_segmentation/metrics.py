import torch
import numpy as np

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

        # 4. Fast PyTorch Tensor Metrics
        # Avoids sklearn UserWarnings and drastically speeds up evaluation
        # By ensuring y_true and y_pred are flattened discrete tensors
        y_true = targets.view(-1)
        y_pred = preds.view(-1)

        tp = (y_true * y_pred).sum().item()
        tn = ((1 - y_true) * (1 - y_pred)).sum().item()
        fp = ((1 - y_true) * y_pred).sum().item()
        fn = (y_true * (1 - y_pred)).sum().item()

        # IoU (Jaccard)
        union = tp + fp + fn
        iou = tp / union if union > 0 else 1.0

        # Dice (F1)
        dice_denom = 2 * tp + fp + fn
        dice = (2 * tp) / dice_denom if dice_denom > 0 else 1.0

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

        # Recall (Sensitivity)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

        # Pixel Accuracy
        total_pixels = y_true.numel()
        pixel_accuracy = (tp + tn) / total_pixels

        # Balanced Accuracy
        balanced_acc = (recall + specificity) / 2.0

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
