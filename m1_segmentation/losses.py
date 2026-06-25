"""
Module: losses.py
Responsibility: Implement topology-aware composite loss functions (BCE + Dice + soft-clDice)
to mathematically enforce road continuity and handle extreme class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    """
    Composite loss combining pixel-wise cross entropy with area-based Dice score.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        # PyTorch BCEWithLogitsLoss is numerically stable compared to Sigmoid + BCELoss
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) float32 raw predictions.
            targets: (B, 1, H, W) float32 binary masks (0.0 or 1.0).
        """
        # Pixel-level accuracy
        bce_loss = self.bce(logits, targets)
        
        # Spatial overlap (Dice)
        probs = torch.sigmoid(logits)
        
        # Computes Dice per-image in the batch to avoid global collapse, then averages
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class SoftSkeletonization(nn.Module):
    """
    Differentiable approximation of topological skeletonization using morphological pooling.
    Requires (B, C, H, W) continuous tensors [0, 1].
    """
    def __init__(self, iterations: int = 5):
        super().__init__()
        self.iterations = iterations
        
    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        # Min pool via inverted max pool to approximate erosion
        # Pad by 1 to maintain tensor dimensions (stride=1, kernel=3)
        p1 = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
        p2 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p3 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.min(torch.min(p1, p2), p3)
        
    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        # Max pool to approximate dilation
        return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
        
    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self.soft_dilate(self.soft_erode(img))
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        skeleton = torch.zeros_like(img)
        eroded = img
        
        for _ in range(self.iterations):
            opened = self.soft_open(eroded)
            # Skeleton is the union of remnants after subtracting opened from eroded
            skeleton = torch.max(skeleton, F.relu(eroded - opened))
            eroded = self.soft_erode(eroded)
            
        return skeleton


class clDiceLoss(nn.Module):
    """
    Centerline-Dice (clDice) Loss for topology preservation.
    Extracts soft skeletons and computes topological overlap (Sensitivity/Precision).
    """
    def __init__(self, iterations: int = 5, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.skeletonize = SoftSkeletonization(iterations=iterations)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Extract 1D mathematical skeletons
        # IMPORTANT: targets must also be soft-skeletonized or pre-skeletonized. 
        # Using soft-skeletonization dynamically to handle batch augmentation natively.
        pred_skeleton = self.skeletonize(probs)
        target_skeleton = self.skeletonize(targets)
        
        # Flatten for spatial computation
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        pred_skel_flat = pred_skeleton.view(pred_skeleton.size(0), -1)
        target_skel_flat = target_skeleton.view(target_skeleton.size(0), -1)
        
        # Precision (Topology): How much predicted skeleton falls on the true mask?
        t_prec = (pred_skel_flat * targets_flat).sum(dim=1) + self.smooth
        prec_denom = pred_skel_flat.sum(dim=1) + self.smooth
        precision = t_prec / prec_denom
        
        # Sensitivity (Topology): How much true skeleton falls on the predicted mask?
        t_sens = (target_skel_flat * probs_flat).sum(dim=1) + self.smooth
        sens_denom = target_skel_flat.sum(dim=1) + self.smooth
        sensitivity = t_sens / sens_denom
        
        cl_dice = (2. * precision * sensitivity) / (precision + sensitivity)
        return 1.0 - cl_dice.mean()


class TopologyLoss(nn.Module):
    """
    The ultimate composite loss orchestrating BCE, area-Dice, and centerline-Dice.
    """
    def __init__(self, bce_w: float = 0.5, dice_w: float = 0.3, cldice_w: float = 0.2):
        super().__init__()
        self.bce_dice_w = bce_w + dice_w
        self.cldice_w = cldice_w
        
        # Normalize BCE and Dice weights to sum to 1.0 for the internal calculation
        norm_bce = bce_w / (bce_w + dice_w)
        norm_dice = dice_w / (bce_w + dice_w)
        
        self.bce_dice = BCEDiceLoss(bce_weight=norm_bce, dice_weight=norm_dice)
        
        if cldice_w > 0:
            self.cldice = clDiceLoss(iterations=5)
        else:
            self.cldice = None
            
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.bce_dice_w * self.bce_dice(logits, targets)
        
        # Only compute computationally expensive soft-skeletonization if weighted > 0
        if self.cldice_w > 0 and self.cldice is not None:
            loss += self.cldice_w * self.cldice(logits, targets)
            
        return loss

if __name__ == "__main__":
    # DRY RUN / TESTING STRATEGY
    print("Initiating Topological Loss Unit Test...")
    
    # 1. Verify Shapes & Mathematical Validity
    dummy_logits = torch.randn(2, 1, 512, 512, requires_grad=True)
    dummy_targets = torch.randint(0, 2, (2, 1, 512, 512)).float()
    
    print("Testing BCE+Dice Loss...")
    bce_dice = BCEDiceLoss()
    loss1 = bce_dice(dummy_logits, dummy_targets)
    assert loss1.shape == torch.Size([]), "BCE+Dice output must be a scalar."
    assert not torch.isnan(loss1), "BCE+Dice produced NaN!"
    
    print("Testing Full Topology Loss (including clDice overhead)...")
    top_loss = TopologyLoss(bce_w=0.4, dice_w=0.4, cldice_w=0.2)
    loss2 = top_loss(dummy_logits, dummy_targets)
    assert loss2.shape == torch.Size([]), "TopologyLoss output must be a scalar."
    assert not torch.isnan(loss2), "TopologyLoss produced NaN!"
    
    # 2. Verify Gradient Flow
    loss2.backward()
    assert dummy_logits.grad is not None, "Gradient failed to flow through morphological pooling!"
    
    print("Topological Loss constraints and backpropagation successfully validated.")
