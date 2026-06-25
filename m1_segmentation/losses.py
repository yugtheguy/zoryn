import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, targets, smooth=1e-6):
        # BCE
        bce_loss = self.bce(logits, targets)
        
        # Dice
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        dice_loss = 1.0 - dice_score
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# Placeholder for Soft clDice (To be implemented in Phase 2/later)
class clDiceLoss(nn.Module):
    def __init__(self):
        super(clDiceLoss, self).__init__()
        # TODO: Implement soft skeletonization and clDice forward pass
        
    def forward(self, logits, targets):
        # Placeholder returning 0
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

class TopologyLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.3, cldice_w=0.2):
        super(TopologyLoss, self).__init__()
        self.bce_dice = BCEDiceLoss(bce_weight=bce_w/(bce_w+dice_w), dice_weight=dice_w/(bce_w+dice_w))
        self.cldice = clDiceLoss()
        self.bce_dice_w = bce_w + dice_w
        self.cldice_w = cldice_w
        
    def forward(self, logits, targets):
        return self.bce_dice_w * self.bce_dice(logits, targets) + self.cldice_w * self.cldice(logits, targets)
