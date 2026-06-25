"""
Module: model.py
Responsibility: Provide a highly robust, research-grade factory for instantiating
segmentation architectures optimized for topological continuity.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional

def get_model(
    architecture: str = "linknet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    freeze_encoder: bool = False
) -> nn.Module:
    """
    Instantiates a PyTorch segmentation model.
    
    Args:
        architecture: 'linknet' or 'unetplusplus'. LinkNet is preferred for 
                      preserving thin continuous structures due to its residual 
                      addition blocks instead of concatenations.
        encoder_name: Backbone architecture (e.g., 'resnet34'). ResNet34 provides
                      an optimal tradeoff between feature capacity and T4 VRAM.
        encoder_weights: Pretrained weights to load (e.g., 'imagenet').
        in_channels: Number of input channels (3 for RGB).
        classes: Number of output channels (1 for binary road mask).
        freeze_encoder: If True, freezes the backbone weights for warm-up training.
        
    Returns:
        torch.nn.Module: The instantiated segmentation model.
    """
    valid_architectures = ["linknet", "unetplusplus"]
    arch = architecture.lower()
    
    if arch not in valid_architectures:
        raise ValueError(f"Architecture '{architecture}' is unsupported. Must be one of {valid_architectures}.")
        
    if arch == "linknet":
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    else:  # unetplusplus
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
        
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
            
    return model

if __name__ == "__main__":
    # DRY RUN / TESTING STRATEGY
    print("Initiating Model Architecture Unit Test...")
    test_model = get_model(architecture="linknet", encoder_name="resnet34", encoder_weights=None)
    
    # Simulate a Kaggle T4 tensor (Batch=4, Channels=3, H=512, W=512)
    dummy_input = torch.randn(4, 3, 512, 512)
    
    # Forward pass
    with torch.no_grad():
        output = test_model(dummy_input)
        
    # Verify Shapes
    expected_shape = (4, 1, 512, 512)
    print(f"Output Shape: {output.shape} | Expected: {expected_shape}")
    assert output.shape == expected_shape, "Model output shape mismatch!"
    
    # Verify Dtype
    print(f"Output Dtype: {output.dtype} | Expected: torch.float32")
    assert output.dtype == torch.float32, "Model output dtype mismatch!"
    
    print("Model Architecture constraints successfully validated.")
