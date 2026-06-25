"""
Module: inference.py
Responsibility: Consume unseen satellite images, execute segmentation, and
strictly enforce the binary M2 output contract (uint8, 0/255).
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from data_loader import get_transforms
from model import get_model

@torch.no_grad()
def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing inference on device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Fatal Error: Image directory {image_dir} not found.")
        return
        
    image_files = list(image_dir.glob('*.jpg'))
    if not image_files:
        print(f"No '.jpg' images found in {image_dir}")
        return
        
    print(f"Found {len(image_files)} images for inference.")
    
    # 1. Model Init
    model = get_model(architecture="linknet", encoder_name="resnet34").to(device)
    model.eval()
    
    # Checkpoint loading
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        # Handle dict from train.py vs raw state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint {args.weights} missing. Proceeding with untrained model.")

    # 2. Pipeline Init
    # Must use is_train=False to get the exact identical Normalization as training.
    # We redefine it here locally to skip the CenterCrop, allowing native 1024x1024 
    # fully convolutional inference directly on the original satellite imagery.
    from albumentations import Compose, Normalize
    from albumentations.pytorch import ToTensorV2
    from data_loader import MEAN, STD
    
    native_transform = Compose([
        Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    
    # 3. Inference Loop
    for img_path in tqdm(image_files, desc="Inference"):
        # OpenCV reads BGR, model expects RGB
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read {img_path}. Skipping.")
            continue
            
        original_h, original_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform (Normalization ONLY, no cropping)
        tensor_img = native_transform(image=image)['image'].unsqueeze(0).to(device)
        
        # Forward pass with AMP
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(tensor_img)
            
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # 4. Strict Binary Conversion (M2 Contract)
        binary_mask = (probs > args.threshold).astype(np.uint8)
        final_mask = binary_mask * 255
        
        # 5. Output Verification (CRITICAL)
        unique_vals = set(np.unique(final_mask))
        if not unique_vals.issubset({0, 255}):
            raise RuntimeError(f"Contract Violation: Non-binary pixels detected in mask: {unique_vals}")
            
        # 6. Save (PNG guarantees lossless binary preservation)
        # DeepGlobe schema mapping
        out_name = img_path.name.replace('_sat.jpg', '_mask.png')
        if '_mask' not in out_name:
            out_name = out_name.replace('.jpg', '_mask.png')
            
        cv2.imwrite(str(output_dir / out_name), final_mask)
        
    print(f"Successfully saved {len(image_files)} binary masks to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 Segmentation Inference Pipeline")
    parser.add_argument('--weights', type=str, required=True, help="Path to best_model.pt")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory of test images")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/predictions")
    parser.add_argument('--threshold', type=float, default=0.5, help="Binary classification threshold")
    
    args = parser.parse_args()
    main(args)
