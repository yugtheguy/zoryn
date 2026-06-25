import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from data_loader import get_transforms
from model import get_model

def infer(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(architecture="linknet", encoder="resnet34").to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"Warning: Weights {args.weights} not found. Using untrained model.")
        
    model.eval()
    transform = get_transforms(is_train=False)
    
    image_files = []
    if os.path.exists(args.image_dir):
        image_files = [f for f in os.listdir(args.image_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images for inference.")
    
    with torch.no_grad():
        for img_name in tqdm(image_files, desc="Inference"):
            img_path = os.path.join(args.image_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Predict
            tensor_img = transform(image=image)['image'].unsqueeze(0).to(device)
            logits = model(tensor_img)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Binary Threshold
            binary_mask = (probs > args.threshold).astype(np.uint8)
            final_mask = binary_mask * 255
            
            # Strict guarantee
            unique_vals = set(np.unique(final_mask))
            assert unique_vals.issubset({0, 255}), f"Mask contains invalid values: {unique_vals}"
            
            # Save
            out_name = img_name.replace('.jpg', '.png')
            cv2.imwrite(os.path.join(args.output_dir, out_name), final_mask)
            
    print(f"Saved {len(image_files)} binary masks to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help="Path to best_model.pt")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to test images")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/predictions/")
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    infer(args)
