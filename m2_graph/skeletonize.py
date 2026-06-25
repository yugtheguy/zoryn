#!/usr/bin/env python3
"""
Module 1 — Skeletonization
Reduces binary road segmentation blobs into 1-pixel thick centerlines while preserving
network topology, connectivity, and junction geometry.
"""

import os
import argparse
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects

def skeletonize_road_mask(binary_mask: np.ndarray, min_component_size: int = 20) -> np.ndarray:
    """
    Performs morphological thinning on a binary road mask.
    
    Args:
        binary_mask: 2D numpy array (uint8 or bool) where roads > 0.
        min_component_size: Minimum pixel area to keep before thinning (removes salt-noise).
        
    Returns:
        uint8 numpy array (0 and 255) representing the road skeleton.
    """
    if binary_mask.ndim > 2:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        
    # Convert to bool (road = True)
    bool_mask = binary_mask > 0
    
    # Remove small isolated noise artifacts
    if min_component_size > 0:
        clean_mask = remove_small_objects(bool_mask, max_size=min_component_size)
    else:
        clean_mask = bool_mask
        
    # Perform morphological skeletonization (Lee's method via scikit-image)
    skel = skeletonize(clean_mask)
    
    # Convert back to uint8 image (0 and 255)
    skel_uint8 = skel.astype(np.uint8) * 255
    return skel_uint8

def main():
    parser = argparse.ArgumentParser(description="M2 Module 1: Skeletonization Pipeline")
    parser.add_argument('--input_mask', type=str, required=True, help="Input binary road mask image")
    parser.add_argument('--output_skel', type=str, default="skeleton.png", help="Output skeleton image")
    parser.add_argument('--min_size', type=int, default=20, help="Min component size noise filter")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_mask):
        raise FileNotFoundError(f"Input mask not found: {args.input_mask}")
        
    mask = cv2.imread(args.input_mask, cv2.IMREAD_GRAYSCALE)
    skel = skeletonize_road_mask(mask, min_component_size=args.min_size)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_skel)), exist_ok=True)
    cv2.imwrite(args.output_skel, skel)
    print(f"Skeleton image saved to: {os.path.abspath(args.output_skel)}")

if __name__ == "__main__":
    main()
