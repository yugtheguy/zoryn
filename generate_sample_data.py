#!/usr/bin/env python3
"""
Route Resilience: Sample Binary Road Mask Generator
Generates a realistic synthetic urban road mask (binary image: road=255, bg=0)
with simulated occlusion gaps (tree canopy, shadows, clutter).
"""

import os
import argparse
import numpy as np
import cv2

def create_synthetic_road_mask(width: int = 512, height: int = 512, road_thickness: int = 10) -> np.ndarray:
    """
    Creates a grid and diagonal road network with artificial gaps.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define grid lines (horizontal and vertical)
    h_coords = [100, 256, 400]
    v_coords = [100, 256, 400]
    
    # Draw horizontal roads
    for y in h_coords:
        cv2.line(mask, (20, y), (width - 20, y), 255, thickness=road_thickness)
        
    # Draw vertical roads
    for x in v_coords:
        cv2.line(mask, (x, 20), (x, height - 20), 255, thickness=road_thickness)
        
    # Draw a diagonal arterial road
    cv2.line(mask, (50, 50), (460, 460), 255, thickness=road_thickness)
    cv2.line(mask, (50, 460), (460, 50), 255, thickness=road_thickness)
    
    # Introduce artificial occlusion gaps (simulating tree canopies, shadows, vehicles)
    # Gaps are created by drawing black rectangles/circles over the roads
    gaps = [
        # (x1, y1, x2, y2)
        (160, 92, 190, 108),   # Horizontal gap on y=100 road
        (310, 92, 335, 108),   # Another horizontal gap on y=100
        (92, 160, 108, 190),   # Vertical gap on x=100 road
        (248, 320, 264, 345),  # Vertical gap on x=256 road
        (380, 392, 420, 408),  # Gap at intersection/road near (400,400)
        (180, 180, 210, 210),  # Gap on main diagonal
        (320, 180, 345, 205)   # Gap on counter diagonal
    ]
    
    for (x1, y1, x2, y2) in gaps:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        
    # Add minor salt-and-pepper noise to test Module 1 robustness
    noise_pts_x = np.random.randint(0, width, size=100)
    noise_pts_y = np.random.randint(0, height, size=100)
    mask[noise_pts_y, noise_pts_x] = 255
    
    # Threshold strictly to binary (0 and 1 or 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def main():
    parser = argparse.ArgumentParser(description="Generate sample binary road mask.")
    parser.add_argument('--output', type=str, default="sample_mask.png", help="Output file path")
    parser.add_argument('--width', type=int, default=512, help="Image width")
    parser.add_argument('--height', type=int, default=512, help="Image height")
    args = parser.parse_args()
    
    mask = create_synthetic_road_mask(args.width, args.height)
    cv2.imwrite(args.output, mask)
    print(f"Successfully generated sample road mask at: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
