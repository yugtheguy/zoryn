import torch
import numpy as np
import albumentations as A
import cv2

def run_verification():
    # 1. Create dummy image and mask
    image = np.ones((512, 512, 3), dtype=np.uint8) * 255  # White image
    mask = np.ones((512, 512), dtype=np.uint8) * 255      # White mask (road)

    # 2. Define the new CoarseDropout
    transform = A.Compose([
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(16, 64),
            hole_width_range=(16, 64),
            fill=0,
            fill_mask=None,
            p=1.0  # Force it to apply every time
        )
    ], additional_targets={'mask': 'mask'})

    # 3. Apply 10 times and verify
    for i in range(10):
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']

        # Prove Image receives occlusions
        # If filled with 0, min should be 0
        assert aug_image.min() == 0, "Image did not receive occlusion (no black pixels found)."

        # Prove Mask remains unchanged (no 0 pixels)
        assert aug_mask.min() == 255, f"Mask was corrupted! Found dropped pixels in mask on iteration {i}."

    print("SUCCESS: Augmentation validation passed. Images receive occlusion, masks remain physically intact.")

if __name__ == '__main__':
    run_verification()
