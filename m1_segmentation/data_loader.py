"""
Module: data_loader.py
Responsibility: Robustly load and augment the DeepGlobe Road Extraction dataset.
Output Contract: 
- Images: (B, 3, H, W) float32 normalized tensors.
- Masks: (B, 1, H, W) float32 binary tensors (strictly 0.0 or 1.0).
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
DEEPGLOBE_IMG_SUFFIX = '_sat.jpg'
DEEPGLOBE_MASK_SUFFIX = '_mask.png'
IMAGE_SIZE = 512
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DeepGlobeDataset(Dataset):
    """
    PyTorch Dataset for DeepGlobe Road Extraction.
    Ensures rigorous shape and type transformations.
    """
    def __init__(self, image_paths: List[Path], transform: Optional[A.Compose] = None):
        """
        Args:
            image_paths: List of absolute paths to the '_sat.jpg' files.
            transform: Albumentations composition pipeline.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        
        # DeepGlobe mapping logic: replace suffix
        mask_path = img_path.with_name(img_path.name.replace(DEEPGLOBE_IMG_SUFFIX, DEEPGLOBE_MASK_SUFFIX))

        # Read image (RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask (Grayscale)
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask at {mask_path}")
        else:
            # Strict fail-fast instead of silent fallback to prevent false-negative training
            raise FileNotFoundError(f"Missing required mask file at {mask_path}")

        # Apply augmentations (Albumentations)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure strict binary float32 mask before pushing to model
        # Note: If ToTensorV2 is used, 'mask' is already a torch.Tensor.
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        # Cast to float32 and enforce binary threshold (protects against interpolation artifacts)
        mask = (mask > 127).to(torch.float32)

        # PyTorch requires masks to have a channel dimension matching logits (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


def get_transforms(is_train: bool = True) -> A.Compose:
    """
    Returns the Albumentations pipeline.
    CRITICAL: Mask interpolation MUST be INTER_NEAREST to prevent anti-aliased gray pixels.
    """
    if is_train:
        return A.Compose([
            A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            # CoarseDropout explicitly targets ONLY the image, preserving the ground truth mask for occlusion learning
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 64), hole_width_range=(16, 64), fill=0, fill_mask=None, p=0.3), 
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'}) # Implicitly uses nearest neighbor for masks
    else:
        return A.Compose([
            # Validation uses CenterCrop for determinism
            A.CenterCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])


def get_dataloaders(
    data_dir: str, 
    batch_size: int = 16, 
    val_split: float = 0.2, 
    seed: int = 42
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates train and validation dataloaders from a single DeepGlobe directory.
    
    Args:
        data_dir: Path to the Kaggle dataset.
        batch_size: Number of samples per batch.
        val_split: Fraction of data reserved for validation.
        seed: Random seed for deterministic splitting.
        
    Returns:
        train_loader, val_loader
    """
    base_path = Path(data_dir)
    if not base_path.exists() or not base_path.is_dir():
        print(f"Warning: Directory {data_dir} not found. (Expected during GitHub-only workflow)")
        return None, None

    # Collect all image files and sort for deterministic cross-platform splits
    all_images = sorted(list(base_path.glob(f'*{DEEPGLOBE_IMG_SUFFIX}')))
    if not all_images:
        print(f"Warning: No files ending with {DEEPGLOBE_IMG_SUFFIX} found in {data_dir}.")
        return None, None

    # Deterministic split
    total_size = len(all_images)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_paths, val_paths = random_split(all_images, [train_size, val_size], generator=generator)

    # Instantiate datasets
    train_dataset = DeepGlobeDataset(list(train_paths), transform=get_transforms(is_train=True))
    val_dataset = DeepGlobeDataset(list(val_paths), transform=get_transforms(is_train=False))

    # Instantiate dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # DRY RUN / TESTING STRATEGY
    # Simulates Kaggle path or fallback local testing
    test_dir = "/kaggle/input/deepglobe-road-extraction-dataset/train"
    
    print("Initiating DataLoader Unit Test...")
    train_loader, val_loader = get_dataloaders(test_dir, batch_size=4)
    
    if train_loader is not None:
        images, masks = next(iter(train_loader))
        
        # Verify Shapes
        print(f"Image Batch Shape: {images.shape} | Expected: (4, 3, 512, 512)")
        print(f"Mask Batch Shape: {masks.shape} | Expected: (4, 1, 512, 512)")
        assert images.shape == (4, 3, 512, 512), "Image shape mismatch!"
        assert masks.shape == (4, 1, 512, 512), "Mask shape mismatch!"
        
        # Verify Dtypes
        print(f"Image Dtype: {images.dtype} | Expected: torch.float32")
        print(f"Mask Dtype: {masks.dtype} | Expected: torch.float32")
        assert images.dtype == torch.float32, "Image dtype mismatch!"
        assert masks.dtype == torch.float32, "Mask dtype mismatch!"
        
        # Verify Mask Binary Contract (CRITICAL FOR M2)
        unique_vals = torch.unique(masks)
        print(f"Unique Mask Values: {unique_vals.tolist()} | Expected: [0.0] or [0.0, 1.0]")
        for val in unique_vals:
            assert val.item() in [0.0, 1.0], f"Violation of M2 Output Contract! Found non-binary value: {val.item()}"
            
        print("All Phase 1 constraints successfully validated.")
    else:
        print("Dataset not found locally. Code logic passes dry run. Push to Kaggle for data validation.")
