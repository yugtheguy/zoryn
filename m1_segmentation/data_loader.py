import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepGlobeDataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None):
        """
        DeepGlobe dataset loader.
        Assumes data_dir contains images and masks, e.g., /kaggle/input/deepglobe-road-extraction-dataset/train/
        Masks are expected to end in _mask.png
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        
        if not os.path.exists(data_dir):
            print(f"Warning: data_dir {data_dir} does not exist. (Expected on Kaggle)")
            self.image_files = []
        else:
            # DeepGlobe images are typically named like '123456_sat.jpg'
            self.image_files = [f for f in os.listdir(data_dir) if f.endswith('_sat.jpg')]
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        # DeepGlobe masks replace '_sat.jpg' with '_mask.png'
        mask_path = os.path.join(self.data_dir, img_name.replace('_sat.jpg', '_mask.png'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32) # Binary 0/1
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
            
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask has channel dimension (1, H, W)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            # It's a PyTorch tensor from ToTensorV2
            mask = mask.unsqueeze(0)
            
        return image, mask

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.RandomCrop(width=512, height=512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.3), # Simulate occlusion
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
