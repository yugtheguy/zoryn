"""
Module: train.py
Responsibility: Orchestrate the training pipeline with AMP optimization, 
checkpoint resilience, and lightweight CSV metric tracking for Kaggle execution.
"""

import os
import argparse
import csv
import random
from pathlib import Path
from typing import Dict
import numpy as np

import torch
from tqdm import tqdm

from data_loader import get_dataloaders
from model import get_model
from losses import TopologyLoss

def train_one_epoch(
    model: torch.nn.Module, 
    loader: torch.utils.data.DataLoader, 
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scaler: torch.amp.GradScaler, 
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> float:
    """Executes one training epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}/{total_epochs}", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad(set_to_none=True) # Performance optimization
        
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(images)
            loss = criterion(logits, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return running_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module, 
    loader: torch.utils.data.DataLoader, 
    criterion: torch.nn.Module, 
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> float:
    """Executes validation without gradient tracking to prevent memory leaks."""
    model.eval()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Val Epoch {epoch}/{total_epochs}", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(images)
            loss = criterion(logits, masks)
            
        running_loss += loss.item()
        pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
    return running_loss / len(loader)

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args: argparse.Namespace):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = checkpoint_dir / "training_metrics.csv"
    
    # 1. Dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        val_split=args.val_split
    )
    
    if train_loader is None or val_loader is None:
        print("Fatal Error: Dataloaders failed to initialize.")
        return

    # 2. Model, Loss, Optimizer
    model = get_model(architecture="linknet", encoder_name="resnet34").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Start Phase 1 training with BCE+Dice. clDice weight can be toggled via args later.
    criterion = TopologyLoss(bce_w=0.5, dice_w=0.5, cldice_w=0.0) 
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
    
    start_epoch = 1
    best_val_loss = float('inf')
    
    # 3. Resume Logic
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # 4. CSV Logger Init
    if start_epoch == 1:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args.epochs)
        val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, args.epochs)
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Log to CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}"])
            
        # Checkpointing
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss
        }
        
        # Save last
        torch.save(checkpoint_state, checkpoint_dir / "last.pt")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_state, checkpoint_dir / "best_model.pt")
            print(f" -> Best model saved with Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 Segmentation Training Pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to /kaggle/input/...")
    parser.add_argument('--checkpoint_dir', type=str, default="/kaggle/working/checkpoints")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (16 fits T4)")
    parser.add_argument('--epochs', type=int, default=50, help="Total epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--fp16', action='store_true', help="Enable mixed precision")
    parser.add_argument('--resume', type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument('--val_split', type=float, default=0.2, help="Validation holdout percentage")
    
    args = parser.parse_args()
    main(args)
