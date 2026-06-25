import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import DeepGlobeDataset, get_transforms
from model import get_model
from losses import TopologyLoss

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = DeepGlobeDataset(args.data_dir, is_train=True, transform=get_transforms(is_train=True))
    if len(train_dataset) == 0:
        print("Warning: Dataset is empty. Are you on Kaggle?")
    else:
        print(f"Loaded {len(train_dataset)} training images.")
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    model = get_model(architecture="linknet", encoder="resnet34").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = TopologyLoss(bce_w=0.5, dice_w=0.5, cldice_w=0.0) # Start with BCE+Dice
    
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        # tqdm pbar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=args.fp16):
                logits = model(images)
                loss = criterion(logits, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/max(1, len(train_loader)):.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, "last.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to /kaggle/input/...")
    parser.add_argument('--checkpoint_dir', type=str, default="/kaggle/working/checkpoints/")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fp16', action='store_true', help="Use mixed precision")
    parser.add_argument('--resume', type=str, default="", help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    train(args)
