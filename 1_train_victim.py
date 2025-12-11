import pandas as pd
import numpy as np
import torch.nn as nn
import torch.functional as F
import torch
from ModelDefinitions import VictimClassifier
import argparse
from torchvision import transforms
from sklearn.model_selection import train_test_split
from CelebADataset import CelebADataset
from torch.utils.data import DataLoader
import torch.optim as optim
import os, json, datetime


def get_args():
    parser = argparse.ArgumentParser(description="Victim Model")

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, 
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--num_epochs", type=int, default=1, 
                        help="Number of epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size (default: 16)")
    parser.add_argument("--patience", type=int, default=3, 
                        help="Scheduler patience (default: 3)")
    
    # System / Logging
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of dataloader workers (default: 0)")
    parser.add_argument("--checkpoint_path", type=str, default="", 
                        help="Path to save checkpoints")
    parser.add_argument("--run_name", type=str, default="", 
                        help="Name for the run (default: empty string)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.5),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_df = pd.read_csv("victim_train.csv")
    val_df = pd.read_csv("victim_val.csv")
    test_df = pd.read_csv("victim_test.csv")

    all_ids = pd.concat([train_df['id'], val_df['id'], test_df['id']]).unique()
    all_ids = sorted(all_ids) # Sort to ensure deterministic order (0=Person A, 1=Person B...)

    # This is the "Master Key" for your project
    global_id_map = {original_id: idx for idx, original_id in enumerate(all_ids)}
    num_classes = len(all_ids)

    print(f"Total Unique Identities (num_classes): {num_classes}")
    
    train_dataset = CelebADataset(df=train_df, root_dir=None, id_map=global_id_map, transform=train_transform)
    val_dataset = CelebADataset(df=val_df, root_dir=None, id_map=global_id_map, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=(device.type == 'cuda')
    )

    # Ensure num_classes matches the number of unique IDs in your victim csv
    print(f"Initializing model for {num_classes} classes")
    model = VictimClassifier(num_classes=num_classes)
    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=args.patience
    )

    # Standard Classification Loss
    criterion = nn.CrossEntropyLoss()

    # Create run directory
    run_dir = args.checkpoint_path + args.run_name
    save_dir = os.path.join("runs", run_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_loss = float('inf')

    print(f"Starting training on device: {device}")

    for epoch in range(args.num_epochs):
        # ==========================
        # 1. Training Phase
        # ==========================
        model.train()
        running_train_loss = 0.0
        
        for batch in train_loader:
            # Unpack the batch (image, label, original_id)
            imgs, labels, _ = batch
            
            imgs = imgs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long) # Labels must be Long for CrossEntropy
            
            optimizer.zero_grad()            
            
            # Forward pass (returns logits)
            outputs = model(imgs)
            
            # Compute Loss (Supervised Classification)
            loss = criterion(outputs, labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # ==========================
        # 2. Validation Phase
        # ==========================
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels, _ = batch
                
                imgs = imgs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                
                # Forward pass
                outputs = model(imgs)
                
                # Compute Loss
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # Step the scheduler based on validation loss
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Logging
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Epoch {epoch+1}/{args.num_epochs}\t "
              f"Train Loss: {epoch_train_loss:.4f}\t "
              f"Val Loss: {epoch_val_loss:.4f}\t "
              f"[{current_time}]")

        # ==========================
        # 3. Checkpointing
        # ==========================
        # Save best model if validation loss improves
        if epoch_val_loss < best_loss:
            print(f"--> Validation loss improved from {best_loss:.4f} to {epoch_val_loss:.4f}. Saving model.")
            best_loss = epoch_val_loss
            save_name = f"best_{args.run_name}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, save_name))
        
        # Save every 10 epochs as a backup
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), os.path.join(save_dir, f"{args.run_name}_epoch_{epoch+1}.pth"))

    print("Training Complete.")

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_loss": best_loss,
        "epochs": args.num_epochs,
        "run_name": args.run_name,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    json_path = os.path.join(save_dir, f"{args.run_name}_history.json")
    
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"Training history saved to: {json_path}")