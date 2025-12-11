import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import argparse
import os

# Import your custom definitions
from ModelDefinitions import VictimClassifier
from CelebADataset import CelebADataset

def get_args():
    parser = argparse.ArgumentParser(description="Victim Inference")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the saved .pth model file")
    
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top k indices
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        # Check if target is in top k predictions
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    args = get_args()
    
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Running inference on: {device}")

    # ==========================================
    # 1. Reconstruct the Global Identity Map
    # ==========================================
    print("Reconstructing Global Identity Map from all splits...")
    train_df = pd.read_csv("victim_train.csv")
    val_df = pd.read_csv("victim_val.csv")
    test_df = pd.read_csv("victim_test.csv")

    # Combine all IDs to get the master list
    all_ids = pd.concat([train_df['id'], val_df['id'], test_df['id']]).unique()
    all_ids = sorted(all_ids) # Sort to ensure deterministic order (0=Person A, 1=Person B...)
    
    # Create the Master Key
    global_id_map = {original_id: idx for idx, original_id in enumerate(all_ids)}
    num_classes = len(all_ids)
    
    print(f"Total Unique Identities (num_classes): {num_classes}")

    # ==========================================
    # 2. Dataset Setup
    # ==========================================
    # Standard Transform (Resize only, no augmentation)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # Initialize Dataset with the GLOBAL MAP
    # We only run inference on the TEST split
    test_dataset = CelebADataset(
        df=test_df, 
        root_dir=None, 
        id_map=global_id_map,  # <--- Passing the map here
        transform=transform
    )
    
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(test_dataset)} test images.")

    # ==========================================
    # 3. Model Setup
    # ==========================================
    print(f"Initializing model for {num_classes} classes...")
    model = VictimClassifier(num_classes=num_classes)
    
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    
    model.to(device)
    model.eval()

    # ==========================================
    # 4. Inference Loop
    # ==========================================
    top1_acc_total = 0
    top5_acc_total = 0
    total_samples = 0
    
    print("\n--- Starting Inference Loop ---")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs, labels, original_ids = batch
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward Pass (Raw Logits)
            logits = model(imgs)
            
            # Calculate Probabilities (Softmax) for inspection
            probs = F.softmax(logits, dim=1)
            
            # Calculate Metrics
            acc1, acc5 = calculate_accuracy(logits, labels, topk=(1, 5))
            
            # Accumulate
            top1_acc_total += acc1.item() * imgs.size(0)
            top5_acc_total += acc5.item() * imgs.size(0)
            total_samples += imgs.size(0)
            
            # Sanity Check: Print first batch details
            if i == 0:
                print(f"\n[Batch 0 Sample Inspection]")
                # Get the predicted class for the first image
                pred_conf, pred_idx = torch.max(probs[0], dim=0)
                actual_label = labels[0].item()
                actual_id = original_ids[0].item()
                
                print(f"Image ID (Original):      {actual_id}")
                print(f"Ground Truth Label (Map): {actual_label}")
                print(f"Predicted Label (Map):    {pred_idx.item()}")
                print(f"Confidence:               {pred_conf.item():.4f}")
                print("-" * 30)

    # 5. Final Report
    avg_top1 = top1_acc_total / total_samples
    avg_top5 = top5_acc_total / total_samples

    print(f"\nResults on {total_samples} test images:")
    print(f"Top-1 Accuracy: {avg_top1:.2f}%")
    print(f"Top-5 Accuracy: {avg_top5:.2f}%")