import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

class Config:
    # UPDATED: Pointing to your local folder
    DATA_DIR = "./orl_data" 
    MODEL_PATH = "orl_victim_model.pth"
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 40  # High enough to ensure overfitting (crucial for this attack)
    
    # UPDATED: Logic for Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("[*] Apple Silicon (MPS) detected and enabled.")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("[*] CUDA GPU detected and enabled.")
    else:
        DEVICE = torch.device("cpu")
        print("[*] No GPU detected. Running on CPU.")

    TARGET_CLASS = 0  # The person we want to reconstruct (Folder 's1' usually maps to index 0)
    TV_WEIGHT = 0.05  # Total Variation weight (higher = smoother image)
    ATTACK_ITERS = 3000

def setup_data():
    """
    Loads data from the local ./orl_data folder.
    Expects subfolders (e.g., s1, s2...) inside orl_data.
    """
    if not os.path.exists(Config.DATA_DIR):
        raise FileNotFoundError(f"Could not find directory: {Config.DATA_DIR}. Please ensure it exists.")

    # ORL images are grayscale. We normalize to [-1, 1] for better attack convergence.
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ImageFolder automatically uses subfolders as class labels
    dataset = datasets.ImageFolder(root=Config.DATA_DIR, transform=transform)
    
    print(f"[*] Loaded {len(dataset)} images across {len(dataset.classes)} classes.")

    # 90% Train (to memorize faces), 10% Test (to check basic functionality)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Fixed seed for reproducibility
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, dataset

class FaceNet(nn.Module):
    def __init__(self, num_classes=40):
        super(FaceNet, self).__init__()
        # Input: 1 x 112 x 92
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 56 x 46
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 28 x 23
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 14 x 11
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 11, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_model(model, train_loader):
    print(f"\n[+] Starting training for {Config.EPOCHS} epochs...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    model.train()

    for epoch in range(Config.EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    print("[+] Training complete. Saving model...")
    torch.save(model.state_dict(), Config.MODEL_PATH)

def evaluate_model(model, test_loader):
    print("\n[+] Running Inference Evaluation...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"    Test Accuracy: {acc:.2f}%")
    if acc < 50:
        print("    [!] Warning: Accuracy is low. The attack may produce poor results.")

def total_variation_loss(img, weight):
    # Total Variation regularization to smooth the image
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w)

def perform_inversion(model):
    print(f"\n[+] Launching Inversion Attack on Class Index {Config.TARGET_CLASS}...")
    model.eval()
    
    # Start with random noise
    # We require gradients for the INPUT image
    reconstructed = torch.randn((1, 1, 112, 92), device=Config.DEVICE, requires_grad=True)
    
    optimizer = optim.Adam([reconstructed], lr=0.1)
    
    for i in range(Config.ATTACK_ITERS):
        optimizer.zero_grad()
        output = model(reconstructed)
        
        # Loss = Minimize ( -Target_Score + TV_Regularization )
        # We want to maximize the score for the specific target class
        target_score = output[0][Config.TARGET_CLASS]
        tv_reg = total_variation_loss(reconstructed, Config.TV_WEIGHT)
        
        loss = -target_score + tv_reg
        loss.backward()
        optimizer.step()
        
        # Clamp to valid pixel range
        with torch.no_grad():
            reconstructed.clamp_(-1, 1)
            
        if i % 500 == 0:
            print(f"    Iter {i}: Target Score: {target_score.item():.2f}")

    return reconstructed.detach().cpu()

# ==========================================
# 5. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    
    # 1. Setup Data (Local Folder)
    train_loader, test_loader, full_dataset = setup_data()
    
    # 2. Init Model
    # Automatically adjust num_classes based on folders found
    num_classes = len(full_dataset.classes)
    print(f"[*] Detected {num_classes} identities (classes).")
    
    model = FaceNet(num_classes=num_classes).to(Config.DEVICE)
    
    # 3. Load or Train
    if os.path.exists(Config.MODEL_PATH):
        print(f"[*] Found saved model at {Config.MODEL_PATH}. Loading...")
        # Note: map_location is crucial when moving models between CPU/MPS/CUDA
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    else:
        train_model(model, train_loader)
        
    # 4. Inference / Evaluation
    evaluate_model(model, test_loader)
    
    # 5. Attack
    inverted_tensor = perform_inversion(model)
    
    # 6. Visualization
    # Retrieve a REAL image of the target class for comparison
    real_img = None
    target_label = Config.TARGET_CLASS
    
    # Find the first image that matches our target class index
    for img, label in full_dataset:
        if label == target_label:
            real_img = img
            break
            
    print("\n[+] Displaying results...")
    
    # Helper to un-normalize [-1, 1] -> [0, 1] for display
    def unnorm(t): return (t.squeeze() * 0.5) + 0.5
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    if real_img is not None:
        ax[0].imshow(unnorm(real_img), cmap='gray')
        ax[0].set_title(f"Real Image (Class {target_label})")
    else:
        ax[0].text(0.5, 0.5, "Real Image Not Found", ha='center')
    ax[0].axis('off')
    
    ax[1].imshow(unnorm(inverted_tensor), cmap='gray')
    ax[1].set_title("Inverted Image (Recovered from Weights)")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("[+] Done.")