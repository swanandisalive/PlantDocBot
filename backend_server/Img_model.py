import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt # Added for graphing

# ==========================================
# 1. Define the Model
# ==========================================
class MyCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(MyCNNModel, self).__init__()
        
        self.con_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout helps prevent overfitting
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.con_layers(x)
        x = self.adaptive_pool(x)
        x = self.layers(x)
        return x

# ==========================================
# 2. Main Execution
# ==========================================
def main():
    # --- A. Check GPU ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ SUCCESS: Detected GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ WARNING: GPU not detected. Using CPU.")
        device = torch.device("cpu")

    # --- B. Data Path ---
    base_path = "C:\PlantDocBot_Dataset"
    
    print(f"\n[1/4] Checking Data at: {base_path}")
    if not os.path.exists(base_path):
         print(f"❌ ERROR: Dataset path '{base_path}' not found!")
         print("Please update 'base_path' in the code to your actual dataset folder.")
         return

    # --- C. Transforms (Fix for Overfitting) ---
    # Training: Heavy augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    # Validation: Simple resize
    valid_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    print("[2/4] Loading Images...")
    
    # Check for specific train/valid folders
    train_dir = os.path.join(base_path, "train")
    valid_dir = os.path.join(base_path, "valid") # or 'val'

    if os.path.exists(train_dir) and os.path.exists(valid_dir):
        print("   Found separate 'train' and 'valid' folders.")
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
        valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)
    else:
        print("   Single folder detected. Splitting automatically (80/20)...")
        full_dataset = datasets.ImageFolder(root=base_path, transform=train_transforms)
        train_size = int(0.8 * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
        # Force valid_dataset to use the simple transform
        valid_dataset.dataset.transform = valid_transforms

    # Optimized for RTX 4060
    BATCH_SIZE = 32 # Reduced to 32 to help generalization
    NUM_WORKERS = 0 # Set to 0 for Windows compatibility if you get errors, else 4
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Get class names safely
    if hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)
    else:
        num_classes = len(train_dataset.dataset.classes)
    print(f"      Found {num_classes} classes.")

    # --- D. Model Setup ---
    model = MyCNNModel(num_classes=num_classes).to(device)
    
    epochs = 20
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    # --- E. Training Loop ---
    print("\n[3/4] Starting Training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = y_pred.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

            if batch_idx % 50 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                valid_loss += loss.item()
                preds = y_pred.argmax(dim=1)
                valid_correct += (preds == y).sum().item()
                valid_total += y.size(0)

        # Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        train_acc = train_correct / train_total
        valid_acc = valid_correct / valid_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_valid_loss)

        print(f"Epoch {epoch+1} Result: Train Acc: {train_acc:.4f} | Val Acc: {valid_acc:.4f} | Val Loss: {avg_valid_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes.")
    
    # --- F. Save Model ---
    print("[4/4] Saving Model...")
    torch.save(model.state_dict(), "plant_disease_model_final.pth")
    print(f"      Model saved successfully.")
    
    # Plot Graph
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.savefig('training_curve.png')
    print("      Graph saved as 'training_curve.png'")

if __name__ == '__main__':
    main()