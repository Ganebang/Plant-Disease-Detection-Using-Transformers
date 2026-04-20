import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.amp import GradScaler
import matplotlib.pyplot as plt

# Ensure that we can import from src when running `python src/train.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import PlantDocDataset
from src.model import PlantDocDETR
from src.loss import HungarianMatcher, SetCriterion
from src.engine import train_one_epoch, evaluate

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(is_train=True):
    transforms = []
    # Data Augmentation: Crucial for PlantDoc as lighting varies drastically
    if is_train:
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    transforms.append(T.Resize((416, 416)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(transforms)

def main():
    print("Initializing DETR Training Pipeline for PlantDoc")
    
    # 1. Paths
    train_dir = os.path.join("data", "raw", "train")
    test_dir = os.path.join("data", "raw", "test")
    train_ann = os.path.join(train_dir, "_annotations.coco.json")
    test_ann = os.path.join(test_dir, "_annotations.coco.json")
    
    if not os.path.exists(train_ann):
        print(f"Error: Could not find annotations at {train_ann}")
        print("Please run `python data/download_data.py` first.")
        return
        
    os.makedirs("output", exist_ok=True)
    os.makedirs("weights", exist_ok=True)  # Used by main.py
    
    # 2. Datasets & Dataloaders
    train_transform = get_transform(is_train=True)
    val_transform = get_transform(is_train=False)
    train_dataset = PlantDocDataset(train_dir, train_ann, transforms=train_transform)
    val_dataset = PlantDocDataset(test_dir, test_ann, transforms=val_transform)
    
    # Using small batch size to fit in 6GB VRAM
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # 3. Model
    # We must ensure num_classes handles the max ID, which requires +1 since IDs are 0-indexed
    cat_ids = train_dataset.coco.getCatIds()
    num_classes = max(cat_ids) + 1 if len(cat_ids) > 0 else 30
    print(f"Max class id detected bounds: {num_classes}. Initializing model with {num_classes} classes.")
    
    # Apple MPS support (for macs) or CUDA for Nvidia
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Training on physical device: {device}")
    
    model = PlantDocDETR(num_classes=num_classes)
    model.to(device)
    
    # 4. Class Imbalance Weights Strategy
    class_counts = {i: 0 for i in range(num_classes)}
    for ann in train_dataset.coco.dataset['annotations']:
        cat_id = ann['category_id']
        if cat_id < num_classes:
            class_counts[cat_id] += 1
            
    total_anns = sum(class_counts.values())
    class_weights = []
    if total_anns > 0:
        for i in range(num_classes):
            if class_counts[i] > 0:
                # Calculate inverse frequency (rarer classes get higher weight)
                weight = total_anns / (num_classes * class_counts[i])
                # Cap the maximum weight to 5.0 so an extremely rare class doesn't dominate gradients
                class_weights.append(min(weight, 5.0)) 
            else:
                class_weights.append(0.0)
    else:
        class_weights = [1.0] * num_classes
        
    class_weights.append(0.1) # The very last element is the background 'no object' class weight (eos_coef)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # 5. Criterion & Optimizer
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes']
    
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses, class_weights=class_weights_tensor)
    criterion.to(device)
    
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
    ]
    optimizer = optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4) # Standard transformer lr
    
    epochs = 20  # Massive reduction: Pre-trained DETRs converge extremely fast
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_maps = []
    
    best_val_loss = float('inf')
    
    # 6. Scaler and Scheduler
    scaler = GradScaler('cuda')
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, scaler, accumulation_steps=4)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
            
        # Val
        avg_val_loss, avg_val_acc, avg_val_map = evaluate(model, criterion, val_loader, device)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        val_maps.append(avg_val_map)
        
        # Step the learning rate at the end of the epoch
        lr_scheduler.step()
        
        print(f"End of Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {avg_val_acc:.4f}")
        
        # Save Best Model to output and weights (for fast streamlit test)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state = model.state_dict()
            torch.save(state, os.path.join("output", "model_20.pth"))
            torch.save(state, os.path.join("weights", "model_20.pth"))
            print(f"Saved Best Model (Loss: {best_val_loss:.4f})")
            
    # Plotting Output - Loss Graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', marker='s')
    plt.title('PlantDoc DETR: Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Total Weighted Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("output", "loss_graph_20.png"))
    plt.close()
    
    # Plotting Output - Accuracy Graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accs, label='Train Acc', marker='o', color='green')
    plt.plot(range(1, epochs + 1), val_accs, label='Val Acc', marker='s', color='orange')
    plt.title('PlantDoc DETR: Subset Match Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("output", "accuracy_graph_20.png"))
    plt.close()
    
    # Plotting Output - Mean Average Precision Graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), val_maps, label='Val mAP', marker='s', color='orange')
    plt.title('Mean Average Precision (mAP)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Average Precision (mAP)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("output", "map_graph_20.png"))
    plt.close()
    
    print("\nTraining completed! Saved three separate graphs to output/loss_graph.png, output/accuracy_graph.png, and output/map_graph.png")

if __name__ == "__main__":
    main()
