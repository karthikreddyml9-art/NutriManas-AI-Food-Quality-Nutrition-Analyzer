"""
Train MobileNetV3-Small on Indian Food Images dataset.
Fine-tunes a pretrained MobileNetV3 for 80 Indian food classes.
Run: python train_food_classifier.py
Output: models/indian_food_mobilenet.pth + models/food_classes.json
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

DATA_DIR = Path("data/food_images")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "indian_food_mobilenet.pth"
CLASSES_PATH = MODEL_DIR / "food_classes.json"

BATCH_SIZE = 16
EPOCHS = 15
LR = 0.001
VAL_SPLIT = 0.2
IMG_SIZE = 224

MODEL_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_data():
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transforms

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    classes = full_dataset.classes
    print(f"Classes: {len(classes)}, Train: {train_size}, Val: {val_size}")
    return train_loader, val_loader, classes


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def train():
    train_loader, val_loader, classes = load_data()
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f)
    print(f"Saved {len(classes)} class labels to {CLASSES_PATH}")

    model = build_model(len(classes))
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam([
        {"params": model.features.parameters(), "lr": LR * 0.1},
        {"params": model.classifier.parameters(), "lr": LR},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = val_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.3f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.3f} Acc: {val_acc:.1f}% | "
              f"Time: {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
            }, MODEL_PATH)
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.1f}%")

    print(f"\n🎉 Training complete! Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
