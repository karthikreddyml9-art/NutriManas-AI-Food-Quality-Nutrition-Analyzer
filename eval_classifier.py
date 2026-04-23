"""
=============================================================
NutriManas — Food Classifier Evaluation Script
=============================================================

WHAT IS AN EVAL?
  An evaluation measures how good your AI model actually is.
  Instead of guessing "it seems to work", we give it images
  it has never seen and measure exact accuracy numbers.

METRICS WE CALCULATE:
  - Top-1 Accuracy: Did the model pick the EXACT correct class?
  - Top-3 Accuracy: Was the correct class in the top 3 guesses?
  - Top-5 Accuracy: Was the correct class in the top 5 guesses?
  - Per-class accuracy: Which foods does the model know best?
  - Confusion: Which foods does it mix up with each other?

HOW TO RUN:
  D:\\NutriManas\\backend\\train_venv\\Scripts\\python.exe eval_classifier.py
=============================================================
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

# ── Config ────────────────────────────────────────────────
DATA_DIR   = Path("data/food_images")      # same images used in training
MODEL_PATH = Path("models/indian_food_mobilenet.pth")
CLASSES_PATH = Path("models/food_classes.json")
BATCH_SIZE = 32
VAL_SPLIT  = 0.2   # must match train_food_classifier.py

# ── Load class labels ──────────────────────────────────────
with open(CLASSES_PATH) as f:
    classes = json.load(f)
num_classes = len(classes)
print(f"Evaluating {num_classes} Indian food classes\n")

# ── Load model ─────────────────────────────────────────────
# This is the same MobileNetV3-Small architecture we trained.
# We load the checkpoint saved during training.
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, num_classes
)
checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()   # eval mode: disables dropout, batchnorm acts differently
print(f"Loaded model from epoch {checkpoint['epoch']} "
      f"(train-time val acc: {checkpoint['val_acc']:.1f}%)\n")

# ── Transforms ─────────────────────────────────────────────
# Must match EXACTLY what we used during validation in training.
# If transforms differ, accuracy will be artificially lower.
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Load validation split ──────────────────────────────────
# We reproduce the same train/val split using the same random seed.
# torch.manual_seed ensures we get the SAME split as training.
torch.manual_seed(42)
full_dataset = ImageFolder(DATA_DIR, transform=val_transforms)
val_size  = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
_, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Validation set: {val_size} images\n")
print("Running evaluation... (this takes ~1-2 minutes)\n")

# ── Evaluation loop ────────────────────────────────────────
# For each batch of images:
#   1. Run model forward pass → get logits (raw scores per class)
#   2. Convert to probabilities with softmax
#   3. Take top-5 predictions
#   4. Check if true label appears in top-1, top-3, top-5

top1_correct = 0
top3_correct = 0
top5_correct = 0
total = 0

# Per-class tracking: how many correct vs total per food type
per_class_correct = defaultdict(int)
per_class_total   = defaultdict(int)

# Confusion tracking: when model guesses wrong, what does it guess?
# confusion[true_class][predicted_class] = count
confusion = defaultdict(lambda: defaultdict(int))

t0 = time.time()

with torch.no_grad():   # no_grad = don't compute gradients (saves memory, faster)
    for images, labels in val_loader:
        outputs = model(images)                        # shape: [batch, num_classes]
        probs   = torch.softmax(outputs, dim=1)        # convert logits → probabilities

        # Get top-5 predictions for each image in batch
        top5_preds = torch.topk(probs, k=5, dim=1).indices  # shape: [batch, 5]

        for i in range(len(labels)):
            true_label = labels[i].item()
            top5       = top5_preds[i].tolist()
            pred_label = top5[0]   # top-1 = most confident prediction

            # Update accuracy counters
            if true_label == top5[0]:
                top1_correct += 1
            if true_label in top5[:3]:
                top3_correct += 1
            if true_label in top5:
                top5_correct += 1

            # Per-class tracking
            per_class_total[true_label]   += 1
            if true_label == pred_label:
                per_class_correct[true_label] += 1
            else:
                # Record confusion: model thought true_label was pred_label
                confusion[true_label][pred_label] += 1

            total += 1

elapsed = time.time() - t0

# ── Results ────────────────────────────────────────────────
print("=" * 55)
print("  EVALUATION RESULTS")
print("=" * 55)
print(f"  Total images evaluated : {total}")
print(f"  Time taken             : {elapsed:.1f}s")
print()
print(f"  Top-1 Accuracy  : {100*top1_correct/total:.1f}%  "
      f"({top1_correct}/{total} exact matches)")
print(f"  Top-3 Accuracy  : {100*top3_correct/total:.1f}%  "
      f"(correct in top 3 guesses)")
print(f"  Top-5 Accuracy  : {100*top5_correct/total:.1f}%  "
      f"(correct in top 5 guesses)")
print("=" * 55)

# ── Per-class breakdown ────────────────────────────────────
print("\n  PER-CLASS ACCURACY (sorted worst → best)\n")
per_class_acc = {}
for cls_idx, total_count in per_class_total.items():
    correct = per_class_correct[cls_idx]
    acc = 100 * correct / total_count
    per_class_acc[cls_idx] = acc

sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])

print(f"  {'Food':<30} {'Accuracy':>10}  {'Correct/Total':>15}")
print(f"  {'-'*30} {'-'*10}  {'-'*15}")
for cls_idx, acc in sorted_classes:
    name    = classes[cls_idx].replace("_", " ").title()
    correct = per_class_correct[cls_idx]
    total_c = per_class_total[cls_idx]
    bar     = "█" * int(acc / 10)
    print(f"  {name:<30} {acc:>9.1f}%  {correct:>5}/{total_c:<5}  {bar}")

# ── Top confusions ─────────────────────────────────────────
print("\n  TOP CONFUSIONS (foods the model mixes up)\n")
all_confusions = []
for true_cls, preds in confusion.items():
    for pred_cls, count in preds.items():
        all_confusions.append((count, true_cls, pred_cls))

all_confusions.sort(reverse=True)
print(f"  {'True Food':<25} {'Predicted As':<25} {'Count':>6}")
print(f"  {'-'*25} {'-'*25} {'-'*6}")
for count, true_cls, pred_cls in all_confusions[:15]:
    true_name = classes[true_cls].replace("_", " ").title()
    pred_name = classes[pred_cls].replace("_", " ").title()
    print(f"  {true_name:<25} {pred_name:<25} {count:>6}")

print("\n  Evaluation complete!")
print("  Tip: Low-accuracy classes need more training images.")
print("  Tip: High confusion pairs look visually similar to the model.\n")
