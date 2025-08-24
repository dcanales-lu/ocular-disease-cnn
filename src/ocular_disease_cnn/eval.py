import os
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .model import OcularCNN

# -----------------------
# Config
# -----------------------
CKPT_PATH = os.getenv("CKPT_PATH", "checkpoints/best_model.pth")
DATA_DIR  = os.getenv("DATA_DIR", "data/dataset")  # usamos dataset completo para validar
IMG_SIZE  = int(os.getenv("IMG_SIZE", 224))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

# -----------------------
# Cargar checkpoint
# -----------------------
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

classes = ckpt["classes"]
img_size = ckpt["img_size"]

model = OcularCNN(num_classes=len(classes))
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"Loaded model trained until epoch {ckpt['epoch']}")
print(f"Validation metrics at save time: loss={ckpt['val_loss']:.4f}, acc={ckpt['val_acc']:.3f}")
print(f"Classes: {classes}")

# -----------------------
# Quick evaluation on full dataset (val split)
# -----------------------
val_tfms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=val_tfms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

correct, total = 0, 0
with torch.inference_mode():
    for images, targets in loader:
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.numel()

print(f"Eval accuracy on whole dataset (no split): {correct/total:.3f}")
