import os
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from .model import OcularCNN

# -----------------------
# Config
# -----------------------
CKPT_PATH = os.getenv("CKPT_PATH", "checkpoints/best_model.pth")
IMG_SIZE  = int(os.getenv("IMG_SIZE", 224))

# -----------------------
# Cargar checkpoint
# -----------------------
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

classes = ckpt["classes"]
model = OcularCNN(num_classes=len(classes))
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"Loaded model trained until epoch {ckpt['epoch']}")
print(f"Classes: {classes}")

# -----------------------
# Transformaciones
# -----------------------
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------
# Predicción sobre imágenes dadas en CLI
# -----------------------
if len(sys.argv) < 2:
    print("Uso: uv run python -m ocular_disease_cnn.predict <imagen1> [imagen2 ...]")
    sys.exit(1)

for img_path in sys.argv[1:]:
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"[!] No existe: {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")
    x = tfms(img).unsqueeze(0)  # batch=1

    with torch.inference_mode():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        prob = torch.softmax(logits, dim=1)[0, pred].item()

    print(f"{img_path.name}: {classes[pred]} (confianza {prob:.2f})")
