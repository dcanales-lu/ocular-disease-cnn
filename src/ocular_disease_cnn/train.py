import os
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from .model import OcularCNN

# -----------------------
# Hiperparámetros (variables de entorno con defaults)
# -----------------------
EPOCHS       = int(os.getenv("EPOCHS", 10))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 32))
LR           = float(os.getenv("LR", 1e-3))
IMG_SIZE     = int(os.getenv("IMG_SIZE", 224))
NUM_WORKERS  = int(os.getenv("NUM_WORKERS", 4))
DATA_DIR     = os.getenv("DATA_DIR", "data")         # puede contener train/val o dataset/
RUNS_DIR     = os.getenv("RUNS_DIR", "runs/exp1")
SEED         = int(os.getenv("SEED", 42))
DEVICE_SEL   = os.getenv("DEVICE", "auto")           # "cuda", "cpu" o "auto"
VAL_RATIO    = float(os.getenv("VAL_RATIO", 0.2))    # usado si solo hay dataset/

# Checkpoints
CKPT_DIR     = os.getenv("CKPT_DIR", "checkpoints")  # carpeta donde guardar pesos
BEST_METRIC  = os.getenv("BEST_METRIC", "val_acc")   # métrica para “best_model.pth” (val_acc o val_loss)

# -----------------------
# Utilidades
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def get_device() -> torch.device:
    if DEVICE_SEL == "cpu":
        return torch.device("cpu")
    if DEVICE_SEL == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_transforms(img_size: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tfms, val_tfms

def load_datasets(data_dir: str, img_size: int) -> Tuple[DataLoader, DataLoader, list]:
    """
    Caso A: data/train y data/val -> usa ImageFolder en ambas.
    Caso B: data/dataset/<clase>/... -> hace split 80/20 (VAL_RATIO) en memoria.
    """
    base = Path(data_dir)
    train_path = base / "train"
    val_path   = base / "val"
    dataset_path = base / "dataset"

    train_tfms, val_tfms = build_transforms(img_size)

    if train_path.exists() and val_path.exists():
        # ---- Caso A: pre-split
        train_ds = datasets.ImageFolder(train_path, transform=train_tfms)
        val_ds   = datasets.ImageFolder(val_path,   transform=val_tfms)
        classes = train_ds.classes
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        return train_loader, val_loader, classes

    elif dataset_path.exists():
        # ---- Caso B: dataset completo
        full_ds = datasets.ImageFolder(dataset_path, transform=train_tfms)
        classes = full_ds.classes

        gen = torch.Generator().manual_seed(SEED)
        n_total = len(full_ds)
        n_val = int(VAL_RATIO * n_total)
        n_train = n_total - n_val

        train_subset, val_subset_idx = random_split(full_ds, [n_train, n_val], generator=gen)

        # Val con transforms distintos
        full_ds_val = datasets.ImageFolder(dataset_path, transform=val_tfms)
        val_subset = Subset(full_ds_val, val_subset_idx.indices)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        return train_loader, val_loader, classes

    else:
        raise RuntimeError(
            f"No se encontró estructura válida en '{data_dir}'. "
            f"Debe existir 'data/train'+'data/val' o 'data/dataset/<clase>/...'"
        )

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.numel()

def save_checkpoint(model, optimizer, epoch, classes, img_size, val_loss, val_acc, ckpt_dir, tag):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(ckpt_dir) / f"{tag}.pth"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "classes": classes,
        "img_size": img_size,
    }, ckpt_path)
    return ckpt_path

# -----------------------
# Entrenamiento
# -----------------------
def train():
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, classes = load_datasets(DATA_DIR, IMG_SIZE)
    num_classes = len(classes)
    print(f"Detected classes: {classes} (num_classes={num_classes})")

    model = OcularCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    writer = SummaryWriter(log_dir=RUNS_DIR)

    best_val_acc = -1.0
    best_val_loss = float("inf")

    global_step = 0
    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        train_loss_sum, train_acc_sum, train_samples = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")

        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            acc = accuracy_from_logits(logits, targets)
            train_loss_sum += loss.item() * batch_size
            train_acc_sum  += acc * batch_size
            train_samples  += batch_size

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/acc",  acc,         global_step)
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}"})

        epoch_train_loss = train_loss_sum / train_samples
        epoch_train_acc  = train_acc_sum  / train_samples

        # ---- Val ----
        model.eval()
        val_loss_sum, val_acc_sum, val_samples = 0.0, 0.0, 0
        with torch.inference_mode():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]")
            for images, targets in pbar_val:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)

                batch_size = targets.size(0)
                acc = accuracy_from_logits(logits, targets)
                val_loss_sum += loss.item() * batch_size
                val_acc_sum  += acc * batch_size
                val_samples  += batch_size

                pbar_val.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}"})

        epoch_val_loss = val_loss_sum / val_samples
        epoch_val_acc  = val_acc_sum  / val_samples

        writer.add_scalars("epoch/loss", {"train": epoch_train_loss, "val": epoch_val_loss}, epoch)
        writer.add_scalars("epoch/acc",  {"train": epoch_train_acc,  "val": epoch_val_acc},  epoch)

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"train_loss={epoch_train_loss:.4f} train_acc={epoch_train_acc:.3f} | "
            f"val_loss={epoch_val_loss:.4f} val_acc={epoch_val_acc:.3f}"
        )

        # ---- Checkpoints (por época + mejor)
        ep_path = save_checkpoint(
            model, optimizer, epoch+1, classes, IMG_SIZE,
            val_loss=epoch_val_loss, val_acc=epoch_val_acc,
            ckpt_dir=CKPT_DIR, tag=f"epoch_{epoch+1:03d}"
        )
        print(f"Saved epoch checkpoint: {ep_path}")

        is_better = (
            (BEST_METRIC == "val_acc"  and epoch_val_acc  > best_val_acc) or
            (BEST_METRIC == "val_loss" and epoch_val_loss < best_val_loss)
        )
        if is_better:
            best_val_acc  = max(best_val_acc,  epoch_val_acc)
            best_val_loss = min(best_val_loss, epoch_val_loss)
            best_path = save_checkpoint(
                model, optimizer, epoch+1, classes, IMG_SIZE,
                val_loss=epoch_val_loss, val_acc=epoch_val_acc,
                ckpt_dir=CKPT_DIR, tag="best_model"
            )
            print(f"Updated best model → {BEST_METRIC}: {best_path}")

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    train()