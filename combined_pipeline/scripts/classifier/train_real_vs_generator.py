#!/usr/bin/env python3
import argparse
import os
import random
import re
import shutil
import json
from typing import Dict, List, Tuple

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ============================
# Global config
# ============================

# Split sizes per class (you said 1000 total -> 800/100/100)
TRAIN_COUNT = 800
VAL_COUNT = 100
TEST_COUNT = 100

BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Map logical class names to relative dirs (from data_root)
# This matches YOUR structure exactly.
CLASS_DIRS = {
    "coco": "real/coco/test",
    "Dall-e2": "generated/Dall-e2/test",
    "Dall-e3": "generated/Dall-e3/test",
    "Firefly": "generated/Firefly/test",
    "MidjourneyV5": "generated/MidjourneyV5/test",
    "MidjourneyV6": "generated/MidjourneyV6/test",
    "SDXL": "generated/SDXL/test",
    "stable_diffusion_1-5": "generated/stable_diffusion_1-5/test",
}

REAL_CLASS_NAME = "coco"  # real baseline class used as negative


# ============================
# Dataset
# ============================

class BinaryImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        """
        samples: list of (absolute_image_path, binary_label)
        binary_label: 1 = positive (target generator), 0 = negative (real/coco)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ============================
# Helper functions
# ============================

def list_images_in_dir(full_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(full_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(root, f))
    return paths


def build_splits(
    data_root: str,
    class_dirs: Dict[str, str],
    train_count: int,
    val_count: int,
    test_count: int,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns splits[class_name] = {"train": [...], "val": [...], "test": [...]}
    """
    splits: Dict[str, Dict[str, List[str]]] = {}

    for cls_name, rel_dir in class_dirs.items():
        full_dir = os.path.join(data_root, rel_dir)
        imgs = list_images_in_dir(full_dir)
        total_needed = train_count + val_count + test_count

        if len(imgs) < total_needed:
            raise ValueError(
                f"Class '{cls_name}' (dir={full_dir}) has only {len(imgs)} images, "
                f"needs at least {total_needed}."
            )

        random.shuffle(imgs)
        train_imgs = imgs[:train_count]
        val_imgs = imgs[train_count:train_count + val_count]
        test_imgs = imgs[train_count + val_count:train_count + val_count + test_count]

        splits[cls_name] = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs,
        }

        print(f"[{cls_name}] total={len(imgs)}, "
              f"train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    return splits


def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf


def sanitize_name_for_filename(name: str) -> str:
    """
    Convert class name to clean token for filename.
    e.g. "Dall-e3" -> "dalle3", "stable_diffusion_1-5" -> "stablediffusion15"
    """
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


def safe_link_or_copy(src: str, dst: str):
    """
    Try to create a symlink; if that fails, copy the file.
    """
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_splits_to_folder(
    splits: Dict[str, Dict[str, List[str]]],
    splits_root: str,
):
    """
    Create a folder structure with symlinks/copies to all split images:

      splits_root/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg
    """
    print(f"\nMaterializing splits to: {splits_root}")
    for split_name in ["train", "val", "test"]:
        for cls_name, cls_splits in splits.items():
            target_dir = os.path.join(splits_root, split_name, cls_name)
            os.makedirs(target_dir, exist_ok=True)
            for src in cls_splits[split_name]:
                base = os.path.basename(src)
                dst = os.path.join(target_dir, base)
                # Handle potential name collisions
                if os.path.exists(dst):
                    name, ext = os.path.splitext(base)
                    k = 1
                    new_dst = os.path.join(target_dir, f"{name}_{k}{ext}")
                    while os.path.exists(new_dst):
                        k += 1
                        new_dst = os.path.join(target_dir, f"{name}_{k}{ext}")
                    dst = new_dst
                safe_link_or_copy(src, dst)


def train_real_vs_generator_classifier(
    target_generator: str,
    splits: Dict[str, Dict[str, List[str]]],
    real_class: str,
    output_dir: str,
    device: str,
    epochs: int,
):
    """
    Train a single binary classifier: target_generator (label 1) vs real_class (label 0).
    This matches Sandra's "single binary classifier for each generator" setup.
    """
    print(f"\n===== Training binary classifier: {target_generator} (1) vs {real_class} (0) =====")

    # ---------- Build train samples ----------
    pos_train = list(splits[target_generator]["train"])
    neg_train = list(splits[real_class]["train"])

    random.shuffle(pos_train)
    random.shuffle(neg_train)

    # Balance positives and negatives
    n = min(len(pos_train), len(neg_train))
    if n == 0:
        raise ValueError(
            f"No training data for {target_generator} or {real_class} (pos={len(pos_train)}, neg={len(neg_train)})"
        )
    pos_train = pos_train[:n]
    neg_train = neg_train[:n]

    train_samples = [(p, 1) for p in pos_train] + [(p, 0) for p in neg_train]
    random.shuffle(train_samples)

    # ---------- Build val/test samples ----------
    def build_eval_samples(split_name: str):
        pos = splits[target_generator][split_name]
        neg = splits[real_class][split_name]
        samples = [(p, 1) for p in pos] + [(p, 0) for p in neg]
        random.shuffle(samples)
        return samples

    val_samples = build_eval_samples("val")
    test_samples = build_eval_samples("test")

    print(f"[{target_generator}] Train: pos={len(pos_train)}, neg={len(neg_train)}, total={len(train_samples)}")
    print(f"[{target_generator}] Val:   {len(val_samples)}")
    print(f"[{target_generator}] Test:  {len(test_samples)}")

    # For JSON stats
    stats = {
        "train_pos": len(pos_train),
        "train_neg": len(neg_train),
        "val_pos": len(splits[target_generator]["val"]),
        "val_neg": len(splits[real_class]["val"]),
        "test_pos": len(splits[target_generator]["test"]),
        "test_neg": len(splits[real_class]["test"]),
    }

    train_tf, eval_tf = build_transforms()

    train_ds = BinaryImageDataset(train_samples, transform=train_tf)
    val_ds = BinaryImageDataset(val_samples, transform=eval_tf)
    test_ds = BinaryImageDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # ---------- Model ----------
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        model = models.resnet50(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Val ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ---- Test ----
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"[{target_generator}] Test accuracy (gen vs real): {test_acc:.4f}")
    stats["test_accuracy"] = test_acc

    # ---- Save ----
    os.makedirs(output_dir, exist_ok=True)
    base_gen = sanitize_name_for_filename(target_generator)
    base_real = sanitize_name_for_filename(real_class)
    filename = f"resnet50_{base_gen}_vs_{base_real}.pth"
    save_path = os.path.join(output_dir, save_path := os.path.join(output_dir, filename))
    torch.save(model.state_dict(), save_path)
    print(f"[{target_generator}] Saved model to: {save_path}")

    return stats


# ============================
# Main
# ============================

def main():
    parser = argparse.ArgumentParser(
        description="Train per-generator binary ResNet50 classifiers "
                    "(generator vs real/coco) for GI dataset structure"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/hdd-data/vaidya/gi_conference_dataset",
        help="Root dataset directory (contains 'real' and 'generated')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save trained .pth files and stats JSON",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=None,
        help="Where to materialize split images (train/val/test). "
             "Default: <data_root>/ovr_splits",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    if args.splits_dir is None:
        args.splits_dir = os.path.join(args.data_root, "ovr_splits")

    print(f"Using data_root: {args.data_root}")
    print("Classes and dirs:")
    for cls_name, rel_dir in CLASS_DIRS.items():
        print(f"  {cls_name:22s} -> {rel_dir}")

    # Build per-class splits
    splits = build_splits(
        data_root=args.data_root,
        class_dirs=CLASS_DIRS,
        train_count=TRAIN_COUNT,
        val_count=VAL_COUNT,
        test_count=TEST_COUNT,
    )

    # Materialize splits to a folder (symlink/copy)
    materialize_splits_to_folder(splits, args.splits_dir)

    all_classes = list(CLASS_DIRS.keys())
    print("\nAll classes:", all_classes)

    # We'll train binary classifiers for GENERATORS only; real_class is NEGATIVE
    generator_classes = [c for c in all_classes if c != REAL_CLASS_NAME]

    print("\nTraining on device:", args.device)
    print(f"Real (negative) class: {REAL_CLASS_NAME}")
    print("Will train per-generator binary classifiers (generator vs real):", generator_classes)

    model_stats = {}

    for gen_cls in generator_classes:
        stats = train_real_vs_generator_classifier(
            target_generator=gen_cls,
            splits=splits,
            real_class=REAL_CLASS_NAME,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.epochs,
        )
        model_stats[gen_cls] = stats

    # Save JSON with counts per model
    stats_path = os.path.join(args.output_dir, "real_vs_generator_model_image_counts.json")
    with open(stats_path, "w") as f:
        json.dump(model_stats, f, indent=2)
    print(f"\nSaved model image-count stats to: {stats_path}")


if __name__ == "__main__":
    main()
