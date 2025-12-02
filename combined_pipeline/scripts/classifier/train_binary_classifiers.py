#!/usr/bin/env python3
"""
Train Binary (Real vs Generator) ResNet classifiers using CSV files
------------------------------------------------------------------
Improved version: supports resnet18/34/50, augmentation, dropout, weight decay,
label smoothing, backbone freezing, and LR scheduler.

Outputs (per generator):
  resnet<arch>_<generator>_vs_real.pth
  cm_test_<generator>.png
  binary_training_summary.json
"""

import os, csv, json, time, argparse, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
try:
    from PIL import UnidentifiedImageError
except ImportError:
    class UnidentifiedImageError(Exception):
        pass


# -----------------------
# Dataset
# -----------------------
class ListImageDataset(Dataset):
    def __init__(self, samples, img_size=224, augment=False):
        self.samples = samples
        if augment:
            self.tx = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            import numpy as np
            img = Image.fromarray(np.zeros((224,224,3),dtype=np.uint8))
        return self.tx(img), label


# -----------------------
# CSV helpers
# -----------------------
def read_csv_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            r["path"] = r["path"].strip().strip('"')
            rows.append(r)
    return rows

def pick_real(rows_level1, limit=None):
    out = [(r["path"], 0) for r in rows_level1 if r.get("label","") == "0"]
    if limit: out = out[:limit]
    return out

def pick_generator(rows_level2, gen_id, limit=None):
    out = []
    for r in rows_level2:
        try:
            label = int(r["label"])
            path = r["path"].strip('"')
        except Exception:
            continue
        if label == gen_id:
            out.append((path, 1))
    if limit: out = out[:limit]
    return out


# -----------------------
# Training helpers
# -----------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return (tot_loss / max(1, total)), (correct / max(1, total))

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        tot_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return (tot_loss / max(1, total)), (correct / max(1, total))

@torch.no_grad()
def test_model(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        preds = out.argmax(1)
        all_y += labels.cpu().tolist()
        all_p += preds.cpu().tolist()
    cm = confusion_matrix(all_y, all_p)
    acc = (cm[0,0] + cm[1,1]) / max(1, cm.sum())
    return acc, cm


# -----------------------
# Train + evaluate per generator
# -----------------------

def build_model(arch, pretrained, dropout):
    arch = arch.lower()
    # load backbone
    if arch == "resnet18":
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            model = models.resnet18(pretrained=pretrained)
    elif arch == "resnet34":
        try:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            model = models.resnet34(pretrained=pretrained)
    elif arch == "resnet50":
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        except Exception:
            model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity(),
        nn.Linear(in_feats, 2)
    )
    return model


def train_binary_for_generator(gen_name, gen_id, args, device):
    print(f"\n{'='*70}\nTraining binary: {gen_name} (label={gen_id}) vs Real\n{'='*70}")

    # Load CSVs
    l1_train = read_csv_rows(args.level1_train)
    l1_val   = read_csv_rows(args.level1_val)
    l1_test  = read_csv_rows(args.level1_test)
    l2_train = read_csv_rows(args.level2_train)
    l2_val   = read_csv_rows(args.level2_val)
    l2_test  = read_csv_rows(args.level2_test)

    # Select samples
    real_train = pick_real(l1_train, limit=args.limit_train)
    real_val   = pick_real(l1_val, limit=args.limit_val)
    real_test  = pick_real(l1_test, limit=args.limit_test)
    gen_train  = pick_generator(l2_train, gen_id, limit=args.limit_train)
    gen_val    = pick_generator(l2_val, gen_id, limit=args.limit_val)
    gen_test   = pick_generator(l2_test, gen_id, limit=args.limit_test)

    train_samples = real_train + gen_train
    val_samples   = real_val   + gen_val
    test_samples  = real_test  + gen_test

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    if len(train_samples) == 0 or len(val_samples) == 0:
        print(f"[WARN] No samples for {gen_name}. Train={len(train_samples)}, Val={len(val_samples)}")
        return None

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_ds = ListImageDataset(train_samples, img_size=args.img_size, augment=args.augment)
    val_ds   = ListImageDataset(val_samples,   img_size=args.img_size, augment=False)
    test_ds  = ListImageDataset(test_samples,  img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    model = build_model(args.arch, args.pretrained, args.dropout)
    model = model.to(device)

    # Optionally freeze backbone
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith('fc'):
                p.requires_grad = False

    # Criterion + optimizer
    # Criterion + optimizer
    try:
        # Newer PyTorch versions accept label_smoothing in CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0.0)
    except TypeError:
        # Fallback for older PyTorch where label_smoothing isn't supported
        if args.label_smoothing and args.label_smoothing > 0:
            print("[WARN] PyTorch build doesn't support label_smoothing in CrossEntropyLoss; continuing without label smoothing.")
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    best_val_acc = 0.0
    best_state = None
    patience_ctr = 0
    hist = []
    t0 = time.time()

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        hist.append({"epoch":epoch,"train_loss":tr_loss,"train_acc":tr_acc,"val_loss":va_loss,"val_acc":va_acc})
        print(f"Epoch {epoch:03d} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f} loss={va_loss:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            patience_ctr = 0
            print(f"  ✓ new best val acc {va_acc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  early stopping at epoch {epoch}")
                break

        if scheduler is not None:
            scheduler.step(va_loss)

    train_time = time.time() - t0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arch}_{gen_name}_vs_real.pth"

    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "model_config": {
            "num_classes": 2,
            "architecture": args.arch,
            "generator": gen_name,
            "real_source": "from level1 csv (label=0)"
        },
        "training_history": hist,
        "best_val_acc": best_val_acc,
        "training_time": train_time
    }, out_path)

    # ----- TEST EVALUATION -----
    model.load_state_dict(best_state or model.state_dict())
    test_acc, cm = test_model(model, test_loader, device)
    print(f"Test acc={test_acc:.3f}")

    # Save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real","Fake"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{gen_name} vs Real – Test Confusion")
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_test_{gen_name}.png", dpi=200)
    plt.close()

    print(f"Saved: {out_path}")
    return {
        "generator": gen_name,
        "model_path": str(out_path),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "training_time": train_time
    }


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level1-train", required=True)
    ap.add_argument("--level1-val",   required=True)
    ap.add_argument("--level1-test",  required=True)
    ap.add_argument("--level2-train", required=True)
    ap.add_argument("--level2-val",   required=True)
    ap.add_argument("--level2-test",  required=True)
    ap.add_argument("--output-dir",   required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--arch", type=str, default="resnet18", choices=["resnet18","resnet34","resnet50"])
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--scheduler", action="store_true")
    ap.add_argument("--limit-train", type=int, default=800)
    ap.add_argument("--limit-val", type=int, default=100)
    ap.add_argument("--limit-test", type=int, default=100)
    args = ap.parse_args()

    label_to_gen = {
        0: "dalle2",
        1: "dalle3",
        2: "firefly",
        3: "midjourneyV5",
        4: "midjourneyV6",
        5: "sdxl",
        6: "stable_diffusion_1_5"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Training binary classifiers for generators:", label_to_gen)

    all_results = {}
    for gen_id, gen_name in label_to_gen.items():
        try:
            res = train_binary_for_generator(gen_name, gen_id, args, device)
            if res: all_results[gen_name] = res
        except Exception as e:
            print(f"[ERROR] {gen_name}: {e}")

    summary_path = os.path.join(args.output_dir, "binary_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nTraining Summary")
    print("="*60)
    for g, r in all_results.items():
        print(f"{g:22s} val_acc={r['best_val_acc']:.3f} test_acc={r['test_acc']:.3f} "
              f"train={r['train_samples']} val={r['val_samples']} test={r['test_samples']} "
              f"time={r['training_time']:.1f}s")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
