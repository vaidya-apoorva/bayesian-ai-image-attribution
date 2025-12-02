import os, csv, random, time, argparse, json
import numpy as np
from typing import Tuple, List, Optional, Dict
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from torchvision import models, transforms

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------
# utils
# -------------------
def safe_mkdir(d):
    os.makedirs(d, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path):
    torch.save(state, path)

# -------------------
# dataset stats helper (human-readable only)
# -------------------
def _count_images_per_class(csv_path: Optional[str]) -> Dict[str, int]:
    """Count samples per numeric class label from a CSV."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    counts: Dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = str(row["label"])
            counts[lbl] = counts.get(lbl, 0) + 1
    return counts

def _write_dataset_stats(out_dir: str,
                         train_csv: Optional[str],
                         val_csv: Optional[str],
                         test_csv: Optional[str],
                         label_names: Optional[List[str]]):
    """Write per-class counts (human-readable) into results/dataset_stats.json."""
    def readable(counts: Dict[str, int]) -> Dict[str, int]:
        if not counts:
            return {}
        if not label_names:
            return {k: v for k, v in counts.items()}
        result = {}
        for k, v in counts.items():
            idx = int(k)
            name = label_names[idx] if 0 <= idx < len(label_names) else f"class_{idx}"
            result[f"{name} ({idx})"] = v
        return result

    stats = {
        "train": readable(_count_images_per_class(train_csv)),
        "val":   readable(_count_images_per_class(val_csv)),
        "test":  readable(_count_images_per_class(test_csv)),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

# -------------------
# confusion matrix helpers (save PNG + PDF)
# -------------------
def _plot_confusion(cm, class_names, acc, out_base, normalize=False, auc=None, title_prefix=""):
    """Draw confusion matrix and save to <out_base>.png and <out_base>.pdf."""
    fig_w, fig_h = (5.2, 4.8) if len(class_names) <= 2 else (6.5, 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    values_fmt = ".2f" if normalize else "d"

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    im = disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=values_fmt)
    plt.xticks(rotation=45 if len(class_names) > 2 else 0,
               ha="right" if len(class_names) > 2 else "center",
               fontsize=9 if len(class_names) > 2 else 10)
    plt.yticks(fontsize=9 if len(class_names) > 2 else 10)

    title = f"{title_prefix} Confusion Matrix"
    if normalize: title += " (Normalized)"
    if auc is not None:
        plt.title(f"{title}\nacc={acc:.3f}, AUC={auc:.3f}", fontsize=11)
    else:
        plt.title(f"{title}\nacc={acc:.3f}", fontsize=11)

    im = ax.images[0]
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8 if len(class_names) > 2 else 9)

    plt.tight_layout(pad=1.2 if len(class_names) > 2 else 1.5)
    for ext in ("png", "pdf"):
        path = f"{out_base}.{ext}"
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)

def _save_confusions(loader, model, device, criterion, num_classes, out_dir, class_names, is_binary=False, title_prefix=""):
    """Evaluate model on loader, then save raw and normalized confusion matrices (PNG+PDF)."""
    loss, acc, auc, cm = evaluate(model, loader, device, criterion, num_classes, is_binary=is_binary)

    # raw
    _plot_confusion(
        cm, class_names, acc,
        os.path.join(out_dir, f"{title_prefix}_confusion_matrix_raw"),
        normalize=False, auc=auc if is_binary else None, title_prefix=title_prefix
    )

    # normalized (row-wise)
    cmf = cm.astype(float)
    rs = cmf.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    cmn = cmf / rs
    _plot_confusion(
        cmn, class_names, acc,
        os.path.join(out_dir, f"{title_prefix}_confusion_matrix_norm"),
        normalize=True, auc=auc if is_binary else None, title_prefix=title_prefix
    )

# -------------------
# dataset
# -------------------
class SimpleImageCSVDataset(Dataset):
    def __init__(self, csv_path: str, is_train=True, img_size=224,
                 normalize="imagenet", augment=True, robust_augs=False,
                 jpeg_p=0.3, jpeg_qmin=60, jpeg_qmax=95):
        self.samples = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['path'], int(row['label'])))

        mean, std = ([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225]) if normalize == "imagenet" else ([0.5]*3, [0.5]*3)

        def jpeg_reencode(img):
            import io
            q = random.randint(jpeg_qmin, jpeg_qmax)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
            buf.seek(0)
            return Image.open(buf).convert("RGB")

        t_aug = []
        if is_train and augment:
            t_aug += [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.RandomGrayscale(p=0.05),
            ]
            if robust_augs:
                t_aug += [
                    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
                    transforms.RandomApply([transforms.RandomPerspective(0.3)], p=0.05),
                    transforms.RandomApply([transforms.Lambda(jpeg_reencode)], p=jpeg_p),
                ]
        else:
            t_aug += [transforms.Resize(int(img_size*1.14)),
                      transforms.CenterCrop(img_size)]

        self.transform = transforms.Compose(t_aug + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.transform(img), label

# -------------------
# model
# -------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.resnet34(weights=weights)
        except Exception:
            m = models.resnet34(pretrained=pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.model = m
    def forward(self, x): return self.model(x)

# -------------------
# evaluation
# -------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes, is_binary=False):
    model.eval()
    all_y, all_p = [], []
    tot_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        tot_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_y += labels.cpu().tolist()
        all_p += out.softmax(1)[:,1].cpu().tolist() if is_binary else out.softmax(1).cpu().tolist()
    acc = correct / max(1, total)
    auc = None
    if is_binary:
        try:
            auc = roc_auc_score(all_y, all_p)
        except Exception:
            pass
        cm = confusion_matrix(all_y, (np.array(all_p)>0.5).astype(int), labels=[0,1])
    else:
        # ensure fixed NxN with the expected class count
        cm = confusion_matrix(all_y, np.argmax(all_p, 1), labels=list(range(num_classes)))
    return tot_loss/max(1,total), acc, auc, cm

# -------------------
# loader setup
# -------------------
def make_loaders_from_csvs(train_csv, val_csv, cfg, num_classes,
                           use_balanced_sampler=False, robust_augs=False, is_binary=False):
    train_ds = SimpleImageCSVDataset(train_csv, is_train=True, img_size=cfg.img_size, robust_augs=robust_augs)
    val_ds = SimpleImageCSVDataset(val_csv, is_train=False, img_size=cfg.img_size)
    counts = [0]*num_classes
    for _, y in train_ds: counts[y]+=1
    total = sum(counts)
    class_w = torch.tensor([(total/(c+1e-8))/num_classes for c in counts], dtype=torch.float)

    if use_balanced_sampler and is_binary:
        inv_freq = [total/(c+1e-8) for c in counts]
        label_weights = [inv_freq[y] for _,y in train_ds]
        sampler = WeightedRandomSampler(label_weights, len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                                  num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, class_w

def make_test_loader(test_csv, cfg):
    test_ds = SimpleImageCSVDataset(test_csv, is_train=False, img_size=cfg.img_size)
    return DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                      num_workers=cfg.num_workers, pin_memory=True)

# -------------------
# training config
# -------------------
class TrainConfig:
    def __init__(self, **kw): self.__dict__.update(kw)

# -------------------
# pipeline evaluation (Level-1 + Level-2 => 8 classes with "Real")
# -------------------
@torch.no_grad()
def evaluate_pipeline_and_save(out_root: str,
                               l1_ckpt_path: str,
                               l2_ckpt_path: str,
                               csv_level1_test: str,
                               csv_level2_test: str,
                               display_gen_names: List[str]):
    """
    Build an 8-class confusion matrix with labels:
    [<7 generator names>, "Real"], by routing images through Level-1 → Level-2.
    Saves PNG + PDF under: <out_root>/baseline/results/
    """
    baseline_dir = os.path.join(out_root, "baseline")
    baseline_results = os.path.join(baseline_dir, "results")
    baseline_runs = os.path.join(baseline_dir, "runs")
    safe_mkdir(baseline_results)
    safe_mkdir(baseline_runs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224
    tx = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def load_img(p):
        x = Image.open(p).convert("RGB")
        return tx(x).unsqueeze(0).to(device)

    # ---- Load Level-1 (binary) ----
    m1 = ResNetClassifier(2, pretrained=False).to(device)
    m1.load_state_dict(torch.load(l1_ckpt_path, map_location=device)["model"])
    m1.eval()

    # ---- Load Level-2 (auto-detect classes) ----
    m2_ckpt = torch.load(l2_ckpt_path, map_location=device)
    num_cls2 = m2_ckpt["model"]["model.fc.weight"].shape[0]
    m2 = ResNetClassifier(num_cls2, pretrained=False).to(device)
    m2.load_state_dict(m2_ckpt["model"])
    m2.eval()

    # Display labels: 7 gens + Real (order comes from display_gen_names then Real)
    # If l2 is 8-class and contained a "Coco"/"Real", we still display as "Real".
    gen_names = display_gen_names[:num_cls2] if num_cls2 <= len(display_gen_names) else display_gen_names
    DISPLAY = gen_names.copy()
    if "Real" not in DISPLAY:
        DISPLAY.append("Real")

    # Map L2 output ids to gen names
    ID2NAME = {i: gen_names[i] for i in range(min(num_cls2, len(gen_names)))}

    # ---- Collect test items ----
    real_paths = []
    with open(csv_level1_test, newline="") as f:
        for r in csv.DictReader(f):
            if r["label"] == "0":
                real_paths.append(r["path"])
    real_paths = real_paths[:100]  # cap for parity

    fake_items = []
    with open(csv_level2_test, newline="") as f:
        for r in csv.DictReader(f):
            fake_items.append((r["path"], int(r["label"])))

    name2idx = {n:i for i,n in enumerate(DISPLAY)}
    def idx_of(name): return name2idx[name]

    y_true, y_pred = [], []

    with torch.no_grad():
        # REAL → gt "Real"
        for p in real_paths:
            x = load_img(p)
            p_ai = torch.softmax(m1(x),1)[0,1].item()
            if p_ai < 0.5:
                pred_name = "Real"
            else:
                g = torch.softmax(m2(x),1)[0].argmax().item()
                pred_name = ID2NAME.get(g, "Real")
                if pred_name.lower() == "coco":
                    pred_name = "Real"
            y_true.append(idx_of("Real"))
            y_pred.append(idx_of(pred_name))

        # FAKE → gt = generator
        for p, gid in fake_items:
            x = load_img(p)
            p_ai = torch.softmax(m1(x),1)[0,1].item()
            if p_ai < 0.5:
                pred_name = "Real"
            else:
                g = torch.softmax(m2(x),1)[0].argmax().item()
                pred_name = ID2NAME.get(g, "Real")
                if pred_name.lower() == "coco":
                    pred_name = "Real"
            if gid >= len(gen_names):
                # skip labels not present in displayed gen names
                continue
            gt_name = ID2NAME.get(gid, "Real")
            if gt_name.lower() == "coco":
                gt_name = "Real"
            y_true.append(idx_of(gt_name))
            y_pred.append(idx_of(pred_name))

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    labels_idx = list(range(len(DISPLAY)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    def save_cm(cm, display_names, out_base, normalize=False):
        acc = (cm.trace() / max(1, cm.sum()))
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        fmt = ".2f" if normalize else "d"
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
        im = disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=fmt)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.title(f"Pipeline Confusion Matrix{' (Normalized)' if normalize else ''}\nacc={acc:.3f}", fontsize=11)
        im = ax.images[0]
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.tight_layout(pad=1.2)
        for ext in ("png","pdf"):
            plt.savefig(f"{out_base}.{ext}", dpi=300 if ext=="png" else None, bbox_inches="tight")
            print(f"Saved → {out_base}.{ext}")
        plt.close(fig)

    # Save raw + normalized to baseline/results
    out_base_raw  = os.path.join(baseline_results, "Pipeline_confusion_matrix_raw")
    out_base_norm = os.path.join(baseline_results, "Pipeline_confusion_matrix_norm")
    save_cm(cm, DISPLAY, out_base_raw)
    cmf = cm.astype(float); rs = cmf.sum(axis=1, keepdims=True); rs[rs==0]=1.0
    save_cm(cmf/rs, DISPLAY, out_base_norm, normalize=True)

# -------------------
# training
# -------------------
def train_with_splits(train_csv, val_csv, out_dir, num_classes, cfg,
                      pretrained=True, is_binary=False,
                      robust_augs=False, balanced_sampler=False,
                      test_csv=None, label_names: Optional[List[str]] = None):
    # structured output folders
    safe_mkdir(out_dir)
    results_dir = os.path.join(out_dir, "results")
    runs_dir = os.path.join(out_dir, "runs")
    safe_mkdir(results_dir)
    safe_mkdir(runs_dir)

    # save dataset stats (human-readable) in results/
    _write_dataset_stats(results_dir, train_csv, val_csv, test_csv, label_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    tr_loader, val_loader, class_w = make_loaders_from_csvs(
        train_csv, val_csv, cfg, num_classes,
        use_balanced_sampler=balanced_sampler,
        robust_augs=robust_augs, is_binary=is_binary)
    te_loader = make_test_loader(test_csv, cfg) if test_csv else None

    model = ResNetClassifier(num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                          weight_decay=cfg.weight_decay, nesterov=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    best_acc = -1
    history = []

    for epoch in range(1, cfg.epochs+1):
        model.train()
        tot_loss, correct, total = 0,0,0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1)==labels).sum().item()
            total += imgs.size(0)
        tr_loss, tr_acc = tot_loss/max(1,total), correct/max(1,total)
        val_loss, val_acc, val_auc, cm = evaluate(model, val_loader, device, criterion, num_classes, is_binary)
        sched.step()
        print(f"Epoch {epoch:03d} | train_acc={tr_acc:.3f} val_acc={val_acc:.3f} loss={val_loss:.3f}")
        print(cm)
        save_checkpoint({"model": model.state_dict()}, os.path.join(runs_dir,"last.pt"))
        if val_acc>best_acc:
            best_acc=val_acc
            save_checkpoint({"model": model.state_dict()}, os.path.join(runs_dir,"best.pt"))
        history.append({"epoch":epoch,"train_acc":tr_acc,"val_acc":val_acc,"val_auc":val_auc})
        with open(os.path.join(results_dir,"metrics.json"),"w") as f: json.dump(history,f,indent=2)
    print("Best val acc:",best_acc)

    # ---- Confusion matrix only for test set (best checkpoint), saved to results/ ----
    best_ckpt = os.path.join(runs_dir, "best.pt")
    if test_csv and os.path.exists(best_ckpt):
        print("Generating test confusion matrices...")
        best_model = ResNetClassifier(num_classes, pretrained=False).to(device)
        best_model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        best_model.eval()
        class_names = ["REAL", "AI"] if is_binary else (label_names or [f"class_{i}" for i in range(num_classes)])
        _save_confusions(te_loader, best_model, device, criterion, num_classes,
                         results_dir, class_names, is_binary=is_binary, title_prefix="Test")

# -------------------
# inference
# -------------------
@torch.no_grad()
def infer_two_level(img_path, ckpt1, ckpt2, class_names, img_size=224):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans=transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img=Image.open(img_path).convert("RGB")
    x=trans(img).unsqueeze(0).to(device)

    m1=ResNetClassifier(2,pretrained=False).to(device)
    m1.load_state_dict(torch.load(ckpt1,map_location=device)["model"])
    m1.eval()
    p_ai=torch.softmax(m1(x),1)[0,1].item()
    if p_ai<0.5:
        return {"decision":"REAL","real_prob":1-p_ai,"ai_prob":p_ai}
    m2=ResNetClassifier(len(class_names),pretrained=False).to(device)
    m2.load_state_dict(torch.load(ckpt2,map_location=device)["model"])
    m2.eval()
    p=torch.softmax(m2(x),1)[0].cpu().numpy()
    top=int(np.argmax(p))
    return {"decision":"AI","ai_prob":p_ai,"ai_class":class_names[top],
            "class_probs":{class_names[i]:float(p[i]) for i in range(len(class_names))}}

# -------------------
# main
# -------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=["train_level1","train_level2","infer"])
    p.add_argument("--train-csv",type=str)
    p.add_argument("--val-csv",type=str)
    p.add_argument("--test-csv",type=str)
    p.add_argument("--out",type=str,default="/mnt/ssd-data/vaidya/gi_conference_results/")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--img-size",type=int,default=224)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--no-amp",action="store_true")
    p.add_argument("--robust-augs",action="store_true")
    p.add_argument("--balanced-sampler",action="store_true")
    p.add_argument("--ckpt-level1",type=str)
    p.add_argument("--ckpt-level2",type=str)
    p.add_argument("--img",type=str)
    p.add_argument("--class-names",nargs="*",default=[
        "Dall-e2","Dall-e3","Firefly","MidjourneyV5","MidjourneyV6","SDXL","stable_diffusion_1-5"])
    # For pipeline 8-class display, pass your preferred generator names (7), "Real" is appended internally.

    args=p.parse_args()

    cfg=TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp)

    # ---- Level-1 (2-class) ----
    if args.mode=="train_level1":
        train_with_splits(args.train_csv,args.val_csv,args.out+"/level1_resnet34",2,cfg,
                          pretrained=True,is_binary=True,robust_augs=args.robust_augs,
                          balanced_sampler=args.balanced_sampler,test_csv=args.test_csv,
                          label_names=["REAL","AI"])

    # ---- Level-2 (7-class) + Pipeline (8-class) ----
    elif args.mode=="train_level2":
        # Train/eval 7-class model and save to level2_resnet34/{results,runs}
        train_with_splits(args.train_csv,args.val_csv,args.out+"/level2_resnet34",
                          len(args.class_names),cfg,pretrained=True,is_binary=False,
                          robust_augs=args.robust_augs,balanced_sampler=False,test_csv=args.test_csv,
                          label_names=args.class_names)

        # After Level-2 test plots, produce Pipeline 8-class (Real + 7 gens) using L1+L2
        # We require best.pt to exist for both levels.
        l1_best = os.path.join(args.out, "level1_resnet34", "runs", "best.pt")
        l2_best = os.path.join(args.out, "level2_resnet34", "runs", "best.pt")
        if os.path.exists(l1_best) and os.path.exists(l2_best) and args.test_csv:
            # For the pipeline we need both test CSVs: L1 test (has REAL) and L2 test (has gens)
            # Reuse the provided test_csv for level2; infer level1 test path from args if passed via --ckpt-level1 ?
            # Prefer explicit: use args.test_csv for L2; for L1, try to find from common naming, else fallback to same path.
            # Best: ask user to pass --train/val/test for both tasks. Here, we assume Level-1 test is alongside Level-2 test root.
            # If you want a different path, change here:
            # By default, assume level1_test.csv is in the same splits directory.
            l2_test_csv = args.test_csv
            l1_test_csv_guess = None
            splits_dir = os.path.dirname(args.test_csv or "")
            cand = os.path.join(splits_dir, "level1_test.csv")
            if os.path.exists(cand):
                l1_test_csv_guess = cand
            else:
                # fallback: use level1 path under the known dataset tree
                l1_test_csv_guess = "/mnt/hdd-data/vaidya/gi_conference_dataset/splits/level1_test.csv"

            if os.path.exists(l1_test_csv_guess):
                print("Generating pipeline 8-class confusion matrices (Real + 7 gens)...")
                evaluate_pipeline_and_save(
                    out_root=args.out,
                    l1_ckpt_path=l1_best,
                    l2_ckpt_path=l2_best,
                    csv_level1_test=l1_test_csv_guess,
                    csv_level2_test=l2_test_csv,
                    display_gen_names=args.class_names  # your 7 generator display names, "Real" gets appended
                )
            else:
                print("WARNING: Could not find Level-1 test CSV for pipeline evaluation; skipped 8-class plots.")

    elif args.mode=="infer":
        res=infer_two_level(args.img,args.ckpt_level1,args.ckpt_level2,args.class_names,args.img_size)
        print(res)
