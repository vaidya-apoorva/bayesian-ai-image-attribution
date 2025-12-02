#!/usr/bin/env python
"""
RIGID: Training-free, model-agnostic detector for AI-generated images
Standalone Python CLI (final version)

What this script does
---------------------
1) **Single-folder / single-image mode**: computes per-image similarity between an image and a tiny noise-perturbed version in a
   vision backbone's feature space (default: DINOv2 ViT-L/14 via `timm`). Optionally classifies with a threshold.
2) **Folder batch mode**: if `--input` is a directory that contains subfolders, it will process *each* subfolder and save
   one CSV per folder + a summary CSV.
3) **Benchmark mode**: `--benchmark` + `--datasets` (first is REAL/ID) runs a multi-dataset benchmark (like in the RIGID paper):
   prints mean similarities and computes **AUC**, **FPR@95**, **AP** per OOD set, and writes a tidy metrics CSV.
4) Auto-generates timestamped outputs when `--output` / `--metrics-out` are not provided.

Examples
--------
# A) Single image
python rigid_cli.py --input path/to/image.jpg

# B) One folder → CSV
python rigid_cli.py --input gen_images/ImageNet/ADM --output results/ADM_results.csv \
  --backbone vit_large_patch14_dinov2

# C) Batch all subfolders under a root (each subfolder saved to results/<root>-<subdir>_results.csv)
python rigid_cli.py --input gen_images/ImageNet --backbone vit_large_patch14_dinov2

# D) Full benchmark (first dataset is REAL/ID)
python rigid_cli.py \
  --benchmark gen_images/ImageNet \
  --datasets Imagenet Imagenet256-ADM Imagenet256-ADMG Imagenet256-LDM Imagenet256-DiT-XL-2 Imagenet256-BigGAN Imagenet256-GigaGAN Imagenet256-StyleGAN-XL Imagenet256-RQ-Transformer Imagenet256-Mask-GIT \
  --backbone vit_large_patch14_dinov2 \
  --limit-per-set 1024 \
  --metrics-out results/imagenet_metrics.csv

Dependencies
------------
- torch, torchvision, timm, pillow, numpy, pandas, tqdm, scikit-learn, opencv-python (optional)
Install:
  pip install torch torchvision timm pillow numpy pandas tqdm scikit-learn opencv-python

Notes
-----
- For CUDA GPUs with driver ~11.x, install PyTorch cu118 wheels: `--index-url https://download.pytorch.org/whl/cu118`.
- DINOv2 model names in `timm>=0.9.16`; in `timm>=1.0` use names like `vit_large_patch14_dinov2` (no suffix).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import timm
except ImportError as e:
    print("[ERROR] timm not installed. Please run: pip install timm", file=sys.stderr)
    raise

try:
    import pandas as pd
except Exception:
    pd = None

from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

# Optional metrics for benchmark mode
try:
    from sklearn import metrics
except Exception:
    metrics = None

# ---------------------------
# Utilities
# ---------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def discover_images(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and is_image(p):
            files.append(p)
    return sorted(files)


# ---------------------------
# Model & preprocessing
# ---------------------------

def _infer_input_size(model) -> int:
    # timm 0.x: model.default_cfg; timm 1.x: model.pretrained_cfg
    size = None
    if hasattr(model, 'pretrained_cfg') and isinstance(model.pretrained_cfg, dict):
        si = model.pretrained_cfg.get('input_size')
        if isinstance(si, (list, tuple)) and len(si) == 3:
            size = int(si[1])
    if size is None and hasattr(model, 'default_cfg') and isinstance(model.default_cfg, dict):
        si = model.default_cfg.get('input_size')
        if isinstance(si, (list, tuple)) and len(si) == 3:
            size = int(si[1])
    return int(size or 224)


def build_transform(img_size: int = 224):
    # Slight up-resize then center-crop to model input size
    resize_to = max(256, int(img_size * 1.15))
    return transforms.Compose([
        transforms.Resize(resize_to, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_backbone(backbone: str, device: str):
    model = timm.create_model(backbone, pretrained=True)
    model.eval().to(device)
    img_size = _infer_input_size(model)
    return model, img_size


def extract_vec(model, x: torch.Tensor) -> torch.Tensor:
    """Extract a single global embedding vector per image.
    Works across most timm backbones; normalizes output.
    """
    with torch.no_grad():
        feats = model.forward_features(x)
        vec = None
        if isinstance(feats, dict):
            for k in [
                'x_norm_clstoken', 'x_norm_clspool', 'pooled', 'last_hidden_state',
                'global_pool', 'feats', 'head', 'embedding'
            ]:
                if k in feats and feats[k] is not None:
                    vec = feats[k]
                    break
        if vec is None:
            if isinstance(feats, torch.Tensor) and feats.ndim == 2:
                vec = feats
            elif isinstance(feats, torch.Tensor) and feats.ndim == 4:
                vec = feats.mean(dim=(2, 3))
            elif hasattr(model, 'forward_head'):
                try:
                    vec = model.forward_head(feats, pre_logits=True)
                except Exception:
                    pass
        if vec is None:
            raise RuntimeError("Could not derive an embedding vector from model outputs.")
        return F.normalize(vec, dim=1)


# ---------------------------
# RIGID scorer
# ---------------------------

def add_noise(img_np: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img_np.astype(np.float32) / 255.0
    noisy = noisy + rng.normal(0.0, noise_std, size=noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def rigid_similarity(
    model, tfm, pil_img: Image.Image, device: str, noise_std: float = 0.02, seed: int = 0
) -> float:
    x0 = tfm(pil_img).unsqueeze(0).to(device)
    v0 = extract_vec(model, x0)

    arr = np.array(pil_img.convert('RGB'))
    arr_noisy = add_noise(arr, noise_std=noise_std, seed=seed)
    pil_noisy = Image.fromarray(arr_noisy)
    x1 = tfm(pil_noisy).unsqueeze(0).to(device)
    v1 = extract_vec(model, x1)

    cos = F.cosine_similarity(v0, v1, dim=1).item()
    return float(cos)


# ---------------------------
# Calibration (optional threshold)
# ---------------------------

def calibrate_threshold(
    model, tfm, real_folder: Path, device: str, noise_std: float, seed: int, quantile: float
) -> float:
    paths = discover_images(real_folder)
    if len(paths) == 0:
        raise ValueError(f"No images found under {real_folder}")
    sims = []
    for p in tqdm(paths, desc="Calibrating on REAL images"):
        try:
            img = Image.open(p).convert('RGB')
            sim = rigid_similarity(model, tfm, img, device, noise_std, seed)
            sims.append(sim)
        except Exception as e:
            print(f"[WARN] Failed {p}: {e}")
    if len(sims) == 0:
        raise RuntimeError("Calibration produced no similarities.")
    return float(np.quantile(np.array(sims), quantile))


# ---------------------------
# Main evaluation loop
# ---------------------------

def _auto_output_path(suffix: str, out_dir: Path = Path("results")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return out_dir / f"rigid_results_{ts}{suffix}"


def evaluate(
    input_path: Path,
    output_csv: Optional[Path],
    backbone: str,
    noise_std: float,
    seed: int,
    threshold: Optional[float],
    calibrate_real: Optional[Path],
    calib_quantile: float,
    save_interval: int,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device: {device}")

    model, img_size = load_backbone(backbone, device)
    print(f"[INFO] Backbone: {backbone}  (input {img_size}px)")
    tfm = build_transform(img_size)

    # If the input directory has subfolders, process each one and write per-folder CSVs
    if input_path.is_dir():
        subdirs = [p for p in sorted(input_path.iterdir()) if p.is_dir()]
        if subdirs:
            print(f"[INFO] Found {len(subdirs)} subfolders in {input_path}")
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            summary_rows = []
            for subdir in subdirs:
                print(f"[INFO] Processing folder: {subdir.name}")
                out_csv = results_dir / f"{input_path.name}-{subdir.name}_results.csv"
                _evaluate_single_folder(
                    folder=subdir,
                    out_csv=out_csv,
                    model=model,
                    tfm=tfm,
                    device=device,
                    noise_std=noise_std,
                    seed=seed,
                    threshold=threshold,
                    save_interval=save_interval,
                )
                # collect quick stats
                try:
                    if pd is not None and out_csv.exists():
                        df = pd.read_csv(out_csv)
                        sims = df["similarity"].dropna().values
                        mean_sim = float(np.mean(sims)) if len(sims) else float('nan')
                        summary_rows.append({
                            "folder": subdir.name,
                            "num_images": int(len(df)),
                            "mean_similarity": mean_sim,
                        })
                except Exception as e:
                    print(f"[WARN] Could not summarize {subdir.name}: {e}")
            if pd is not None and summary_rows:
                summary_path = results_dir / f"{input_path.name}_summary.csv"
                pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
                print(f"[INFO] Summary written to {summary_path}")
            return

    # Single file or a flat directory (no subfolders)
    _evaluate_single_folder(
        folder=input_path,
        out_csv=output_csv,
        model=model,
        tfm=tfm,
        device=device,
        noise_std=noise_std,
        seed=seed,
        threshold=threshold,
        save_interval=save_interval,
    )


def _evaluate_single_folder(
    folder: Path,
    out_csv: Optional[Path],
    model,
    tfm,
    device: str,
    noise_std: float,
    seed: int,
    threshold: Optional[float],
    save_interval: int,
):
    paths = discover_images(folder)
    if len(paths) == 0:
        raise ValueError(f"No input images found at {folder}")

    rows = []
    for i, p in enumerate(tqdm(paths, desc=f"Scoring [{folder}]") ):
        try:
            img = Image.open(p).convert('RGB')
            sim = rigid_similarity(model, tfm, img, device, noise_std, seed)
            pred = None
            if threshold is not None:
                pred = 'REAL' if sim >= threshold else 'AI'
            rows.append({
                'path': str(p),
                'similarity': sim,
                'prediction': pred,
            })
        except Exception as e:
            rows.append({'path': str(p), 'similarity': np.nan, 'prediction': None, 'error': str(e)})

        if out_csv and save_interval > 0 and (i + 1) % save_interval == 0:
            _save_rows(rows, out_csv)
            print(f"[INFO] Checkpoint saved: {out_csv} ({i+1} rows)")

    # Final save + summary
    if out_csv is None:
        out_csv = _auto_output_path(suffix=".csv")
        print(f"[INFO] No output file specified — saving to {out_csv}")
    _save_rows(rows, out_csv)

    sims = [r['similarity'] for r in rows if isinstance(r.get('similarity'), (int, float))]
    if sims:
        sims_np = np.array(sims, dtype=np.float32)
        print(f"[SUMMARY] Processed {len(sims_np)} images | mean {sims_np.mean():.4f} ± {sims_np.std():.4f}")
        if any(r.get('prediction') for r in rows):
            if pd is not None:
                df = pd.DataFrame(rows)
                print(df['prediction'].value_counts())


def _save_rows(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if pd is None:
        # Minimal TSV fallback
        with open(out_csv, 'w', encoding='utf-8') as f:
            header = ['path', 'similarity', 'prediction']
            f.write('\t'.join(header) + '\n')
            for r in rows:
                f.write('\t'.join(str(r.get(k, '')) for k in header) + '\n')
    else:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[INFO] Results written to {out_csv}")


# ---------------------------
# Benchmark mode (ID + OOD sets → AUC/FPR95/AP)
# ---------------------------

def _calc_auc_fpr95(id_conf: np.ndarray, ood_conf: np.ndarray):
    if metrics is None:
        raise RuntimeError("scikit-learn not installed. Please: pip install scikit-learn")
    all_conf = np.concatenate([id_conf, ood_conf])
    labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])
    fpr, tpr, _ = metrics.roc_curve(labels, all_conf)
    auroc = metrics.auc(fpr, tpr)
    idx = np.argmax(tpr >= 0.95) if np.any(tpr >= 0.95) else -1
    fpr95 = fpr[idx] if idx != -1 else fpr[-1]
    return float(auroc), float(fpr95)


def _calc_ap(id_conf: np.ndarray, ood_conf: np.ndarray):
    if metrics is None:
        raise RuntimeError("scikit-learn not installed. Please: pip install scikit-learn")
    all_conf = np.concatenate([id_conf, ood_conf])
    labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])
    return float(metrics.average_precision_score(labels, all_conf))


def run_benchmark(
    bench_root: Path,
    dataset_names: List[str],
    backbone: str,
    noise_std: float,
    seed: int,
    limit_per_set: int,
    metrics_out: Optional[Path],
    save_sims_dir: Optional[Path] = None,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device: {device}")
    model, img_size = load_backbone(backbone, device)
    print(f"[INFO] Backbone: {backbone}  (input {img_size}px)")
    tfm = build_transform(img_size)

    sims_per_set: List[np.ndarray] = []
    means, counts = [], []

    for name in dataset_names:
        folder = bench_root / name
        paths = [p for p in discover_images(folder)][:limit_per_set]
        if not paths:
            raise ValueError(f"No images in {folder}")
        sims = []
        for p in tqdm(paths, desc=f"{name}"):
            try:
                img = Image.open(p).convert("RGB")
                sim = rigid_similarity(model, tfm, img, device, noise_std, seed)
                sims.append(sim)
            except Exception as e:
                print(f"[WARN] {name}: failed {p}: {e}")
        sims = np.array(sims, dtype=np.float32)
        sims_per_set.append(sims)
        means.append(float(np.mean(sims)))
        counts.append(int(len(sims)))
        print(f"{name}, Image number: {len(sims)}, similarity is {np.mean(sims):.10f}")

        if save_sims_dir is not None:
            save_sims_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_sims_dir / f"{name}_sims.npy", sims)

    id_conf = sims_per_set[0]
    rows = []
    aucs, fprs, aps = [], [], []

    print("Detection Results AUC:")
    for name, ood in zip(dataset_names[1:], sims_per_set[1:]):
        auc, fpr95 = _calc_auc_fpr95(id_conf, ood)
        aucs.append(auc); fprs.append(fpr95)
        print(f"Dataset: {name:<25} | AUC: {auc:.4f} | FPR95: {fpr95:.4f}")
        rows.append({"dataset": name, "metric": "AUC", "value": auc})
        rows.append({"dataset": name, "metric": "FPR95", "value": fpr95})
    print("-" * 60)
    print(f"Average AUC: {np.mean(aucs):.4f} | Average FPR95: {np.mean(fprs):.4f}")

    print("Detection Results AP:")
    for name, ood in zip(dataset_names[1:], sims_per_set[1:]):
        ap = _calc_ap(id_conf, ood)
        aps.append(ap)
        print(f"Dataset: {name:<25} | AP: {ap:.4f}")
        rows.append({"dataset": name, "metric": "AP", "value": ap})
    print("-" * 40)
    print(f"Average AP: {np.mean(aps):.4f}")

    if metrics_out is None:
        metrics_out = _auto_output_path(suffix=f"_{bench_root.name}_metrics.csv", out_dir=Path("results"))
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    # Add mean similarity & counts, including ID set
    for name, m, c in zip(dataset_names, means, counts):
        rows.append({"dataset": name, "metric": "mean_similarity", "value": m})
        rows.append({"dataset": name, "metric": "num_images", "value": c})

    if pd is None:
        # write TSV fallback
        with open(metrics_out, 'w', encoding='utf-8') as f:
            f.write("dataset\tmetric\tvalue\n")
            for r in rows:
                f.write(f"{r['dataset']}\t{r['metric']}\t{r['value']}\n")
    else:
        pd.DataFrame(rows).to_csv(metrics_out, index=False)
    print(f"[INFO] Metrics saved to {metrics_out}")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RIGID training-free detector: measure robustness to tiny noise in feature space.")

    # Regular evaluation
    parser.add_argument('--input', type=str, default=None,
                        help='Path to an image file or a directory (recursively scanned). If a directory with subfolders, each subfolder will be processed to its own CSV.')
    parser.add_argument('--output', type=str, default=None,
                        help='CSV output file. If omitted, a timestamped file is created under results/.')

    # Model & scoring
    parser.add_argument('--backbone', type=str, default='vit_large_patch14_dinov2',
                        help='timm model name (e.g., vit_large_patch14_dinov2, vit_base_patch14_dinov2, vit_small_patch14_dinov2).')
    parser.add_argument('--noise-std', type=float, default=0.02,
                        help='Std of Gaussian pixel noise (0..1 scale, default 0.02).')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for noise.')

    # Optional classification & calibration
    parser.add_argument('--threshold', type=float, default=None,
                        help='If set, classify as REAL if similarity >= threshold else AI.')
    parser.add_argument('--calibrate-real', type=str, default=None,
                        help='Folder with REAL images to estimate threshold from.')
    parser.add_argument('--calib-quantile', type=float, default=0.05,
                        help='Quantile of REAL similarities used as threshold (default 0.05).')

    parser.add_argument('--save-interval', type=int, default=0,
                        help='Rows between interim CSV checkpoints (0 = none).')

    # Benchmark mode
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Root folder that contains multiple dataset subfolders. First in --datasets is treated as ID (real).')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Dataset subfolder names under --benchmark; first is ID.')
    parser.add_argument('--limit-per-set', type=int, default=1024,
                        help='Max images to read per dataset (default 1024).')
    parser.add_argument('--metrics-out', type=str, default=None,
                        help='CSV file to save benchmark metrics (AUC/FPR95/AP). Defaults to results/<auto>.csv')
    parser.add_argument('--save-sims-dir', type=str, default=None,
                        help='If set, saves per-dataset raw similarities as .npy files here (benchmark mode only).')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Benchmark mode takes precedence
    if args.benchmark and args.datasets:
        run_benchmark(
            bench_root=Path(args.benchmark),
            dataset_names=args.datasets,
            backbone=args.backbone,
            noise_std=args.noise_std,
            seed=args.seed,
            limit_per_set=args.limit_per_set,
            metrics_out=Path(args.metrics_out) if args.metrics_out else None,
            save_sims_dir=Path(args.save_sims_dir) if args.save_sims_dir else None,
        )
        sys.exit(0)

    # Regular evaluation
    if not args.input:
        print("[ERROR] Please provide --input (file or folder), or use --benchmark with --datasets.")
        sys.exit(2)

    try:
        evaluate(
            input_path=Path(args.input),
            output_csv=Path(args.output) if args.output else None,
            backbone=args.backbone,
            noise_std=args.noise_std,
            seed=args.seed,
            threshold=args.threshold,
            calibrate_real=Path(args.calibrate_real) if args.calibrate_real else None,
            calib_quantile=args.calib_quantile,
            save_interval=args.save_interval,
        )
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

