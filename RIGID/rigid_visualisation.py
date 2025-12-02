#!/usr/bin/env python
"""
Simple visualization helper for RIGID results.

Inputs
------
1) A benchmark metrics CSV produced by rigid_cli.py --benchmark (AUC, FPR95, AP, mean_similarity, num_images)
2) (Optional) A directory of per-dataset similarity arrays saved with --save-sims-dir (e.g., <name>_sims.npy)

Outputs
-------
Saves PNGs into --outdir:
- bar_auc.png               : AUC per OOD dataset
- bar_ap.png                : AP per OOD dataset
- bar_mean_similarity.png   : Mean similarity for all datasets (incl. ID)
- roc_curves.png            : ROC curves for a few selected OOD datasets vs ID
- pr_curves.png             : Precision–Recall curves for the same selection

Examples
--------
python viz_rigid.py \
  --metrics results/cifar10_metrics.csv \
  --sims-dir results/cifar10_quick_sims \
  --id-name CIFAR10 \
  --roc-datasets CIFAR10-StyleGAN2-ada CIFAR10-BigGAN-Deep CIFAR10-iDDPM-DDIM \
  --outdir figures

If you omit --roc-datasets, the script will automatically pick up to 3 OOD datasets (best/worst/median AUC) for the curves.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

try:
    from sklearn import metrics
except Exception as e:
    metrics = None

plt.rcParams.update({
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.5,
    'savefig.bbox': 'tight',
})


def load_metrics_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect columns: dataset, metric, value
    if not set(['dataset', 'metric', 'value']).issubset(df.columns):
        raise ValueError('Metrics CSV must have columns: dataset, metric, value')
    return df


def infer_id_name(df: pd.DataFrame) -> Optional[str]:
    # ID (real) dataset generally has no AUC/AP entries, only mean_similarity / num_images.
    ood_with_auc = set(df[df['metric'] == 'AUC']['dataset'].unique())
    all_names = set(df['dataset'].unique())
    candidates = list(all_names - ood_with_auc)
    return candidates[0] if candidates else None


def bar_plot(df: pd.DataFrame, metric: str, title: str, out: Path, exclude: Optional[List[str]] = None):
    dd = df[df['metric'] == metric].copy()
    if exclude:
        dd = dd[~dd['dataset'].isin(exclude)]
    dd = dd.sort_values('value', ascending=(metric == 'FPR95'))
    plt.figure()
    plt.bar(dd['dataset'], dd['value'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def choose_roc_sets(df: pd.DataFrame, k: int = 3, id_name: Optional[str] = None) -> List[str]:
    # Pick best/worst/median by AUC
    auc = df[df['metric'] == 'AUC'][['dataset', 'value']].sort_values('value')
    if auc.empty:
        return []
    names = list(auc['dataset'].values)
    picks = []
    if len(names) >= 1:
        picks.append(names[0])           # worst
    if len(names) >= 2:
        picks.append(names[-1])          # best
    if len(names) >= 3:
        picks.append(names[len(names)//2]) # median
    return list(dict.fromkeys(picks))[:k]


def load_sims(sims_dir: Path, name: str) -> np.ndarray:
    path = sims_dir / f"{name}_sims.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing sims file: {path}")
    return np.load(path)


def plot_roc_pr(id_name: str, ood_names: List[str], sims_dir: Path, outdir: Path):
    if metrics is None:
        raise RuntimeError('scikit-learn not installed. `pip install scikit-learn`')
    id_conf = load_sims(sims_dir, id_name)
    # ROC
    plt.figure()
    for name in ood_names:
        ood_conf = load_sims(sims_dir, name)
        y = np.concatenate([np.ones_like(id_conf), np.zeros_like(ood_conf)])
        s = np.concatenate([id_conf, ood_conf])
        fpr, tpr, _ = metrics.roc_curve(y, s)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves (ID vs OOD)')
    plt.legend()
    plt.savefig(outdir / 'roc_curves.png')
    plt.close()

    # PR
    plt.figure()
    for name in ood_names:
        ood_conf = load_sims(sims_dir, name)
        y = np.concatenate([np.ones_like(id_conf), np.zeros_like(ood_conf)])
        s = np.concatenate([id_conf, ood_conf])
        prec, rec, _ = metrics.precision_recall_curve(y, s)
        ap = metrics.average_precision_score(y, s)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curves (ID vs OOD)')
    plt.legend()
    plt.savefig(outdir / 'pr_curves.png')
    plt.close()


def main():
    p = argparse.ArgumentParser(description='Visualize RIGID benchmark results')
    p.add_argument('--metrics', type=str, required=True, help='CSV from rigid_cli.py --benchmark')
    p.add_argument('--sims-dir', type=str, default=None, help='Directory with *_sims.npy files')
    p.add_argument('--id-name', type=str, default=None, help='Name of the ID (real) dataset')
    p.add_argument('--roc-datasets', nargs='*', default=None, help='Specific OOD names for ROC/PR (default: auto-pick 3)')
    p.add_argument('--outdir', type=str, default='figures', help='Where to save PNGs')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_metrics_table(Path(args.metrics))
    id_name = args.id_name or infer_id_name(df)

    # Bars: AUC, AP, mean similarity
    bar_plot(df, 'AUC', 'ROC-AUC per OOD dataset', outdir / 'bar_auc.png', exclude=[id_name] if id_name else None)
    bar_plot(df, 'AP', 'Average Precision per OOD dataset', outdir / 'bar_ap.png', exclude=[id_name] if id_name else None)
    bar_plot(df, 'mean_similarity', 'Mean Similarity (ID and OOD)', outdir / 'bar_mean_similarity.png')

    # Curves: use sims if available
    if args.sims_dir:
        sims_dir = Path(args.sims_dir)
        if not id_name:
            raise ValueError('Cannot infer ID dataset name; please provide --id-name')
        if args.roc_datasets:
            ood_names = args.roc_datasets
        else:
            ood_names = choose_roc_sets(df, k=3, id_name=id_name)
        plot_roc_pr(id_name, ood_names, sims_dir, outdir)
        print(f"Saved ROC/PR curves for: {ood_names}")

    print(f"Saved figures to: {outdir}")


if __name__ == '__main__':
    main()

