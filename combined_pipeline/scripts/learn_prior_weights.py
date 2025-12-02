#!/usr/bin/env python3
"""Learn prior combination weights for SREC, RIGID and Aeroblade.

Reads per-image results JSONs produced earlier (they contain 'priors' and 'likelihoods'),
performs a grid search over weights on a validation split, and writes a combined
results JSON using the best weights.

Outputs:
 - results/BAYESIAN_RESULTS/bayesian_combined_results.json
 - prints best weights and accuracy

Usage:
  python learn_prior_weights.py --results-dir /mnt/ssd-data/vaidya/combined_pipeline/results/BAYESIAN_RESULTS \
      --step 0.05 --val-frac 0.2

"""
import json
from pathlib import Path
import numpy as np
import argparse
import random


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def extract_true_generator_from_path(p):
    # heuristics similar to analyzer
    p = str(p)
    gens = ['dalle2','dalle3','firefly','midjourneyV5','midjourneyV6','sdxl','stable_diffusion_1_5','coco']
    pl = p.lower()
    for g in gens:
        if g in pl:
            return g
    # fallback: try parent segment after dataset_512
    parts = Path(p).parts
    if 'dataset_512' in parts:
        try:
            idx = parts.index('dataset_512')
            cand = parts[idx+1]
            if cand in gens:
                return cand
        except Exception:
            pass
    return None


def combine_and_predict(likelihoods, p_srec, p_rigid, p_aero, w):
    # w = (w1,w2,w3) weights sum to 1
    eps = 1e-12
    gens = sorted(list(likelihoods.keys()))
    # compute combined log prior
    logp1 = np.log(np.array([p_srec.get(g, eps) + eps for g in gens]))
    logp2 = np.log(np.array([p_rigid.get(g, eps) + eps for g in gens]))
    logp3 = np.log(np.array([p_aero.get(g, eps) + eps for g in gens]))
    combined_log = w[0]*logp1 + w[1]*logp2 + w[2]*logp3
    loglik = np.log(np.array([likelihoods.get(g, eps) + eps for g in gens]))
    logpost = loglik + combined_log
    # normalize
    a = np.exp(logpost - np.max(logpost))
    post = a / np.sum(a)
    idx = int(np.argmax(post))
    return gens[idx], float(post[idx]), {g: float(p) for g,p in zip(gens, post)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Base results directory containing bayesian_{method}_results.json')
    parser.add_argument('--step', type=float, default=0.05, help='Grid step for weights (simplex)')
    parser.add_argument('--val-frac', type=float, default=0.2, help='Fraction to use as validation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    srec_file = results_dir / 'bayesian_srec_results.json'
    rigid_file = results_dir / 'bayesian_rigid_results.json'
    aero_file = results_dir / 'bayesian_aeroblade_results.json'

    assert srec_file.exists() and rigid_file.exists() and aero_file.exists(), 'Missing one of the method result files'

    srec = load_results(srec_file)
    rigid = load_results(rigid_file)
    aero = load_results(aero_file)

    # Build mapping by image_path
    data = {}
    for item in srec:
        data[item['image_path']] = {'likelihoods': item['likelihoods'], 'priors_srec': item['priors']}
    for item in rigid:
        if item['image_path'] not in data:
            continue
        data[item['image_path']]['priors_rigid'] = item['priors']
    for item in aero:
        if item['image_path'] not in data:
            continue
        data[item['image_path']]['priors_aero'] = item['priors']

    # Keep only entries that have all three priors
    images = []
    for k,v in data.items():
        if 'priors_srec' in v and 'priors_rigid' in v and 'priors_aero' in v:
            images.append(k)
    images.sort()

    # Build dataset list with true label
    rows = []
    for img in images:
        true = extract_true_generator_from_path(img)
        if true is None:
            continue
        rows.append((img, data[img]['likelihoods'], data[img]['priors_srec'], data[img]['priors_rigid'], data[img]['priors_aero'], true))

    random.seed(args.seed)
    np.random.seed(args.seed)
    random.shuffle(rows)

    n = len(rows)
    n_val = int(n * args.val_frac)
    val = rows[:n_val]
    train = rows[n_val:]

    print(f"Total images: {n}, train: {len(train)}, val: {len(val)}")

    # grid search over simplex weights
    step = args.step
    ws = []
    wvals = np.arange(0, 1+1e-8, step)
    for w1 in wvals:
        for w2 in wvals:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            if w3 < 0:
                w3 = 0.0
            ws.append((w1, w2, w3))

    best = None
    best_acc = -1.0
    print(f"Searching {len(ws)} weight combinations (step={step})...")
    for w in ws:
        correct = 0
        for img, lik, p1, p2, p3, true in val:
            pred, conf, _ = combine_and_predict(lik, p1, p2, p3, w)
            if pred == true:
                correct += 1
        acc = correct / len(val) if len(val) else 0.0
        if acc > best_acc:
            best_acc = acc
            best = w
    print(f"Best weights on val: w_srec={best[0]:.3f}, w_rigid={best[1]:.3f}, w_aero={best[2]:.3f} -> acc={best_acc:.4f}")

    # Apply best weights to full set and write combined results
    combined = []
    for img, lik, p1, p2, p3, true in rows:
        pred, conf, post = combine_and_predict(lik, p1, p2, p3, best)
        combined.append({'image_path': img, 'likelihoods': lik, 'priors_combined': {
            'srec_weight': best[0], 'rigid_weight': best[1], 'aeroblade_weight': best[2]
        }, 'posteriors': post, 'predicted_generator': pred, 'confidence': conf})

    out_file = results_dir / 'bayesian_combined_results.json'
    with open(out_file, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"Combined results written to: {out_file}")


if __name__ == '__main__':
    main()
