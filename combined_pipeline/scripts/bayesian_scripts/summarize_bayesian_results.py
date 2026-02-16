#!/usr/bin/env python3
"""
Summarize Bayesian attribution JSON results.
Outputs:
 - bayesian_summary.csv  (per-class metrics)
 - confusion_matrix.csv  (rows=true, cols=pred)
 - per_image_summary.csv (image_path, true_label, pred_label, confidence)
Prints overall accuracy and per-class precision/recall.

Usage: python summarize_bayesian_results.py /path/to/bayesian_results.json
"""
import sys
import json
import csv
import re
from collections import Counter, defaultdict

KNOWN_LABELS = [
    'dalle2', 'dalle3', 'firefly', 'midjourneyV5', 'midjourneyV6', 'sdxl', 'stable_diffusion_1_5', 'coco'
]
# create a regex to find any known label in filename (case-insensitive)
LABEL_RE = re.compile(r'(' + '|'.join(re.escape(l) for l in KNOWN_LABELS) + r')', re.IGNORECASE)


def infer_true_label(path):
    # try to find known label token in path
    m = LABEL_RE.search(path)
    if m:
        lab = m.group(1)
        # normalize stable_diffusion variations
        # match case-insensitively to known labels list
        for k in KNOWN_LABELS:
            if lab.lower() == k.lower():
                return k
    # fallback: unknown
    return 'unknown'


def load_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def summarize(data, out_prefix):
    # counters and confusion matrix
    classes = KNOWN_LABELS + ['unknown']
    conf = {t: Counter() for t in classes}  # conf[true][pred] = count
    support = Counter()
    pred_counts = Counter()
    correct = 0
    total = 0
    confidences_by_class = defaultdict(list)  # true_label -> list of confidence for correct predictions

    per_image_rows = []

    for item in data:
        img = item.get('image_path')
        pred = item.get('predicted_generator') or 'unknown'
        conf_score = item.get('confidence', 0.0)
        true = infer_true_label(img)
        conf[true][pred] += 1
        support[true] += 1
        pred_counts[pred] += 1
        total += 1
        if pred == true:
            correct += 1
            confidences_by_class[true].append(conf_score)
        per_image_rows.append((img, true, pred, conf_score))

    accuracy = correct / total if total else 0.0

    # per-class precision, recall, f1
    per_class = []
    for cls in classes:
        tp = conf[cls][cls]
        fn = sum(conf[cls].values()) - tp
        fp = sum(conf[t][cls] for t in classes) - tp
        support_cls = support[cls]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_conf_correct = (sum(confidences_by_class[cls]) / len(confidences_by_class[cls])) if confidences_by_class[cls] else 0.0
        per_class.append({'class': cls, 'support': support_cls, 'precision': precision, 'recall': recall, 'f1': f1, 'mean_conf_correct': mean_conf_correct})

    # write per-image CSV
    per_image_csv = out_prefix + '_per_image_summary.csv'
    with open(per_image_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'true_label', 'predicted_label', 'confidence'])
        for r in per_image_rows:
            writer.writerow(r)

    # write per-class metrics
    per_class_csv = out_prefix + '_per_class_metrics.csv'
    with open(per_class_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class','support','precision','recall','f1','mean_conf_correct'])
        for r in per_class:
            writer.writerow([r['class'], r['support'], f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['mean_conf_correct']:.4f}"])

    # write confusion matrix CSV (rows=true, cols=pred)
    conf_csv = out_prefix + '_confusion_matrix.csv'
    with open(conf_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['true\\pred'] + classes
        writer.writerow(header)
        for t in classes:
            row = [t] + [conf[t].get(p, 0) for p in classes]
            writer.writerow(row)

    # print summary
    print('RESULTS SUMMARY')
    print('Total images:', total)
    print('Overall accuracy: {:.4f} ({}/{})'.format(accuracy, correct, total))
    print('\nPer-class metrics:')
    for r in per_class:
        print("{class}: support={support}, precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}, mean_conf_correct={mean_conf_correct:.3f}".format(**r))

    print('\nWrote:')
    print(' -', per_image_csv)
    print(' -', per_class_csv)
    print(' -', conf_csv)

    return {'total': total, 'accuracy': accuracy, 'per_class': per_class, 'confusion_csv': conf_csv, 'per_image_csv': per_image_csv, 'per_class_csv': per_class_csv}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python summarize_bayesian_results.py /path/to/bayesian_results.json [out_prefix]')
        sys.exit(1)
    path = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else '/mnt/ssd-data/vaidya/combined_pipeline/results/BAYESIAN_RESULTS/bayesian_attribution_20_images_summary'
    data = load_results(path)
    summarize(data, out_prefix)
