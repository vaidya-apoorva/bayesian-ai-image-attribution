import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def infer_label(dataset_name):
    return 0 if dataset_name in ['coco', 'raise'] else 1

def load_data(input_dir):
    y_true, distances = [], []
    for fname in os.listdir(input_dir):
        if fname.endswith('_distances.csv'):
            dataset = fname.replace('_distances.csv', '')
            label = infer_label(dataset)
            df = pd.read_csv(os.path.join(input_dir, fname))
            y_true.extend([label] * len(df))
            distances.extend(df['distance'].tolist())
    return np.array(y_true), np.array(distances)

def plot_roc(y_true, distances, output_path=None):
    fpr, tpr, _ = roc_curve(y_true, distances)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AEROBLADE ROC Curve')
    plt.legend(loc='lower right')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()

def plot_pr(y_true, distances, output_path=None):
    precision, recall, _ = precision_recall_curve(y_true, distances)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AEROBLADE Precision–Recall Curve')
    plt.legend(loc='upper right')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot ROC and PR curves for AEROBLADE")
    parser.add_argument('--input_dir', required=True,
                        help='Folder containing *_distances.csv files')
    parser.add_argument('--roc_out', default=None,
                        help='Optional path to save the ROC plot (PNG)')
    parser.add_argument('--pr_out', default=None,
                        help='Optional path to save the PR plot (PNG)')
    args = parser.parse_args()

    y_true, distances = load_data(args.input_dir)
    plot_roc(y_true, distances, args.roc_out)
    plot_pr(y_true, distances, args.pr_out)
