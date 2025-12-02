"""
Analyze Bayesian Attribution Performance
Analyzes how well the Bayesian approach (prior + likelihood) performs for generator attribution
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def extract_true_generator(image_path):
    """Extract true generator from filename"""
    path = Path(image_path)
    filename = path.name

    generators = ['dalle2', 'dalle3', 'firefly', 'midjourneyV5', 'midjourneyV6',
                  'sdxl', 'stable_diffusion_1_5', 'coco']

    for gen in generators:
        if gen in filename:
            return gen

    return None

def load_results(json_path):
    """Load and parse results"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = []
    for item in data:
        true_gen = extract_true_generator(item['image_path'])
        if true_gen:
            results.append({
                'image_path': item['image_path'],
                'true_generator': true_gen,
                'predicted_generator': item['predicted_generator'],
                'confidence': item['confidence'],
                'posteriors': item['posteriors'],
                'correct': true_gen == item['predicted_generator']
            })

    return results

def analyze_performance(results):
    """Analyze overall performance"""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100

    confidences = [r['confidence'] for r in results]
    correct_confidences = [r['confidence'] for r in results if r['correct']]
    wrong_confidences = [r['confidence'] for r in results if not r['correct']]

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'mean_confidence': np.mean(confidences),
        'median_confidence': np.median(confidences),
        'mean_confidence_correct': np.mean(correct_confidences) if correct_confidences else 0,
        'mean_confidence_wrong': np.mean(wrong_confidences) if wrong_confidences else 0
    }

def per_generator_analysis(results):
    """Analyze performance per generator"""
    generator_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'confidences': []})

    for r in results:
        gen = r['true_generator']
        generator_stats[gen]['total'] += 1
        if r['correct']:
            generator_stats[gen]['correct'] += 1
        generator_stats[gen]['confidences'].append(r['confidence'])

    # Calculate accuracy and mean confidence per generator
    per_gen_results = {}
    for gen, stats in generator_stats.items():
        per_gen_results[gen] = {
            'count': stats['total'],
            'accuracy': stats['correct'] / stats['total'] * 100,
            'mean_confidence': np.mean(stats['confidences']),
            'median_confidence': np.median(stats['confidences'])
        }

    return per_gen_results

def create_confusion_matrix(results):
    """Create confusion matrix"""
    generators = sorted(list(set([r['true_generator'] for r in results])))
    n = len(generators)
    confusion = np.zeros((n, n))

    gen_to_idx = {gen: i for i, gen in enumerate(generators)}

    for r in results:
        true_idx = gen_to_idx[r['true_generator']]
        pred_idx = gen_to_idx[r['predicted_generator']]
        confusion[true_idx, pred_idx] += 1

    return confusion, generators

def plot_accuracy_by_generator(per_gen_results, output_dir):
    """Plot accuracy by generator"""
    generators = sorted(per_gen_results.keys())
    accuracies = [per_gen_results[g]['accuracy'] for g in generators]
    counts = [per_gen_results[g]['count'] for g in generators]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(generators, accuracies, color='steelblue', alpha=0.8)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Generator', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Bayesian Attribution Accuracy by Generator', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_generator.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_by_generator.pdf', bbox_inches='tight')
    plt.close()

def plot_confidence_by_generator(per_gen_results, output_dir):
    """Plot confidence by generator"""
    generators = sorted(per_gen_results.keys())
    confidences = [per_gen_results[g]['mean_confidence'] for g in generators]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(generators, confidences, color='coral', alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Generator', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Mean Prediction Confidence by Generator', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_by_generator.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'confidence_by_generator.pdf', bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(confusion, generators, output_dir):
    """Plot confusion matrix"""
    # Normalize by row (true labels)
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(confusion, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=generators, yticklabels=generators,
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_xlabel('Predicted Generator', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Generator', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)

    # Normalized
    sns.heatmap(confusion_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=generators, yticklabels=generators,
                cbar_kws={'label': 'Proportion'}, ax=ax2, vmin=0, vmax=1)
    ax2.set_xlabel('Predicted Generator', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Generator', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized by True Label)', fontsize=14, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'confusion_matrix.pdf', bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(results, output_dir):
    """Plot confidence distribution for correct vs incorrect predictions"""
    correct_confidences = [r['confidence'] for r in results if r['correct']]
    wrong_confidences = [r['confidence'] for r in results if not r['correct']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    ax1.hist(wrong_confidences, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    ax1.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Box plot
    box_data = [correct_confidences, wrong_confidences]
    bp = ax2.boxplot(box_data, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.7)
    ax2.set_ylabel('Confidence', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'confidence_distribution.pdf', bbox_inches='tight')
    plt.close()

def plot_accuracy_confidence_scatter(per_gen_results, output_dir):
    """Plot accuracy vs confidence scatter"""
    generators = sorted(per_gen_results.keys())
    accuracies = [per_gen_results[g]['accuracy'] for g in generators]
    confidences = [per_gen_results[g]['mean_confidence'] for g in generators]
    counts = [per_gen_results[g]['count'] for g in generators]

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(confidences, accuracies, s=[c*2 for c in counts],
                        alpha=0.6, c=range(len(generators)), cmap='tab10')

    # Add generator labels
    for i, gen in enumerate(generators):
        ax.annotate(gen, (confidences[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    ax.set_xlabel('Mean Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Confidence by Generator', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add diagonal reference line
    ax.plot([0, 1], [0, 100], 'k--', alpha=0.3, label='Perfect calibration')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_confidence.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_vs_confidence.pdf', bbox_inches='tight')
    plt.close()

def save_summary_report(overall_stats, per_gen_results, output_dir):
    """Save detailed summary report"""
    report_path = output_dir / 'performance_summary.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BAYESIAN ATTRIBUTION PERFORMANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write("OVERALL PERFORMANCE:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total images analyzed: {overall_stats['total']}\n")
        f.write(f"Correct predictions: {overall_stats['correct']}\n")
        f.write(f"Overall accuracy: {overall_stats['accuracy']:.2f}%\n")
        f.write(f"Mean confidence: {overall_stats['mean_confidence']:.4f}\n")
        f.write(f"Median confidence: {overall_stats['median_confidence']:.4f}\n")
        f.write(f"Mean confidence (correct): {overall_stats['mean_confidence_correct']:.4f}\n")
        f.write(f"Mean confidence (incorrect): {overall_stats['mean_confidence_wrong']:.4f}\n\n")

        f.write("PER-GENERATOR PERFORMANCE:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Generator':<25} {'Count':<8} {'Accuracy':<12} {'Mean Conf':<12}\n")
        f.write("-"*80 + "\n")

        for gen in sorted(per_gen_results.keys()):
            stats = per_gen_results[gen]
            f.write(f"{gen:<25} {stats['count']:<8} {stats['accuracy']:>10.2f}% {stats['mean_confidence']:>11.4f}\n")

        f.write("\n" + "="*80 + "\n")

def main():
    # Paths
    results_path = Path('/Users/apoorvavaidya/Desktop/Thesis/GITLAB/mt_apoorva_zero_shot_detector/combined_pipeline/results/BAYESIAN_RESULTS/bayesian_attribution_200_images_results.json')
    output_dir = Path('/Users/apoorvavaidya/Desktop/Thesis/GITLAB/mt_apoorva_zero_shot_detector/combined_pipeline/results/bayesian_performance_analysis')
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_results(results_path)
    print(f"Loaded {len(results)} results")

    print("\nAnalyzing overall performance...")
    overall_stats = analyze_performance(results)

    print("\nAnalyzing per-generator performance...")
    per_gen_results = per_generator_analysis(results)

    print("\nCreating confusion matrix...")
    confusion, generators = create_confusion_matrix(results)

    print("\nGenerating visualizations...")
    plot_accuracy_by_generator(per_gen_results, output_dir)
    print("  ✓ Accuracy by generator")

    plot_confidence_by_generator(per_gen_results, output_dir)
    print("  ✓ Confidence by generator")

    plot_confusion_matrix(confusion, generators, output_dir)
    print("  ✓ Confusion matrix")

    plot_confidence_distribution(results, output_dir)
    print("  ✓ Confidence distribution")

    plot_accuracy_confidence_scatter(per_gen_results, output_dir)
    print("  ✓ Accuracy vs confidence scatter")

    print("\nSaving summary report...")
    save_summary_report(overall_stats, per_gen_results, output_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOverall Accuracy: {overall_stats['accuracy']:.2f}%")
    print(f"Mean Confidence: {overall_stats['mean_confidence']:.4f}")
    print(f"\nAll outputs saved to: {output_dir}")

if __name__ == '__main__':
    main()
