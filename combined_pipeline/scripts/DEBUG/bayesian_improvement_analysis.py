#!/usr/bin/env python3
"""
Bayesian Attribution Improvement Analysis and Fixes

This script implements several improvement strategies based on the performance analysis:
1. Generator-specific method selection
2. Confidence calibration
3. Ensemble methods
4. Diagnostic tools for underperforming cases
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt


class BayesianAttributionImprover:
    """Implements improvement strategies for Bayesian attribution."""

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.method_results = {}
        self.load_all_results()

    def load_all_results(self):
        """Load results from all methods."""
        methods = ['aeroblade', 'srec', 'rigid']
        for method in methods:
            result_file = self.results_dir / f"bayesian_{method}_results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    self.method_results[method] = json.load(f)
                print(f"Loaded {len(self.method_results[method])} results for {method}")

    def analyze_per_generator_performance(self):
        """Analyze performance per generator for each method."""
        generators = ['coco', 'dall-e2', 'dall-e3', 'firefly', 'midjourneyV5', 'midjourneyV6', 'sdxl',
                      'stable_diffusion_1-5']

        performance_matrix = {}

        for method, results in self.method_results.items():
            performance_matrix[method] = {}

            for gen in generators:
                # Filter results for this generator
                gen_results = [r for r in results if self._extract_generator_from_path(r['image_path']) == gen]

                if gen_results:
                    # Calculate accuracy
                    correct = sum(1 for r in gen_results if r['predicted_generator'] == gen)
                    accuracy = correct / len(gen_results)

                    # Calculate confidence stats
                    confidences = [r['confidence'] for r in gen_results]
                    mean_conf = np.mean(confidences)

                    performance_matrix[method][gen] = {
                        'accuracy': accuracy,
                        'mean_confidence': mean_conf,
                        'count': len(gen_results),
                        'correct': correct
                    }

        return performance_matrix

    def _extract_generator_from_path(self, image_path):
        """Extract generator name from image path."""
        path_lower = image_path.lower()
        generators = ['coco', 'dall-e2', 'dall-e3', 'firefly', 'midjourneyv5', 'midjourneyv6', 'sdxl',
                      'stable_diffusion_1-5']

        for gen in generators:
            if gen.replace('-', '').replace('_', '') in path_lower.replace('-', '').replace('_', ''):
                return gen.replace('midjourneyv5', 'midjourneyV5').replace('midjourneyv6', 'midjourneyV6')
        return 'unknown'

    def create_optimal_ensemble(self):
        """Create ensemble based on best method for each generator."""
        performance = self.analyze_per_generator_performance()

        # Find best method for each generator
        best_methods = {}
        for gen in performance[list(performance.keys())[0]].keys():
            best_acc = 0
            best_method = None

            for method in performance:
                if gen in performance[method]:
                    acc = performance[method][gen]['accuracy']
                    if acc > best_acc:
                        best_acc = acc
                        best_method = method

            best_methods[gen] = {
                'method': best_method,
                'accuracy': best_acc
            }

        return best_methods

    def diagnose_failures(self, method='aeroblade', top_n=10):
        """Diagnose worst performing cases for a method."""
        if method not in self.method_results:
            print(f"No results found for method: {method}")
            return

        results = self.method_results[method]

        # Find misclassified cases
        misclassified = []
        for result in results:
            true_gen = self._extract_generator_from_path(result['image_path'])
            pred_gen = result['predicted_generator']

            if true_gen != pred_gen:
                misclassified.append({
                    'image_path': result['image_path'],
                    'true_generator': true_gen,
                    'predicted_generator': pred_gen,
                    'confidence': result['confidence'],
                    'posteriors': result.get('posteriors', {})
                })

        # Sort by confidence (lowest first - most uncertain failures)
        misclassified.sort(key=lambda x: x['confidence'])

        print(f"\\n{method.upper()} - Top {top_n} Most Uncertain Misclassifications:")
        print("=" * 80)

        for i, case in enumerate(misclassified[:top_n]):
            print(
                f"{i + 1:2d}. True: {case['true_generator']:15s} | Pred: {case['predicted_generator']:15s} | Conf: {case['confidence']:.3f}")
            print(f"    Image: {case['image_path'].split('/')[-1]}")

            # Show top 3 posteriors
            if case['posteriors']:
                top_post = sorted(case['posteriors'].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"    Top posteriors: {top_post}")
            print()

    def generate_improvement_report(self):
        """Generate comprehensive improvement report."""
        print("\\n" + "=" * 80)
        print("BAYESIAN ATTRIBUTION IMPROVEMENT ANALYSIS")
        print("=" * 80)

        # Performance matrix
        performance = self.analyze_per_generator_performance()

        print("\\n1. PERFORMANCE MATRIX:")
        print("-" * 40)
        print(f"{'Generator':<20} {'Aeroblade':<10} {'SReC':<10} {'RIGID':<10} {'Best Method'}")
        print("-" * 70)

        best_methods = self.create_optimal_ensemble()

        for gen in sorted(performance[list(performance.keys())[0]].keys()):
            row = f"{gen:<20}"
            for method in ['aeroblade', 'srec', 'rigid']:
                if gen in performance[method]:
                    acc = performance[method][gen]['accuracy']
                    row += f"{acc:>8.1%}  "
                else:
                    row += f"{'N/A':>8s}  "

            best_info = best_methods.get(gen, {})
            row += f"{best_info.get('method', 'N/A')} ({best_info.get('accuracy', 0):.1%})"
            print(row)

        # Improvement recommendations
        print("\\n2. CRITICAL IMPROVEMENT AREAS:")
        print("-" * 40)

        critical_issues = []
        for gen, methods in performance['aeroblade'].items():
            for method in performance:
                if gen in performance[method]:
                    acc = performance[method][gen]['accuracy']
                    if acc < 0.6:  # Less than 60% accuracy
                        critical_issues.append(f"{method.upper()} + {gen}: {acc:.1%}")

        if critical_issues:
            for issue in critical_issues:
                print(f"  ❌ {issue}")
        else:
            print("  ✅ No critical accuracy issues found")

        # Ensemble potential
        print("\\n3. ENSEMBLE IMPROVEMENT POTENTIAL:")
        print("-" * 40)

        # Calculate potential ensemble accuracy
        total_best_acc = sum(best_methods[gen]['accuracy'] for gen in best_methods)
        avg_best_acc = total_best_acc / len(best_methods)

        # Current best single method
        current_best = max(
            (method, np.mean([performance[method][gen]['accuracy'] for gen in performance[method]]))
            for method in performance
        )

        improvement = avg_best_acc - current_best[1]
        print(f"  Current best method: {current_best[0].upper()} ({current_best[1]:.1%})")
        print(f"  Optimal ensemble potential: {avg_best_acc:.1%}")
        print(f"  Potential improvement: +{improvement:.1%}")

        return {
            'performance_matrix': performance,
            'best_methods': best_methods,
            'critical_issues': critical_issues,
            'ensemble_improvement': improvement
        }


def main():
    """Main execution function."""
    results_dir = "/mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/bayesian_pipeline_results"

    improver = BayesianAttributionImprover(results_dir)

    # Generate comprehensive report
    report = improver.generate_improvement_report()

    # Diagnose specific failures
    for method in ['aeroblade', 'srec', 'rigid']:
        improver.diagnose_failures(method, top_n=5)


if __name__ == "__main__":
    main()