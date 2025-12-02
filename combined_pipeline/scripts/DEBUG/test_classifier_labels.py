#!/usr/bin/env python3
"""
Test script to determine correct class label mapping for binary classifiers.
Tests individual classifiers on known real vs AI samples.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import json

# Configuration
models_dir = '/mnt/ssd-data/vaidya/combined_pipeline/models/model_with_gi_data'
test_data_dir = '/mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_classifier(generator):
    """Load a single binary classifier."""
    # Try ResNet50 first, then ResNet18
    model_path = Path(models_dir) / f"resnet50_{generator}_vs_real.pth"
    architecture = 'resnet50'

    if not model_path.exists():
        model_path = Path(models_dir) / f"resnet18_{generator}_vs_real.pth"
        architecture = 'resnet18'

    if not model_path.exists():
        print(f"Model not found for {generator}")
        return None, None

    # Create and load model
    if architecture == 'resnet50':
        model = models.resnet50(pretrained=False)
    else:
        model = models.resnet18(pretrained=False)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Check which fc structure was used during training
    fc_keys = [k for k in state_dict.keys() if k.startswith('fc.')]

    if any('fc.1.' in k for k in fc_keys):
        # Sequential structure: Dropout + Linear
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 2)
        )
        print(f"{generator}: Using Sequential fc structure")
    else:
        # Simple Linear structure
        model.fc = nn.Linear(model.fc.in_features, 2)
        print(f"{generator}: Using Linear fc structure")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded {architecture} classifier: {generator}")
    return model, architecture


def test_classifier_on_samples(model, samples):
    """Test classifier on a list of image samples."""
    results = []

    with torch.no_grad():
        for img_path, true_label in samples:
            try:
                # Load and transform image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Get predictions
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)[0]

                class_0_prob = probs[0].item()
                class_1_prob = probs[1].item()
                predicted_class = 0 if class_0_prob > class_1_prob else 1

                results.append({
                    'image': str(img_path),
                    'true_label': true_label,
                    'class_0_prob': class_0_prob,
                    'class_1_prob': class_1_prob,
                    'predicted_class': predicted_class,
                    'correct': predicted_class == (0 if true_label == 'AI' else 1)  # Assuming class 0 = AI
                })

                print(f"  {Path(img_path).name:30s} | True: {true_label:4s} | "
                      f"Class0: {class_0_prob:.3f} | Class1: {class_1_prob:.3f} | "
                      f"Pred: {predicted_class}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    return results


def main():
    # Test all generators to determine class mappings
    test_generators = ['dall-e2', 'dall-e3', 'firefly', 'midjourneyV5', 'midjourneyV6', 'sdxl', 'stable_diffusion_1-5']

    # Store summary results
    classifier_mappings = {}

    for generator in test_generators:
        print(f"\n{'=' * 80}")
        print(f"TESTING CLASSIFIER: {generator.upper()}")
        print(f"{'=' * 80}")

        # Load classifier
        model, arch = load_classifier(generator)
        if model is None:
            continue

        # Prepare test samples - mix of real and AI
        samples = []

        # Real samples from COCO
        coco_dir = Path(test_data_dir) / 'coco'
        if coco_dir.exists():
            coco_images = list(coco_dir.glob('*.jpg'))[:3]  # First 3 COCO images for faster testing
            for img in coco_images:
                samples.append((img, 'Real'))

        # AI samples from this generator
        gen_dir = Path(test_data_dir) / generator
        if gen_dir.exists():
            gen_images = list(gen_dir.glob('*.png'))[:3]  # First 3 AI images for faster testing
            for img in gen_images:
                samples.append((img, 'AI'))

        if not samples:
            print(f"No test samples found for {generator}")
            continue

        print(f"\nTesting on {len(samples)} samples:")
        print(f"Image Name                     | True | Class0 | Class1 | Pred")
        print(f"-" * 70)

        # Test classifier
        results = test_classifier_on_samples(model, samples)

        # Analyze results
        if results:
            real_results = [r for r in results if r['true_label'] == 'Real']
            ai_results = [r for r in results if r['true_label'] == 'AI']

            print(f"\n{'-' * 40}")
            print(f"ANALYSIS FOR {generator.upper()}:")
            print(f"{'-' * 40}")

            if real_results:
                real_class0_avg = sum(r['class_0_prob'] for r in real_results) / len(real_results)
                real_class1_avg = sum(r['class_1_prob'] for r in real_results) / len(real_results)
                print(f"Real images - Avg Class0: {real_class0_avg:.3f}, Avg Class1: {real_class1_avg:.3f}")

            if ai_results:
                ai_class0_avg = sum(r['class_0_prob'] for r in ai_results) / len(ai_results)
                ai_class1_avg = sum(r['class_1_prob'] for r in ai_results) / len(ai_results)
                print(f"AI images   - Avg Class0: {ai_class0_avg:.3f}, Avg Class1: {ai_class1_avg:.3f}")

            # Determine likely class mapping
            if real_results and ai_results:
                real_prefers_class1 = real_class1_avg > real_class0_avg
                ai_prefers_class0 = ai_class0_avg > ai_class1_avg

                if real_prefers_class1 and ai_prefers_class0:
                    mapping = "Class 0 = AI, Class 1 = Real"
                    print(f"✓ LIKELY MAPPING: {mapping}")
                    classifier_mappings[generator] = {"mapping": mapping, "status": "correct"}
                elif not real_prefers_class1 and not ai_prefers_class0:
                    mapping = "Class 0 = Real, Class 1 = AI"
                    print(f"✓ LIKELY MAPPING: {mapping}")
                    classifier_mappings[generator] = {"mapping": mapping, "status": "inverted"}
                else:
                    mapping = "Mixed results"
                    print(f"? UNCLEAR MAPPING - {mapping}")
                    classifier_mappings[generator] = {"mapping": mapping, "status": "unclear"}

            # Save detailed results
            output_file = f"classifier_test_{generator}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to: {output_file}")

    # Print summary of all classifiers
    print(f"\n{'=' * 80}")
    print("SUMMARY OF ALL BINARY CLASSIFIERS")
    print(f"{'=' * 80}")
    for generator, info in classifier_mappings.items():
        status_icon = "✅" if info["status"] == "correct" else "❌" if info["status"] == "inverted" else "❓"
        print(f"{status_icon} {generator:20s}: {info['mapping']}")

    # Save summary
    with open("classifier_mappings_summary.json", 'w') as f:
        json.dump(classifier_mappings, f, indent=2)
    print(f"\nSummary saved to: classifier_mappings_summary.json")


if __name__ == "__main__":
    main()