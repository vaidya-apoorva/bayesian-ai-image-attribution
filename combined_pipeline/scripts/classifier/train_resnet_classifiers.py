#!/usr/bin/env python3
"""
Binary ResNet Classifier Training for Multiple Generators

Train binary classifiers (Real vs Generator) for each generator using the same
simple approach as the previous multi-class training script.

Usage:
    python train_binary_classifiers.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import json
import time

# Set up transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Base data directory
data_dir = '/mnt/hdd-data/vaidya/dataset'
output_dir = '/mnt/ssd-data/vaidya/combined_pipeline/models'
os.makedirs(output_dir, exist_ok=True)

# List of generators to train binary classifiers for
generators = ['dalle2', 'dalle3', 'firefly', 'midjourneyV5', 'midjourneyV6', 'sdxl', 'stable_diffusion_1_5']
real_dataset = 'coco'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def create_binary_dataset(real_folder, generator_folder):
    """Create a binary dataset with Real (0) and Generator (1) classes."""

    # Create temporary directory structure for ImageFolder
    import tempfile
    import shutil
    from pathlib import Path

    temp_dir = tempfile.mkdtemp()
    real_temp = Path(temp_dir) / "real"
    gen_temp = Path(temp_dir) / "generator"

    real_temp.mkdir()
    gen_temp.mkdir()

    # Create symbolic links to original images
    real_images = list(Path(real_folder).glob("*.png"))
    gen_images = list(Path(generator_folder).glob("*.png"))

    for i, img in enumerate(real_images):
        (real_temp / f"real_{i}.png").symlink_to(img)

    for i, img in enumerate(gen_images):
        (gen_temp / f"gen_{i}.png").symlink_to(img)

    print(f"  Real images: {len(real_images)}")
    print(f"  Generator images: {len(gen_images)}")

    # Create ImageFolder dataset
    dataset = ImageFolder(temp_dir, transform=transform)

    return dataset, temp_dir


def train_one_epoch(model, dataloader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 20 == 0:
            print(f"    Batch {batch_idx}: Loss={loss.item():.4f}")

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion):
    """Evaluate the model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_binary_classifier(generator_name):
    """Train a binary classifier for one generator vs real images."""

    print(f"\n{'=' * 60}")
    print(f"Training binary classifier: {generator_name} vs {real_dataset}")
    print(f"{'=' * 60}")

    # Paths
    real_folder = os.path.join(data_dir, real_dataset)
    generator_folder = os.path.join(data_dir, generator_name)

    if not os.path.exists(real_folder):
        print(f"Error: Real folder not found: {real_folder}")
        return None

    if not os.path.exists(generator_folder):
        print(f"Error: Generator folder not found: {generator_folder}")
        return None

    # Create binary dataset
    print("Creating binary dataset...")
    dataset, temp_dir = create_binary_dataset(real_folder, generator_folder)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 20  # Allow up to 20 but use early stopping
    best_val_acc = 0.0
    training_history = []
    patience = 3  # Stop if no improvement for 3 epochs
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # Store history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0  # Reset patience
            print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping triggered after {epoch + 1} epochs")
            break

    training_time = time.time() - start_time

    # Load best model
    model.load_state_dict(best_model_state)

    # Save model
    model_path = os.path.join(output_dir, f"resnet50_{generator_name}_vs_{real_dataset}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': 2,
            'architecture': 'resnet50',
            'generator': generator_name,
            'real_dataset': real_dataset
        },
        'training_history': training_history,
        'best_val_acc': best_val_acc,
        'training_time': training_time
    }, model_path)

    print(f"\nModel saved: {model_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Training time: {training_time:.1f} seconds")

    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)

    return {
        'generator': generator_name,
        'model_path': model_path,
        'best_val_acc': best_val_acc,
        'training_time': training_time,
        'training_history': training_history
    }


def main():
    """Train binary classifiers for all generators."""

    print("Starting Binary ResNet Classifier Training Pipeline")
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Generators: {generators}")
    print(f"Real dataset: {real_dataset}")

    all_results = {}

    for generator in generators:
        try:
            result = train_binary_classifier(generator)
            if result:
                all_results[generator] = result
                print(f"✓ Successfully trained {generator}")
            else:
                print(f"✗ Failed to train {generator}")
        except Exception as e:
            print(f"✗ Error training {generator}: {e}")
            continue

    # Save training summary
    summary_path = os.path.join(output_dir, "binary_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")

    for generator, result in all_results.items():
        print(f"{generator:20s}: {result['best_val_acc']:6.2f}% ({result['training_time']:6.1f}s)")

    print(f"\nTraining summary saved to: {summary_path}")
    print(f"Total classifiers trained: {len(all_results)}")
    print("Training pipeline completed!")


if __name__ == "__main__":
    main()