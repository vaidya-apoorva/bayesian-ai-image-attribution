#!/usr/bin/env python3
"""
Conservative DALL-E 2 Classifier Training

Use more conservative settings to avoid CUDA errors.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import json
import time
import gc

# Conservative settings to avoid CUDA issues
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Paths
data_dir = '/mnt/hdd-data/vaidya/dataset'
output_dir = '/mnt/ssd-data/vaidya/combined_pipeline/models'
generator_name = 'dalle2'
real_dataset = 'coco'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def create_binary_dataset(real_folder, generator_folder):
    """Create a binary dataset with Real (0) and Generator (1) classes."""
    import tempfile
    from pathlib import Path

    temp_dir = tempfile.mkdtemp()
    real_temp = Path(temp_dir) / "real"
    gen_temp = Path(temp_dir) / "generator"

    real_temp.mkdir()
    gen_temp.mkdir()

    # Create symbolic links to original images
    real_images = list(Path(real_folder).glob("*.png"))
    gen_images = list(Path(generator_folder).glob("*.png"))

    # Limit to smaller dataset to reduce memory pressure
    max_images = 300  # Reduced from 400
    if len(real_images) > max_images:
        real_images = real_images[:max_images]
    if len(gen_images) > max_images:
        gen_images = gen_images[:max_images]

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
    """Train for one epoch with conservative settings."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        # Clear cache every few batches
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        try:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}: Loss={loss.item():.4f}")

        except RuntimeError as e:
            print(f"    Error in batch {batch_idx}: {e}")
            # Clear cache and continue
            torch.cuda.empty_cache()
            continue

    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0


def evaluate(model, dataloader, criterion):
    """Evaluate the model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

            images, labels = images.to(device), labels.to(device)

            try:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
            except RuntimeError as e:
                print(f"    Eval error in batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                continue

    return total_loss / total if total > 0 else 0, correct / total if total > 0 else 0


def train_dalle2_classifier():
    """Train DALL-E 2 classifier with conservative settings."""

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

    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)  # Reduced
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model with ResNet18 (lighter than ResNet50)
    model = models.resnet18(pretrained=True)  # Lighter model
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Loss and optimizer with lower learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Lower LR

    # Training loop with fewer epochs
    num_epochs = 10  # Reduced epochs
    best_val_acc = 0.0
    training_history = []
    patience = 3
    patience_counter = 0

    print(f"Starting conservative training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Clear cache before each epoch
        torch.cuda.empty_cache()
        gc.collect()

        try:
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
                patience_counter = 0
                print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")

            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping triggered after {epoch + 1} epochs")
                break

        except Exception as e:
            print(f"  Error in epoch {epoch + 1}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

    training_time = time.time() - start_time

    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    # Save model
    model_path = os.path.join(output_dir, f"resnet18_{generator_name}_vs_{real_dataset}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': 2,
            'architecture': 'resnet18',  # Note: ResNet18 not ResNet50
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
    """Main execution."""
    print("Starting conservative DALL-E 2 training...")

    result = train_dalle2_classifier()

    if result:
        print(f"✓ Successfully trained {generator_name}")

        # Update training summary
        summary_path = os.path.join(output_dir, "binary_training_summary.json")
        try:
            with open(summary_path, 'r') as f:
                all_results = json.load(f)
        except:
            all_results = {}

        all_results[generator_name] = result

        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Updated training summary")

    else:
        print(f"✗ Failed to train {generator_name}")


if __name__ == "__main__":
    main()