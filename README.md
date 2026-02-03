# GI Conference: AI-Generated Image Detection & Attribution Pipeline

This repository contains a comprehensive multi-method pipeline for detection and attribution of AI-generated images, combining four state-of-the-art approaches with Bayesian inference for robust probabilistic attribution. The pipeline was developed for the GI Conference research on zero-shot and training-free AI-generated image forensics.

## Overview

The pipeline integrates four complementary detection methods:

1. **AEROBLADE**: Training-free detection using autoencoder reconstruction error from latent diffusion models
2. **SReC (Super-Resolution Compression)**: Lossless image compression through super-resolution for detecting compression artifacts
3. **RIGID**: Training-free, model-agnostic detection using robustness to noise perturbations in vision foundation model representations
4. **Baseline Classifier**: Two-level ResNet-based supervised classifier for Real vs AI detection and generator identification
5. **Bayesian Attribution Framework**: Combines evidence from multiple detectors to compute posterior probabilities for generator attribution

### Key Features

- **Multi-method approach**: Combines training-free (AEROBLADE, RIGID, SReC) and supervised (Baseline Classifier) methods
- **Zero-shot detection**: Training-free methods require no retraining on specific datasets
- **Multi-generator attribution**: Distinguishes between Real and 7 AI generators: DALL-E 2, DALL-E 3, Firefly, MidJourney V5, MidJourney V6, SDXL, and Stable Diffusion 1.5
- **Probabilistic approach**: Bayesian framework provides confidence scores and posterior probabilities
- **Comprehensive evaluation**: Includes ROC analysis, confusion matrices, accuracy metrics, and visualization tools
- **Unified pipeline runners**: Scripts for easy execution across all datasets and methods

## Repository Structure

```
gi_conference_code/
├── aeroblade/                      # AEROBLADE implementation
│   ├── scripts/                    # Detection and analysis scripts
│   │   ├── run_aeroblade.py       # Primary detection script
│   │   ├── bayesian_posterior_inference.py  # Bayesian analysis
│   │   ├── calculate_prior.py     # Prior probability computation
│   │   └── compute_aeroblade_distances_combined.py  # Batch distance computation
│   ├── experiments/               # Experimental notebooks and analysis
│   │   ├── 01_detect.ipynb       # Detection performance analysis
│   │   ├── 02_analyze_patches.ipynb  # Patch-based analysis
│   │   └── 03_deeper_reconstructions.ipynb  # Reconstruction quality
│   ├── models/                    # Pre-trained KDE models
│   │   └── generator_kdes/       # Generator-specific KDE models
│   ├── results/                   # Analysis results and outputs
│   ├── visualisation/            # Generated plots and visualizations
│   └── src/aeroblade/           # Core AEROBLADE modules
│
├── SReC/                          # Super-Resolution Compression
│   ├── srec_detector.py          # Main SReC detection script (called via srec_runner.py)
│   ├── zero_shot_detector.py     # Legacy detection interface
│   ├── src/                      # SReC core implementation
│   ├── models/                   # Pre-trained compression models
│   │   ├── openimages.pth       # Open Images model (2.70 bpsp)
│   │   └── imagenet64.pth       # ImageNet64 model (4.29 bpsp)
│   ├── scripts/                  # Utility scripts (not directly used by combined_pipeline)
│   └── datasets/                 # Training and evaluation file lists
│
├── RIGID/                         # Training-free detection via noise robustness
│   ├── rigid_detector.py         # Main RIGID detection script
│   ├── rigid_cli.py             # Command-line interface
│   ├── rigid_visualisation.py   # Visualization utilities
│   ├── RIGID.ipynb              # Demonstration notebook
│   ├── gen_images/              # Generated image test sets
│   └── results/                 # Detection results and visualizations
│
├── baseline_classifier/          # Supervised ResNet-based classifier
│   ├── two_level_diffusion_forensics.py  # Two-level training script
│   ├── plot_confusion_matrix_level1.py   # Level-1 (Real vs AI) visualization
│   ├── plot_confusion_matrix_level2.py   # Level-2 (Generator ID) visualization
│   └── README.md                # Classifier-specific documentation
│
├── combined_pipeline/            # Unified pipeline integration
│   ├── scripts/                  # Pipeline runners and integration scripts
│   │   ├── srec_runner.py       # SReC pipeline runner
│   │   ├── aeroblade_runner.py  # AEROBLADE pipeline runner
│   │   ├── rigid_runner.py      # RIGID pipeline runner
│   │   ├── generate_probability_matrix.py  # Classifier likelihood matrix
│   │   ├── confusion_from_likelihoods.py   # Bayesian confusion matrices
│   │   ├── learn_prior_weights.py          # Prior weight optimization
│   │   ├── compare_priors.py    # Compare different prior methods
│   │   └── BAYESIAN_SCRIPTS/    # Bayesian attribution scripts
│   │       ├── bayesian_attribution.py          # Main attribution script
│   │       ├── bayesian_attribution_srec_weights.py  # SReC-weighted version
│   │       ├── generator_attribution.py         # Legacy attribution
│   │       ├── analyze_bayesian_performance.py  # Performance analysis
│   │       └── summarize_bayesian_results.py    # Results summarization
│   ├── models/                   # Trained classifier models
│   │   └── models_with_new_srec_weights/  # SReC-weighted models
│   └── results/                  # Combined pipeline results
│       ├── AEROBLADE/           # AEROBLADE outputs per dataset
│       ├── SREC/                # SReC outputs per dataset
│       ├── RIGID/               # RIGID outputs per dataset
│       └── BAYESIAN_RESULTS/    # Final Bayesian attribution results
│
├── gi_conference_results/        # Published results and analysis
│   ├── attribution_summary.json  # Overall attribution summary
│   ├── bayesian_results/        # Bayesian analysis outputs
│   ├── bayesian_results_with_new_model/  # Updated model results
│   └── detectors_only_likelihoods_analysis/  # Detector-only analysis
│
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.7+ (AEROBLADE tested with 3.10, SReC with 3.7, RIGID with 3.8+)
- CUDA-compatible GPU (recommended for faster processing)
- Git for cloning the repository

### Setup Instructions

#### 1. Clone the repository
```bash
git clone git@gitlab.cs.fau.de:op44ogoh/gi_conference_code.git
cd gi_conference_code
```

#### 2. Set up AEROBLADE environment
```bash
cd aeroblade
python -m venv aeroblade_env
source aeroblade_env/bin/activate  # On Windows: aeroblade_env\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

#### 3. Set up SReC environment
```bash
cd ../SReC
conda env create -f environment.yml
conda activate SReC
pip install -r requirements.txt
```

#### 4. Install torchac for SReC compression (optional)
Required for actual compression/decompression:
```bash
pip install torchac
```

#### 5. Set up RIGID environment
```bash
cd ../RIGID
# RIGID uses standard PyTorch with DINOv2
pip install torch torchvision
pip install timm  # For vision transformers
```

#### 6. Set up Baseline Classifier environment
```bash
cd ../baseline_classifier
conda create -n baseline python=3.10 -y
conda activate baseline
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --index-url https://download.pytorch.org/whl/cu113
pip install numpy==1.24.4 pillow==9.5.0 scikit-learn==1.2.2 tqdm pandas matplotlib
```

#### 7. Set up Combined Pipeline environment
The combined pipeline can use any of the above environments, or create a unified environment:
```bash
cd ../combined_pipeline
# Use the baseline classifier environment for consistency
conda activate baseline
# Additional dependencies for Bayesian integration
pip install scipy joblib
```

## Usage

### Dataset Structure

The pipeline expects datasets organized as follows:
```
/path/to/dataset/
├── coco/                          # Real images from COCO
├── imagenet/                      # Real images from ImageNet (optional)
├── dalle2/                        # DALL-E 2 generated images
├── dalle3/                        # DALL-E 3 generated images
├── firefly/                       # Adobe Firefly generated images
├── midjourneyV5/                 # MidJourney V5 generated images
├── midjourneyV6/                 # MidJourney V6 generated images
├── sdxl/                         # Stable Diffusion XL generated images
└── stable_diffusion_1-5/         # Stable Diffusion 1.5 generated images
```

### Pipeline Runners (Recommended Approach)

The combined pipeline provides unified runners for each detection method that handle multiple datasets automatically.

#### 1. Run SReC Detection

Process multiple datasets through SReC to compute D(l) compression scores.
**Note**: `srec_runner.py` calls `SReC/srec_detector.py` as a subprocess:

```bash
python /path/to/combined_pipeline/scripts/srec_runner.py \
    --images-list \
        /path/to/dataset/coco \
        /path/to/dataset/dalle2 \
        /path/to/dataset/dalle3 \
        /path/to/dataset/firefly \
        /path/to/dataset/midjourneyV5 \
        /path/to/dataset/midjourneyV6 \
        /path/to/dataset/sdxl \
        /path/to/dataset/stable_diffusion_1-5 \
    --model /path/to/SReC/models/openimages.pth \
    --resblocks 3
```

**Example command (as used in GI Conference experiments):**
```bash
python /mnt/ssd-data/vaidya/combined_pipeline/scripts/srec_runner.py \
    --images-list \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/coco \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/dalle2 \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/dalle3 \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/firefly \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/midjourneyV5 \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/midjourneyV6 \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/sdxl \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/stable_diffusion_1-5 \
    --model /mnt/ssd-data/vaidya/SReC/models/openimages.pth \
    --resblocks 3
```

**Key Parameters:**
- `--images-list`: Space-separated list of dataset directories to process
- `--model`: Path to pre-trained SReC model (`openimages.pth` or `imagenet64.pth`)
- `--resblocks`: Number of residual blocks (default: 5, use 3 for faster processing)
- `--n-feats`: Number of features (default: 64)
- `--scale`: Downsampling scale factor (default: 3)
- `--K`: Clusters in logistic mixture model (default: 10)

**Outputs:**
- JSON files with D(l) scores for each image: `combined_pipeline/results/SREC/{DATASET}/JSON/{dataset}_d0.json`
- SReC compressed files: `combined_pipeline/results/SREC/{DATASET}/SREC_IMAGES/`

#### 2. Run AEROBLADE Detection

Process multiple datasets through AEROBLADE to compute reconstruction distances:

```bash
python /path/to/combined_pipeline/scripts/aeroblade_runner.py \
    --images-list \
        /path/to/dataset/coco \
        /path/to/dataset/dalle2 \
        /path/to/dataset/dalle3 \
        /path/to/dataset/firefly \
        /path/to/dataset/midjourneyV5 \
        /path/to/dataset/midjourneyV6 \
        /path/to/dataset/sdxl \
        /path/to/dataset/stable_diffusion_1-5 \
    --amount 100 \
    --repo-ids CompVis/stable-diffusion-v1-1
```

**Example command (as used in GI Conference experiments):**
```bash
python /mnt/ssd-data/vaidya/combined_pipeline/scripts/aeroblade_runner.py \
    --images-list \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/coco \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/dalle2 \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/dalle3 \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/firefly \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/midjourneyV5 \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/midjourneyV6 \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/sdxl \
        /mnt/hdd-data/vaidya/gi_conference_dataset/bayesian_test_folder/stable_diffusion_1-5 \
    --amount 100 \
    --repo-ids CompVis/stable-diffusion-v1-1
```

**Key Parameters:**
- `--images-list`: Space-separated list of dataset directories to process
- `--amount`: Number of images to process per dataset (default: None = all images)
- `--repo-ids`: Diffusion model autoencoders to use (default: SD1-1, SD2-base, Kandinsky-2-1)
- `--output`: Base output directory (default: auto-organized in results/)

**Outputs:**
- Parquet files with distances: `combined_pipeline/results/AEROBLADE/{DATASET}/distances.parquet`
- CSV format: `combined_pipeline/results/AEROBLADE/{DATASET}/distances.csv`
- JSON format for Bayesian integration: `combined_pipeline/results/AEROBLADE/{DATASET}/{dataset}_aeroblade.json`
- Reconstruction images: `combined_pipeline/results/AEROBLADE/{DATASET}/reconstructions/`

#### 3. Run RIGID Detection

Process multiple datasets through RIGID to compute noise robustness scores:

```bash
python /path/to/combined_pipeline/scripts/rigid_runner.py \
    --images-list \
        /path/to/dataset/coco \
        /path/to/dataset/dalle2 \
        /path/to/dataset/dalle3 \
        /path/to/dataset/firefly \
        /path/to/dataset/midjourneyV5 \
        /path/to/dataset/midjourneyV6 \
        /path/to/dataset/sdxl \
        /path/to/dataset/stable_diffusion_1-5 \
    --real-datasets coco \
    --ai-datasets dalle2 dalle3 firefly midjourneyV5 midjourneyV6 sdxl stable_diffusion_1-5
```

**Key Parameters:**
- `--images-list`: Space-separated list of dataset directories to process
- `--real-datasets`: Names of real datasets (used for calibration)
- `--ai-datasets`: Names of AI-generated datasets
- `--noise-intensity`: Noise perturbation strength (default: 0.05)
- `--batch-size`: Processing batch size (default: 256)
- `--max-images`: Maximum images per dataset (default: 1000)

**Outputs:**
- JSON files with RIGID scores: `combined_pipeline/results/RIGID/rigid_results.json`
- Per-dataset distributions and statistics

#### 4. Generate Classifier Likelihood Matrix

Create a probability matrix from trained binary classifiers:

```bash
python /path/to/combined_pipeline/scripts/generate_probability_matrix.py
```

This script tests all trained ResNet classifiers on all datasets to generate P(Generator|Image) likelihoods.

**Outputs:**
- `combined_pipeline/results/classifier_probability_matrix.csv`: Full probability matrix
- `combined_pipeline/results/classifier_probability_matrix.json`: JSON format
- Heatmap visualizations

#### 5. Bayesian Attribution

Combine all detector outputs to compute final Bayesian posteriors.
**Note**: Uses Bayesian scripts from `combined_pipeline/scripts/BAYESIAN_SCRIPTS/`, not `SReC/scripts/`:

```bash
# Using SReC priors
python /path/to/combined_pipeline/scripts/BAYESIAN_SCRIPTS/bayesian_attribution_srec_weights.py \
    --batch \
    --input-dir /path/to/test/images \
    --method srec \
    --model openimages

# Using RIGID priors
python /path/to/combined_pipeline/scripts/BAYESIAN_SCRIPTS/bayesian_attribution_srec_weights.py \
    --batch \
    --input-dir /path/to/test/images \
    --method rigid

# Using AEROBLADE priors
python /path/to/combined_pipeline/scripts/BAYESIAN_SCRIPTS/bayesian_attribution_srec_weights.py \
    --batch \
    --input-dir /path/to/test/images \
    --method aeroblade
```

**Key Parameters:**
- `--method`: Prior method to use (`srec`, `rigid`, or `aeroblade`)
- `--model`: Model type for SReC (`openimages` or `imagenet`)
- `--batch`: Process batch of images
- `--input-dir`: Directory containing test images
- `--output`: Output file for results

**Outputs:**
- JSON files with posterior probabilities for each image
- Predicted generator labels
- Confidence scores

#### 6. Compare Prior Methods

Run comprehensive comparison of all prior methods:

```bash
python /path/to/combined_pipeline/scripts/compare_priors.py
```

This script runs Bayesian attribution with SReC, RIGID, and AEROBLADE priors and generates comparative analysis.

### Architecture Notes

**SReC Integration**: 
- `combined_pipeline/scripts/srec_runner.py` is the entry point
- It internally calls `SReC/srec_detector.py` as a subprocess
- **`SReC/scripts/` is NOT directly used by combined_pipeline** (it contains legacy utilities)
- Results are saved to `combined_pipeline/results/SREC/`

**Bayesian Attribution**:
- Located in: `combined_pipeline/scripts/BAYESIAN_SCRIPTS/`
- Main script: `bayesian_attribution_srec_weights.py`
- **NOT in `SReC/scripts/bayesian_generator_attribution.py`** (legacy reference - do not use)
- Loads detector outputs from: AEROBLADE, SReC, and RIGID result directories

## Output Format

### SReC Outputs

**Per-dataset JSON files:**
- Location: `combined_pipeline/results/SREC/{DATASET}/JSON/{dataset}_d0.json`
- Content: D(l) = NLL - Entropy scores for each image
- Format:
```json
{
  "image_001.png": -0.0245,
  "image_002.png": -0.0312,
  ...
}
```

**SReC compressed files:**
- Location: `combined_pipeline/results/SREC/{DATASET}/SREC_IMAGES/`
- Format: `.srec` binary files


### AEROBLADE Outputs

**Distance files:**
- Parquet: `combined_pipeline/results/AEROBLADE/{DATASET}/distances.parquet`
- CSV: `combined_pipeline/results/AEROBLADE/{DATASET}/distances.csv`
- JSON: `combined_pipeline/results/AEROBLADE/{DATASET}/{dataset}_aeroblade.json`

**JSON format:**
```json
{
  "image_001.png": 0.1234,
  "image_002.png": 0.1567,
  ...
}
```

**Reconstruction images:**
- Location: `combined_pipeline/results/AEROBLADE/{DATASET}/reconstructions/`
- Visual comparison of original and reconstructed images

### RIGID Outputs

**Results JSON:**
- Location: `combined_pipeline/results/RIGID/rigid_results.json`
- Per-image similarity scores
- Dataset statistics and distributions
- Real vs AI classification metrics

**Format:**
```json
{
  "coco": {
    "scores": [0.985, 0.978, ...],
    "mean": 0.981,
    "std": 0.012
  },
  "dalle2": {
    "scores": [0.823, 0.845, ...],
    "mean": 0.834,
    "std": 0.045
  }
}
```

### Classifier Outputs

**Probability Matrix:**
- Location: `combined_pipeline/results/classifier_probability_matrix.csv`
- Dimensions: 7 classifiers × 8 datasets (7 AI + 1 Real)
- Content: P(Generator|Image) probabilities

**Trained Models:**
- Location: `combined_pipeline/models/`
- Format: `resnet{18|50}_{generator}_vs_coco.pth`
- Includes: model weights, training history, validation accuracy

**Level-1 (Real vs AI) Results:**
- Confusion matrices
- Accuracy: ~92-95%

**Level-2 (Generator ID) Results:**
- Multi-class confusion matrices
- Per-class accuracy
- Overall accuracy: ~74-82%

### Bayesian Integration Outputs

**Attribution results:**
- Location: `gi_conference_results/bayesian_results/bayesian_pipeline_results/{method}/`
- Per-image posterior probabilities P(Generator|Image)
- Predicted generator labels
- Confidence scores

**Format:**
```json
{
  "image_001.png": {
    "predicted_generator": "dalle2",
    "confidence": 0.87,
    "posteriors": {
      "coco": 0.02,
      "dalle2": 0.87,
      "dalle3": 0.05,
      "firefly": 0.01,
      "midjourneyV5": 0.02,
      "midjourneyV6": 0.01,
      "sdxl": 0.01,
      "stable_diffusion_1_5": 0.01
    }
  }
}
```

**Summary files:**
- Overall accuracy per method (SReC, RIGID, AEROBLADE priors)
- Per-generator accuracy
- Confusion matrices
- Comparison across prior methods
