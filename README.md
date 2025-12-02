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
gi_conference/
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
│   ├── srec_detector.py          # Main SReC detection script
│   ├── zero_shot_detector.py     # Legacy detection interface
│   ├── src/                      # SReC core implementation
│   ├── models/                   # Pre-trained compression models
│   │   ├── openimages.pth       # Open Images model (2.70 bpsp)
│   │   └── imagenet64.pth       # ImageNet64 model (4.29 bpsp)
│   ├── scripts/                  # Utility and analysis scripts
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
git clone <repository-url>
cd gi_conference
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

Process multiple datasets through SReC to compute D(l) compression scores:

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

Combine all detector outputs to compute final Bayesian posteriors:

```bash
# Using SReC priors
python /path/to/combined_pipeline/scripts/BAYESIAN_SCRIPTS/bayesian_attribution.py \
    --batch \
    --input-dir /path/to/test/images \
    --method srec \
    --model openimages

# Using RIGID priors
python /path/to/combined_pipeline/scripts/BAYESIAN_SCRIPTS/bayesian_attribution.py \
    --batch \
    --input-dir /path/to/test/images \
    --method rigid

# Using AEROBLADE priors
python /path/to/combined_pipeline/scripts/BAYESIAN_SCRIPTS/bayesian_attribution.py \
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

### Direct Method Usage (Individual Components)

#### AEROBLADE Standalone

Run AEROBLADE on sample images:
```bash
cd aeroblade
python scripts/run_aeroblade.py --files-or-dirs path/to/images/
```

For custom configuration:
```bash
python scripts/run_aeroblade.py \
    --files-or-dirs path/to/images/ \
    --repo-ids CompVis/stable-diffusion-v1-1 stabilityai/stable-diffusion-2-base \
    --distance-metrics lpips_vgg_2 \
    --batch-size 4
```

#### SReC Standalone

Run SReC-based detection:
```bash
cd SReC
python srec_detector.py \
    --path path/to/images \
    --file image_list.txt \
    --load models/openimages.pth \
    --save-path output/ \
    --resblocks 3
```

#### RIGID Standalone

Run RIGID detection:
```bash
cd RIGID
python rigid_detector.py \
    --dataset-paths /path/to/dataset1 /path/to/dataset2 \
    --real-datasets coco \
    --ai-datasets dalle2 dalle3
```

Or use the CLI:
```bash
python rigid_cli.py --images /path/to/images --output results/
```

#### Baseline Classifier Training

Train two-level classifier:
```bash
cd baseline_classifier
python two_level_diffusion_forensics.py \
    --data-dir /path/to/dataset \
    --output-dir runs/ \
    --epochs 50 \
    --batch-size 32
```

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

**Statistics:**
- Mean D(l) per dataset
- Standard deviation
- Distribution analysis

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
- ROC curves
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

## Experiments and Evaluation

The repository includes comprehensive experimental frameworks for each method.

### Detection Method Experiments

#### AEROBLADE Experiments

Located in `aeroblade/experiments/`:

1. **Detection Performance** (`01_detect.ipynb`, `01_detect.py`):
   - ROC analysis across generators
   - Distance distribution analysis
   - Optimal threshold selection
   - Cross-generator performance

2. **Patch Analysis** (`02_analyze_patches.ipynb`, `02_analyze_patches.py`):
   - Patch-level detection
   - Image complexity vs reconstruction error
   - Spatial analysis of artifacts

3. **Deep Reconstructions** (`03_deeper_reconstructions.ipynb`, `03_deeper_reconstructions.py`):
   - Multi-layer reconstruction analysis
   - Reconstruction quality metrics
   - Generator-specific patterns

**Run experiments:**
```bash
cd aeroblade
python experiments/01_detect.py  # Default detection evaluation
python experiments/01_detect.py --experiment-id robustness --transforms clean jpeg_90 blur_1.0  # Robustness
```

#### SReC Experiments

Located in `SReC/scripts/`:

1. **D(l) Distribution Analysis**:
   - KDE plots of D(l) values per generator
   - Statistical significance tests
   - Real vs AI separation analysis

2. **Model Performance**:
   - ImageNet64 vs Open Images models
   - Compression rate comparisons
   - Resolution impact analysis

#### RIGID Experiments

Located in `RIGID/`:

1. **Noise Robustness** (`RIGID.ipynb`):
   - Noise intensity ablation
   - Robustness curves
   - Vision model comparison (DINOv2, CLIP, etc.)

2. **Cross-dataset Evaluation**:
   - Generalization to unseen generators
   - ImageNet vs LSUN-Bedroom
   - GenImage benchmark results

#### Baseline Classifier Experiments

Located in `baseline_classifier/`:

1. **Two-Level Classification**:
   - Level-1: Real vs AI (binary)
   - Level-2: Generator identification (7-class)
   - Confusion matrix analysis

2. **Architecture Comparison**:
   - ResNet18 vs ResNet50
   - Transfer learning evaluation
   - Data augmentation impact

### Bayesian Integration Experiments

Located in `combined_pipeline/scripts/`:

1. **Prior Method Comparison** (`compare_priors.py`):
   ```bash
   python scripts/compare_priors.py
   ```
   - SReC-based priors
   - RIGID-based priors
   - AEROBLADE-based priors
   - Uniform priors (baseline)

2. **Likelihood Matrix Generation** (`generate_probability_matrix.py`):
   - 7×8 classifier probability matrix
   - Per-generator likelihood distributions
   - Heatmap visualizations

3. **Prior Weight Learning** (`learn_prior_weights.py`):
   - Optimize prior weights for maximum accuracy
   - Cross-validation
   - Weight sensitivity analysis

4. **Confusion Matrix Analysis** (`confusion_from_likelihoods.py`):
   - Multi-class confusion matrices
   - Per-class precision/recall
   - Misclassification patterns

### Performance Metrics

**Detection Metrics:**
- True Positive Rate (TPR) / Recall
- False Positive Rate (FPR)
- Precision
- F1-Score
- ROC-AUC

**Attribution Metrics:**
- Multi-class accuracy
- Per-generator accuracy
- Macro/Micro F1-scores
- Confusion matrices
- Top-K accuracy

**Bayesian Metrics:**
- Posterior probability calibration
- Confidence correlation with accuracy
- Expected Calibration Error (ECE)
- Brier Score

### Evaluation Scripts

Key evaluation scripts in `combined_pipeline/scripts/BAYESIAN_SCRIPTS/`:

1. **analyze_bayesian_performance.py**: Comprehensive performance analysis
2. **summarize_bayesian_results.py**: Generate summary reports
3. **confusion_from_likelihoods.py**: Create confusion matrices from likelihoods

**Run evaluation:**
```bash
python scripts/BAYESIAN_SCRIPTS/analyze_bayesian_performance.py \
    --results-dir /path/to/bayesian/results \
    --method srec
```

## Model Information

### Pre-trained Models

#### AEROBLADE Models
Location: `aeroblade/models/generator_kdes/`

**KDE (Kernel Density Estimation) Models:**
- `kde_model_Real.joblib`: Real image distance distribution
- `kde_model_DALL-E.joblib`: DALL-E generated images  
- `kde_model_MidJourney.joblib`: MidJourney generated images
- `kde_model_StableDiffusion.joblib`: Stable Diffusion variants

**Diffusion Model Autoencoders (downloaded automatically):**
- CompVis/stable-diffusion-v1-1
- stabilityai/stable-diffusion-2-base
- kandinsky-community/kandinsky-2-1

#### SReC Models
Location: `SReC/models/`

- **openimages.pth**: Trained on Open Images dataset
  - Compression rate: 2.70 bits per subpixel (bpsp)
  - Recommended for diverse natural images
  
- **imagenet64.pth**: Trained on ImageNet64
  - Compression rate: 4.29 bpsp
  - Optimized for ImageNet-style images

#### RIGID Models
- Uses pre-trained vision foundation models (no additional training)
- Default: DINOv2 ViT-L/14 (downloaded automatically from HuggingFace)
- Alternatives: CLIP, OpenCLIP (configurable)

#### Baseline Classifier Models
Location: `combined_pipeline/models/` and `combined_pipeline/models_ovr/`

**Binary Classifiers (Generator vs Real):**
- `resnet18_dalle2_vs_coco.pth`: DALL-E 2 detector
- `resnet50_dalle3_vs_coco.pth`: DALL-E 3 detector
- `resnet50_firefly_vs_coco.pth`: Firefly detector
- `resnet50_midjourneyV5_vs_coco.pth`: MidJourney V5 detector
- `resnet50_midjourneyV6_vs_coco.pth`: MidJourney V6 detector
- `resnet50_sdxl_vs_coco.pth`: SDXL detector
- `resnet50_stable_diffusion_1_5_vs_coco.pth`: SD 1.5 detector

**Architecture:** ResNet18/50 with binary classification head
**Training:** Supervised learning on balanced Real vs Generator datasets
**Validation Accuracy:** ~90-95% per generator

### Supported Generators

**Real Image Sources:**
- COCO (Common Objects in Context)
- ImageNet (256×256)
- Open Images (optional)
- RAISE (optional, removed due to distribution shift)

**AI Generators:**
1. **DALL-E 2** (OpenAI, 2022)
2. **DALL-E 3** (OpenAI, 2023)
3. **Adobe Firefly** (Adobe, 2023)
4. **MidJourney V5** (Midjourney, 2023)
5. **MidJourney V6** (Midjourney, 2024)
6. **Stable Diffusion XL (SDXL)** (Stability AI, 2023)
7. **Stable Diffusion 1.5** (Stability AI, 2022)

**Additional Generators (for extended evaluation):**
- ADM, ADMG (Diffusion Models)
- BigGAN, GigaGAN (GAN-based)
- DiT-XL-2 (Diffusion Transformer)
- LDM (Latent Diffusion Models)
- StyleGAN-XL
- RQ-Transformer, Mask-GIT (Token-based)

## Technical Details

### Detection Methods

#### AEROBLADE (Training-Free)
- **Principle**: Exploits autoencoder reconstruction error as a detection signal
- **Method**: 
  - Uses pre-trained autoencoders from diffusion models (SD1, SD2, Kandinsky)
  - Measures perceptual distance using LPIPS (Learned Perceptual Image Patch Similarity)
  - AI-generated images have lower reconstruction error in their native autoencoders
- **Advantages**: 
  - No training required
  - Model-agnostic (works across generators)
  - Interpretable (visual reconstructions)
- **Limitations**: 
  - Requires multiple autoencoder models
  - Computationally intensive for large batches
  - Best for diffusion-based generators

#### SReC (Training-Free)
- **Principle**: Frames detection as a lossless compression problem
- **Method**:
  - Uses super-resolution networks for compression
  - Measures D(l) = NLL - Entropy as compression efficiency
  - Real images typically compress better than generated images
- **Advantages**:
  - Single model for all generators
  - Fast inference
  - Strong theoretical foundation
- **Limitations**:
  - Requires pre-trained compression model
  - Performance depends on model training data
  - May be sensitive to image resolution

#### RIGID (Training-Free)
- **Principle**: Real images are more robust to noise perturbations in feature space
- **Method**:
  - Adds small Gaussian noise to images
  - Extracts features using vision foundation models (DINOv2)
  - Compares similarity between original and perturbed versions
  - Real images maintain higher similarity than AI-generated
- **Advantages**:
  - Extremely simple and fast
  - No training or fine-tuning
  - Works with any vision model
  - Strong generalization
- **Limitations**:
  - Requires careful noise calibration
  - Performance depends on foundation model choice
  - May struggle with highly realistic generators

#### Baseline Classifier (Supervised)
- **Principle**: Learns discriminative features through supervised training
- **Architecture**: ResNet18/50 with binary classification heads
- **Training**:
  - Level-1: Real vs AI (binary classification)
  - Level-2: Generator identification (7-class)
  - Balanced datasets, data augmentation, transfer learning
- **Advantages**:
  - High accuracy on known generators
  - Fast inference
  - Direct probability outputs
- **Limitations**:
  - Requires labeled training data
  - May not generalize to unseen generators
  - Needs retraining for new generators

### Bayesian Integration Framework

The pipeline combines multiple detectors using Bayesian inference:

**Formula:**
```
P(Generator|Image) ∝ P(Image|Generator) × P(Generator)
```

**Components:**
1. **Likelihoods** P(Image|Generator):
   - From binary classifier outputs
   - 7 trained classifiers provide P(Generator|Image) directly
   - Inverted using Bayes' theorem and priors

2. **Priors** P(Generator):
   - **SReC-based**: Learned from D(l) score distributions
   - **RIGID-based**: Learned from robustness score distributions
   - **AEROBLADE-based**: Learned from reconstruction distance distributions
   - **Uniform**: Equal probability baseline

3. **Posterior** P(Generator|Image):
   - Computed for all 8 classes (7 AI + 1 Real)
   - Normalized to sum to 1
   - Prediction = argmax posterior

**Prior Learning:**
- Train on validation set of known images
- Fit KDE to detector scores per generator
- Convert KDE to probability distributions
- Optionally learn weights to optimize accuracy

**Inference:**
```python
for each generator g:
    prior[g] = KDE_score(detector_output, generator=g)
    likelihood[g] = classifier_probability(image, generator=g)
    posterior[g] = likelihood[g] × prior[g]

posterior = normalize(posterior)
prediction = argmax(posterior)
```

**Benefits:**
- Combines complementary information from multiple detectors
- Provides calibrated confidence scores
- More robust than single-method detection
- Interpretable posterior distributions


## Troubleshooting

### Common Issues

#### 1. CUDA out of memory
**Symptoms**: CUDA OOM errors during processing

**Solutions**:
- **AEROBLADE**: Reduce `--batch-size 1` or use CPU mode
- **SReC**: Process fewer images at once, reduce `--n-feats`
- **RIGID**: Reduce `--batch-size`
- **Classifier**: Reduce batch size in training script

#### 2. Missing dependencies
**Symptoms**: Import errors, module not found

**Solutions**:
```bash
# Verify environment activation
conda activate <env_name>

# Reinstall requirements
pip install -r requirements.txt

# For torchac (SReC compression):
pip install torchac

# For DINOv2 (RIGID):
pip install timm
```

#### 3. Model loading errors
**Symptoms**: FileNotFoundError, checkpoint loading fails

**Solutions**:
- Verify model paths in configuration
- Check file permissions: `chmod 644 models/*.pth`
- Ensure models downloaded:
  - AEROBLADE: KDE models in `aeroblade/models/generator_kdes/`
  - SReC: `openimages.pth` or `imagenet64.pth` in `SReC/models/`
  - Classifier: Binary classifiers in `combined_pipeline/models/`

#### 4. File path issues
**Symptoms**: Dataset not found, image list empty

**Solutions**:
- Use absolute paths: `/full/path/to/dataset/`
- Check directory structure matches expected format
- Verify image extensions (`.png`, `.jpg`, `.jpeg`)
- Check file permissions: `chmod -R 755 /path/to/dataset/`

#### 5. Pipeline runner errors
**Symptoms**: Runner script fails, subprocess errors

**Solutions**:
- Ensure all paths in runner scripts point to correct locations
- Update hardcoded paths (search for `/mnt/ssd-data/` and `/mnt/hdd-data/`)
- Check that detector scripts exist and are executable
- Verify Python path includes necessary modules

**Example path updates:**
```python
# In srec_runner.py, aeroblade_runner.py, rigid_runner.py
self.detector_dir = Path("/your/actual/path/to/detector")
```

#### 6. Bayesian attribution errors
**Symptoms**: Missing JSON files, KeyError for datasets

**Solutions**:
- Ensure detectors have been run and generated outputs
- Check JSON file locations:
  - SReC: `combined_pipeline/results/SREC/{DATASET}/JSON/`
  - AEROBLADE: `combined_pipeline/results/AEROBLADE/{DATASET}/`
  - RIGID: `combined_pipeline/results/RIGID/`
- Verify classifier models are trained and available
- Check dataset naming consistency (lowercase, no spaces)

#### 7. Image format issues
**Symptoms**: PIL errors, unsupported format, corrupted images

**Solutions**:
- Convert images to standard formats:
```bash
# Convert all to PNG
for img in *.jpg; do convert "$img" "${img%.jpg}.png"; done
```
- Check image integrity:
```bash
# Find corrupted images
find /path/to/dataset -name "*.png" -exec file {} \; | grep -v "PNG image"
```
- Remove or fix corrupted images

#### 8. Slow processing
**Symptoms**: Very slow inference, taking hours

**Solutions**:
- **AEROBLADE**: 
  - Use `--amount 100` to limit images
  - Use single autoencoder: `--repo-ids CompVis/stable-diffusion-v1-1`
  - Set `--batch-size 1` for variable-size images
- **SReC**: 
  - Reduce `--resblocks` from 5 to 3
  - Use CPU if GPU memory is bottleneck
- **RIGID**: 
  - Increase `--batch-size 256` if memory allows
  - Reduce `--max-images` for testing
- **Use pipeline runners** instead of individual scripts (more optimized)

#### 9. Results not matching expected format
**Symptoms**: JSON structure different, missing fields

**Solutions**:
- Update to latest version of detector scripts
- Check output parsers in Bayesian attribution scripts
- Manually verify JSON structure matches expected format
- Use provided example JSON files as reference

#### 10. Environment conflicts
**Symptoms**: Version conflicts, incompatible packages

**Solutions**:
```bash
# Create fresh environments
conda create -n gi_conference python=3.10 -y
conda activate gi_conference

# Install minimal requirements first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113

# Then install method-specific packages
pip install -r requirements.txt
```

### Getting Help

1. **Check individual component READMEs**:
   - `aeroblade/README.md`
   - `SReC/README.md`
   - `RIGID/README.md`
   - `baseline_classifier/README.md`

2. **Review experiment notebooks**:
   - `aeroblade/experiments/*.ipynb`
   - `RIGID/RIGID.ipynb`

3. **Check example commands** in this README

4. **Verify system specs**:
```bash
# Python version
python --version

# PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# GPU info
nvidia-smi
```

5. **Open an issue** with:
   - Full error traceback
   - Command used
   - System specifications
   - Python and package versions

## Citation

If you use this pipeline or any of its components in your research, please cite the relevant papers:

### AEROBLADE
```bibtex
@inproceedings{ricker2024aeroblade,
  title={AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error},
  author={Ricker, Jonas and Lukovnikov, Denis and Fischer, Asja},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

### SReC
```bibtex
@article{cao2020lossless,
  title={Lossless Image Compression through Super-Resolution},
  author={Cao, Sheng and Wu, Chao-Yuan and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:2004.02872},
  year={2020}
}
```

### RIGID
```bibtex
@article{kui2024rigid,
  title={RIGID: A Training-free and Model-Agnostic Framework for Robust AI-Generated Image Detection},
  author={Kui, Zhiyuan and others},
  journal={arXiv preprint},
  year={2024}
}
```

### Baseline Classifier & Bayesian Framework
If you use the combined pipeline, baseline classifier, or Bayesian attribution framework:
```bibtex
@article{gi_conference_2025,
  title={Multi-Method Bayesian Framework for AI-Generated Image Detection and Attribution},
  author={[Your Name/Team]},
  journal={GI Conference},
  year={2025}
}
```

## Related Work

This pipeline builds upon and integrates several key works in AI-generated image detection:

- **L3C**: [Practical Full Resolution Learned Lossless Image Compression](https://github.com/fab-jul/L3C-PyTorch)
- **LPIPS**: [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://github.com/richzhang/PerceptualSimilarity)
- **DINOv2**: [DINOv2: Learning Robust Visual Features without Supervision](https://github.com/facebookresearch/dinov2)
- **Stable Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/CompVis/stable-diffusion)
- **GenImage**: [A Million-Scale Benchmark for Detecting AI-Generated Image](https://github.com/GenImage-Dataset/GenImage)

## License

This project combines code from multiple sources, each with their own licenses:

- **AEROBLADE**: [MIT License](https://github.com/jonasricker/aeroblade/blob/main/LICENSE) - Original repository
- **SReC**: [MIT License](https://github.com/caoscott/SReC/blob/master/LICENSE) - Original repository  
- **RIGID**: [License](https://github.com/huggingface/transformers/blob/main/LICENSE) - Uses HuggingFace transformers
- **Baseline Classifier**: Custom implementation (specify your license)
- **Combined Pipeline**: Custom implementation (specify your license)

Please refer to individual component directories for specific license information.

## Contributing

Contributions are welcome! We encourage improvements in the following areas:

### Areas for Contribution
1. **New Detection Methods**: Integration of additional detection techniques
2. **Generator Support**: Adding support for new AI image generators
3. **Optimization**: Speed and memory efficiency improvements
4. **Evaluation**: Additional metrics and benchmarks
5. **Documentation**: Tutorials, examples, and guides
6. **Visualization**: Better result visualization and analysis tools

### How to Contribute
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Update documentation (README, docstrings)
6. Submit a pull request with detailed description

### Code Guidelines
- Follow PEP 8 for Python code
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write descriptive commit messages
- Test on multiple datasets before submitting

### Reporting Issues
When reporting issues, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU)
- Full error traceback
- Minimal reproducible example if possible

## Acknowledgments

This project would not be possible without:

- **Jonas Ricker et al.** for AEROBLADE
- **Sheng Cao et al.** for SReC
- **RIGID authors** for training-free detection framework
- **PyTorch** and **HuggingFace** communities
- **Dataset providers**: COCO, ImageNet, Open Images, GenImage
- All contributors and users of this pipeline

Special thanks to the GI Conference organizers and reviewers for their feedback and support.

## Support

For questions, issues, and discussions:

### Documentation
1. Review this README thoroughly
2. Check component-specific READMEs:
   - `aeroblade/README.md`
   - `SReC/README.md` and `SReC/INSTALL.md`
   - `RIGID/README.md`
   - `baseline_classifier/README.md`
3. Explore Jupyter notebooks in `experiments/` directories
4. Review example commands and outputs

### Getting Help
1. **GitHub Issues**: For bugs and feature requests
2. **GitHub Discussions**: For questions and general discussion
3. **Email**: [contact email if applicable]

### Before Asking for Help
- [ ] Checked troubleshooting section
- [ ] Reviewed relevant documentation
- [ ] Verified environment setup
- [ ] Tested with provided example data
- [ ] Searched existing issues

### Useful Information to Provide
When seeking help, include:
- Full command or script used
- Complete error message/traceback
- System specifications (OS, GPU, Python version)
- Package versions: `pip freeze` or `conda list`
- Dataset information (size, format, structure)
- Expected vs actual behavior

---

## Quick Reference

### Key Commands

```bash
# SReC Runner (OpenImages model, 3 resblocks)
python combined_pipeline/scripts/srec_runner.py \
    --images-list dataset1/ dataset2/ dataset3/ \
    --model SReC/models/openimages.pth \
    --resblocks 3

# AEROBLADE Runner (SD1-1 autoencoder, 100 images)
python combined_pipeline/scripts/aeroblade_runner.py \
    --images-list dataset1/ dataset2/ dataset3/ \
    --amount 100 \
    --repo-ids CompVis/stable-diffusion-v1-1

# RIGID Runner
python combined_pipeline/scripts/rigid_runner.py \
    --images-list dataset1/ dataset2/ dataset3/ \
    --real-datasets coco \
    --ai-datasets dalle2 dalle3 firefly

# Bayesian Attribution (SReC priors, OpenImages model)
python combined_pipeline/scripts/BAYESIAN_SCRIPTS/bayesian_attribution.py \
    --batch \
    --input-dir test_images/ \
    --method srec \
    --model openimages

# Compare All Prior Methods
python combined_pipeline/scripts/compare_priors.py
```

### Directory Organization

```
Results Structure:
combined_pipeline/results/
├── AEROBLADE/{DATASET}/distances.json
├── SREC/{DATASET}/JSON/{dataset}_d0.json
├── RIGID/rigid_results.json
└── classifier_probability_matrix.csv

gi_conference_results/bayesian_results/
├── bayesian_pipeline_results/
│   ├── srec/attribution_results.json
│   ├── rigid/attribution_results.json
│   └── aeroblade/attribution_results.json
└── detectors_only_likelihoods_analysis/
```

### Important Notes

⚠️ **Path Configuration**: Update hardcoded paths in runner scripts to match your system
⚠️ **GPU Memory**: Use batch_size=1 for AEROBLADE if processing variable-size images  
⚠️ **Resblocks**: SReC resblocks=3 is faster, resblocks=5 is more accurate
⚠️ **Dataset Naming**: Keep consistent lowercase naming (no spaces) across all components
⚠️ **Image Formats**: Supports .png, .jpg, .jpeg - ensure uniform format per dataset

---

**Note**: This pipeline is designed for research purposes in AI-generated image forensics. Performance varies depending on:
- Image quality and resolution
- Generator characteristics and training data
- Prior method selection
- Classifier training quality
- Test dataset similarity to training data

Always validate results on your specific dataset and use case. For production deployment, consider additional robustness testing and adversarial evaluation.
