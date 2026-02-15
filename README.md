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


## Installation

### Prerequisites

- Python 3.7+ (AEROBLADE tested with 3.10, SReC with 3.7, RIGID with 3.8+)
- CUDA-compatible GPU (recommended for faster processing)
- Git for cloning the repository

### Setup Instructions

#### 1. Clone the repository
```bash
git clone https://github.com/vaidya-apoorva/bayesian-ai-image-attribution.git
cd bayesian-ai-image-attribution
```


#### 2. Set up AEROBLADE environment
Clone the official AEROBLADE implementation:
https://github.com/jonasricker/aeroblade

```bash
cd aeroblade
python -m venv aeroblade_env
source aeroblade_env/bin/activate  # On Windows: aeroblade_env\Scripts\activate
pip install -r requirements.txt
pip install -e .
```



#### 3. Set up SReC environment
Clone the official SReC implementation:
https://github.com/caoscott/SReC

```bash
cd ../SReC
# The following extra file has been added to the official SReC implementation to enable integration with the combined_pipeline:
#   SReC/srec_detector.py
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
Clone the official RIGID implementation and add the pipeline-specific files:
```bash
git clone https://github.com/IBM/RIGID.git RIGID
# The following two files are added on top for pipeline integration:
#   RIGID/rigid_detector.py
#   RIGID/rigid_visualisation.py
cd RIGID
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
```bash
python combined_pipeline/scripts/bayesian_scripts/bayesian_attribution.py \
  --batch \
  --input-dir /path/to/test/images \
  --method srec \
  --model openimages

python combined_pipeline/scripts/bayesian_scripts/bayesian_attribution.py \
  --batch \
  --input-dir /path/to/test/images \
  --method rigid

python combined_pipeline/scripts/bayesian_scripts/bayesian_attribution.py \
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

## Output Format

### Classifier Outputs

**Probability Matrix:**
- Location: `combined_pipeline/results/classifier_probability_matrix.csv`
- Dimensions: 7 classifiers × 8 datasets (7 AI + 1 Real)
- Content: P(Generator|Image) probabilities

**Trained Models:**
- Location: `combined_pipeline/models/`
- Format: `resnet{18|50}_{generator}_vs_coco.pth`
- Includes: model weights, training history, validation accuracy

## Citations

If you use this repository in your research, please cite the following works:

**AEROBLADE**
```bibtex
@InProceedings{Ricker_2024_CVPR,
    author    = {Ricker, Jonas and Lukovnikov, Denis and Fischer, Asja},
    title     = {AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {9130-9140}
}
```

**RIGID**
```bibtex
@misc{he2024rigidtrainingfreemodelagnosticframework,
  title={RIGID: A Training-free and Model-Agnostic Framework for Robust AI-Generated Image Detection}, 
  author={Zhiyuan He and Pin-Yu Chen and Tsung-Yi Ho},
  year={2024},
  eprint={2405.20112},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2405.20112}, 
}
```

**SReC**
```bibtex
@misc{cao2020losslessimagecompressionsuperresolution,
  title={Lossless Image Compression through Super-Resolution}, 
  author={Sheng Cao and Chao-Yuan Wu and Philipp Krähenbühl},
  year={2020},
  eprint={2004.02872},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2004.02872}, 
}
```
