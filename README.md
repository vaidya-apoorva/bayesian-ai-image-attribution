# Zero-Shot AI-Generated Image Detection Pipeline

This repository contains a comprehensive pipeline for zero-shot detection and attribution of AI-generated images, combining two state-of-the-art methods: **AEROBLADE** and **SReC (Super-Resolution Compression)**. The pipeline implements a Bayesian approach to probabilistically determine whether an image is AI-generated and, if so, attribute it to specific generators.

## Overview

The pipeline integrates:

1. **AEROBLADE**: Training-free detection using autoencoder reconstruction error from latent diffusion models
2. **SReC**: Lossless image compression through super-resolution for detecting compression artifacts
3. **Bayesian Inference**: Combines evidence from both methods to compute posterior probabilities for generator attribution

### Key Features

- **Zero-shot detection**: No training required on specific datasets
- **Multi-generator attribution**: Distinguishes between Real, DALL-E, MidJourney, Stable Diffusion, and Unknown sources
- **Probabilistic approach**: Provides confidence scores rather than binary classifications
- **Comprehensive evaluation**: Includes ROC analysis, accuracy metrics, and visualization tools

## Repository Structure

```
mt_apoorva_zero_shot_detector/
├── aeroblade/                      # AEROBLADE implementation
│   ├── scripts/                    # Main execution scripts
│   │   ├── run_aeroblade.py       # Primary detection script
│   │   ├── bayesian_posterior_inference.py  # Bayesian analysis
│   │   └── calculate_prior.py     # Prior probability computation
│   ├── experiments/               # Experimental notebooks and analysis
│   ├── models/                    # Pre-trained KDE models
│   ├── example_images/            # Sample images for testing
│   ├── results/                   # Analysis results and outputs
│   ├── visualisation/             # Generated plots and visualizations
│   └── src/aeroblade/            # Core AEROBLADE modules
├── SReC/                          # Super-Resolution Compression
│   ├── zero_shot_detector.py     # SReC-based detection
│   ├── src/                       # SReC core implementation
│   ├── models/                    # Pre-trained compression models
│   ├── results/                   # Organized analysis outputs
│   │   ├── performance/           # ROC curves and accuracy metrics
│   │   ├── histograms/           # D(0) distributions and data analysis
│   │   ├── models/               # Model performance plots
│   │   ├── KDE_plot/             # KDE analysis visualizations
│   │   └── aeroblade_plots/      # AEROBLADE integration results
│   ├── figs/                     # Concept figures and documentation
│   └── scripts/                  # Utility and analysis scripts
│       └── bayesian_generator_attribution.py  # Main pipeline integration
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.7+ (AEROBLADE tested with 3.10, SReC with 3.7)
- CUDA-compatible GPU (recommended)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mt_apoorva_zero_shot_detector
   ```

2. **Set up AEROBLADE environment**:
   ```bash
   cd aeroblade
   python -m venv aeroblade_env
   source aeroblade_env/bin/activate  # On Windows: aeroblade_env\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Set up SReC environment**:
   ```bash
   cd ../SReC
   conda env create -f environment.yml
   conda activate SReC
   pip install -r requirements.txt
   ```

4. **Install torchac for compression** (optional, required for SReC compression):
   ```bash
   pip install torchac
   ```

## Usage

### Quick Start

#### 1. AEROBLADE Detection

Run AEROBLADE on sample images:
```bash
cd aeroblade
python scripts/run_aeroblade.py --files-or-dirs example_images/
```

For custom images:
```bash
python scripts/run_aeroblade.py --files-or-dirs path/to/your/images/
```

#### 2. SReC Analysis

Run SReC-based detection:
```bash
cd SReC
python zero_shot_detector.py --path path/to/images --file image_list.txt --load models/openimages.pth --save-path output/
```

#### 3. Bayesian Integration

Combine both methods for final attribution:
```bash
cd SReC
python scripts/bayesian_generator_attribution.py
```

### Advanced Usage

#### Custom Model Configuration

**AEROBLADE parameters**:
- `--autoencoders`: Specify autoencoder models (default: SD1, SD2, Kandinsky)
- `--distance-metric`: Distance calculation method (default: lpips_vgg_2)
- `--batch-size`: Processing batch size

**SReC parameters**:
- `--resblocks`: Number of residual blocks (default: 5)
- `--scale`: Downsampling scale factor (default: 3)
- `--K`: Clusters in logistic mixture model (default: 10)

#### Batch Processing

Process multiple image directories:
```bash
# AEROBLADE
python scripts/run_aeroblade.py --files-or-dirs dir1/ dir2/ dir3/

# SReC with file list
echo "image1.jpg" > file_list.txt
echo "image2.jpg" >> file_list.txt
python zero_shot_detector.py --path images/ --file file_list.txt --load models/openimages.pth
```

### Pipeline Integration

The complete pipeline workflow:

1. **Generate AEROBLADE distances**:
   ```bash
   cd aeroblade
   python scripts/run_aeroblade.py --files-or-dirs dataset/
   ```

2. **Compute SReC scores**:
   ```bash
   cd ../SReC
   python zero_shot_detector.py --path dataset/ --file image_list.txt --load models/openimages.pth
   ```

3. **Calculate priors and posteriors**:
   ```bash
   cd ../aeroblade
   python scripts/calculate_prior.py
   python scripts/bayesian_posterior_inference.py
   ```

4. **Final attribution**:
   ```bash
   cd ../SReC
   python scripts/bayesian_generator_attribution.py
   ```

## Output Format

### AEROBLADE Output
- **CSV files**: Distance measurements for each autoencoder
- **Reconstruction images**: Visual reconstruction results (saved in `results/`)
- **JSON files**: Prior probabilities for each generator
- **Visualizations**: KDE plots and analysis charts (organized in `visualisation/`)

### SReC Output
- **SREC files**: Compressed representations
- **D(l) scores**: Reconstruction error measurements
- **Performance plots**: ROC curves and accuracy metrics (in `results/performance/`)
- **Distribution analysis**: D(0) histograms and generator comparisons (in `results/histograms/`)
- **Model diagnostics**: Training and debug visualizations (in `results/models/`)

### Bayesian Integration
- **Posterior probabilities**: Final attribution confidence scores
- **Classification results**: Most likely generator for each image
- **Visualization plots**: ROC curves, confusion matrices, histograms

## Experiments and Evaluation

The repository includes comprehensive experimental frameworks:

### Notebooks
- `aeroblade/experiments/01_detect.ipynb`: Detection performance analysis
- `aeroblade/experiments/02_analyze_patches.ipynb`: Patch-based analysis
- `aeroblade/experiments/03_deeper_reconstructions.ipynb`: Reconstruction quality

### Evaluation Scripts
- `scripts/compute_accuracy.py`: Classification accuracy metrics
- `scripts/compare_posterior_vs_classifier.py`: Method comparison
- `plot.py`: ROC curve and performance visualization

### Performance Metrics
- **Detection accuracy**: Binary real vs. fake classification
- **Attribution accuracy**: Multi-class generator identification
- **ROC analysis**: True positive vs. false positive rates
- **Bayesian metrics**: Posterior probability calibration

## Model Information

### Pre-trained Models

**AEROBLADE KDE Models**:
- `aeroblade/models/generator_kdes/`: Core KDE models (4 generators)
  - `kde_model_Real.joblib`: Real image distance distribution
  - `kde_model_DALL-E.joblib`: DALL-E generated images  
  - `kde_model_MidJourney.joblib`: MidJourney V5 images
  - `kde_model_StableDiffusion.joblib`: Stable Diffusion variants

**SReC Models** (located in `SReC/models/`):
- `openimages.pth`: Trained on Open Images dataset (2.70 bpsp)
- `imagenet64.pth`: Trained on ImageNet64 (4.29 bpsp)

### Output Organization

**AEROBLADE Outputs**:
- `results/`: Distance measurements and analysis results
- `visualisation/`: Generated plots and KDE visualizations

**SReC Outputs**:
- `results/performance/`: ROC curves and classification metrics
- `results/histograms/`: D(0) distribution analysis plots  
- `results/models/`: Model performance and debug visualizations
- `results/KDE_plot/`: KDE analysis and coding cost plots
- `results/aeroblade_plots/`: Integration analysis with AEROBLADE

### Supported Generators
- **Real images**: COCO, Open Images, RAISE dataset
- **DALL-E**: DALL-E 2, DALL-E 3
- **MidJourney**: MidJourney V5, V5.1
- **Stable Diffusion**: SD1.1, SD1.5, SD2.1, SDXL

## Technical Details

### AEROBLADE Method
- Uses autoencoder reconstruction error as detection signal
- Leverages pre-trained autoencoders from diffusion models
- Measures perceptual distance using LPIPS (Learned Perceptual Image Patch Similarity)
- Training-free approach suitable for zero-shot scenarios

### SReC Method
- Frames detection as a compression problem
- Uses super-resolution networks for lossless compression
- Measures D(l) = NLL - Entropy as compression efficiency
- Real images typically compress better than generated ones

### Bayesian Integration
- Combines evidence from both methods using Bayes' theorem
- Computes posterior probabilities: P(generator|evidence)
- Uses KDE-estimated priors from distance distributions
- Provides probabilistic rather than deterministic classifications


## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size: `--batch-size 1`
   - Use CPU mode if necessary

2. **Missing dependencies**:
   - Ensure all requirements are installed
   - Check CUDA compatibility for torchac

3. **File path issues**:
   - Use absolute paths for directories
   - Ensure image files have supported extensions (.jpg, .png)

4. **Model loading errors**:
   - Verify model files are present in `models/` directories
   - For AEROBLADE: KDE models are organized in `generator_kdes/`
   - For SReC: Compression models are in `SReC/models/`
   - Check file permissions and disk space

5. **Output file organization**:
   - Results are automatically organized into appropriate subdirectories
   - AEROBLADE: outputs to `results/` and `visualisation/`
   - SReC: outputs organized by type in `results/performance/`, `results/histograms/`, etc.

## Citation

If you use this pipeline in your research, please cite the original papers:

**AEROBLADE**:
```bibtex
@inproceedings{ricker2024aeroblade,
  title={AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error},
  author={Ricker, Jonas and Lukovnikov, Denis and Fischer, Asja},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

**SReC**:
```bibtex
@article{cao2020lossless,
  title={Lossless Image Compression through Super-Resolution},
  author={Cao, Sheng and Wu, Chao-Yuan and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:2004.02872},
  year={2020}
}
```

## License

This project combines code from multiple sources. Please refer to individual component licenses:
- AEROBLADE: [Original repository license](https://github.com/jonasricker/aeroblade)
- SReC: [Original repository license](https://github.com/caoscott/SReC)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review individual component READMEs in `aeroblade/` and `SReC/`
3. Open an issue with detailed error information and system specs

---

**Note**: This pipeline is designed for research purposes. Performance may vary depending on image quality, resolution, and specific use cases. Always validate results on your specific dataset and use case.
