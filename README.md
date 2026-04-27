# ViT-Robustness
Comparing robustness of Vision Transformers vs CNNs under image corruption

# Robustness of Vision Transformers Under Image Corruption

**Course:** Advanced Topics in Deep Learning, LLMs, and LVMs  
**Track:** B: LVM Applications (Vision)  
**Team:** Shruthi Saravanan, Sharvani Bhavanam, CharanSai Kamsani

## Project Overview
This project compares how well a Vision Transformer (ViT-Base/16) and a 
CNN (ResNet-18) hold up when test images are corrupted, using four corruption 
types at three intensity levels on the CIFAR-10 dataset.

Important Note: We do NOT aim to improve either model, only to measure and analyze 
robustness behavior under corruption.

## Research Questions

1. How much does performance drop under corruption?
2. Are Vision Transformers more or less robust than CNNs?
3. Which corruption types cause the largest degradations?

---

## Key Results

| Corruption | Severity | CNN Acc | ViT Acc | Gap |
|---|---|---|---|---|
| Gaussian Noise | 1 | 26.7% | 21.8% | -4.8% |
| Gaussian Noise | 2 | 13.2% | 10.2% | -3.0% |
| Gaussian Noise | 3 | 10.9% | 10.0% | -0.9% |
| Gaussian Blur | 1 | 53.9% | 77.6% | +23.6% |
| Gaussian Blur | 2 | 29.9% | 30.9% | +1.0% |
| Gaussian Blur | 3 | 23.5% | 16.4% | -7.1% |
| Brightness | 1 | 77.0% | 96.0% | +19.1% |
| Brightness | 2 | 62.0% | 89.4% | +27.4% |
| Brightness | 3 | 14.2% | 30.3% | +16.1% |
| Rotation | 1 | 68.2% | 92.2% | +24.1% |
| Rotation | 2 | 46.1% | 84.7% | +38.5% |
| Rotation | 3 | 35.8% | 72.2% | +36.4% |

**Clean Accuracy:** CNN = 83.84% | ViT = 97.88%

---

## Repository Structure

```
ViT-Robustness/
├── README.md
├── requirements.txt
├── RepoLink.txt
├── configs/
│   ├── cnn_config.yaml
│   └── vit_config.yaml
├── src/
│   ├── 01_Setup_and_Data.ipynb
│   ├── 02_Train_Models.ipynb
│   ├── 03_Evaluation.ipynb
│   └── 04_Plots_and_Analysis.ipynb
├── results/
│   ├── metrics/
│   │   └── accuracy_table.csv
│   └── plots/
│       ├── degradation_curves.png
│       ├── confusion_matrices.png
│       ├── per_class_accuracy.png
│       ├── robustness_comparison.png
│       └── average_robustness.png
└── report/
    └── FinalReport.pdf
```

## Repository Structure

## Setup Instructions

### 1. Clone this repo
```bash
git clone https://github.com/sshooey/ViT-Robustness.git
cd ViT-Robustness
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Reproduce Results

### Step 1: Data Setup
Run `src/01_Setup_and_Data.ipynb` on **Google Colab or Kaggle**
- Downloads CIFAR-10 automatically
- Creates fixed train (45k) / val (5k) / test (10k) split
- Saves split indices using random seed 42

> Run this notebook first before any other notebook

### Step 2: Train Models
Run `src/02_Train_Models.ipynb` on **Google Colab (T4 GPU) or Kaggle (T4 GPU)**
- Trains ResNet-18: 15 epochs, batch size 128, learning rate 0.0001
- Fine-tunes ViT-Base/16: 10 epochs, batch size 64, learning rate 0.00002
- Saves best checkpoints based on validation accuracy
- CNN runtime: ~30 minutes | ViT runtime: ~90 minutes

> A GPU is required for this step, CPU is too slow for ViT training.  
> We recommend starting with Google Colab T4 GPU.  
> If you hit Colab's free GPU quota limit, Kaggle T4 x2 GPU is a free alternative.  
> To enable GPU on Kaggle: Session options → Accelerator → GPU T4 x2 (requires phone verification).

### Step 3: Evaluate Robustness
Run `src/03_Evaluation.ipynb` on **Google Colab (T4 GPU) or Kaggle (T4 GPU)**
- Applies 4 corruption types x 3 severity levels to the test set only
- Saves all results to `results/metrics/`
- Runtime: ~15 minutes

> Same GPU recommendation as Step 2: try Colab first, use Kaggle if GPU is unavailable.

### Step 4: Generate Plots
Run `src/04_Plots_and_Analysis.ipynb` on **Google Colab (CPU is fine) or Kaggle**
- No GPU needed for this step
- Generates all plots and saves them to `results/plots/`
- Runtime: ~2 minutes

## Outputs
All results saved to `results/metrics/` and `results/plots/`

## Hardware & Runtime

| Step | Recommended | Alternative | Runtime |
|---|---|---|---|
| Data Setup | Google Colab CPU | Kaggle CPU | ~2 minutes |
| CNN Training | Google Colab T4 GPU | Kaggle T4 x2 GPU | ~30 minutes |
| ViT Training | Google Colab T4 GPU | Kaggle T4 x2 GPU | ~90 minutes |
| Evaluation | Google Colab T4 GPU | Kaggle T4 x2 GPU | ~15 minutes |
| Plotting | Google Colab CPU | Kaggle CPU | ~2 minutes |

> **Note:** Google Colab free tier has GPU usage limits that reset every 12-24 hours.  
> If you see "Cannot connect to GPU backend" on Colab, switch to Kaggle as your backup.  
> Kaggle gives 30 free GPU hours per week and requires phone verification to unlock GPU access.

Our group used Kaggle to generate our final training and Corruption evalutions due to using up all of our Colab GPU usage during our numerous attempts of getting the most efficient training and corruption evaluation.

## Reproducibility (Further information)

All experiments use `seed = 42` for reproducibility.  
Seed set via `torch.manual_seed(42)` and `numpy.random.seed(42)`.
Fixed train/val split documented in `configs/` folder

---

## Checkpoints

Saved model checkpoints available at: https://drive.google.com/drive/folders/1bpb-aRyqiTvootc9Z3-V56J81wOeRRVF?usp=drive_link

---

## Dataset

- **Name:** CIFAR-10
- **Source:** University of Toronto (Krizhevsky, 2009)
- **Size:** 60,000 images, 10 classes, 32x32 pixels
- **License:** Publicly available for academic research
- **Download:** Handled automatically via `torchvision.datasets.CIFAR10`

---

## Corruption Types

| Corruption | Severity 1 | Severity 2 | Severity 3 |
|---|---|---|---|
| Gaussian Noise | σ = 0.25 | σ = 0.50 | σ = 0.75 |
| Gaussian Blur | σ = 1.0 | σ = 2.0 | σ = 4.0 |
| Brightness | scale = 0.5 | scale = 0.3 | scale = 0.1 |
| Rotation | 15° | 30° | 45° |

All corruptions are applied to the test set only. Training data is always clean.

---

## Models

| Model | Architecture | Pretrained On | Parameters | Clean Test Accuracy |
|---|---|---|---|---|
| CNN Baseline | ResNet-18 | ImageNet-1k | ~11M | 83.84% |
| Main Model | ViT-Base/16 | ImageNet-21k | ~86M | 97.88% |

---

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=9.5.0
pyyaml>=6.0
tqdm>=4.65.0
seaborn>=0.12.0
```



## References

- Dosovitskiy et al. (2020). An Image is Worth 16x16 Words. arXiv:2010.11929
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Hendrycks & Dietterich (2019). Benchmarking Neural Network Robustness. arXiv:1903.12261
- Karkala et al. (2025). Measuring Robustness of Vision Transformers Under Realistic Corruptions.
- Krizhevsky (2009). Learning Multiple Layers of Features from Tiny Images. U of Toronto.
- Zhou et al. (2022). Understanding the Robustness in Vision Transformers. ICML (PMLR 162).

## AI Tool Disclosure

AI tools (Claude, Anthropic) were used to assist with code structure and debugging. All experimental results, analysis, and writing represent the team's own work 
