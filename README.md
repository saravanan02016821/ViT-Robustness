# ViT-Robustness
Comparing robustness of Vision Transformers vs CNNs under image corruption

# Robustness of Vision Transformers Under Image Corruption

**Course:** Advanced Topics in Deep Learning, LLMs, and LVMs  
**Track:** B — LVM Applications (Vision)  
**Team:** Shruthi Saravanan, Sharvani Bhavanam, CharanSai Kamsani

## Project Overview
We compare the robustness of Vision Transformers (ViT) vs CNNs (ResNet-18) 
under realistic image corruptions (Gaussian noise, blur, rotation, brightness) 
on CIFAR-10.

## Setup Instructions

### 1. Clone this repo
```bash
git clone https://github.com/YourUsername/ViT-Robustness.git
cd ViT-Robustness
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run training
```bash
# Train CNN baseline
python src/train.py --config configs/cnn_config.yaml

# Train ViT
python src/train.py --config configs/vit_config.yaml
```

### 4. Run evaluation
```bash
python src/evaluate.py --config configs/cnn_config.yaml
python src/evaluate.py --config configs/vit_config.yaml
```

### 5. Generate plots
```bash
python src/plot_results.py
```

## Outputs
All results saved to `results/metrics/` and `results/plots/`

## Hardware
Trained on Google Colab T4 GPU. CNN: ~45 min. ViT: ~2 hrs.

## Random Seed
All experiments use `seed = 42` for reproducibility.

## Checkpoints
Saved model checkpoints available at: [add your Google Drive link here]
