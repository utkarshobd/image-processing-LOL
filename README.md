# CRISP Low-Light Image Enhancement - Training Repository

## Quick Start Training

Clone and train in 3 commands:

```bash
git clone https://github.com/utkarshobd/image-processing-LOL.git
cd image-processing-LOL
python train.py --config configs/config_lol_ready.py
```

## What's Included

### Essential Training Files
- `train.py` - Main training script
- `model.py` - CRISP model architecture
- `encoder.py` - Encoder network
- `decoder.py` - Decoder network
- `isp.py` - Image signal processing
- `dataset.py` - Dataset loader
- `configs/config_lol_ready.py` - Training configuration

### Dataset
- `datasets/LOL_our485/` - Training data (485 paired images)
  - `low/` - Low-light input images
  - `high/` - Normal-light target images
- `datasets/LOL_eval15/` - Evaluation data (15 test images)

### Utilities
- `utils/metrics.py` - PSNR/SSIM calculation
- `utils/visualization.py` - Result visualization

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- torch >= 1.7.0
- torchvision
- opencv-python
- pillow
- scikit-image
- matplotlib
- tqdm
- tensorboard

## Training

### Basic Training
```bash
python train.py --config configs/config_lol_ready.py
```

### Resume Training
```bash
python train.py --config configs/config_lol_ready.py --resume checkpoints_lol/latest.pth
```

### Monitor Training
```bash
tensorboard --logdir logs_lol
```

## Training Configuration

Edit `configs/config_lol_ready.py` to customize:

```python
# Dataset
train_dir = 'datasets/LOL_our485'
val_dir = 'datasets/LOL_eval15'

# Training
num_epochs = 300
batch_size = 4
learning_rate = 1e-4

# Model
style_dim = 3
use_lq_encoder = False
```

## Output Structure

After training:
```
checkpoints_lol/
├── latest.pth          # Latest checkpoint
├── best.pth            # Best model (highest PSNR)
└── style_presets.pth   # Style presets

logs_lol/               # TensorBoard logs
outputs_lol/            # Validation visualizations
```

## Expected Results

- Training time: 2-4 hours on GPU
- Expected PSNR: 20-22 dB on LOL dataset
- Model size: ~5MB

## Dataset Details

### LOL Dataset
- Training: 485 paired low/normal-light images
- Evaluation: 15 test images
- Resolution: 400x600 pixels
- Format: PNG

## Hardware Requirements

- GPU: NVIDIA GPU with 4GB+ VRAM (recommended)
- RAM: 8GB+
- Storage: 2GB for dataset + models

## Citation

If you use this code, please cite the LOL dataset:

```
@inproceedings{wei2018deep,
  title={Deep retinex decomposition for low-light enhancement},
  author={Wei, Chen and Wang, Wenjing and Yang, Wenhan and Liu, Jiaying},
  booktitle={BMVC},
  year={2018}
}
```