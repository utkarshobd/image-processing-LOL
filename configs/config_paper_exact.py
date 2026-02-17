# PAPER EXACT Configuration
# Section IV-B: Training Details

import os

# PAPER SPEC: Style dimension D=3
style_dim = 3
use_lq_encoder = False

# Dataset paths (LOL instead of MIT-FiveK)
dataset_type = 'custom'
lq_train_dir = 'datasets/LOL_our485/low'
hq_train_dir = 'datasets/LOL_our485/high'
lq_val_dir = 'datasets/LOL_eval15/low'
hq_val_dir = 'datasets/LOL_eval15/high'

# PAPER SPEC: Training parameters
# Iterations: 1.6×10^5
# Batch size: 16
# Crop size: 200×200
num_epochs = 330  # ~160k iterations with 485 images
batch_size = 16
learning_rate = 1e-4
weight_decay = 0.0  # Not mentioned in paper

# PAPER SPEC: LR schedule - halve every 25% training
# At 40k, 80k, 120k iterations
# With 485 images, batch 16: ~30 images/iter
# 160k iter = ~5333 epochs (but we use 330 for practical training)
lr_milestones = [82, 165, 247]  # 25%, 50%, 75% of 330 epochs
lr_gamma = 0.5

# PAPER SPEC: Loss = MSE only
# No perceptual, no GAN, no SSIM

# PAPER SPEC: Data augmentation
# Random crop 200×200
# Flip + rotation
crop_size = 200
use_augmentation = True

# Data loading
num_workers = 4

# Output directories
checkpoint_dir = 'checkpoints_lol_paper_exact'
output_dir = 'outputs_lol_paper_exact'
log_dir = 'logs_lol_paper_exact'

# Create directories
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Validation frequency
val_every = 10  # epochs
save_every = 10  # epochs
