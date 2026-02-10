# Configuration for LOL Dataset - Ready to Train
import os

# Model parameters
style_dim = 3
use_lq_encoder = False

# LOL Dataset paths (corrected)
dataset_type = 'custom'
lq_train_dir = 'datasets/LOL_our485/low'
hq_train_dir = 'datasets/LOL_our485/high'
lq_val_dir = 'datasets/LOL_eval15/low'
hq_val_dir = 'datasets/LOL_eval15/high'

# Training parameters
num_epochs = 300
batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-5

# Learning rate scheduler
lr_step_size = 75
lr_gamma = 0.5

# Loss weights
mse_weight = 1.0
style_reg_weight = 0.005

# Data loading
crop_size = 256
num_workers = 2

# Output directories
checkpoint_dir = 'checkpoints_lol'
output_dir = 'outputs_lol'
log_dir = 'logs_lol'

# Create directories
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)