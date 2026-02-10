import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os

def save_comparison_images(lq, hq, pred, save_path, max_images=4):
    """
    Save comparison images (LQ, HQ, Predicted)
    
    Args:
        lq: Low quality images [B,C,H,W]
        hq: High quality images [B,C,H,W]
        pred: Predicted images [B,C,H,W]
        save_path: Path to save the comparison
        max_images: Maximum number of images to show
    """
    batch_size = min(lq.shape[0], max_images)
    
    # Convert to PIL images
    to_pil = transforms.ToPILImage()
    
    images = []
    for i in range(batch_size):
        lq_img = to_pil(lq[i].cpu())
        hq_img = to_pil(hq[i].cpu())
        pred_img = to_pil(pred[i].cpu())
        
        images.extend([lq_img, hq_img, pred_img])
    
    # Create grid
    grid_img = create_image_grid(images, cols=3, labels=['LQ', 'HQ', 'Predicted'])
    grid_img.save(save_path)

def save_enhancement_grid(images, save_path, grid_size=3):
    """
    Save a grid of enhanced images with different styles
    
    Args:
        images: List of PIL images
        save_path: Path to save the grid
        grid_size: Size of the grid
    """
    # Pad images list if needed
    total_slots = grid_size * grid_size
    while len(images) < total_slots:
        images.append(images[-1])  # Repeat last image
    
    images = images[:total_slots]  # Trim if too many
    
    # Create labels
    labels = ['Original'] + [f'Style {i}' for i in range(1, len(images))]
    
    grid_img = create_image_grid(images, cols=grid_size, labels=labels)
    grid_img.save(save_path)

def create_image_grid(images, cols=3, labels=None, img_size=(256, 256)):
    """
    Create a grid of images
    
    Args:
        images: List of PIL images
        cols: Number of columns
        labels: List of labels for each image
        img_size: Size to resize images to
        
    Returns:
        grid_img: PIL Image of the grid
    """
    if not images:
        return None
    
    # Resize all images
    resized_images = [img.resize(img_size, Image.LANCZOS) for img in images]
    
    # Calculate grid dimensions
    rows = (len(resized_images) + cols - 1) // cols
    
    # Create grid
    grid_width = cols * img_size[0]
    grid_height = rows * img_size[1]
    
    # Add space for labels if provided
    label_height = 30 if labels else 0
    grid_height += label_height * rows
    
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        
        x = col * img_size[0]
        y = row * (img_size[1] + label_height)
        
        grid_img.paste(img, (x, y))
        
        # Add label if provided
        if labels and idx < len(labels):
            draw = ImageDraw.Draw(grid_img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            label_y = y + img_size[1] + 5
            draw.text((x + 5, label_y), labels[idx], fill='black', font=font)
    
    return grid_img

def plot_training_curves(log_dir, save_path):
    """
    Plot training curves from tensorboard logs
    
    Args:
        log_dir: Directory containing tensorboard logs
        save_path: Path to save the plot
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Load tensorboard data
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        # Get scalar data
        train_loss = event_acc.Scalars('Epoch/Train_Loss')
        val_loss = event_acc.Scalars('Epoch/Val_Loss')
        train_psnr = event_acc.Scalars('Epoch/Train_PSNR')
        val_psnr = event_acc.Scalars('Epoch/Val_PSNR')
        
        # Extract values
        epochs = [x.step for x in train_loss]
        train_loss_vals = [x.value for x in train_loss]
        val_loss_vals = [x.value for x in val_loss]
        train_psnr_vals = [x.value for x in train_psnr]
        val_psnr_vals = [x.value for x in val_psnr]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(epochs, train_loss_vals, label='Train Loss', color='blue')
        ax1.plot(epochs, val_loss_vals, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # PSNR plot
        ax2.plot(epochs, train_psnr_vals, label='Train PSNR', color='blue')
        ax2.plot(epochs, val_psnr_vals, label='Val PSNR', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Training and Validation PSNR')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_path}")
        
    except ImportError:
        print("tensorboard not available for plotting")
    except Exception as e:
        print(f"Error plotting training curves: {e}")

def visualize_isp_parameters(params_dict, save_path):
    """
    Visualize ISP parameters as a bar chart
    
    Args:
        params_dict: Dictionary of parameter names and values
        save_path: Path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    names = list(params_dict.keys())
    values = list(params_dict.values())
    
    bars = ax.bar(names, values)
    
    # Color bars based on parameter type
    colors = []
    for name in names:
        if 'gain' in name or 'wb' in name:
            colors.append('skyblue')
        elif 'ccm' in name:
            colors.append('lightcoral')
        elif 'offset' in name:
            colors.append('lightgreen')
        elif 'gamma' in name:
            colors.append('gold')
        elif 'tone' in name:
            colors.append('plum')
        else:
            colors.append('lightgray')
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('ISP Parameters')
    ax.set_ylabel('Parameter Values')
    ax.set_title('ISP Parameter Values')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ISP parameters visualization saved to {save_path}")

def create_style_space_visualization(style_vectors, labels=None, save_path=None):
    """
    Create 2D visualization of style space using PCA
    
    Args:
        style_vectors: Array of style vectors [N, style_dim]
        labels: Optional labels for each vector
        save_path: Path to save the plot
        
    Returns:
        fig: Matplotlib figure
    """
    from sklearn.decomposition import PCA
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    style_2d = pca.fit_transform(style_vectors)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(style_2d[mask, 0], style_2d[mask, 1], 
                      c=[colors[i]], label=label, alpha=0.7)
        
        ax.legend()
    else:
        ax.scatter(style_2d[:, 0], style_2d[:, 1], alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('Style Space Visualization (PCA)')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Style space visualization saved to {save_path}")
    
    return fig