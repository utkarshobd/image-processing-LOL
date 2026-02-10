import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images
    
    Args:
        img1, img2: Images as tensors [B,C,H,W] or [C,H,W]
        max_val: Maximum pixel value
        
    Returns:
        psnr: PSNR value
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images
    
    Args:
        img1, img2: Images as tensors [B,C,H,W] or [C,H,W]
        
    Returns:
        ssim_val: SSIM value
    """
    # Convert to numpy
    if img1.dim() == 4:
        img1 = img1.squeeze(0)
        img2 = img2.squeeze(0)
    
    # Handle batch dimension properly
    if img1.dim() == 4:
        # Take first image from batch
        img1 = img1[0]
        img2 = img2[0]
    
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    
    # Calculate SSIM with appropriate window size
    try:
        ssim_val = ssim(img1_np, img2_np, multichannel=True, data_range=1.0)
    except ValueError:
        # If image too small, use smaller window
        min_dim = min(img1_np.shape[0], img1_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        ssim_val = ssim(img1_np, img2_np, multichannel=True, data_range=1.0, win_size=win_size)
    
    return ssim_val

def calculate_lpips(img1, img2, lpips_model):
    """
    Calculate LPIPS perceptual distance
    
    Args:
        img1, img2: Images as tensors [B,C,H,W]
        lpips_model: Pre-loaded LPIPS model
        
    Returns:
        lpips_val: LPIPS value
    """
    # Normalize to [-1, 1] for LPIPS
    img1_norm = img1 * 2.0 - 1.0
    img2_norm = img2 * 2.0 - 1.0
    
    lpips_val = lpips_model(img1_norm, img2_norm)
    return lpips_val.item()

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    numpy_img = tensor.permute(1, 2, 0).cpu().numpy()
    numpy_img = np.clip(numpy_img, 0, 1)
    
    return numpy_img

def numpy_to_tensor(numpy_img, device='cpu'):
    """Convert numpy array to tensor"""
    if numpy_img.ndim == 3:
        tensor = torch.from_numpy(numpy_img).permute(2, 0, 1).float()
    else:
        tensor = torch.from_numpy(numpy_img).float()
    
    return tensor.to(device)

class MetricsCalculator:
    """
    Class for calculating various image quality metrics
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize LPIPS model if available
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.has_lpips = True
        except ImportError:
            self.lpips_model = None
            self.has_lpips = False
            print("LPIPS not available. Install with: pip install lpips")
    
    def calculate_all_metrics(self, pred, target):
        """
        Calculate all available metrics
        
        Args:
            pred: Predicted image [B,C,H,W]
            target: Target image [B,C,H,W]
            
        Returns:
            metrics: Dictionary of metrics
        """
        metrics = {}
        
        # PSNR
        metrics['psnr'] = calculate_psnr(pred, target)
        
        # SSIM
        metrics['ssim'] = calculate_ssim(pred, target)
        
        # LPIPS (if available)
        if self.has_lpips:
            metrics['lpips'] = calculate_lpips(pred, target, self.lpips_model)
        
        # MSE
        metrics['mse'] = F.mse_loss(pred, target).item()
        
        # MAE
        metrics['mae'] = F.l1_loss(pred, target).item()
        
        return metrics

def batch_metrics(pred_batch, target_batch, metrics_calculator):
    """
    Calculate metrics for a batch of images
    
    Args:
        pred_batch: Predicted images [B,C,H,W]
        target_batch: Target images [B,C,H,W]
        metrics_calculator: MetricsCalculator instance
        
    Returns:
        avg_metrics: Dictionary of average metrics
    """
    batch_size = pred_batch.shape[0]
    all_metrics = []
    
    for i in range(batch_size):
        pred = pred_batch[i:i+1]
        target = target_batch[i:i+1]
        
        metrics = metrics_calculator.calculate_all_metrics(pred, target)
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics