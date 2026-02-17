import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from tqdm import tqdm
import numpy as np

from model import CRISP, CRISPLoss
from dataset import create_dataloaders
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import save_comparison_images

class CRISPTrainer:
    """
    Trainer for CRISP model
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = CRISP(
            style_dim=config.style_dim,
            use_lq_encoder=config.use_lq_encoder
        ).to(self.device)
        
        # Loss function - PAPER EXACT: MSE ONLY
        self.criterion = CRISPLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler - PAPER EXACT: halve every 25%
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.lr_milestones,
            gamma=config.lr_gamma
        )
        
        # Data loaders
        self.train_loader, self.val_loader = create_dataloaders(config)
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.epoch = 0
        self.best_psnr = 0.0
        self.style_vectors = []  # Store for preset generation
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            lq = batch['lq'].to(self.device)
            hq = batch['hq'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output, style_vec = self.model(lq, hq, mode='train')
            
            # Compute loss
            loss, loss_dict = self.criterion(output, hq, style_vec)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            psnr = calculate_psnr(output, hq)
            
            # Update running averages
            total_loss += loss.item()
            total_psnr += psnr
            
            # Store style vectors for preset generation
            if batch_idx % 10 == 0:  # Sample every 10 batches
                self.style_vectors.extend(style_vec.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PSNR': f'{psnr:.2f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to tensorboard
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/PSNR', psnr, global_step)
            
            for key, value in loss_dict.items():
                self.writer.add_scalar(f'Train/{key}', value.item(), global_step)
        
        # Epoch averages
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        
        return avg_loss, avg_psnr
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(self.val_loader)
        
        # Collect outputs for diverse style visualization
        sample_outputs = []
        sample_lq = None
        sample_hq = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                lq = batch['lq'].to(self.device)
                hq = batch['hq'].to(self.device)
                
                # Forward pass
                output, style_vec = self.model(lq, hq, mode='train')
                
                # Compute loss
                loss, _ = self.criterion(output, hq, style_vec)
                
                # Metrics
                psnr = calculate_psnr(output, hq)
                ssim = 0.0  # Skip SSIM for now to avoid validation errors
                
                total_loss += loss.item()
                total_psnr += psnr
                total_ssim += ssim
                
                # Save first batch for diverse style generation
                if batch_idx == 0:
                    sample_lq = lq[0:1]  # Take first image
                    sample_hq = hq[0:1]
                    sample_outputs.append(output[0:1])
            
            # Generate diverse styles for the same image
            if sample_lq is not None:
                # Generate 5 different styles using MORE diverse random style vectors
                for i in range(5):
                    # More diverse random style vector (wider range)
                    random_style = torch.rand(1, self.config.style_dim).to(self.device) * 5.0 + torch.randn(1, self.config.style_dim).to(self.device) * 0.5
                    random_style = torch.clamp(random_style, 0.1, 10.0)  # Wider range
                    diverse_output, _ = self.model(sample_lq, style=random_style, mode='inference_manual')
                    sample_outputs.append(diverse_output)
                
                # Save comparison with diverse styles
                self._save_diverse_styles(sample_lq, sample_hq, sample_outputs)
        
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        
        return avg_loss, avg_psnr, avg_ssim
    
    def _save_diverse_styles(self, lq, hq, outputs):
        """Save comparison of diverse styles for the same image"""
        from torchvision.utils import make_grid
        import torchvision.transforms as T
        
        # Prepare images
        all_images = [lq, hq] + outputs
        grid = make_grid(torch.cat(all_images, dim=0), nrow=len(all_images), padding=2, normalize=False)
        
        # Save
        save_path = os.path.join(self.config.output_dir, f'epoch_{self.epoch}.jpg')
        T.ToPILImage()(grid.cpu()).save(save_path)
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        # Extract only serializable config attributes
        config_dict = {
            'style_dim': getattr(self.config, 'style_dim', 3),
            'use_lq_encoder': getattr(self.config, 'use_lq_encoder', False),
            'learning_rate': getattr(self.config, 'learning_rate', 1e-4),
            'num_epochs': getattr(self.config, 'num_epochs', 300),
            'batch_size': getattr(self.config, 'batch_size', 4)
        }
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'config': config_dict
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'best.pth'))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_psnr = checkpoint['best_psnr']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def generate_style_presets(self, k=10):
        """Generate style presets using K-means"""
        if len(self.style_vectors) == 0:
            print("No style vectors collected yet")
            return
        
        style_array = np.array(self.style_vectors)
        presets = self.model.get_style_presets(
            torch.tensor(style_array, dtype=torch.float32), k=k
        )
        
        # Save presets
        torch.save(presets, os.path.join(self.config.checkpoint_dir, 'style_presets.pth'))
        print(f"Generated {k} style presets")
        
        return presets
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Start from current epoch (for resuming)
        start_epoch = self.epoch + 1 if hasattr(self, 'epoch') and self.epoch > 0 else 0
        print(f"Starting from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_psnr = self.train_epoch()
            
            # Validate
            val_loss, val_psnr, val_ssim = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch results
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}")
            print(f"           Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}")
            
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_PSNR', train_psnr, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Val_PSNR', val_psnr, epoch)
            self.writer.add_scalar('Epoch/Val_SSIM', val_ssim, epoch)
            
            # Save checkpoint
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
            
            self.save_checkpoint(is_best)
            
            # Generate presets periodically
            if (epoch + 1) % 20 == 0:
                self.generate_style_presets()
        
        print(f"Training completed. Best PSNR: {self.best_psnr:.2f}")
        
        # Final preset generation
        self.generate_style_presets()
        
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train CRISP model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Create trainer
    trainer = CRISPTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()