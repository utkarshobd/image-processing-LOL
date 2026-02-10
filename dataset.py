import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import cv2

class CRISPDataset(Dataset):
    """
    Dataset for CRISP training
    Loads (LQ, HQ) image pairs
    """
    
    def __init__(self, lq_dir, hq_dir, crop_size=256, mode='train'):
        self.lq_dir = lq_dir
        self.hq_dir = hq_dir
        self.crop_size = crop_size
        self.mode = mode
        
        # Get image pairs
        self.image_pairs = self._get_image_pairs()
        
        # Transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ])
    
    def _get_image_pairs(self):
        """Get list of (LQ, HQ) image pairs"""
        lq_images = sorted([f for f in os.listdir(self.lq_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
        
        pairs = []
        for lq_name in lq_images:
            hq_name = lq_name  # Assuming same filename
            lq_path = os.path.join(self.lq_dir, lq_name)
            hq_path = os.path.join(self.hq_dir, hq_name)
            
            if os.path.exists(hq_path):
                pairs.append((lq_path, hq_path))
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        lq_path, hq_path = self.image_pairs[idx]
        
        # Load images
        lq_img = self._load_image(lq_path)
        hq_img = self._load_image(hq_path)
        
        # Ensure same size
        if lq_img.size != hq_img.size:
            hq_img = hq_img.resize(lq_img.size, Image.LANCZOS)
        
        # Apply transforms with same random seed for paired data
        if self.mode == 'train':
            # Use same random crop for both images
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            lq_tensor = self.transform(lq_img)
            
            torch.manual_seed(seed)
            hq_tensor = self.transform(hq_img)
        else:
            lq_tensor = self.transform(lq_img)
            hq_tensor = self.transform(hq_img)
        
        return {
            'lq': lq_tensor,
            'hq': hq_tensor,
            'lq_path': lq_path,
            'hq_path': hq_path
        }
    
    def _load_image(self, path):
        """Load image with proper handling of different formats"""
        if path.lower().endswith(('.tif', '.tiff')):
            # Handle TIFF files (potentially 16-bit)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img.dtype == np.uint16:
                img = (img / 65535.0 * 255).astype(np.uint8)
            elif img.dtype == np.float32:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            img = Image.open(path).convert('RGB')
        
        return img

class FiveKDataset(CRISPDataset):
    """
    Specialized dataset for MIT-Adobe FiveK
    """
    
    def __init__(self, data_root, expert='C', split='train', crop_size=256):
        """
        Args:
            data_root: Root directory of FiveK dataset
            expert: Expert photographer ('A', 'B', 'C', 'D', 'E')
            split: 'train' or 'test'
            crop_size: Crop size for training
        """
        self.data_root = data_root
        self.expert = expert
        self.split = split
        
        # FiveK directory structure
        lq_dir = os.path.join(data_root, 'input')
        hq_dir = os.path.join(data_root, f'expert{expert}')
        
        super().__init__(lq_dir, hq_dir, crop_size, split)
        
        # Load train/test split
        self._load_split()
    
    def _load_split(self):
        """Load train/test split indices"""
        if self.split == 'train':
            split_file = os.path.join(self.data_root, 'train_input.txt')
        else:
            split_file = os.path.join(self.data_root, 'test_input.txt')
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                valid_names = set(line.strip() for line in f)
            
            # Filter image pairs based on split
            self.image_pairs = [
                (lq_path, hq_path) for lq_path, hq_path in self.image_pairs
                if os.path.basename(lq_path).split('.')[0] in valid_names
            ]

def create_dataloaders(config):
    """
    Create train and validation dataloaders
    
    Args:
        config: Configuration object
        
    Returns:
        train_loader, val_loader
    """
    if config.dataset_type == 'fivek':
        train_dataset = FiveKDataset(
            data_root=config.data_root,
            expert=config.expert,
            split='train',
            crop_size=config.crop_size
        )
        
        val_dataset = FiveKDataset(
            data_root=config.data_root,
            expert=config.expert,
            split='test',
            crop_size=config.crop_size
        )
    else:
        train_dataset = CRISPDataset(
            lq_dir=config.lq_train_dir,
            hq_dir=config.hq_train_dir,
            crop_size=config.crop_size,
            mode='train'
        )
        
        val_dataset = CRISPDataset(
            lq_dir=config.lq_val_dir,
            hq_dir=config.hq_val_dir,
            crop_size=config.crop_size,
            mode='val'
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader