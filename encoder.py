import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleEncoder(nn.Module):
    """
    Style Encoder for CRISP
    Encodes (LQ, HQ) image pairs into low-dimensional style vectors
    """
    
    def __init__(self, style_dim=3, input_channels=6):
        super(StyleEncoder, self).__init__()
        
        self.style_dim = style_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, style_dim)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, lq, hq):
        """
        Forward pass
        
        Args:
            lq: Low quality image [B,3,H,W]
            hq: High quality image [B,3,H,W]
            
        Returns:
            style: Style vector [B,style_dim]
        """
        # Concatenate LQ and HQ
        x = torch.cat([lq, hq], dim=1)  # [B,6,H,W]
        
        # Convolutional layers with downsampling
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # [B,32,H/2,W/2]
        
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # [B,64,H/4,W/4]
        
        x = self.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # [B,64,H/8,W/8]
        
        x = self.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)  # [B,128,H/16,W/16]
        
        # Global average pooling
        x = self.global_pool(x)  # [B,128,1,1]
        x = x.view(x.size(0), -1)  # [B,128]
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        style = self.fc2(x)
        
        # Ensure non-negative (as mentioned in paper)
        style = F.relu(style)
        
        return style

class LQOnlyEncoder(nn.Module):
    """
    Optional LQ-only encoder for inference when HQ is not available
    """
    
    def __init__(self, style_dim=3):
        super(LQOnlyEncoder, self).__init__()
        
        self.style_dim = style_dim
        
        # Lighter architecture for LQ-only
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, style_dim)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, lq):
        """
        Forward pass for LQ-only encoding
        
        Args:
            lq: Low quality image [B,3,H,W]
            
        Returns:
            style: Style vector [B,style_dim]
        """
        x = lq
        
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        style = self.fc2(x)
        
        style = F.relu(style)
        
        return style