import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleEncoder(nn.Module):
    """
    PAPER EXACT: 12 convolution layers, 64 channels, ResNet-style
    Section III-B: Encoder f
    """
    
    def __init__(self, style_dim=3, input_channels=6):
        super(StyleEncoder, self).__init__()
        
        self.style_dim = style_dim
        
        # PAPER SPEC: 12 conv layers, 64 channels
        # ResNet-style with skip connections
        
        # Initial conv
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        
        # 11 more conv layers in residual blocks
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv10 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv11 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # FC to style vector
        self.fc = nn.Linear(64, style_dim)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, lq, hq):
        """
        Args:
            lq: Low quality [B,3,H,W]
            hq: High quality [B,3,H,W]
        Returns:
            style: [B,3] non-negative
        """
        # Concatenate LQ + HQ
        x = torch.cat([lq, hq], dim=1)  # [B,6,H,W]
        
        # 12 conv layers with residual connections
        x = self.relu(self.conv1(x))
        
        # Block 1
        identity = x
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.relu(x + identity)
        
        # Block 2
        identity = x
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.relu(x + identity)
        
        # Block 3
        identity = x
        x = self.relu(self.conv6(x))
        x = self.conv7(x)
        x = self.relu(x + identity)
        
        # Block 4
        identity = x
        x = self.relu(self.conv8(x))
        x = self.conv9(x)
        x = self.relu(x + identity)
        
        # Block 5
        identity = x
        x = self.relu(self.conv10(x))
        x = self.conv11(x)
        x = self.relu(x + identity)
        
        # Final conv
        x = self.relu(self.conv12(x))
        
        # Global pooling
        x = self.global_pool(x)  # [B,64,1,1]
        x = x.view(x.size(0), -1)  # [B,64]
        
        # FC to style
        style = self.fc(x)  # [B,3]
        
        # Non-negative (paper requirement) - use Softplus for unbounded growth
        style = F.softplus(style)
        
        return style

class LQOnlyEncoder(nn.Module):
    """
    LQ-only encoder for inference (not in paper, but useful)
    """
    
    def __init__(self, style_dim=3):
        super(LQOnlyEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, style_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, lq):
        x = self.relu(self.conv1(lq))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        style = self.fc(x)
        style = F.softplus(style)  # Non-negative with growth
        
        return style
