import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DifferentiableISP(nn.Module):
    """
    Differentiable Image Signal Processor with 19 parameters
    Based on CRISP paper: Learning Controllable ISP for Image Enhancement
    """
    
    def __init__(self):
        super(DifferentiableISP, self).__init__()
        
        # Initialize default ISP parameters
        self.register_buffer('default_params', self._get_default_params())
        
    def _get_default_params(self):
        """Default ISP parameters for initialization"""
        params = torch.zeros(19)
        
        # Digital gain (1 param)
        params[0] = 1.0
        
        # White balance R,B (2 params) 
        params[1] = 1.0  # R gain
        params[2] = 1.0  # B gain
        
        # Color correction matrix (9 params) - identity
        params[3:12] = torch.tensor([1,0,0,0,1,0,0,0,1], dtype=torch.float32)
        
        # Color offset (3 params)
        params[12:15] = 0.0
        
        # Gamma (1 param)
        params[15] = 2.2
        
        # Tone mapping (3 params)
        params[16] = 1.0  # s
        params[17] = 1.0  # p1  
        params[18] = 1.0  # p2
        
        return params
    
    def forward(self, x, params):
        """
        Apply ISP pipeline to input image
        
        Args:
            x: Input image [B,3,H,W] in range [0,1]
            params: ISP parameters [B,19] or [19]
            
        Returns:
            Enhanced image [B,3,H,W]
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)
            
        B, C, H, W = x.shape
        
        # 1. Digital gain
        dg = params[:, 0:1].view(B, 1, 1, 1)
        x = x * dg
        
        # 2. White balance
        wb_r = params[:, 1:2].view(B, 1, 1, 1)
        wb_b = params[:, 2:3].view(B, 1, 1, 1)
        
        x_wb = x.clone()
        x_wb[:, 0:1] = x[:, 0:1] * wb_r
        x_wb[:, 2:3] = x[:, 2:3] * wb_b
        x = x_wb
        
        # 3. Color correction matrix + offset
        ccm = params[:, 3:12].view(B, 3, 3)
        offset = params[:, 12:15].view(B, 3, 1, 1)
        
        # Reshape for matrix multiplication
        x_flat = x.permute(0, 2, 3, 1).reshape(B, -1, 3)
        x_corrected = torch.bmm(x_flat, ccm.transpose(1, 2))
        x = x_corrected.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        x = x + offset
        
        # 4. Gamma correction
        gamma = params[:, 15:16].view(B, 1, 1, 1)
        x = torch.clamp(x, 1e-8, 1.0)
        x = torch.pow(x, 1.0 / gamma)
        
        # 5. Tone mapping
        s = params[:, 16:17].view(B, 1, 1, 1)
        p1 = params[:, 17:18].view(B, 1, 1, 1)
        p2 = params[:, 18:19].view(B, 1, 1, 1)
        
        x = torch.clamp(x, 1e-8, 1.0)
        x1 = torch.pow(x, p1)
        x2 = torch.pow(x, p2)
        x = s * x1 - (s - 1.0) * x2
        
        return torch.clamp(x, 0.0, 1.0)
    
    def get_param_names(self):
        """Get parameter names for visualization"""
        return [
            'digital_gain',
            'wb_r', 'wb_b',
            'ccm_00', 'ccm_01', 'ccm_02',
            'ccm_10', 'ccm_11', 'ccm_12', 
            'ccm_20', 'ccm_21', 'ccm_22',
            'offset_r', 'offset_g', 'offset_b',
            'gamma',
            'tone_s', 'tone_p1', 'tone_p2'
        ]