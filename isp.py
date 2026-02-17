import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableISP(nn.Module):
    """
    PAPER EXACT: Differentiable ISP with 19 parameters
    Section III-C: ISP functions
    
    Order (MUST NOT CHANGE):
    1. Digital Gain
    2. White Balance
    3. Color Correction Matrix + Offset
    4. Gamma Correction
    5. Tone Mapping
    """
    
    def __init__(self):
        super(DifferentiableISP, self).__init__()
        
        # PAPER EXACT initialization (Sec III-C-6)
        self.register_buffer('default_params', self._get_default_params())
        
    def _get_default_params(self):
        """
        PAPER EXACT initialization values
        """
        params = torch.zeros(19)
        
        # Digital gain: φ_dg = 1.2
        params[0] = 1.2
        
        # White balance: identity (R=1, B=1)
        params[1] = 1.0  # R
        params[2] = 1.0  # B
        
        # CCM: identity matrix
        params[3:12] = torch.tensor([1,0,0, 0,1,0, 0,0,1], dtype=torch.float32)
        
        # Offsets: zero
        params[12:15] = 0.0
        
        # Gamma: φ_γ = 1/2.2
        params[15] = 1.0 / 2.2
        
        # Tone mapping: φ_s=3, φ_p1=2, φ_p2=3
        params[16] = 3.0  # s
        params[17] = 2.0  # p1
        params[18] = 3.0  # p2
        
        return params
    
    def forward(self, x, params):
        """
        Apply ISP pipeline
        
        Args:
            x: [B,3,H,W] in [0,1]
            params: [B,19]
        Returns:
            enhanced: [B,3,H,W]
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)
            
        B, C, H, W = x.shape
        
        # 1. Digital Gain (Eq. in paper)
        dg = params[:, 0:1].view(B, 1, 1, 1)
        x = x * dg
        
        # 2. White Balance (per-channel)
        wb_r = params[:, 1:2].view(B, 1, 1, 1)
        wb_b = params[:, 2:3].view(B, 1, 1, 1)
        
        x_wb = x.clone()
        x_wb[:, 0:1] = x[:, 0:1] * wb_r  # R channel
        x_wb[:, 2:3] = x[:, 2:3] * wb_b  # B channel
        x = x_wb
        
        # 3. Color Correction Matrix + Offset
        # CRITICAL: CCM row sum constraint (Sec III-C)
        ccm = params[:, 3:12].view(B, 3, 3)
        
        # Enforce Σφᵢⱼ = 1 for each row
        row_sums = ccm.sum(dim=2, keepdim=True) + 1e-8
        ccm = ccm / row_sums
        
        offset = params[:, 12:15].view(B, 3, 1, 1)
        
        # Apply CCM: M * x + o
        x_flat = x.permute(0, 2, 3, 1).reshape(B, -1, 3)
        x_corrected = torch.bmm(x_flat, ccm.transpose(1, 2))
        x = x_corrected.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        x = x + offset
        
        # 4. Gamma Correction: max(x, 1e-8)^φ_γ
        gamma = params[:, 15:16].view(B, 1, 1, 1)
        x = torch.clamp(x, 1e-8, 1.0)
        x = torch.pow(x, gamma)
        
        # 5. Tone Mapping: φ_s * x^φ_p1 - (φ_s-1) * x^φ_p2
        s = params[:, 16:17].view(B, 1, 1, 1)
        p1 = params[:, 17:18].view(B, 1, 1, 1)
        p2 = params[:, 18:19].view(B, 1, 1, 1)
        
        x = torch.clamp(x, 1e-8, 1.0)
        x1 = torch.pow(x, p1)
        x2 = torch.pow(x, p2)
        x = s * x1 - (s - 1.0) * x2
        
        return torch.clamp(x, 0.0, 1.0)
    
    def get_param_names(self):
        """Parameter names for visualization"""
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
