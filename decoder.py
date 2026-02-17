import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleDecoder(nn.Module):
    """
    PAPER EXACT: 5 FC layers, 64 hidden units
    Section III-B: Style decoder g
    Outputs ISP parameter RESIDUALS
    """
    
    def __init__(self, style_dim=3, isp_params=19):
        super(StyleDecoder, self).__init__()
        
        # PAPER SPEC: 5 FC layers, 64 units each
        self.fc1 = nn.Linear(style_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, isp_params)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Small initialization for residual learning
        self._init_weights()
        
    def _init_weights(self):
        """Small initialization for stable residual learning"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, style):
        """
        Args:
            style: [B,3]
        Returns:
            residuals: [B,19] scaled for stability
        """
        x = self.relu(self.fc1(style))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        residuals = self.fc5(x)
        
        # Scale residuals for stable training (implicit in paper)
        residuals = 0.1 * residuals
        
        return residuals

class ParameterConstraints:
    """
    Apply constraints to ISP parameters
    PAPER: φ = φ_init + residual
    """
    
    @staticmethod
    def apply_constraints(params_residual, default_params):
        """
        Args:
            params_residual: [B,19]
            default_params: [19]
        Returns:
            params: [B,19] constrained
        """
        B = params_residual.shape[0]
        default = default_params.unsqueeze(0).expand(B, -1)
        
        # Residual learning: φ = φ_init + Δφ
        params = default + params_residual
        
        # Apply observed ranges from paper
        
        # Digital gain: [0.85, 2.17]
        params[:, 0] = torch.clamp(params[:, 0], 0.85, 2.17)
        
        # White balance R: [0.73, 1.07]
        params[:, 1] = torch.clamp(params[:, 1], 0.73, 1.07)
        
        # White balance B: [0.80, 2.41]
        params[:, 2] = torch.clamp(params[:, 2], 0.80, 2.41)
        
        # CCM: sum constraint handled in ISP
        # Just clamp to reasonable range
        params[:, 3:12] = torch.clamp(params[:, 3:12], -2.0, 2.0)
        
        # Offsets: small values
        params[:, 12:15] = torch.clamp(params[:, 12:15], -0.2, 0.2)
        
        # Gamma: typically < 1
        params[:, 15] = torch.clamp(params[:, 15], 0.1, 2.0)
        
        # Tone mapping: positive
        params[:, 16:19] = torch.clamp(params[:, 16:19], 0.1, 5.0)
        
        return params
