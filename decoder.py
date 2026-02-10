import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleDecoder(nn.Module):
    """
    Style Decoder for CRISP
    Converts low-dimensional style vectors to ISP parameters
    """
    
    def __init__(self, style_dim=3, isp_params=19):
        super(StyleDecoder, self).__init__()
        
        self.style_dim = style_dim
        self.isp_params = isp_params
        
        # Multi-layer perceptron
        self.fc1 = nn.Linear(style_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, isp_params)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize with small weights for residual learning
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, style):
        """
        Forward pass
        
        Args:
            style: Style vector [B,style_dim]
            
        Returns:
            params: ISP parameter residuals [B,isp_params]
        """
        x = style
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        
        # Output residual parameters (small values)
        params_residual = self.fc4(x)
        
        return params_residual

class ParameterConstraints:
    """
    Apply constraints to ISP parameters to keep them in valid ranges
    """
    
    @staticmethod
    def apply_constraints(params_residual, default_params):
        """
        Apply constraints to parameter residuals
        
        Args:
            params_residual: Residual parameters [B,19]
            default_params: Default ISP parameters [19]
            
        Returns:
            params: Constrained ISP parameters [B,19]
        """
        B = params_residual.shape[0]
        default = default_params.unsqueeze(0).expand(B, -1)
        
        # Start with default parameters
        params = default.clone()
        
        # Apply residuals with constraints
        
        # Digital gain: [0.1, 3.0]
        params[:, 0] = torch.clamp(default[:, 0] + params_residual[:, 0], 0.1, 3.0)
        
        # White balance: [0.5, 2.0]
        params[:, 1:3] = torch.clamp(default[:, 1:3] + params_residual[:, 1:3], 0.5, 2.0)
        
        # Color correction matrix: [-0.5, 1.5] for off-diagonal, [0.5, 1.5] for diagonal
        ccm_residual = params_residual[:, 3:12].view(B, 3, 3)
        ccm_default = default[:, 3:12].view(B, 3, 3)
        
        # Diagonal elements
        for i in range(3):
            params[:, 3 + i*3 + i] = torch.clamp(
                ccm_default[:, i, i] + ccm_residual[:, i, i], 0.5, 1.5
            )
        
        # Off-diagonal elements
        for i in range(3):
            for j in range(3):
                if i != j:
                    params[:, 3 + i*3 + j] = torch.clamp(
                        ccm_default[:, i, j] + ccm_residual[:, i, j], -0.5, 0.5
                    )
        
        # Color offset: [-0.1, 0.1]
        params[:, 12:15] = torch.clamp(
            default[:, 12:15] + params_residual[:, 12:15], -0.1, 0.1
        )
        
        # Gamma: [0.5, 3.0]
        params[:, 15] = torch.clamp(default[:, 15] + params_residual[:, 15], 0.5, 3.0)
        
        # Tone mapping s: [0.5, 2.0]
        params[:, 16] = torch.clamp(default[:, 16] + params_residual[:, 16], 0.5, 2.0)
        
        # Tone mapping p1, p2: [0.5, 2.0]
        params[:, 17:19] = torch.clamp(
            default[:, 17:19] + params_residual[:, 17:19], 0.5, 2.0
        )
        
        return params