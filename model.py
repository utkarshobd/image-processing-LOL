import torch
import torch.nn as nn
import torch.nn.functional as F
from isp import DifferentiableISP
from encoder import StyleEncoder, LQOnlyEncoder
from decoder import StyleDecoder, ParameterConstraints

class CRISP(nn.Module):
    """
    Complete CRISP Model: Controllable Image Signal Processor
    
    Paper: Learning Controllable ISP for Image Enhancement
    """
    
    def __init__(self, style_dim=3, use_lq_encoder=False):
        super(CRISP, self).__init__()
        
        self.style_dim = style_dim
        self.use_lq_encoder = use_lq_encoder
        
        # Core components
        self.isp = DifferentiableISP()
        self.style_encoder = StyleEncoder(style_dim=style_dim)
        self.style_decoder = StyleDecoder(style_dim=style_dim)
        
        # Optional LQ-only encoder for inference
        if use_lq_encoder:
            self.lq_encoder = LQOnlyEncoder(style_dim=style_dim)
        
        # Parameter constraints
        self.constraints = ParameterConstraints()
        
    def forward(self, lq, hq=None, style=None, mode='train'):
        """
        Forward pass with different modes
        
        Args:
            lq: Low quality image [B,3,H,W]
            hq: High quality image [B,3,H,W] (only for training)
            style: Manual style vector [B,style_dim] (for inference)
            mode: 'train', 'inference_manual', 'inference_lq'
            
        Returns:
            output: Enhanced image [B,3,H,W]
            style_vec: Style vector [B,style_dim] (if computed)
        """
        
        if mode == 'train':
            # Training mode: use LQ+HQ encoder
            assert hq is not None, "HQ image required for training"
            style_vec = self.style_encoder(lq, hq)
            
        elif mode == 'inference_manual':
            # Manual style control
            assert style is not None, "Style vector required for manual inference"
            style_vec = style
            
        elif mode == 'inference_lq':
            # LQ-only inference
            assert self.use_lq_encoder, "LQ encoder not available"
            style_vec = self.lq_encoder(lq)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Decode style to ISP parameters
        params_residual = self.style_decoder(style_vec)
        
        # Apply constraints
        params = self.constraints.apply_constraints(
            params_residual, self.isp.default_params
        )
        
        # Apply ISP
        output = self.isp(lq, params)
        
        return output, style_vec
    
    def get_style_presets(self, style_vectors, k=10):
        """
        Generate style presets using K-means clustering
        
        Args:
            style_vectors: Collection of style vectors [N,style_dim]
            k: Number of clusters
            
        Returns:
            presets: Style presets [k,style_dim]
        """
        from sklearn.cluster import KMeans
        
        style_np = style_vectors.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(style_np)
        
        presets = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        return presets
    
    def enhance_with_preset(self, lq, preset_idx, presets):
        """
        Enhance image with preset style
        
        Args:
            lq: Low quality image [B,3,H,W]
            preset_idx: Preset index
            presets: Style presets [k,style_dim]
            
        Returns:
            output: Enhanced image [B,3,H,W]
        """
        B = lq.shape[0]
        style = presets[preset_idx].unsqueeze(0).expand(B, -1)
        
        if lq.is_cuda:
            style = style.cuda()
            
        output, _ = self.forward(lq, style=style, mode='inference_manual')
        return output
    
    def get_isp_parameters(self, style):
        """
        Get ISP parameters for given style
        
        Args:
            style: Style vector [B,style_dim]
            
        Returns:
            params: ISP parameters [B,19]
        """
        params_residual = self.style_decoder(style)
        params = self.constraints.apply_constraints(
            params_residual, self.isp.default_params
        )
        return params

class CRISPLoss(nn.Module):
    """
    Loss function for CRISP training - PAPER EXACT: MSE ONLY
    """
    
    def __init__(self):
        super(CRISPLoss, self).__init__()
        
    def forward(self, output, target, style_vec=None):
        """
        Compute CRISP loss - PAPER EXACT: MSE ONLY
        
        Args:
            output: Enhanced image [B,3,H,W]
            target: Target HQ image [B,3,H,W]
            style_vec: Unused (kept for compatibility)
            
        Returns:
            loss: MSE loss
            loss_dict: Dictionary with loss
        """
        # PAPER SPECIFICATION: ONLY MSE LOSS
        mse_loss = F.mse_loss(output, target)
        
        loss_dict = {
            'total': mse_loss,
            'mse': mse_loss
        }
        
        return mse_loss, loss_dict