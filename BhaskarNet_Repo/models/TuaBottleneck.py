import torch
import torch.nn as nn
from models.TuaAttention import TuaAttention

class TuaBottleneck(nn.Module):
    """
    TuaBottleneck: A Vision-Transformer (ViT) style bottleneck block operating on 
    2D feature maps. Uses Pre-LayerNorm (Pre-LN), TuaAttention, and MLP blocks 
    with GELU activations. Replaces the standard C2f blocks in the backbone.
    """
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)
        self.c1 = c1
        self.c2 = c2
        self.add = shortcut and c1 == c2
        
        # Pre-LN normalizations
        self.ln1 = nn.LayerNorm(c1)
        self.ln2 = nn.LayerNorm(c2)
        self.attn = TuaAttention(c1)
        
        # MLP Block
        self.mlp = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=1) 
        )

    def forward(self, x):
        residual = x
        
        # Attention pass with LayerNorm trick for 2D tensors
        y = x.permute(0, 2, 3, 1)
        y = self.ln1(y)
        y = y.permute(0, 3, 1, 2)
        y = self.attn(y)
        if self.add:
            y = y + residual
            
        residual = y
        
        # MLP pass with LayerNorm trick
        z = y.permute(0, 2, 3, 1)
        z = self.ln2(z)
        z = z.permute(0, 3, 1, 2)
        z = self.mlp(z)
        if self.add:
            z = z + residual
            
        return z
