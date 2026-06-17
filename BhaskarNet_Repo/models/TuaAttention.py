import torch
import torch.nn as nn

class DilatedConv2d(nn.Module):
    """
    A dilated convolution block (dilation=2, padding=2) that serves as the 
    receptive-field-expanding approximation for deformable convolutions on CPU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=2,
            dilation=2,
            stride=stride,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class TuaAttention(nn.Module):
    """
    TuaAttention: Vision-Transformer inspired local attention module using 
    parallel convolution blocks and GELU activations.
    """
    def __init__(self, in_channels, use_dilation=True):
        super().__init__()
        in_channels = int(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gelu = nn.GELU()
        
        # Can toggle between dilated conv (d=2) or standard conv (d=1)
        # Note: Ablation results show standard conv (use_dilation=False) performs better
        if use_dilation:
            self.dcn1 = DilatedConv2d(in_channels, in_channels, kernel_size=3)
            self.dcn2 = DilatedConv2d(in_channels, in_channels, kernel_size=3)
        else:
            self.dcn1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU()
            )
            self.dcn2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU()
            )
            
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x_init = self.conv1(x)
        x_act = self.gelu(x_init)
        x_dcn = self.dcn1(x_act)
        x_dcn = self.dcn2(x_dcn)
        x_dcn = self.conv2(x_dcn)
        x_combined = x_act + x_dcn
        return self.conv_final(x_combined)
