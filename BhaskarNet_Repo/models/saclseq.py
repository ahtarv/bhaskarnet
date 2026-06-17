import torch
import torch.nn as nn

class Scalseq(nn.Module):
    """
    Scalseq: A hardware-friendly 2D convolution-based neck module that fuses
    hierarchical features across three distinct pyramid levels (P3, P4, P5).
    """
    def __init__(self, *args):
        super().__init__()
        flat_args = []
        for a in args:
            if isinstance(a, (list, tuple)):
                flat_args.extend(a)
            else:
                flat_args.append(a)
                
        if len(flat_args) >= 3:
            c_p3 = int(flat_args[-3])
            c_p4 = int(flat_args[-2])
            c_p5 = int(flat_args[-1])
        else:
            c_p3, c_p4, c_p5 = 256, 512, 1024
            
        self.c_p3 = c_p3
        self.c_p4 = c_p4
        self.c_p5 = c_p5
        out_channels = self.c_p3
        
        # 1x1 convolutions to project all scales to uniform channels
        self.conv_p3 = nn.Conv2d(self.c_p3, out_channels, 1)
        self.conv_p4 = nn.Conv2d(self.c_p4, out_channels, 1)
        self.conv_p5 = nn.Conv2d(self.c_p5, out_channels, 1)
        
        # Channel fusion conv
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) 
        self.act = nn.SiLU()

    def forward(self, x):
        p3, p4, p5 = torch.split(x, [self.c_p3, self.c_p4, self.c_p5], dim=1)
        
        f3 = self.conv_p3(p3)
        f4 = self.conv_p4(p4)
        f5 = self.conv_p5(p5)
        
        f_cat = torch.cat([f3, f4, f5], dim=1)
        y = self.fusion_conv(f_cat)
        y = self.bn(y)
        y = self.act(y)
        return y
