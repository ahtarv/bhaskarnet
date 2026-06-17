import torch
import torch.nn as nn
import torch.nn.functional as F

class Zoomcat(nn.Module):
    """
    Zoomcat: Captures local (zoomed-in), native, and global context (zoomed-out) 
    spatial features simultaneously using dynamic pooling and interpolation.
    """
    def __init__(self, *args):
        super().__init__()
        flat_args = []
        for a in args:
            if isinstance(a, (list, tuple)):
                flat_args.extend(a)
            else:
                flat_args.append(a)
                
        if len(flat_args) > 0:
            c1 = int(flat_args[0])
        else:
            c1 = 256
            
        in_channels = c1
        reduced_c = in_channels // 2 
        
        self.conv_l = nn.Conv2d(in_channels, reduced_c, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv_m = nn.Conv2d(in_channels, reduced_c, 1)
        
        self.conv_s = nn.Conv2d(in_channels, reduced_c, 1)
        
        self.final_conv = nn.Conv2d(reduced_c * 3, in_channels, kernel_size=1)

    def forward(self, x):
        # Contextual zoom-out branch (pooling then interpolation)
        xl = self.conv_l(x)
        xl_pooled = self.max_pool(xl) + self.avg_pool(xl)
        xl_out = F.interpolate(xl_pooled, size=x.shape[2:], mode='nearest')
        
        # Native zoom branch
        xm_out = self.conv_m(x)
        
        # Fine zoom-in branch (double scaling then matching)
        xs = self.conv_s(x)
        xs_up = F.interpolate(xs, scale_factor=2, mode='nearest')
        xs_out = F.interpolate(xs_up, size=x.shape[2:], mode='nearest')
        
        y = torch.cat([xl_out, xm_out, xs_out], dim=1)
        return self.final_conv(y)
