import torch.nn as nn
import torch
import torch.nn.functional as F

class PixelShuffle(nn.Module):
    def __init__(self, D, alpha, H):
        super().__init__()
        upsample_factor = H//alpha
        
        self.upsample = nn.Sequential(
            nn.Conv2d(D, D * upsample_factor ** 2, kernel_size=1, groups=D),
            nn.PixelShuffle(upsample_factor),
        )
    
    def forward(self, feat):
        return self.upsample(feat)
    
class FiLMLayer(nn.Module):
    def __init__(self, c, B):
        super().__init__()
        self.gamma = nn.Conv2d(B, c, kernel_size=1)  # Scale
        self.beta = nn.Conv2d(B, c, kernel_size=1)   # Shift

        nn.init.constant_(self.gamma.weight, 1.0)
        nn.init.constant_(self.beta.weight, 0.0)

    def forward(self, x, x_hr):

        gamma = self.gamma(x_hr)
        beta = self.beta(x_hr)
        return x * gamma + beta
    
class FiLMUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, B, alpha, H, group_channels=False):
        super().__init__()
        
        if group_channels:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=(H // alpha), mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=(H // alpha), mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        self.film = FiLMLayer(out_channels, B)

    def forward(self, x_lr, x_hr):
        x_upsampled = self.upsample(x_lr)
        x_fused = self.film(x_upsampled, x_hr)
        return x_fused

class ConvTranspose(nn.Module):
    def __init__(self, H, D, alpha):
        super().__init__()

        upsample_factor = H//alpha
        self.upsampler = nn.Sequential(nn.ConvTranspose2d(
            in_channels=D,
            out_channels=D,
            kernel_size=upsample_factor,
            stride=upsample_factor,
            padding=0,
            output_padding=0,
            groups=D),
            nn.Conv2d(D,D, kernel_size=3, padding=1)
        )

    def forward(self, features):
        return self.upsampler(features)