"""Localization modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
import os

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, encoder_init: str = ''):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super(VGG11Localizer, self).__init__()
        self.encoder = VGG11Encoder(in_channels)
        if encoder_init.endswith('.pth') and os.path.exists(encoder_init):
            pretrained = torch.load(encoder_init)
            self.encoder.load_state_dict(pretrained, strict=False)
        else:
            print('VGG11 encoder not initialized with Pretrained model.')

        self.localize_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            CustomDropout(0.2),
            nn.Linear(256*3*3, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            CustomDropout(dropout_p),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        xf = self.encoder(x)

        out = self.localize_head(xf)   

        xy = out[:, :2]                
        wh = out[:, 2:]                

        wh = F.softplus(wh)            

        bbox = torch.cat([xy, wh], dim=1) 

        return bbox