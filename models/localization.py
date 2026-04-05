"""Localization modules
"""

import torch
import torch.nn as nn
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
            nn.Flatten(),
            nn.Linear(7*7*512, 4096), nn.ReLU(), CustomDropout(dropout_p),
            nn.Linear(4096, 4096), nn.ReLU(), CustomDropout(dropout_p),
            nn.Linear(4096, 4), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        
        xf = self.encoder(x)

        return self.localize_head(xf)