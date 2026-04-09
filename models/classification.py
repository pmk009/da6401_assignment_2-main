"""Classification components
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):

        super(VGG11Classifier, self).__init__()      
        self.encoder = VGG11Encoder(in_channels)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.classify = nn.Sequential(
            nn.Flatten(),
            CustomDropout(p=0.2),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            CustomDropout(dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        xf = self.encoder(x)

        xavg = self.adaptivepool(xf)
        xmax = self.maxpool(xf)
        xf_ = torch.cat([xavg,xmax], dim=1)
        pred = self.classify(xf_)

        return pred