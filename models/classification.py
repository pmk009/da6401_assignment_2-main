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
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512, 4096), nn.BatchNorm1d, nn.ReLU(), CustomDropout(dropout_p),
            nn.Linear(4096, 4096), nn.BatchNorm1d(), nn.ReLU(), CustomDropout(dropout_p),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """

        xf = self.encoder(x)
        pred = self.classify(xf)

        return pred