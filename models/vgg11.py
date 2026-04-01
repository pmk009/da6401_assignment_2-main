"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(VGG11Encoder, self).__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.BatchNorm2d(128), 
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.BatchNorm2d(256), 
            nn.ReLU()
        )

        self.conv_4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.BatchNorm2d(512), 
            nn.ReLU()
        )

        self.conv_5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )


    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """

        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x4 = self.conv_4(x3)
        x5 = self.conv_5(x4)

        if return_features:
        
            return (x5, x1, x2, x3, x4)
        
        else:
        
            return x5
        

