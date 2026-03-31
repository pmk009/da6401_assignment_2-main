"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        
        if not 0<=p<=1:
            ValueError("p is not a valid probability.")

        self.dropout_prob = p
        self.keep_prob = 1-p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.dropout_prob == 0. or not nn.Module.training:

            return x
        
        else:

            mask = (torch.rand(x.shape, device=x.device) < self.keep_prob).float

            return mask * x / self.keep_prob