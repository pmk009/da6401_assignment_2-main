"""Unified multi-task model
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.layers import CustomDropout
import gdown
import os

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "checkpoints/best_classification_gap.pth", localizer_path: str = "checkpoints/best_localization.pth", unet_path: str = "checkpoints/checkpoints_best_segmentation.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super(MultiTaskPerceptionModel, self).__init__()
        if not os.path.exists(classifier_path):
            gdown.download(id="1GqqCQFkkr3dZ_7w7su8rtUsEV1hsvHtO", output=classifier_path, quiet=False)
        if not os.path.exists(localizer_path):
            gdown.download(id="14DKqE0AVgwRQ7Zp6oMyfcWOR6HSDZUfp", output=localizer_path, quiet=False)
        if not os.path.exists(unet_path):
            gdown.download(id="1ICorgugGHQt7VdOqw-5JP1rDYKztCBaR", output=unet_path, quiet=False)

        self.classify = VGG11Classifier()
        pretrained = torch.load(classifier_path, map_location='cpu')
        self.classify.load_state_dict(pretrained)

        self.localize = VGG11Localizer()
        pretrained = torch.load(localizer_path, map_location='cpu')
        self.localize.load_state_dict(pretrained)
        
        self.segment  = VGG11UNet()
        pretrained = torch.load(unet_path, map_location='cpu')
        self.segment.load_state_dict(pretrained)


    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """

        pet_class = self.classify(x)
        localize = self.localize(x)
        segment = self.segment(x)

        return pet_class, localize, segment