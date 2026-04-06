"""Segmentation model
"""
import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, encoder_init:str =''):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """

        super(VGG11UNet, self).__init__()
        
        self.encoder = VGG11Encoder(in_channels)

        if encoder_init.endswith('.pth') and os.path.exists(encoder_init):

            pretrained = torch.load(encoder_init)

            self.encoder.load_state_dict(pretrained, strict= False)
        
        self.decode_1 = self.decoder_block(512, 512)
        self.decode_2 = self.decoder_block(1024, 256)
        self.decode_3 = self.decoder_block(512, 128)
        self.decode_4 = self.decoder_block(256, 64)
        self.decode_5 = self.decoder_block(128, 32)
        self.pool = nn.MaxPool2d(2,2)
        self.conv_final = nn.Conv2d(32, 3, kernel_size=1)
    
    def decoder_block(self, in_channels, out_channels):

        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        
        x5, x1, x2, x3, x4 = self.encoder(x , return_features=True)

        d1 = self.decode_1(x5)
        d2 = self.decode_2(torch.cat((d1,self.pool(x4)), dim=1))
        d3 = self.decode_3(torch.cat((d2,self.pool(x3)), dim=1))
        d4 = self.decode_4(torch.cat((d3,self.pool(x2)), dim=1))
        d5 = self.decode_5(torch.cat((d4,self.pool(x1)), dim=1))

        return self.conv_final(d5)