"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        if reduction not in ['mean', 'sum']:
            raise ValueError('reduction should be one of [mean, sum]')

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        
        xa,ya,wa,ha = pred_boxes.split(1, dim=-1)
        xb,yb,wb,hb = target_boxes.split(1, dim=-1)

        Xa = torch.max(torch.concat((xa-wa/2, xb-wb/2), dim=1), dim=1, keepdim=True).values
        Ya = torch.max(torch.concat((ya-ha/2, yb-hb/2), dim=1), dim=1, keepdim=True).values
        Xb = torch.min(torch.concat((xa+wa/2, xb+wb/2), dim=1), dim=1, keepdim=True).values
        Yb = torch.min(torch.concat((ya+ha/2, yb+hb/2), dim=1), dim=1, keepdim=True).values

        h_intersection = (Xb-Xa); h_intersection = h_intersection.masked_fill(h_intersection<0, 0.)

        w_intersection = (Yb-Ya); w_intersection = w_intersection.masked_fill(w_intersection<0, 0.)

        Area_intersection = h_intersection * w_intersection

        Area_pred = ha * wa ; Area_target = hb * wb

        IOU = 1 - Area_intersection / (Area_pred+ Area_target- Area_intersection+self.eps)

        if self.reduction == 'sum':
            return torch.sum(IOU)

        else:
            return torch.mean(IOU)
             


class Localize_loss(nn.Module):

    def __init__(self,iou_w = 0.8, reduction='mean'):
        super(Localize_loss, self).__init__()

        self.iou = IoULoss(reduction=reduction)
        self.mse = nn.MSELoss(reduction=reduction)
        self.iouw = iou_w

    def forward(self, pred, target):

        return self.iouw*self.iou.forward(pred,target) + (1-self.iouw)*self.mse.forward(pred,target)/224**2