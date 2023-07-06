from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from measure import ssim



class Mix(nn.Module):
    def __init__(self, losses, coefficients: Optional[dict]=None):
        super(Mix, self).__init__()
        self.losses = losses
        self.coefficients = coefficients

        if self.coefficients is None:
            self.coefficients = { k:1 for k in self.losses}

        assert self.coefficients is not None

    def forward(self, target, predict):
        losses_results = { k:self.losses[k](target, predict) for k in self.losses }

        loss = sum(
            [
                losses_results[k] * self.coefficients[k] # type: ignore
                for k in losses_results
                if losses_results[k] is not None]
        ) / len(losses_results)

        return loss, losses_results


class BCELoss(nn.Module):
    def __init__(self, output_key: str, target_key: str, bg_weight=1):
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.bg_weight=bg_weight

    def forward(self, target, predict):

        assert (target[self.target_key].shape == predict[self.output_key].shape)

        pred = predict[self.output_key].view(-1)
        gt = target[self.target_key].view(-1)

        loss = F.binary_cross_entropy(pred, gt, reduction='mean')

        return loss


class DiceLoss(nn.Module):
    def __init__(self, output_key: str, target_key: str):
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, target, predict):

        assert (target[self.target_key].shape == predict[self.output_key].shape)
        shape = target[self.target_key].shape

        pred = predict[self.output_key].view(shape[0], shape[1], -1)
        gt = target[self.target_key].view(shape[0], shape[1], -1)

        intersection = (pred*gt).sum(dim=(0,2)) + 1e-6
        union = (pred**2 + gt).sum(dim=(0,2)) + 2e-6
        dice = 2.0*intersection / union

        return (1.0 - torch.mean(dice))


class SSIMLoss(nn.Module):
    def __init__(self, output_key: str, target_key: str):
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.loss = ssim.SSIM()

    def forward(self, target, predict):
        assert (target[self.target_key].shape == predict[self.output_key].shape)

        pred = predict[self.output_key][:,:,:,0,:]
        gt = target[self.target_key][:,:,:,0,:]

        error = self.loss(pred, gt)

        return error
