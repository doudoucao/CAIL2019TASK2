import torch.nn.functional as F
import torch.nn as nn
import torch

def binary_cross_entropy(inputs, target, weight=None, reduction='mean', smooth_eps=None, from_logits=False):
    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target = target.float()
        # target.add_(smooth_eps).div_(2.)
        target = (target + smooth_eps) / 2.

    if from_logits:
        return F.binary_cross_entropy_with_logits(inputs, target, weight=weight, reduction=reduction)
    else:
        return F.binary_cross_entropy(inputs, target, weight=weight, reduction=reduction)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha*(1-pt)**self.gamma*BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss