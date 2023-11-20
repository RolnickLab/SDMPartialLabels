# Src code of loss functions
import torch
from torchmetrics import Metric
import torch.nn as nn

eps = 1e-7


class CustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1):
        super().__init__()
        # print('in my custom')
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres

    def __call__(self, pred, target, mask=None,  weights=1, reduction='mean'):
        """
        target: ground truth
        pred: prediction
        reduction: mean, sum, none
        """
        if mask != None:
            target = target[mask.bool()]
            pred = pred[mask.bool()]

        loss = weights*(-self.lambd_pres * target * torch.log(pred + eps) - self.lambd_abs * (1 - target) * torch.log(1 - pred + eps))

        if reduction == 'mean':
            if mask != None:
                loss = loss.sum() / mask.sum().item()
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else: # reduction = None
            loss = loss
        # print('inside_loss',loss)
        return loss


class BCE(nn.Module):
    """
    Binary cross entropy with logits, used with binary inputs
    """
    def __init__(self):
        super().__init__()

    def __call__(self, pred, target, mask=None,  weights=1, reduction='mean'):
        """
        target: ground truth
        pred: prediction
        reduction: mean, sum, none
        """
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        if mask != None:
            target = target[mask.bool()]
            pred = pred[mask.bool()]

        loss = loss_fn(pred, target)

        if reduction == 'mean':
            if mask != None:
                loss = loss.sum() / mask.sum().item()
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else: # reduction = None
            loss = loss

        return loss


class RMSLELoss(nn.Module):
    """
    root mean squared log error
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target, mask=None, reduction='None'):
        if mask is not None:
            target = target[mask.bool()]
            pred = pred[mask.bool()]
            squared_diff = (torch.log1p(pred) - torch.log1p(target)) ** 2
            loss = squared_diff.sum() / mask.sum().item()
            loss = torch.sqrt(loss)
        else:
            loss = torch.sqrt(self.mse(torch.log1p(pred), torch.log1p(target)))

        return loss


class CustomFocalLoss:
    def __init__(self, alpha=1, gamma=2):
        """
        build on top of binary cross entropy as implemented before
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, pred, target, mask=None, reduction='mean'):
        if mask != None:
            target = target[mask.bool()]
            pred = pred[mask.bool()]

        ce_loss = (- target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps))
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if reduction == 'mean':
            if mask != None:
                loss = loss.sum() / mask.sum().item()
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else: # reduction = None
            loss = loss
        return loss


class CustomCrossEntropy(Metric):
    def __init__(self, lambd_pres=1, lambd_abs=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres
        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        target: target distribution
        pred: predicted distribution
        """
        self.correct += (-self.lambd_pres * target * torch.log(pred) - self.lambd_abs * (1 - target) * torch.log(1 - pred)).sum()
        self.total += target.numel()

    def compute(self):
        return (self.correct / self.total)


class WeightedCustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1):
        super().__init__()
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres

    def __call__(self, pred, target, weights=1):
        loss = (weights * (
                -self.lambd_pres * target * torch.log(pred + eps) - self.lambd_abs * (1 - target) * torch.log(
            1 - pred + eps))).mean()

        return loss
