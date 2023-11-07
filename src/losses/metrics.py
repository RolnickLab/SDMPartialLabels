# Src code for all metrics used
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import Metric

from src.losses.losses import CustomCrossEntropy


class CustomKL(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, p: torch.Tensor, q: torch.Tensor, mask=None):
        """
        p: target distribution
        q: predicted distribution
        """
        self.correct += (torch.nansum(p * torch.log(p / q)) + torch.nansum((1 - p) * torch.log((1 - p) / (1 - q))))
        self.total += p.numel()

    def compute(self):
        return (self.correct / self.total)


class Presence_k(nn.Module):
    """
    compare accuracy by binarizing targets  1 if species are present with proba > k
    """

    def __init__(self, k):
        super().__init__()
        self.k = k

    def __call__(self, target, pred):
        pres = ((pred > self.k) == (target > self.k)).mean()
        return (pres)


class CustomTopK(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, preds: torch.Tensor, mask=None):

        assert preds.shape == target.shape
        non_zero_counts = torch.count_nonzero(target, dim=1)
        for i, elem in enumerate(target):
            ki = non_zero_counts[i]
            v_pred, i_pred = torch.topk(preds[i], k=ki)
            v_targ, i_targ = torch.topk(elem, k=ki)
            if ki == 0:
                pass
            else:
                count = torch.tensor(len([k for k in i_pred if k in i_targ]))
                self.correct += count / ki
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class CustomTop10(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, preds: torch.Tensor):

        assert preds.shape == target.shape
        non_zero_counts = torch.count_nonzero(target, dim=1)
        for i, elem in enumerate(target):
            ki = non_zero_counts[i]
            if ki >= 10:
                v_pred, i_pred = torch.topk(preds[i], k=10)
                v_targ, i_targ = torch.topk(elem, k=10)
            else:
                v_pred, i_pred = torch.topk(preds[i], 10)
                v_targ, i_targ = torch.topk(elem, ki)
            if ki == 0:
                pass
            else:
                count = torch.tensor(len([k for k in i_pred if k in i_targ]))
                if ki >= 10:
                    self.correct += count / 10
                else:
                    self.correct += count / ki
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class CustomTop30(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        # self.count_ki_30 = 0

    def update(self, target: torch.Tensor, preds: torch.Tensor):
        assert preds.shape == target.shape
        non_zero_counts = torch.count_nonzero(target, dim=1)
        for i, elem in enumerate(target):
            ki = non_zero_counts[i]
            if ki >= 30:
                v_pred, i_pred = torch.topk(preds[i], k=30)
                v_targ, i_targ = torch.topk(elem, k=30)
            else:
                v_pred, i_pred = torch.topk(preds[i], 30)
                v_targ, i_targ = torch.topk(elem, k=ki)
            if ki == 0:
                pass
            else:
                count = torch.tensor(len([k for k in i_pred if k in i_targ]))
                if ki >= 30:
                    self.correct += count / 30
                else:
                    self.correct += count / ki
                self.total += 1
            # if ki >= 30:
            #     self.count_ki_30 += 1

    def compute(self):
        return self.correct.float() / self.total


class MaskedMSE(Metric):
    def __init__(self):
        super().__init__()

    def update(self, target: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor=None):
        squared_diffs = (preds - target) ** 2
        self.masked_squared_diffs = squared_diffs * mask
        self.non_zero_sum = mask.sum()

    def compute(self):
        mse = self.masked_squared_diffs.sum() / self.non_zero_sum
        return mse


class MaskedMAE(Metric):
    def __init__(self):
        super().__init__()

    def update(self, target: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor=None):
        squared_diffs = torch.abs(preds - target)
        # Apply the mask
        self.masked_abs_diffs = squared_diffs * mask
        self.non_zero_sum = mask.sum()

    def compute(self):
        mae = self.masked_abs_diffs.sum() / self.non_zero_sum
        return mae


def get_metric(metric, masked=False):
    """Returns the transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if metric.name == "mae" and not metric.ignore is True:
        if not masked:
            return torchmetrics.MeanAbsoluteError()
        else:
            return MaskedMAE()
    elif metric.name == "mse" and not metric.ignore is True:
        if not masked:
            return torchmetrics.MeanSquaredError()
        else:
            return MaskedMSE()
    elif metric.name == "topk" and not metric.ignore is True:
        return CustomTopK()
    elif metric.name == "top10" and not metric.ignore is True:
        return CustomTop10()
    elif metric.name == "top30" and not metric.ignore is True:
        return CustomTop30()
    elif metric.name == "ce" and not metric.ignore is True:
        return CustomCrossEntropy(metric.lambd_pres, metric.lambd_abs)
    elif metric.name == 'r2' and not metric.ignore is True:
        return torchmetrics.ExplainedVariance(
            multioutput='variance_weighted')
    elif metric.name == "kl" and not metric.ignore is True:
        return CustomKL()
    elif metric.name == "accuracy" and not metric.ignore is True:
        return torchmetrics.Accuracy()
    elif metric.ignore is True:
        return None
    else:
        return (None)  # raise ValueError("Unknown metric_item {}".format(metric))


def get_metrics(config):
    metrics = []
    for m in config.losses.metrics:
        metrics.append((m.name, get_metric(m, config.Rtran.mask_eval_metrics), m.scale))
    metrics = [(a, b, c) for (a, b, c) in metrics if b is not None]
    return metrics
