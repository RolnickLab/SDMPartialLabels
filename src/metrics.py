# Src code for all metrics used
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import Metric

from src.losses import CustomCrossEntropy


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
        self.correct += torch.nansum(p * torch.log(p / q)) + torch.nansum(
            (1 - p) * torch.log((1 - p) / (1 - q))
        )
        self.total += p.numel()

    def compute(self):
        return self.correct / self.total


class Presence_k(nn.Module):
    """
    compare accuracy by binarizing targets  1 if species are present with proba > k
    """

    def __init__(self, k):
        super().__init__()
        self.k = k

    def __call__(self, target, pred):
        pres = ((pred > self.k) == (target > self.k)).mean()
        return pres


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
                count = torch.tensor(
                    len([k for v, k in zip(v_pred, i_pred) if k in i_targ and v > 0])
                )
                self.correct += count / ki
                self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class CustomTopK_bounded(Metric):
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
                count = torch.tensor(
                    len(
                        [
                            k
                            for v, k in zip(v_pred, i_pred)
                            if k in i_targ and torch.abs(v - elem[k]) <= 0.5
                        ]
                    )
                )
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
                count = torch.tensor(
                    len([k for v, k in zip(v_pred, i_pred) if k in i_targ and v > 0])
                )
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
                count = torch.tensor(
                    len([k for v, k in zip(v_pred, i_pred) if k in i_targ and v > 0])
                )
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
        self.add_state("squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, target: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor = None
    ):
        squared_diffs = (preds - target) ** 2
        self.squared_error += squared_diffs.sum()
        if mask is None:
            self.num_samples += preds.size(0) * preds.size(1)
        else:
            self.num_samples += (mask>=0).long().sum()

    def compute(self):
        mse = self.squared_error / self.num_samples
        return mse


class NonZeroMSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, target: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor = None
    ):
        # squared_diffs = (preds - target) ** 2
        nonzero_indices = target > 0
        preds = preds[nonzero_indices]
        target = target[nonzero_indices]
        squared_diffs = (preds - target) ** 2
        self.squared_error += squared_diffs.sum()
        if mask is None:
            self.num_samples += preds.size(0)
        else:
            mask = (mask>=0).long()
            self.num_samples += mask[nonzero_indices].sum()

    def compute(self):
        mse = self.squared_error / self.num_samples
        return mse


class MaskedMAE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, target: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor = None
    ):
        diffs = torch.abs(preds - target)
        # Apply the mask
        self.abs_error += diffs.sum()
        if mask is None:
            self.num_samples += preds.size(0) * preds.size(1)
        else:
            
            self.num_samples += (mask>=0).long().sum()

    def compute(self):
        mae = self.abs_error / self.num_samples
        return mae


class NonZeroMAE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, target: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor = None
    ):
        nonzero_indices = target > 0
        preds = preds[nonzero_indices]
        target = target[nonzero_indices]
        diffs = torch.abs(preds - target)
        # Apply the mask
        self.abs_error += diffs.sum()
        if mask is None:
            self.num_samples += preds.size(0)
        else:
            self.num_samples += (mask>=0).long()[nonzero_indices].sum()

    def compute(self):
        mae = self.abs_error / self.num_samples
        return mae


class CustomMultiLabelAcc(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    def update(
        self, target: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor = None
    ):
        """
        Compute the accuracy for multi-label classification.

        Returns:
        float: The accuracy score.
        """
        # Compare with true labels
        correct_pred = (preds == target).float()

        # Compute the accuracy per sample
        self.correct += correct_pred.sum()

        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


def get_metric(metric):
    """Returns the transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if metric.name == "mae" and not metric.ignore is True:
        return MaskedMAE()
    elif metric.name == "mse" and not metric.ignore is True:
        return MaskedMSE()
    elif metric.name == "nonzero_mae" and not metric.ignore is True:
        return NonZeroMAE()
    elif metric.name == "nonzero_mse" and not metric.ignore is True:
        return NonZeroMSE()
    elif metric.name == "topk" and not metric.ignore is True:
        return CustomTopK()
    elif metric.name == "topk2" and not metric.ignore is True:
        return CustomTopK_bounded()
    elif metric.name == "top10" and not metric.ignore is True:
        return CustomTop10()
    elif metric.name == "top30" and not metric.ignore is True:
        return CustomTop30()
    elif metric.name == "ce" and not metric.ignore is True:
        return CustomCrossEntropy(metric.lambd_pres, metric.lambd_abs)
    elif metric.name == "r2" and not metric.ignore is True:
        return torchmetrics.ExplainedVariance(multioutput="variance_weighted")
    elif metric.name == "kl" and not metric.ignore is True:
        return CustomKL()
    elif metric.name == "accuracy" and not metric.ignore is True:
        return CustomMultiLabelAcc()
    elif metric.ignore is True:
        return None
    else:
        return None  # raise ValueError("Unknown metric_item {}".format(metric))


def get_metrics(config):
    metrics = []
    for m in config.metrics:
        metrics.append((m.name, get_metric(m), m.scale))
    metrics = [(a, b, c) for (a, b, c) in metrics if b is not None]
    return metrics
