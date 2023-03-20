import torch
import torch.nn as nn
import torch.nn.functional as F
from motion.loss.builder import LOSSES, build_loss


class BaseLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.loss = None

    def _forward_loss(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def forward_loss(self, *args, **kwargs):
        return self._forward_loss(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.weight != 0:
            loss = self.weight * self.forward_loss(*args, **kwargs)
        else:
            return 0.0
        if torch.isinf(loss).any():
            print("[ERROR] INF LOSS!")
            __import__("ipdb").set_trace()
        if torch.isnan(loss).any():
            print("[ERROR] NAN LOSS!")
            __import__("ipdb").set_trace()
        return loss


@LOSSES.register_module()
class crossentropyloss(BaseLoss):
    def __init__(self, weight, *args, **kwargs):
        super().__init__(weight)
        self.loss = nn.CrossEntropyLoss()


@LOSSES.register_module()
class binarycrossentropy(BaseLoss):
    def __init__(self, weight, *args, **kwargs):
        super().__init__(weight)
        self.loss = nn.BCELoss()


class ReductionLoss(BaseLoss):
    def __init__(self, weight, reduction="mean", average_dim=None):
        super().__init__(weight)
        self.reduction = reduction
        self.average_dim = average_dim

    def reduction_loss(self, loss, pred):
        if self.reduction == "sum" and self.average_dim is not None:
            avg_num = 1.0
            if isinstance(pred, list):
                pred_ = pred[0]
            elif isinstance(pred, dict):
                # Sometimes, dict can contains a list of tensor
                for k in pred.keys():
                    if isinstance(pred[k], torch.Tensor):
                        pred_ = pred[k]
                        break
            else:
                pred_ = pred

            for d in self.average_dim:
                avg_num *= pred_.shape[d]
            loss = loss / avg_num
        return loss

    def forward_loss(self, pred, gt, *args, **kwargs):
        # Sometimes we simply need a regularizer
        if gt is None and isinstance(pred, torch.Tensor):
            gt = torch.zeros_like(pred)
        loss = super().forward_loss(pred, gt, *args, **kwargs)
        loss = self.reduction_loss(loss, pred)
        return loss


@LOSSES.register_module()
class mseloss(ReductionLoss):
    def __init__(self, weight, reduction="mean", *args, **kwargs):
        super().__init__(weight, reduction, *args, **kwargs)
        self.loss = nn.MSELoss(reduction=reduction)


@LOSSES.register_module()
class l1loss(ReductionLoss):
    def __init__(self, weight, reduction="mean", *args, **kwargs):
        super().__init__(weight, reduction, *args, **kwargs)
        self.loss = nn.L1Loss(reduction=reduction)


class WeightedLoss(BaseLoss):
    def __init__(self, weight):
        super().__init__(weight)

    def forward_loss(self, pred, gt, loss_w, *args, **kwargs):
        # Sometimes we simply need a regularizer
        if gt is None and isinstance(pred, torch.Tensor):
            gt = torch.zeros_like(pred)
        loss = super().forward_loss(pred, gt, *args, **kwargs)
        loss = loss * loss_w
        return loss.mean()


@LOSSES.register_module()
class weightedmseloss(WeightedLoss):
    def __init__(self, weight, *args, **kwargs):
        super().__init__(weight, *args, **kwargs)
        self.loss = nn.MSELoss(reduction="none")


@LOSSES.register_module()
class weightedl1loss(WeightedLoss):
    def __init__(self, weight, *args, **kwargs):
        super().__init__(weight, *args, **kwargs)
        self.loss = nn.L1Loss(reduction="none")


@LOSSES.register_module()
class smoothl1loss(ReductionLoss):
    def __init__(self, weight, reduction="mean", *args, **kwargs):
        super().__init__(weight, reduction, *args, **kwargs)
        self.loss = nn.SmoothL1Loss(reduction=reduction)


@LOSSES.register_module()
class kldloss(ReductionLoss):
    def __init__(self, weight, reduction="mean", *args, **kwargs):
        super().__init__(weight, reduction, *args, **kwargs)
        self.reduction_func = getattr(torch, reduction)

    def _forward_loss(self, pred, gt, *args, **kwargs):
        var = pred["var"] + 1e-9  # Sometimes var is so small that could be zero
        logvar = pred["logvar"]
        mu = pred["mu"]
        if gt is None:
            loss = -0.5 * self.reduction_func(1 + logvar - mu.pow(2) - var)
        else:
            p_var = gt["prior_var"]
            p_mu = gt["prior_mu"]
            logvar = var.log()
            p_logvar = p_var.log()
            loss = -0.5 * self.reduction_func(
                logvar - p_logvar - var / p_var - (mu - p_mu).pow(2) / p_var + 1
            )

        return loss
