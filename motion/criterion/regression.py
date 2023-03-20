import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from motion.utils.loss import compute_kld
from .criterion import SimpleCriterion
from .builder import CRITERIONS


@CRITERIONS.register_module()
class RegressionCriterion(SimpleCriterion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_types = ["reconstruction", "kld"]

    def _compute_result(self, pred, gt):
        rec_error = F.mse_loss(pred["y_hat"], gt["y"]).item()
        kld = self._compute_kld(pred)
        errors = {
            "reconstruction": rec_error,
            "kld": kld,
        }

        return errors, None

    def _compute_kld(self, pred):
        if "mu" not in pred.keys():
            return 0.0
        mu = pred["mu"]
        var = pred["var"]
        if "prior_mu" in pred.keys():
            p_mu = pred["prior_mu"]
            p_var = pred["prior_var"]
            kld = compute_kld(mu, var, p_mu, p_var)
        else:
            kld = compute_kld(mu, var)
        return kld.item()
