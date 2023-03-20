import torch
import torch.nn as nn
import torch.nn.functional as F

from .pipeline import SLPipeline
from motion.utils.samp import transform_output as samp_transform_output

from .builder import PIPELINES


@PIPELINES.register_module()
class RegressionPipeline(SLPipeline):
    def output_loss(self, data, pred):
        loss_pairs = {
            "reconstruction": [pred["y_hat"], data["y"]],
        }
        return self.calculate_loss(loss_pairs)


@PIPELINES.register_module()
class RegressionSSPipeline(RegressionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_actions = self.cfg.num_actions

    def scheduled_sampling_processing(self, prev_output, data):
        p_prev = prev_output["y_hat"]
        N = -13 * (6 + self.num_actions)
        p_prev = torch.cat((p_prev[:, :N], data["p_prev"][:, N:]), dim=-1)
        data["p_prev"] = p_prev
        return data

    def select_saved_output(self, data, output, *args, **kwargs):
        y_hat = output["y_hat"].clone().detach()
        y_hat = self._transform_output(y_hat, data)
        return {"y_hat": y_hat}

    def _transform_output(self, y_hat, data):
        y_hat = samp_transform_output(y_hat=y_hat, **data)
        return y_hat


@PIPELINES.register_module()
class VAEPipeline(RegressionSSPipeline):
    def output_loss(self, data, pred):
        loss_pairs = {
            "reconstruction": [pred["y_hat"], data["y"]],
            "kld": [pred],
        }
        return self.calculate_loss(loss_pairs)
