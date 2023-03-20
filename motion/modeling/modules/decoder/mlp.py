import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motion.modeling.modules.encoder.samp import INet
from motion.modeling.operators.mlp import MLPBlock
from motion.modeling.modules.module import BaseModule

from motion.modeling.modules.builder import MODULES


@MODULES.register_module()
class MLPDecoder(BaseModule):
    def __init__(
        self,
        state_dim=647,
        I_in_dim=2048,
        I_out_dim=256,
        pred_in_dim=903,
        h_dim=256,
        dropout=0.3,
        activation="ELU",
        **kwargs
    ):
        self.state_dim = state_dim
        self.I_in_dim = I_in_dim
        self.I_out_dim = I_out_dim
        self.pred_in_dim = pred_in_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.activation = activation
        self.kwargs = kwargs
        super().__init__()

    def _build(self):
        self.INet = INet(self.I_in_dim, self.I_out_dim)
        self._build_prediction_network()

    def _build_prediction_network(self):
        self.prediction_net = nn.Sequential(
            MLPBlock(
                self.pred_in_dim,
                self.h_dim,
                dropout=self.dropout,
                activation=self.activation,
                reset_paramter=True,
            ),
            MLPBlock(
                self.h_dim,
                self.h_dim,
                dropout=self.dropout,
                activation=self.activation,
                reset_paramter=True,
            ),
            MLPBlock(
                self.h_dim,
                self.state_dim,
                dropout=self.dropout,
                activation=self.activation,
                reset_paramter=True,
            ),
        )

    def forward(self, p_prev, I, **kawrgs):
        I = self.INet(I)
        x = torch.cat((p_prev, I), dim=-1)
        y_hat = self.prediction_net(x)
        return {"y_hat": y_hat}


@MODULES.register_module()
class MLPDecoderONNX(MLPDecoder):
    def forward(self, p_prev, I):
        I = self.INet(I)
        x = torch.cat((p_prev, I), dim=-1)
        y_hat = self.prediction_net(x)
        return y_hat
