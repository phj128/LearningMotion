import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motion.modeling.operators.mlp import MLPBlock
from motion.modeling.modules.module import BaseModule
from motion.modeling.operators.activation import ACTIVATIONS
from motion.modeling.modules.builder import MODULES


class INet(nn.Module):
    def __init__(self, in_dim=2048, out_dim=256, **kwargs):
        super().__init__()
        self.I = nn.Sequential(
            MLPBlock(in_dim, 256, activation="ELU"),
            MLPBlock(256, 256, activation="ELU"),
            MLPBlock(256, out_dim, activation="ELU"),
        )

    def forward(self, I):
        return self.I(I)


@MODULES.register_module()
class SAMPEncoder(BaseModule):
    def __init__(
        self,
        state_dim=647,
        z_dim=32,
        I_in_dim=2048,
        I_out_dim=256,
        activation="ELU",
        **kwargs
    ):
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.I_in_dim = I_in_dim
        self.I_out_dim = I_out_dim
        self.activation = activation
        self.kwargs = kwargs
        super().__init__()

    def _build(self):
        self._build_encoder_network()
        self._build_predcition_layer()

    def _build_encoder_network(self):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################

    def _build_predcition_layer(self):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################

    def forward(self, p_prev, y, I, **kwargs):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################
        return {"mu": mu, "logvar": logvar, "var": var}
