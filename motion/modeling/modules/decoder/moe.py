import numpy as np
import torch
import torch.nn as nn

from motion.modeling.modules.decoder.mlp import MLPDecoder

from motion.modeling.modules.MoE.MoE import GatingNetwork, PredictionNet
from motion.modeling.modules.encoder.samp import INet

from motion.modeling.modules.builder import MODULES


@MODULES.register_module()
class MoEDecoder(MLPDecoder):
    def __init__(self, num_experts=5, h_dim_gate=10, *args, **kwargs):
        self.num_experts = num_experts
        self.h_dim_gate = h_dim_gate
        super().__init__(*args, **kwargs)

    def _build(self):
        super()._build()
        self._build_gating_network()

    def _build_gating_network(self):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################

    def _build_prediction_network(self):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################

    def forward(self, p_prev, I, **kawrgs):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################
        return {"y_hat": y_hat}


@MODULES.register_module()
class MoEDecoderONNX(MoEDecoder):
    def forward(self, p_prev, I, **kawrgs):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################
        return y_hat
