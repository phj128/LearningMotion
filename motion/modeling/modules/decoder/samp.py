import numpy as np
import torch
import torch.nn as nn

from motion.modeling.modules.MoE.MoE import GatingNetwork, PredictionNet
from motion.modeling.modules.decoder.moe import MoEDecoder

from motion.modeling.modules.builder import MODULES


@MODULES.register_module()
class SAMPDecoder(MoEDecoder):
    def __init__(self, z_dim=64, *args, **kwargs):
        self.z_dim = z_dim
        super().__init__(*args, **kwargs)

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

    def forward(self, z, p_prev, I, **kawrgs):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################
        return {"y_hat": y_hat}
