import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    def __init__(
        self,
        input_size=None,
        output_size=None,
        hidden_size=None,
        final_softmax=True,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################

    def forward(self, inputs):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################


class PredictionNet(nn.Module):
    def __init__(
        self,
        num_experts=6,
        input_size=1664,
        hidden_size=512,
        output_size=618,
        z_dim=32,
        dropout=0.0,
        MoE_layernum=3,
        **kwargs,
    ):
        super().__init__()
        self.rng = np.random.RandomState(23456)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.z_dim = z_dim
        self.dropout = nn.Dropout(dropout)
        self.layer_num = MoE_layernum
        self._build_model()

    def _build_model(self):
        # Motion network expert parameters
        self._build_layer(self.input_size, self.hidden_size, "l1")
        for i in range(1, self.layer_num - 1):
            self._build_layer(
                self.hidden_size + self.z_dim, self.hidden_size, f"l{i + 1}"
            )
        self._build_layer(
            self.hidden_size + self.z_dim,
            self.output_size,
            f"l{self.layer_num}",
            is_last=True,
        )

    def _build_layer(self, in_channels, out_channels, layer_name, is_last=False):
        w, b = self.init_params(self.num_experts, in_channels, out_channels)
        setattr(self, "w_" + layer_name, w)
        setattr(self, "b_" + layer_name, b)

    def init_params(self, num_experts, input_size, output_size):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################
        return w, b

    def dropout_and_linearlayer(self, inputs, weights, bias):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################
        return x

    def calculate_weight(self, blending_coef, layer_name):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################
        return w, b

    def _forward_layer(self, x, blending_coef, layer_name, use_act=True):
        w, b = self.calculate_weight(blending_coef, layer_name)
        x = self.dropout_and_linearlayer(x, w, b)
        if use_act:
            x = F.elu(x)
        return x

    def forward(self, p_prev, blending_coef, z=None):
        pass
        ############################
        #
        #
        # TODO: Implement here!
        #
        #
        ############################
        return x
