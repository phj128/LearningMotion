import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm=None,
        activation=None,
        dropout=None,
        bias=True,
        reset_paramter=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        if norm is not None:
            norm = getattr(nn, norm)(out_channels)
        if activation is not None:
            activation = getattr(nn, activation)()
        if dropout is not None:
            assert isinstance(dropout, (float, int))
            dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = norm
        self.activation = activation
        self.dropout = dropout
        if reset_paramter:
            self._reset_paramters()

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): [*, in_channels]

        Returns:
            x (tensor): [*, out_channels]
        """
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def _reset_paramters(self):
        w_bound = np.sqrt(6.0 / np.prod([self.in_channels, self.out_channels]))
        nn.init.uniform_(self.fc.weight, -w_bound, w_bound)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
