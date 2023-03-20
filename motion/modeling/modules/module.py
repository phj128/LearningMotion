import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._build()

    def _build(self):
        raise NotImplementedError
