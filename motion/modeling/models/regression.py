import torch
import torch.nn as nn
import torch.nn.functional as F
from motion.modeling.model import BaseVAE, BasePredictor, BasePredictorONNX
from motion.modeling.builder import MODELS


@MODELS.register_module()
class MotionPredictor(BasePredictor):
    def forward(self, data):
        output = self.decoder(**data)
        return output


@MODELS.register_module()
class MotionNet(BaseVAE):
    def forward(self, data):
        output = {}
        enc_output = self.encoder(**data)
        output.update(enc_output)
        z = self.reparameterize(enc_output)
        data["z"] = z
        dec_output = self.decoder(**data)
        output.update(dec_output)
        return output


@MODELS.register_module()
class MotionPredictorONNX(BasePredictorONNX):
    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


@MODELS.register_module()
class MotionNetONNX(MotionPredictorONNX):
    pass
