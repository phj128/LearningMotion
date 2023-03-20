import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from motion.modeling.modules.builder import build_module


class BaseModel(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.setup(cfg)
        self.init_build(cfg)
        self.build_model(cfg)

    def setup(self, cfg):
        pass

    def init_build(self, cfg):
        pass

    def build_model(self, cfg):
        raise NotImplementedError


class BasePredictor(BaseModel):
    def build_model(self, cfg):
        decoder_cfg = cfg.clone()
        decoder_cfg.TYPE = cfg.DECODER_TYPE
        self.decoder = build_module(decoder_cfg)


class BaseAE(BasePredictor):
    def build_model(self, cfg):
        super().build_model(cfg)
        encoder_cfg = cfg.clone()
        encoder_cfg.TYPE = cfg.ENCODER_TYPE
        self.encoder = build_module(encoder_cfg)


class BaseVAE(BaseAE):
    def reparameterize(self, dist):
        if self.training:
            mu = dist["mu"]
            var = dist["var"]
        else:
            mu = torch.zeros_like(dist["mu"])
            var = torch.ones_like(dist["var"])
        std = var ** 0.5
        eps = torch.randn_like(std)
        return mu + eps * std


class BasePredictorONNX(BaseModel):
    def build_model(self, cfg):
        decoder_cfg = cfg.clone()
        decoder_cfg.TYPE = cfg.DECODER_TYPE + "ONNX"
        self.decoder = build_module(decoder_cfg)
