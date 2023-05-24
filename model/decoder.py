import numpy as np

import torch
import torch.nn as nn

from torch.distributions.normal import Normal

from model.modules import get_backbone


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.decoder_params = params.decoder_params

        self.backbone, out_shape = get_backbone(params, encoding=False)

        # (c, h, w)
        image_spec = np.concatenate([obs for obs in params.obs_spec.values() if obs.ndim == 3], axis=0).shape
        assert out_shape == image_spec

        self.to(params.device)

    def forward(self, feature):
        reshaped = False
        if feature.ndim > 2:
            reshaped = True
            bs_shape, feature_dim = feature.shape[:-1], feature.shape[-1]
            feature = feature.view(-1, feature_dim)

        reconstruction = self.backbone(feature)

        if reshaped:
            reconstruction = reconstruction.view(*bs_shape, *reconstruction.shape[-3:])

        return Normal(reconstruction, torch.ones_like(reconstruction))
