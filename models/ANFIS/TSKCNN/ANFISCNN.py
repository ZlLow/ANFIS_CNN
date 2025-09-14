from typing import Optional

import torch
from torch import nn

from models.ANFIS.AbstractANFIS import AbstractANFIS, _MFLayer, _ConsequenceLayer
from models.CNN.Encoder import Encoder
from models.CNN.fuzzy.DefuzzyLayer import NormalizationLayer
from models.CNN.fuzzy.FuzzyLayer import FuzzyLayer

class ANFISCNN(AbstractANFIS):
    def __init__(self, n_input: int,membfuncs: Optional[list], to_device: Optional[str] = None,
                 drop_out_rate: Optional[int] = 0.2):
        super(ANFISCNN).__init__(n_input, to_device=to_device, drop_out_rate=drop_out_rate)

        self.layers = nn.ModuleDict({
            # Layer 1 - Membership Function
            'fuzzylayer': _MFLayer(membfuncs),

            # Layer 2 & 3 - CNN
            'cnn': _CNNLayer(self.num_rules, self.n_input, self.n_statevars),

            # Layer 4 - Consequence
            'consequence': _ConsequenceLayer(self._n, self._rules)
        })

    def _reset_model_parameter(self):
        optlclass = self.optimizer.__class__
        self.optimizer = optlclass(self.parameters(), lr=self.optimizer.__dict__['param_groups'][0]['lr'])

        with torch.no_grad():
            for layer in self.layers.values():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()


class _CNNLayer(nn.Module):
    def __init__(self, fuzzy_dims, in_channel, out_channel, drop_out_rate=0.2):
        super(_CNNLayer, self).__init__()
        self.encoder = Encoder(fuzzy_dims, in_channel, drop_out_rate=drop_out_rate)
        self.fuzzy = nn.Sequential(
            FuzzyLayer.from_dimensions(fuzzy_dims, in_channel),
            NormalizationLayer.from_dimensions(in_channel, out_channel),
        )

    def forward(self, input_):
        batch_size = input_.shape[0]
        x, mu, logvar, z = self.encoder(input_)
        output = self.fuzzy(x)
        output = output.reshape(batch_size, self._out_channel)
        return output, mu, logvar, z
