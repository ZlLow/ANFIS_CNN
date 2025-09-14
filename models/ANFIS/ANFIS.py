import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from models.ANFIS.AbstractANFIS import AbstractANFIS, _MFLayer, _RuleLayer, _ConsequenceLayer


class ANFIS(AbstractANFIS):
    def __init__(self, n_input: int, membfuncs: list, to_device:Optional[str] = None, drop_out_rate: Optional[int]=0.2):
        super(ANFIS, self).__init__(n_input=n_input, to_device=to_device, drop_out_rate=drop_out_rate, membfuncs=membfuncs)

        self.layers = nn.ModuleDict({
            # Layer 1 - Membership Function
            'fuzzylayer': _MFLayer(membfuncs),

            # Layer 2 - Compute Firing
            'rules': _RuleLayer(),

            # Layer 4 - Consequence
            'consequence': _ConsequenceLayer(self._rules, self._n)
        })

    def _reset_model_parameter(self):
        optlclass = self.optimizer.__class__
        self.optimizer = optlclass(self.parameters(), lr=self.optimizer.__dict__['param_groups'][0]['lr'])

        with torch.no_grad():
            for layer in self.layers.values():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()


    def forward(self, S_batch: torch.Tensor, X_batch: torch.Tensor) -> torch.Tensor:
        # Layer 1 - fuzzyfication
        fuzzification = self.layers['fuzzylayer'](S_batch)

        # Layer 2 - Compute Firing Strength
        firing_strength = self.layers['rules'](fuzzification)

        # Layer 3 - Normalization Layer
        normalized = F.normalize(firing_strength, p=1, dim=1)

        # Layer 4 - Consequence Layer
        consequences = self.layers['consequence'](X_batch, normalized)

        # Layer 5 - Summation
        summation = consequences.sum(axis=1).reshape(-1,1)

        return summation

    @property
    def premise(self):
        return [level.coeffs for level in self.layers.fuzzylayer.fuzzyfication]

    @premise.setter
    def premise(self, new_memberships: list):
        self.layers.fuzzylayer = _MFLayer(new_memberships)
        self._initial_premise = self.premise

    @property
    def consequence(self):
        return self.layers['consequence'].coeffs

    @consequence.setter
    def consequence(self, new_consequence: dict):
        self.layers['consequence'].coeffs = new_consequence

    @property
    def scaling_params(self):
        return self.scaler.__dict__