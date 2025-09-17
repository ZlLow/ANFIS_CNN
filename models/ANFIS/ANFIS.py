from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, from_numpy
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional import r2_score
from tqdm import tqdm

from utils.plots import plot_actual_vs_predicted
from models.ANFIS.AbstractANFIS import AbstractANFIS


class ANFIS(AbstractANFIS):
    def __init__(self,  input_dim: int, num_mfs: int, scaler: Optional = None,
                 criterion: Optional[nn.Module] = None):
        super(ANFIS, self).__init__(input_dim, num_mfs, num_mfs ** input_dim,
                 criterion)

        rule_premises = torch.zeros(self.num_rules, self.input_dim, dtype=torch.long)
        for i in range(self.num_rules):
            temp = i
            for j in range(self.input_dim - 1, -1, -1):
                rule_premises[i, j] = temp % self.num_mfs
                temp //= self.num_mfs
        self.register_buffer('rule_premises', rule_premises)

    def forward(self, x):
        batch_size = x.shape[0]

        # --- Layer 1: Fuzzification ---
        # O_1,i = mu_Ai(x)
        memberships = self.membership_funcs(x)  # (batch_size, input_dim, num_mfs)

        # --- Layer 2: Rule Firing Strengths (Product T-norm) ---
        # w_i = mu_A1(x1) * mu_B1(x2) * ...
        # We use gather to select the appropriate membership value for each rule
        firing_strengths = torch.ones(batch_size, self.num_rules)
        for i in range(self.num_rules):
            # For each rule, get the premise MF indices for each input
            premise_indices = self.rule_premises[i, :]
            for j in range(self.input_dim):
                # Get the membership values for the j-th input
                input_memberships = memberships[:, j, :]
                # Select the specific membership value for this rule's premise
                selected_membership = input_memberships[:, premise_indices[j]]
                firing_strengths[:, i] *= selected_membership

        # --- Layer 3: Normalization ---
        # w_i_bar = w_i / sum(w)
        sum_firing_strengths = torch.sum(firing_strengths, dim=1, keepdim=True)
        normalized_firing_strengths = firing_strengths / (sum_firing_strengths + 1e-10)  # (batch_size, num_rules)

        # --- Layer 4: Consequent ---
        # O_4,i = w_i_bar * f_i = w_i_bar * (p*x + q*y + ... + r)
        # Augment input tensor for the constant term 'r'
        x_aug = torch.cat([x, torch.ones(batch_size, 1)], dim=1)
        # Rule outputs are the linear functions
        rule_outputs = (self.consequent @ x_aug.T).T  # (batch_size, num_rules)

        # --- Layer 5: Aggregation ---
        # O_5,i = sum(w_i_bar * f_i)
        output = torch.sum(normalized_firing_strengths * rule_outputs, dim=1, keepdim=True)

        return output

