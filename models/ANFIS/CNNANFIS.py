from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from models.ANFIS.AbstractANFIS import AbstractANFIS, GeneralizedBellMembershipFunc


class HybridCnnAnfis(AbstractANFIS):
    def __init__(self, input_dim: int, num_mfs: int, num_filters: int,
                 criterion: Optional = None, kernel_size:int = 3):
        super(HybridCnnAnfis, self).__init__(input_dim, num_mfs, num_filters, criterion)
        # --- Layer 1: Fuzzification ---
        self.membership_funcs = GeneralizedBellMembershipFunc(num_mfs, input_dim)

        # --- Layer 2 & 3: Firing Strength & Normalization ---
        self.cnn_layers = CNNLayer(self.input_dim, self.num_rules, kernel_size)

        # --- Layer 4: Consequent Parameters ---
        # The size is now based on the number of filters, not the old rule calculation
        self.consequent = nn.Parameter(torch.randn(self.num_rules, self.input_dim + 1))

    def forward(self, x):
        batch_size = x.shape[0]

        # --- Layer 1: Fuzzification ---
        # Output shape: (batch_size, input_dim, num_mfs)
        memberships = self.membership_funcs(x)

        # --- Layer 2 & 3: Firing Strength & Normalization ---
        # Shape: (batch_size, num_rules)
        normalized_firing_strengths = self.cnn_layers(memberships)

        # --- Layer 4 & 5: Consequent and Aggregation  ---
        x_aug = torch.cat([x, torch.ones(batch_size, 1)], dim=1)
        rule_outputs = (self.consequent @ x_aug.T).T
        output = torch.sum(normalized_firing_strengths * rule_outputs, dim=1, keepdim=True)

        return output

    def rolling_prediction(self, X_train_np, X_test_np,feature_scaler, target_scaler):
        self.eval()
        predictions_unscaled = []

        # Use the unscaled training data as the starting history
        historical_X = X_train_np.tolist()

        with torch.no_grad():
            for i in tqdm(range(len(X_test_np)), desc="Rolling Prediction"):
                # Get the last known feature set from history
                last_features = np.array(historical_X[-1]).reshape(1, -1)

                # Scale it using the scaler FITTED ON THE INITIAL TRAINING DATA
                last_features_scaled = feature_scaler.transform(last_features)
                features_tensor = torch.tensor(last_features_scaled, dtype=torch.float32)

                # Predict
                prediction_scaled = self(features_tensor)

                # Inverse transform the prediction to its real price value
                prediction_unscaled = target_scaler.inverse_transform(prediction_scaled.numpy())
                predictions_unscaled.append(prediction_unscaled.item())

                # Update history with the new, actual feature set from the test data
                historical_X.append(X_test_np[i])
        return predictions_unscaled, historical_X

class CNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, drop_out_rate: float = 0.2):
        super(CNNLayer, self).__init__()

        self.activation = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
        )


        self.normalization = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # --- Layer 2: Compute Firing Strength ---
        x = self.activation(x)
        # Reduce the size & get global mean of the pool
        x = x.mean(dim=2)
        # --- Layer 3: Normalization ---
        x = self.normalization(x)
        return x
