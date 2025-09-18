from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from models.ANFIS.AbstractANFIS import AbstractANFIS, GeneralizedBellMembershipFunc


class HybridCnnAnfis(AbstractANFIS):
    def __init__(self, input_dim: int, num_mfs: int, num_filters: int,
                 criterion: Optional = None):
        super(HybridCnnAnfis, self).__init__(input_dim, num_mfs, num_filters, criterion)
        # --- Layer 1: Fuzzification ---
        self.membership_funcs = GeneralizedBellMembershipFunc(num_mfs, input_dim)

        # --- Layer 2 & 3: Firing Strength & Normalization ---
        self.cnn_layers = CNNLayer(self.input_dim, self.num_rules)

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

    def rolling_prediction(self, prices_train, prices_test, windows, scaler):
        # 5. Rolling Forecast Prediction Loop
        print("\n--- Starting Rolling Forecast Evaluation ---")
        self.eval()
        predictions_scaled = []
        # This history will be updated with actual values from the test set
        historical_data = prices_train
        with torch.no_grad():
            for i in tqdm(range(len(prices_test)), desc="Rolling Prediction"):
                # 1. Create features for the current step
                # Convert the history to a pandas Series to use .rolling()
                history_series = pd.Series(historical_data)
                current_features = []
                for w in windows:
                    # Calculate the rolling mean for the LAST point in the series
                    current_features.append(history_series.rolling(window=w).mean().iloc[-1])

                # 2. Scale the features and convert to tensor
                current_features_np = np.array(current_features).reshape(1,-1)
                current_features_scaled = scaler.transform(current_features_np)
                features_tensor = torch.tensor(current_features_scaled, dtype=torch.float32)

                # 3. Make a prediction
                prediction = self(features_tensor)
                predictions_scaled.append(prediction.item())

                # 4. Update history with the ACTUAL value from the test set
                historical_data.append(prices_test[i])
        return predictions_scaled, historical_data

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
