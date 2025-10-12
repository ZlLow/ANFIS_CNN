from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models.ANFIS.AbstractANFIS import AbstractANFIS, GeneralizedBellMembershipFunc
from trading.features import calculate_vanilla_macd, calculate_rsi, calculate_bollinger_width


class ConsequentGenerator(nn.Module):
    def __init__(self, input_dim, num_mfs, num_rules, num_conv_filters):
        super(ConsequentGenerator, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_conv_filters, kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(start_dim=1)  # Flatten the output for the linear layer
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, num_mfs)
            dummy_output = self.conv_net(dummy_input)
            linear_input_size = dummy_output.shape[1]

        self.fc_net = nn.Linear(linear_input_size, num_rules * (input_dim + 1))

        # Store these for reshaping in the forward pass
        self.num_rules = num_rules
        self.output_dim_per_rule = input_dim + 1

    def forward(self, memberships):
        # memberships shape: (batch_size, input_dim, num_mfs)
        conv_features = self.conv_net(memberships)

        # fc_net outputs a flat vector of all parameters for all rules
        # Shape: (batch_size, num_rules * (input_dim + 1))
        flat_params = self.fc_net(conv_features)

        # Reshape to get the parameters for each rule separately
        # Shape: (batch_size, num_rules, input_dim + 1)
        dynamic_params = flat_params.view(-1, self.num_rules, self.output_dim_per_rule)

        return dynamic_params


class HybridCnnAnfis(AbstractANFIS):
    def __init__(self, input_dim: int, num_mfs, num_rules, firing_conv_filters, consequent_conv_filters,
                 feature_scaler: Optional[MinMaxScaler] = None, target_scaler: Optional[MinMaxScaler] = None, device=torch.device('cpu')):
        super().__init__(input_dim=input_dim, num_mfs=num_mfs, num_rules=num_rules, feature_scaler=feature_scaler, target_scaler=target_scaler)
        self.device = device
        # Layer 1: Fuzzification
        self.membership_funcs = GeneralizedBellMembershipFunc(num_mfs, input_dim)

        # Head 1: Firing Strength Network
        self.firing_strength_net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=firing_conv_filters, kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(device)
        self.firing_fc = nn.Linear(firing_conv_filters, num_rules).to(device)
        self.batch_norm = nn.BatchNorm1d(num_rules).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

        # Head 2: Dynamic Consequent Parameter Network
        self.consequent_generator = ConsequentGenerator(
            input_dim, num_mfs, num_rules, consequent_conv_filters
        ).to(device)

        self.output_projection = nn.Linear(1, 1).to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        # Always run through the standard path unless there's a specific reason for _forward_single
        # The original code's _forward_single path only removes batch_norm if batch_size == 1
        # For evaluation during GA, we want consistent behavior.

        # Layer 1: Fuzzification
        memberships = self.membership_funcs(x)

        # Head 1: Calculate Firing Strengths (w̄ᵢ)
        # Ensure memberships has appropriate dimensions for Conv1d: (batch_size, input_dim, num_mfs)
        # It comes out as (batch_size, input_dim, num_mfs) from membership_funcs
        firing_features = self.firing_strength_net(memberships)

        # Mean across the num_mfs dimension (the "time" dimension for Conv1d here)
        firing_features = firing_features.mean(dim=2)

        firing_strength_logits = self.firing_fc(firing_features)

        # Apply BatchNorm only if batch_size > 1, otherwise it will error
        if batch_size > 1:
            normalized_firing_strengths = self.softmax(self.batch_norm(firing_strength_logits))
        else:
            normalized_firing_strengths = self.softmax(firing_strength_logits)

        # Head 2: Generate Consequent Parameters (a, b, c...)
        dynamic_consequent_params = self.consequent_generator(memberships)

        # Layer 4 & 5: Consequent Calculation and Aggregation
        # Add a column of ones to x for the constant term in the consequent
        x_aug = torch.cat([x, torch.ones(batch_size, 1, device=self.device)], dim=1).unsqueeze(1)

        # Multiply consequent parameters by augmented input and sum
        # dynamic_consequent_params: (batch_size, num_rules, input_dim + 1)
        # x_aug: (batch_size, 1, input_dim + 1)
        # The broadcasting works here: x_aug is applied to each rule's params
        rule_outputs = (dynamic_consequent_params * x_aug).sum(dim=2)  # Result (batch_size, num_rules)

        # Weighted sum of rule outputs based on firing strengths
        # normalized_firing_strengths: (batch_size, num_rules)
        # rule_outputs: (batch_size, num_rules)
        aggregated_output = (normalized_firing_strengths * rule_outputs).sum(dim=1, keepdim=True)
        return aggregated_output

    def rolling_prediction(self, initial_unscaled_close_history, train_X_scaled, train_y_scaled,
                           X_test_scaled, y_test_scaled, test_dates,
                           look_forward_period):
        all_predictions, all_actuals, all_dates = [], [], []

        # History will be managed with scaled data
        historical_X_scaled = train_X_scaled.copy()
        historical_y_scaled = train_y_scaled.copy()

        # Unscaled history is needed ONLY for dynamic feature calculation
        unscaled_close_history = initial_unscaled_close_history.copy()

        for i in tqdm(range(0, len(X_test_scaled), look_forward_period), desc="Rolling Prediction Windows"):
            # 1. Retrain the model on all scaled data seen so far
            print(f"\nRetraining model with {len(historical_X_scaled)} data points...")

            # 2. Initialize for the prediction window
            last_features_scaled = historical_X_scaled[-1, :].reshape(1, -1)
            temp_unscaled_close_history = unscaled_close_history.copy()

            for j in range(look_forward_period):
                current_idx = i + j
                if current_idx >= len(X_test_scaled): break

                # 3. Predict one step ahead (output is scaled)
                self.eval()
                with torch.no_grad():
                    features_tensor = torch.tensor(last_features_scaled, dtype=torch.float32).to(self.device)
                    scaled_prediction = self(features_tensor)

                # 4. Inverse transform prediction to get unscaled value for feature calculation
                unscaled_prediction = self.target_scaler.inverse_transform(scaled_prediction.cpu().numpy())[0, 0]
                all_predictions.append(unscaled_prediction)
                temp_unscaled_close_history = pd.concat([temp_unscaled_close_history, pd.Series([unscaled_prediction])],
                                                        ignore_index=True)

                # 5. Store actual values for metrics
                unscaled_actual = self.target_scaler.inverse_transform(y_test_scaled[current_idx].reshape(-1, 1))[0, 0]
                all_actuals.append(unscaled_actual)
                all_dates.append(test_dates.iloc[current_idx])

                # 6. Calculate next features using the unscaled history
                temp_df = pd.DataFrame({'Close': temp_unscaled_close_history})
                macd, signal = calculate_vanilla_macd(temp_df['Close'])
                rsi = calculate_rsi(temp_df)
                bb_width = calculate_bollinger_width(temp_df['Close'])

                next_features_unscaled = np.array(
                    [unscaled_prediction, macd.iloc[-1], signal.iloc[-1], rsi.iloc[-1], bb_width.iloc[-1]]).reshape(1,
                                                                                                                    -1)

                # 7. Scale the new features to be the input for the next prediction
                if not np.all(np.isfinite(next_features_unscaled)):
                    print(f"Warning: NaN/inf features at step {current_idx}. Using last valid features.")
                else:
                    last_features_scaled = self.feature_scaler.transform(next_features_unscaled)

            # 8. Update history with actual scaled data from the test set for the next retraining cycle
            actual_X_chunk_scaled = X_test_scaled[i: i + look_forward_period]
            actual_y_chunk_scaled = y_test_scaled[i: i + look_forward_period]

            historical_X_scaled = np.concatenate((historical_X_scaled, actual_X_chunk_scaled), axis=0)
            historical_y_scaled = np.concatenate((historical_y_scaled, actual_y_chunk_scaled), axis=0)

            # Also update the unscaled history with the true unscaled values
            actual_unscaled_chunk = self.target_scaler.inverse_transform(actual_y_chunk_scaled)
            unscaled_close_history = pd.concat([unscaled_close_history, pd.Series(actual_unscaled_chunk.flatten())],
                                               ignore_index=True)

        return np.array(all_predictions), np.array(all_actuals), all_dates