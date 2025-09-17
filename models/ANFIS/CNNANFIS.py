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
from models.ANFIS.AbstractANFIS import AbstractANFIS, GeneralizedBellMembershipFunc


class HybridCnnAnfis(AbstractANFIS):
    def __init__(self, input_dim: int, num_mfs: int, num_filters: int, scaler: Optional = None,
                 criterion: Optional = None):
        super(HybridCnnAnfis, self).__init__(input_dim, num_mfs, num_filters, scaler, criterion)
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

    def fit(self, x_train_data: pd.DataFrame, y_train_data: pd.DataFrame, x_val_data: pd.DataFrame,
            y_val_data: pd.DataFrame, optimizer: Optional[Optimizer], epochs: int = 300, batch_size: int = 64, fold: int = 0):
        self.optimizer = optimizer

        x_train_data_scaled = self.scaler.fit_transform(x_train_data)
        y_train_data_scaled = self.scaler.fit_transform(y_train_data)
        x_train_tensor = torch.tensor(x_train_data_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_data_scaled, dtype=torch.float32)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        x_val_data_scaled = self.scaler.fit_transform(x_val_data)
        y_val_data_scaled = self.scaler.fit_transform(y_val_data)
        x_val_tensor = torch.tensor(x_val_data_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_data_scaled, dtype=torch.float32)

        epoch_bar = tqdm(range(epochs), desc=f"Fold {fold + 1} Training", leave=False)
        for _ in epoch_bar:
            self.train()
            epoch_train_loss = 0.0
            # Wrap train_loader with tqdm for the inner batch loop
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            # Calculate losses and update the epoch progress bar with live metrics
            avg_epoch_train_loss = np.sqrt(epoch_train_loss / len(train_loader))

            self.eval()
            with torch.no_grad():
                val_output = self(x_val_tensor)
                val_loss = torch.sqrt(self.criterion(val_output, y_val_tensor)).item()

            # Use set_postfix to display the latest metrics
            epoch_bar.set_postfix(train_rmse=f"{avg_epoch_train_loss:.4f}", val_rmse=f"{val_loss:.4f}")


    def predict(self, x_val_data: pd.DataFrame, y_val_data: pd.DataFrame, save_path: Optional[str] = None):
        x_val_data_scaled = self.scaler.fit_transform(x_val_data)
        y_val_data_scaled = self.scaler.fit_transform(y_val_data)
        x_val_tensor = torch.tensor(x_val_data_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_data_scaled, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            val_output_scaled = self(x_val_tensor)

        val_output_prices = self.scaler.inverse_transform(val_output_scaled.numpy())
        y_val_prices = self.scaler.inverse_transform(y_val_tensor.numpy())

        val_loss_unscaled = np.sqrt(
            nn.MSELoss()(torch.tensor(val_output_prices), torch.tensor(y_val_prices))).item()
        print(f"r2 score: {r2_score(from_numpy(y_val_prices), from_numpy(val_output_prices)):6f}")
        plot_actual_vs_predicted(y_val_prices, val_output_prices, save_path)
        return val_loss_unscaled, r2_score(from_numpy(y_val_prices), from_numpy(val_output_prices))


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
