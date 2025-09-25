from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, from_numpy, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.functional import r2_score
from tqdm import tqdm

from trade_utils.plotter import plot_actual_vs_predicted


class AbstractANFIS(ABC, nn.Module):
    def __init__(self, input_dim: int, num_mfs: int, num_rules: int, criterion: Optional[nn.Module] = None ):
        super(AbstractANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        self.num_rules = num_rules

        # --- Optimizer ---
        self.optimizer = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.criterion = nn.MSELoss(reduction='mean') if criterion is None else criterion

        # --- Layer 1: Fuzzification ---
        self.membership_funcs = GeneralizedBellMembershipFunc(num_mfs, input_dim)

        # -- Layer 4: Consequent ---
        self.consequent = nn.Parameter(torch.randn(self.num_rules, self.input_dim + 1))


    @abstractmethod
    def forward(self, x):
        pass

    def fit(self, train_loader: DataLoader, optimizer: Optional[Optimizer],
            x_val_tensor: Optional[Tensor] = None, y_val_tensor: Optional[Tensor] = None,
            epochs: int = 300, fold: int = 0):
        self.optimizer = optimizer

        # --- ADD THIS: Initialize the Learning Rate Scheduler ---
        # It will reduce the LR if `val_loss` does not improve for 10 epochs.
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-6)

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
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()

            # Calculate losses and update the epoch progress bar with live metrics
            avg_epoch_train_loss = np.sqrt(epoch_train_loss / len(train_loader))
            if x_val_tensor is None or y_val_tensor is None:
                epoch_bar.set_postfix(train_rmse=f"{avg_epoch_train_loss:.4f}")
            else:
                self.eval()
                with torch.no_grad():
                    val_output = self(x_val_tensor)
                    val_loss = torch.sqrt(self.criterion(val_output, y_val_tensor)).item()

                # --- ADD THIS: Step the scheduler with the validation loss ---
                scheduler.step(val_loss)

                # Use set_postfix to display the latest metrics, including the current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                epoch_bar.set_postfix(train_rmse=f"{avg_epoch_train_loss:.4f}",
                                      val_rmse=f"{val_loss:.4f}",
                                      lr=f"{current_lr:.1e}")

    def predict(self, x_val_tensor: Tensor, y_val_tensor: Tensor,target_scaler, dates: Optional[str] = None, save_path: Optional[str] = None):
        self.eval()
        with torch.no_grad():
            val_output_scaled = self(x_val_tensor)

        val_output_prices = target_scaler.inverse_transform(val_output_scaled.numpy())
        y_val_prices = target_scaler.inverse_transform(y_val_tensor.numpy())

        val_loss_unscaled = np.sqrt(
            nn.MSELoss()(torch.tensor(val_output_prices), torch.tensor(y_val_prices))).item()
        print(f"r2 score: {r2_score(from_numpy(y_val_prices), from_numpy(val_output_prices)):6f}")
        plot_actual_vs_predicted(y_val_prices, val_output_prices,dates=dates, save_path=save_path)
        return val_loss_unscaled, r2_score(from_numpy(y_val_prices), from_numpy(val_output_prices))

class GeneralizedBellMembershipFunc(nn.Module):
    def __init__(self, num_mfs, input_dim):
        super(GeneralizedBellMembershipFunc, self).__init__()
        # ... (new initialization for a, b, c as described above) ...
        self.a = nn.Parameter(torch.rand(num_mfs, input_dim) * 0.5 + 0.1)
        self.b = nn.Parameter(torch.rand(num_mfs, input_dim) * 2 + 0.5)
        self.c = nn.Parameter(torch.rand(num_mfs, input_dim) * 1.0)  # Assuming inputs scaled 0-1

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(2)
        a_transposed = self.a.t().unsqueeze(0)
        b_transposed = self.b.t().unsqueeze(0)
        c_transposed = self.c.t().unsqueeze(0)

        # Add a small epsilon to 'a' to prevent division by absolute zero
        # Use a_transposed + 1e-6 to ensure it's always slightly positive
        denominator_term = torch.abs((x_unsqueezed - c_transposed) / (a_transposed + 1e-6))

        # Handle potential large exponents
        # Clamp b_transposed to a reasonable range if it tends to explode
        b_clamped = torch.clamp(b_transposed, min=0.1, max=5.0)  # Example range, adjust if needed

        return 1. / (1. + denominator_term ** (2 * b_clamped))