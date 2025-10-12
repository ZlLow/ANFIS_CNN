from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn, from_numpy, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class AbstractANFIS(ABC, nn.Module):
    def __init__(self, input_dim: int, num_mfs: int, num_rules: int, criterion: Optional[nn.Module] = None, feature_scaler: Optional[MinMaxScaler] = None, target_scaler: Optional[MinMaxScaler] = None):
        super(AbstractANFIS, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        self.num_rules = num_rules

        # --- Optimizer ---
        self.optimizer = None
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1)) if feature_scaler is None else feature_scaler
        self.target_scaler = MinMaxScaler(feature_range=(0, 1)) if target_scaler is None else target_scaler
        self.criterion = nn.MSELoss(reduction='mean') if criterion is None else criterion

        # --- Layer 1: Fuzzification ---
        self.membership_funcs = GeneralizedBellMembershipFunc(num_mfs, input_dim)

        # -- Layer 4: Consequent ---
        self.consequent = nn.Parameter(torch.randn(self.num_rules, self.input_dim + 1))


    @abstractmethod
    def forward(self, x):
        pass

    def fit(self, features_X, target_Y, optimizer: Optional[Optimizer], batch_size = 32, epochs= 50):
        self.optimizer = optimizer

        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-6)


        x_val_tensor = torch.tensor(features_X, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(target_Y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(x_val_tensor, y_val_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        epoch_bar = tqdm(range(epochs), desc=f"Training Model", leave=True, dynamic_ncols=True, position=0)
        for _ in epoch_bar:
            self.train()
            epoch_loss = 0.0
            num_batches = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            # Calculate losses and update the epoch progress bar with live metrics
            avg_epoch_train_loss = np.sqrt(epoch_loss / len(loader))
            if x_val_tensor is None or y_val_tensor is None:
                epoch_bar.set_postfix(train_rmse=f"{avg_epoch_train_loss:.4f}")
            else:
                self.eval()
                with torch.no_grad():
                    val_output = self(x_val_tensor)
                    val_loss = torch.sqrt(self.criterion(val_output, y_val_tensor)).item()

                # Step the scheduler with the validation loss
                scheduler.step(val_loss)

                # Use set_postfix to display the latest metrics, including the current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                epoch_bar.set_postfix(train_rmse=f"{avg_epoch_train_loss:.4f}",
                                      val_rmse=f"{val_loss:.4f}",
                                      lr=f"{current_lr:.1e}")
                epoch_bar.update()


        return self, self.feature_scaler, self.target_scaler

    def predict(self, features_X, target_Y):
        self.eval()
        x_val_tensor = torch.tensor(features_X, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(target_Y, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            val_pred_scaled = self(x_val_tensor)
            rmse_scaled = torch.sqrt(self.criterion(val_pred_scaled, y_val_tensor)).item()

            val_preds_np = val_pred_scaled.cpu().numpy().flatten()  # Move to CPU for numpy operations
            y_val_np = y_val_tensor.cpu().numpy().flatten()  # Move to CPU for numpy operations

            if np.std(val_preds_np) == 0 or np.std(y_val_np) == 0:
                pearson_correlation_score = 0.0  # Or -1.0, or a penalty value
            else:
                pearson_correlation_score = np.corrcoef(val_preds_np, y_val_np)[0, 1]

            r2 = r2_score(y_val_np, val_preds_np)

        return val_preds_np, rmse_scaled, pearson_correlation_score, r2

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