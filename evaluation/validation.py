from torchmetrics.functional import r2_score

from models.ANFIS.CNNANFIS import HybridCnnAnfis

# trainer.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score


def k_fold(k_folds, X, y, model_params, epochs, batch_size, lr):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_rmse_results = []
    fold_r2_results = []

    for fold, (train_ids, val_ids) in tqdm(enumerate(kfold.split(X)), total=k_folds, desc="K-Folds"):
        # Split data for the current fold
        X_train_fold, X_val_fold = X[train_ids], X[val_ids]
        y_train_fold, y_val_fold = y[train_ids], y[val_ids]

        # --- CORRECT SCALING PROCEDURE ---
        # 1. Fit scalers ONLY on the training data of this fold
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        X_train_scaled = feature_scaler.fit_transform(X_train_fold)
        y_train_scaled = target_scaler.fit_transform(y_train_fold)

        # 2. Transform the validation data using the FITTED scalers
        X_val_scaled = feature_scaler.transform(X_val_fold)
        y_val_scaled = target_scaler.transform(y_val_fold)
        # --- END CORRECT SCALING ---

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = HybridCnnAnfis(**model_params)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training Loop
        epoch_bar = tqdm(range(epochs), desc=f"Fold {fold + 1} Training", leave=False)
        epoch_train_loss = 0.0
        for epoch in epoch_bar:
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

        avg_epoch_train_loss = np.sqrt(epoch_train_loss / len(train_loader))


        # Evaluation
        model.eval()
        with torch.no_grad():
            val_output_scaled = model(X_val_tensor)

            # Inverse transform to get loss in the original price scale
            val_output_prices = target_scaler.inverse_transform(val_output_scaled.numpy())
            y_val_prices = target_scaler.inverse_transform(y_val_tensor.numpy())

            rmse = np.sqrt(np.mean((val_output_prices - y_val_prices) ** 2))
            r2 = r2_score(torch.tensor(y_val_prices), torch.tensor(val_output_prices))
            fold_rmse_results.append(rmse)
            fold_r2_results.append(r2)
        epoch_bar.set_postfix(train_rmse=f"{avg_epoch_train_loss:.4f}")

    return fold_rmse_results, fold_r2_results



def print_r2_and_rmse(predictions, actuals):
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    r2 = r2_score(torch.tensor(actuals), torch.tensor(predictions))
    pearson = np.corrcoef(predictions, actuals)[0, 1]

    print(f"\n--- Rolling Forecast Results ---")
    print(f"Final RMSE on Test Set: ${rmse:.6f}")
    print(f"Final R2 on Test Set: {r2:.6f}")
    print(f"FInal Pearson on Test Set: {pearson:.6f}")