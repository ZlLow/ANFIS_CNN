from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional import r2_score

from models.ANFIS.AbstractANFIS import AbstractANFIS
from models.ANFIS.CNNANFIS import HybridCnnAnfis


def k_fold(n_splits: int, X, y, model: AbstractANFIS, optimizer: torch.optim.Optimizer, batch_size: int, epochs: int, save_path: Optional[str] = None):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    r2_scores = []
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        X_train_fold, X_val_fold = X[train_ids], X[val_ids]
        y_train_fold, y_val_fold = y[train_ids], y[val_ids]
        X_train_scaled = feature_scaler.fit_transform(X_train_fold)
        X_val_scaled = feature_scaler.transform(X_val_fold)

        y_train_scaled = target_scaler.fit_transform(y_train_fold)
        y_val_scaled = target_scaler.fit_transform(y_val_fold)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.fit(train_loader=train_loader, x_val_tensor=X_val_tensor, y_val_tensor=y_val_tensor,
                  optimizer=optimizer, fold=fold, epochs=epochs)
        if save_path is None:
            save_path = "normal"
        val_loss, r2_score = model.predict(X_val_tensor, y_val_tensor, target_scaler=target_scaler,save_path=f"img/{fold + 1}_{save_path}_cnn_anfis_prediction.jpg")
        fold_results.append(val_loss)
        r2_scores.append(r2_score)

    print('--- K-Fold Cross-Validation Results ---')
    print(f'Scores for each fold: {[f"{score:.2f}" for score in fold_results]}')
    mean_rmse = np.mean(fold_results)
    std_rmse = np.std(fold_results)
    print(f'Average RMSE: {mean_rmse:.2f}')
    print(f'Standard Deviation: {std_rmse:.2f}')
    print("-" * 50)
    print(f'Scores for each fold: {[f"{r2_score:.6f}" for r2_score in r2_scores]}')
    mean_r2_score = np.mean(r2_scores)
    std_r2_score = np.std(r2_scores)
    print(f"Average R2 score: {mean_r2_score:.6f}")
    print(f"Standard Deviation: {std_r2_score:.6f}")
    return fold_results, r2_scores


def run_rolling_prediction(X, y, dates, model: HybridCnnAnfis, optimizer: torch.optim.Optimizer, batch_size: int, epochs: int):
    # 1. Split data into initial train set and a test set for rolling
    split_index = int(len(X) * 0.8)
    X_train_np, X_test_np = X[:split_index], X[split_index:]
    y_train_np, y_test_np = y[:split_index], y[split_index:]
    dates_test = dates[split_index:]

    # 2. Fit scalers ONLY on the initial training data
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = feature_scaler.fit_transform(X_train_np)
    y_train_scaled = target_scaler.fit_transform(y_train_np)

    # 3. Initial model training
    print("--- Starting Initial Model Training for Rolling Forecast ---")
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.fit(train_loader=train_loader, optimizer=optimizer, epochs=epochs)
    print("--- Initial Training Complete ---")

    # 4. Rolling prediction loop
    print("\n--- Starting Rolling Forecast ---")
    predictions_unscaled, historical_data = model.rolling_prediction(X_train_np, X_test_np, feature_scaler, target_scaler)

    # 5. Evaluate results
    predictions = np.array(predictions_unscaled)
    actuals = y_test_np.flatten()

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    r2 = r2_score(torch.tensor(actuals), torch.tensor(predictions))

    return predictions, actuals, dates_test, rmse, r2, historical_data

