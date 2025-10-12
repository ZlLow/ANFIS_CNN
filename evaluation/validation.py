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


def run_rolling_prediction(X, y, dates, model_params, epochs, batch_size, lr, forward_days=1):
    """
    Performs a rolling forecast evaluation.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray): Target data.
        dates (list): Dates corresponding to the data.
        model_params (dict): Parameters for the HybridCnnAnfis model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        forward_days (int): The number of days to predict forward in each step.
                             A value of 1 is a simple one-day-ahead rolling forecast.
                             A value > 1 predicts a sequence, using its own previous
                             predictions as input for subsequent steps in the sequence.
    """
    # 1. Split data into initial train set and a test set for rolling
    split_index = int(len(X) * 0.8)
    X_train_np, X_test_np = X[:split_index], X[split_index:]
    y_train_np, y_test_np = y[:split_index], y[split_index:]
    dates_test = dates[split_index:]

    # 2. Fit scalers ONLY on the initial training data to prevent look-ahead bias
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

    # This assumes HybridCnnAnfis is defined and imported
    model = HybridCnnAnfis(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')
    epoch_bar = tqdm(range(epochs), desc="Initial Training", leave=False)

    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_train_loss = np.sqrt(epoch_loss / len(train_loader))
        epoch_bar.set_postfix(train_rmse=f"{avg_epoch_train_loss:.4f}")

    print("--- Initial Training Complete ---")

    # 4. Rolling prediction loop
    print(f"\n--- Starting {forward_days}-Day Forward Rolling Forecast ---")
    model.eval()
    all_predictions_unscaled = []
    historical_X = X_train_np.tolist()

    epoch_bar = tqdm(range(0, len(X_test_np) - forward_days, forward_days), desc="Rolling Prediction", leave=False)

    with torch.no_grad():
        # Iterate through the test set, jumping forward by the forecast horizon
        for i in epoch_bar:
            # Get the last known feature set from the actual historical data
            last_known_features = np.array(historical_X[-1]).reshape(1, -1)

            # This inner loop generates the multi-day forecast sequence
            for _ in range(forward_days):
                # Scale the current feature set
                features_scaled = feature_scaler.transform(last_known_features)
                features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

                # Predict the next step
                prediction_scaled = model(features_tensor)

                # Inverse transform the prediction to its real value
                prediction_unscaled = target_scaler.inverse_transform(prediction_scaled.numpy())
                all_predictions_unscaled.append(prediction_unscaled.item())

                # --- This is the key change for multi-step prediction ---
                # Create the next set of features using the current prediction.
                # We shift the feature window and insert the new prediction as the most recent value.
                last_known_features = np.roll(last_known_features, -1)
                last_known_features[0, -1] = prediction_unscaled.item()

            # Update history with the new, actual feature sets from the test data for the next sequence
            actual_chunk = X_test_np[i: i + forward_days]
            if len(actual_chunk) > 0:
                historical_X.extend(actual_chunk)

    # 5. Evaluate results
    # Ensure predictions and actuals are of the same length for fair evaluation
    num_predictions = len(all_predictions_unscaled)
    predictions = np.array(all_predictions_unscaled)
    actuals = y_test_np.flatten()[:num_predictions]
    eval_dates = dates_test[:num_predictions]

    return predictions, actuals, eval_dates, historical_X

def print_r2_and_rmse(predictions, actuals):
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    r2 = r2_score(torch.tensor(actuals), torch.tensor(predictions))
    pearson = np.corrcoef(predictions, actuals)[0, 1]

    print(f"\n--- Rolling Forecast Results ---")
    print(f"Final RMSE on Test Set: ${rmse:.6f}")
    print(f"Final R2 on Test Set: {r2:.6f}")
    print(f"FInal Pearson on Test Set: {pearson:.6f}")