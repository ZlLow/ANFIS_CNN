import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional import r2_score

from models.ANFIS.CNNANFIS import HybridCnnAnfis
from utils.plots import plot_actual_vs_predicted, plt_rmse_table, plt_r2_score_table, plot_graph
from utils.validation import k_fold

price_scaler = MinMaxScaler(feature_range=(0, 1))

def load_data():
    # 1. Load Data
    try:
        df = pd.read_csv('stock_data.csv')
        df.dropna(how='all', inplace=True)
        # Ensure 'Close' is numeric
        df['Close'] = pd.to_numeric(df['Close'])
    except FileNotFoundError:
        print("Error: 'stock_data.csv' not found. This script requires the specific file.")
        exit()
    return df

def scale_data(df):
    # 2. Feature Engineering with Rolling Means
    df['Close_Scaled'] = price_scaler.fit_transform(df[['Close']])

    # Calculate rolling means for different windows
    windows = [1, 5, 10, 15]
    feature_cols = []
    for w in windows:
        col_name = f'SMA_{w}'
        # Calculate rolling mean and shift it to prevent data leakage
        df[col_name] = df['Close_Scaled'].rolling(window=w).mean().shift(1)
        feature_cols.append(col_name)

    # Drop rows with NaN values created by the rolling windows
    df.dropna(inplace=True)

    # Define the target column
    target_col = 'Close_Scaled'
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.reshape(-1, 1).astype(np.float32)
    print(f"Feature engineering complete. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols, windows

def split_data(df, X, y):
    split_index = int(len(X) * 0.8)
    X_train_np, X_test_np = X[:split_index], X[split_index:]
    y_train_np, y_test_np = y[:split_index], y[split_index:]

    prices_train = df['Close_Scaled'].iloc[:split_index].tolist()
    prices_test = df['Close_Scaled'].iloc[split_index:].tolist()
    return X_train_np, X_test_np, y_train_np, y_test_np, prices_train, prices_test

def eval_rolling_predict():
    df = load_data()
    X,y, feature_cols, windows = scale_data(df)
    X_train_np, X_test_np, y_train_np, y_test_np, prices_train, prices_test = split_data(df, X, y)

    batch_size = 64
    epochs = 200
    learning_rate = 0.02
    num_filters = 27
    input_dim = len(feature_cols)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale features based ONLY on the training set
    X_train_scaled = feature_scaler.fit_transform(X_train_np)

    # Convert to Tensors for training
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss(reduction='mean')
    model = HybridCnnAnfis(input_dim=input_dim, num_mfs=3, num_filters=num_filters, criterion=criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.fit(train_loader=train_loader,optimizer=optimizer, epochs=epochs, batch_size=batch_size)
    predictions_scaled, historical_data = model.rolling_prediction(prices_train, prices_test, windows, feature_scaler)
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)

    # Inverse transform both predictions and actuals to the original price scale
    predicted_prices = price_scaler.inverse_transform(predictions_scaled)
    actual_prices = price_scaler.inverse_transform(y_test_np)

    # Calculate final RMSE on unscaled prices
    rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
    print(f"\n--- Rolling Forecast Results ---")
    print(f"Final RMSE on Test Set: ${rmse:.2f}")
    r2 = r2_score(torch.tensor(predicted_prices), torch.tensor(actual_prices))
    print(f"Final R2 on Test Set: {r2:.6f}")

    plot_actual_vs_predicted(actual_prices, predicted_prices, "img/rolling_prediction.jpg")
    plot_graph(historical_data, "img/historical_data.jpg")


def eval_k_fold():
    df = load_data()
    X,y, feature_cols, _ = scale_data(df)

    # --- K-Fold Cross-Validation Setup ---
    k_folds = 5
    epochs = 200
    batch_size = 64
    learning_rate = 0.01
    num_filters = 27
    input_dim = len(feature_cols)

    criterion = nn.MSELoss(reduction='mean')
    model = HybridCnnAnfis(input_dim=input_dim, num_mfs=3, num_filters=num_filters, criterion=criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    fold_results, r2_scores, mean_rmse, std_rmse, mean_r2_score, std_r2_score = k_fold(k_folds, X, y, model, optimizer, batch_size, epochs)
    plt_rmse_table(fold_results, mean_rmse, std_rmse, "img/k_fold_rmse_tables.jpg")
    plt_r2_score_table(fold_results, mean_r2_score, std_r2_score, "img/k_fold_r2_score_tables.jpg")

def main():
    #eval_k_fold()
    eval_rolling_predict()

if __name__ == '__main__':
    main()