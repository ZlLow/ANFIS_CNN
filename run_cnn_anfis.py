import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from models.ANFIS.CNNANFIS import HybridCnnAnfis
from utils.validation import k_fold

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
    # First, create a scaled version of the close price for feature calculation
    price_scaler = MinMaxScaler(feature_range=(0, 1))
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
    return X, y, feature_cols



def main():
    df = load_data()
    X,y, feature_cols = scale_data(df)

    # --- K-Fold Cross-Validation Setup ---
    k_folds = 5
    epochs = 350
    batch_size = 100
    learning_rate = 0.5
    num_filters = 27
    input_dim = len(feature_cols)

    criterion = nn.MSELoss(reduction='mean')
    model = HybridCnnAnfis(input_dim=input_dim, num_mfs=3, num_filters=num_filters, criterion=criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    k_fold(k_folds, X, y, model, optimizer, batch_size, epochs)

if __name__ == '__main__':
    main()