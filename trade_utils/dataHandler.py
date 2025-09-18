import os

import numpy as np
import pandas as pd
import yfinance as yf

from trade_utils.trading_indicator import target_log_return, log_return, intraday_log_return, over_night_return, \
    day_range, notional_traded, notional_traded_change, up_days_count, momentum, momentum_change, returns_std, \
    coefficient_of_variation


def get_data(ticker, start_date, end_date, num_horizons):
    """
    Main function to download, cache, and preprocess data for a single ticker.
    """
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = f"{data_dir}/{ticker}_{start_date}_{end_date}.pkl"

    if os.path.exists(file_path):
        data = pd.read_pickle(file_path)
        print(f"Loaded cached data for {ticker} from {file_path}")
    else:
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data downloaded for ticker {ticker}")
        data.to_pickle(file_path)
        print(f"Data downloaded and cached at {file_path}")

    return preprocess(data, num_horizons)


def load_data_from_excel(filepath:str, df_name: str):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.replace('^\ufeff', '', regex=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date']).reset_index(drop=True)
    for col in ['Close', 'Volume', 'Open', 'High', 'Low']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Close'],inplace=True)
    print(f"Loaded {df_name} data from Excel.")
    return df


def load_and_engineer_features(filepath, windows):
    """
    Loads stock data from a CSV and engineers rolling mean features.
    NO SCALING is performed here to prevent look-ahead bias.
    """
    try:
        df = pd.read_csv(filepath)
        df.dropna(how='all', inplace=True)
        df['Close'] = pd.to_numeric(df['Close'])
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        return None, None

    # Calculate features based on the unscaled 'Close' price
    feature_cols = []
    for w in windows:
        col_name = f'SMA_{w}'
        df[col_name] = df['Close'].rolling(window=w).mean().shift(1)
        feature_cols.append(col_name)

    df.dropna(inplace=True)

    X = df[feature_cols].values.astype(np.float32)
    # The target is the 'Close' price itself. Scaling will be handled later.
    y = df['Close'].values.reshape(-1, 1).astype(np.float32)

    # Store original dates for plotting
    dates = df.index

    print(f"Data prepared successfully. Feature shape: {X.shape}")

    return X, y, dates


def preprocess(df, num_horizons):
    """
    Creates target variables and a rich feature set from the raw stock data.
    """
    # Create the target DataFrame (y)
    y = pd.DataFrame(index=df.index)
    for i in range(num_horizons):
        y[f"y_log_return_{i}"] = target_log_return(df, i + 1)

    # Create the features DataFrame (X)
    X = pd.DataFrame(index=df.index)

    # Calculate all features using functions from trading_indicators
    for i in range(num_horizons):
        X[f'daily_log_return_{i}'] = log_return(df, 1, i)
        X[f'intraday_log_return_{i}'] = intraday_log_return(df, i)
        X[f'overnight_log_return_{i}'] = over_night_return(df, i)
        X[f'day_range_{i}'] = day_range(df, i)
        X[f'notional_traded_change_{i}'] = notional_traded_change(df, 1, i)

    X['notional_traded'] = notional_traded(df)
    X['num_up_days_1m'] = up_days_count(df, 21)
    X['1m_mom'] = momentum(df, 21)
    X['3m_mom'] = momentum(df, 63)
    X['6m_mom'] = momentum(df, 126)
    X['12m_mom'] = momentum(df, 252)
    X['mom_change_1m_3m'] = momentum_change(df, 21, 63)
    X['mom_change_3m_6m'] = momentum_change(df, 63, 126)
    X['mom_change_6m_12m'] = momentum_change(df, 126, 252)
    X['returns_volatility_1m'] = returns_std(df, 21)
    X['returns_volatility_3m'] = returns_std(df, 63)
    X['close_cv_1m'] = coefficient_of_variation(df, 21)
    X['close_cv_3m'] = coefficient_of_variation(df, 63)

    # --- Critical final step: Align and clean ---
    # Align indices to ensure no look-ahead bias in labels
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    # Drop NaNs that result from rolling calculations
    X.dropna(inplace=True)
    y.dropna(inplace=True)

    final_common_index = X.index.intersection(y.index)
    X = X.loc[final_common_index].astype(np.float32)
    y = y.loc[final_common_index].astype(np.float32)

    print(f"Preprocessing complete. Feature shape: {X.shape}")
    return X, y