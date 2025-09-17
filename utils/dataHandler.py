import os

import pandas as pd
import torch
import yfinance as yf

from typing import List

from sklearn.preprocessing import MinMaxScaler

from utils.tradingIndicator import target_log_return, log_return, intraday_log_return, over_night_return, \
    day_range, notional_traded, notional_traded_change, up_days_count, momentum, momentum_change, returns_std, \
    coefficient_of_variation


def get_tickers(symbols: List[str], start_train: str, end_train: str, start_test: str, end_test: str, num_horizons: int):
    train_features = []
    train_targets = []
    test_features = []
    test_targets = []

    for symbol in symbols:
        train_f, train_t, test_f, test_t = download_stock_data(symbol, start_train, end_train, start_test, end_test, num_horizons)
        train_f['symbol'] = symbol
        train_t['symbol'] = symbol
        test_f['symbol'] = symbol
        test_t['symbol'] = symbol
        train_features.append(train_f)
        train_targets.append(train_t)
        test_features.append(test_f)
        test_targets.append(test_t)

    train_features = pd.concat(train_features)
    train_targets = pd.concat(train_targets)
    test_features = pd.concat(test_features)
    test_targets = pd.concat(test_targets)

    train_features.sort_index(inplace=True)
    train_targets.sort_index(inplace=True)
    test_features.sort_index(inplace=True)
    test_targets.sort_index(inplace=True)

    return train_features, train_targets, test_features, test_targets


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


def download_stock_data(ticker: str, start_train: str, end_train: str, start_test: str, end_test: str, num_horizons: int) -> tuple:
    data = pd.DataFrame()
    data_dir = "./data/main"
    os.makedirs(data_dir, exist_ok=True)

    file_path = f"{data_dir}/{ticker}_{start_train}_{end_test}.pkl"

    if os.path.exists(file_path):
        data = pd.read_pickle(file_path)
    else:
        try:
            data = yf.download(ticker, start=start_train, end=end_test, auto_adjust=True)
            if data.empty:
                print(f"Warning: No data downloaded for ticker {ticker}")
            print(f"{ticker} data has been downloaded successfully.")
        except Exception as e:
            print(f"{ticker} data could not be downloaded. Error: {e}")

    data.to_pickle(file_path)

    features, targets = preprocess(data, num_horizons)
    features.dropna(inplace=True)
    targets.dropna(inplace=True)

    # Align the indices of df and y
    common_index = features.index.intersection(targets.index)
    features = features.loc[common_index]
    targets = targets.loc[common_index]

    # LOOK FORWARD CHECK
    #look_forward_bias_check(features, targets)

    train_features, train_targets = features.loc[start_train:end_train], targets.loc[start_train:end_train]
    test_features, test_targets = features.loc[start_test:end_test], targets.loc[start_test:end_test]
    return train_features, train_targets, test_features, test_targets


def preprocess(df: pd.DataFrame, num_horizons: int) -> tuple:
    y = pd.DataFrame()

    for i in range(num_horizons):
        y[f"y_log_return_{i}"] = target_log_return(df, i + 1)

    for i in range(num_horizons):
        df[f'daily_log_return_{i}'] = log_return(df,1, i)

    for i in range(num_horizons):
        df[f'intraday_log_return_{i}'] = intraday_log_return(df, i)

    for i in range(num_horizons):
        df[f'overnight_log_return_{i}'] = over_night_return(df, i)

    for i in range(num_horizons):
        df[f'day_range_{i}'] = day_range(df, i)

    df['notional_traded'] = notional_traded(df)

    for i in range(num_horizons):
        df[f'notional_traded_change_{i}'] = notional_traded_change(df, 1, i)

    df['num_up_days_1m'] = up_days_count(df, 21)

    df['1m_mom'] = momentum(df, 21)
    df['3m_mom'] = momentum(df, 63)
    df['6m_mom'] = momentum(df, 126)
    df['12m_mom'] = momentum(df, 252)
    df['18m_mom'] = momentum(df, 378)
    df['mom_change_1m_3m'] = momentum_change(df, 21, 63)
    df['mom_change_3m_6m'] = momentum_change(df, 63, 126)
    df['mom_change_6m_12m'] = momentum_change(df, 126, 252)
    df['returns_volatility_1m'] = returns_std(df, 21)
    df['returns_volatility_3m'] = returns_std(df, 63)
    df['close_cv_1m'] = coefficient_of_variation(df, 21)
    df['close_cv_3m'] = coefficient_of_variation(df, 63)

    # Handle NaN values in both df and y
    df.dropna(inplace=True)
    y.dropna(inplace=True)

    # Align the indices of df and y
    common_index = df.index.intersection(y.index)
    df = df.loc[common_index]
    y = y.loc[common_index]

    return df, y

def look_forward_bias_check(features: pd.DataFrame, target: pd.DataFrame):
    return_col = ['daily_log_return_0',
       'daily_log_return_1', 'daily_log_return_2', 'daily_log_return_3',
       'daily_log_return_4', 'daily_log_return_5', 'daily_log_return_6',
       'daily_log_return_7', 'daily_log_return_8', 'daily_log_return_9',
       'daily_log_return_10', 'daily_log_return_11', 'daily_log_return_12']
    t_col = ['y_log_return_0', 'y_log_return_1', 'y_log_return_2', 'y_log_return_3',
               'y_log_return_4', 'y_log_return_5', 'y_log_return_6', 'y_log_return_7',
               'y_log_return_8', 'y_log_return_9', 'y_log_return_10',
               'y_log_return_11', 'y_log_return_12']

    for r1 in return_col:
        for r2 in t_col:
            if (features[r1] == target[r2]).all():
                raise Exception(f"Feature ColumnL: {r1} matches Target Column: {r2}!")

def transform_data_to_torch(df: pd.DataFrame):
    return torch.tensor(df.values)

def scale_data(df: pd.DataFrame):
    tmp_cpy = df.copy()
    values = tmp_cpy.values
    values = values.reshape(len(values), 1)
    scaler = MinMaxScaler()
    scaler.fit(values)
    normalized_values = scaler.transform(values)
    return pd.DataFrame(normalized_values)

def inverse_scale_data(df: pd.DataFrame):
    tmp_cpy = df.copy()
    values = tmp_cpy.values
    values = values.reshape(len(values), 1)
    inverse_scaler = MinMaxScaler()
    inverse_scaler.fit(values)
    normalized_values = inverse_scaler.inverse_transform(values)
    return pd.DataFrame(normalized_values)

