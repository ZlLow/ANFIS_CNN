import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torchmetrics.functional import r2_score
from tqdm.auto import tqdm
import warnings

from models.ANFIS.CNNANFIS import HybridCnnAnfis, train_anfis_model
from trade_utils.dataHandler import get_data
from trade_utils.features import calculate_rsi, calculate_vanilla_macd, calculate_hindsight_macd,calculate_bollinger_width
from trade_utils.plotter import plot_predicted_comparison, plot_compensated_comparison

warnings.filterwarnings('ignore')

# --- Helper Functions ---
y_horizon = 13


def pearson_correlation(preds, targets):
    """Calculates Pearson correlation coefficient using PyTorch tensors."""
    preds_mean = torch.mean(preds)
    targets_mean = torch.mean(targets)
    cov = torch.mean((preds - preds_mean) * (targets - targets_mean))
    preds_std = torch.std(preds)
    targets_std = torch.std(targets)
    return cov / (preds_std * targets_std + 1e-6)  # Add epsilon for stability


def generate_predicted_macd(input_df, full_history_df, desc, price_predictor, scaler_X_price, scaler_y_price):
    """
    Encapsulated function to run the direct multi-step MACD prediction.
    `input_df` is the dataframe to generate predictions for (train or test).
    `full_history_df` is the complete dataframe (df) for historical lookups.
    """
    predicted_macd_list = []
    predicted_signal_list = []
    backward_days = 13

    price_predictor.eval()
    with torch.no_grad():
        for i in tqdm(range(len(input_df)), desc=desc, leave=False):
            current_day_index = input_df.index[i]

            # Features are taken from the day *before* the current day
            current_features_np = full_history_df[['Close', 'MACD_Vanilla', 'Signal_Vanilla']].iloc[
                current_day_index - 1].values.reshape(1, -1)

            # Predict all future prices in a single step
            input_scaled = scaler_X_price.transform(current_features_np)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            pred_scaled = price_predictor(input_tensor)
            future_price_predictions = scaler_y_price.inverse_transform(pred_scaled.numpy()).flatten()

            # Construct hybrid series and calculate MACD
            past_prices = full_history_df['Close'].iloc[current_day_index - backward_days: current_day_index]
            hybrid_price_series = pd.concat([
                past_prices,
                pd.Series(future_price_predictions,
                          index=range(len(past_prices), len(past_prices) + len(future_price_predictions)))
            ], ignore_index=True)

            hybrid_macd, hybrid_signal = calculate_vanilla_macd(hybrid_price_series)
            predicted_macd_list.append(hybrid_macd.iloc[backward_days - 1])
            predicted_signal_list.append(hybrid_signal.iloc[backward_days - 1])

    return predicted_macd_list, predicted_signal_list

# --- Main Implementation Function ---
def run_all_models(ticker, start_date, end_date, anfis_params, batch_size: int, epochs: int = 50, lr=0.001):
    df = get_data(ticker, start_date, end_date)
    df = df['AAPL']
    # --- Indicator Calculations ---
    print("Calculating Model 1: Vanilla MACD...")
    df['MACD_Vanilla'], df['Signal_Vanilla'] = calculate_vanilla_macd(df['Close'])
    print("Calculating Model 2: Hindsight MACD...")
    df['MACD_Hindsight'], df['Signal_Hindsight'] = calculate_hindsight_macd(df['Close'])
    df['RSI'] = calculate_rsi(df, 14)
    df['BB_Width'] = calculate_bollinger_width(df['Close'], window=20)

    # --- Multi-Step Target Generation ---
    target_cols = [f'Close_t+{i + 1}' for i in range(y_horizon)]
    for i in range(y_horizon):
        df[target_cols[i]] = df['Close'].shift(-(i + 1))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    # --- Building Model 3: Predicted MACD (with DIRECT Multi-Step Prediction) ---
    print("\n--- Building Model 3: Multi-Step Price Predictor ---")
    price_model_features = df_train[['Close', 'MACD_Vanilla', 'Signal_Vanilla']].values
    price_model_target = df_train[target_cols].values

    # Add output_dim to model parameters for multi-step training
    anfis_params['output_dim'] = y_horizon

    price_predictor, scaler_X_price, scaler_y_price = train_anfis_model(
        price_model_features, price_model_target, anfis_params,
        epochs=epochs, lr=lr, batch_size=batch_size
    )

    # Generate predictions using the direct multi-step method
    macd_pred_train, signal_pred_train = generate_predicted_macd(df_train, df, "Generating Predicted MACD (Train)",
                                                                 price_predictor, scaler_X_price, scaler_y_price)
    df_train['MACD_Predicted'] = macd_pred_train
    df_train['Signal_Predicted'] = signal_pred_train

    predicted_macd_list, predicted_signal_list = generate_predicted_macd(df_test, df,
                                                                         "Generating Predicted MACD (Test)",
                                                                         price_predictor, scaler_X_price,
                                                                         scaler_y_price)
    df_test['MACD_Predicted'] = predicted_macd_list
    df_test['Signal_Predicted'] = predicted_signal_list

    # --- Building Model 4: Compensated MACD ---
    print("\n--- Building Model 4: Compensated MACD ---")
    # Remove output_dim as this is a single-output model
    anfis_params.pop('output_dim', None)

    comp_model_features = df_train[
        ['Close', 'MACD_Predicted', 'Signal_Predicted', "BB_Width", "RSI"]].values
    comp_model_target = df_train['MACD_Hindsight'].values
    comp_predictor, scaler_X_comp, scaler_y_comp = train_anfis_model(
        comp_model_features, comp_model_target, anfis_params,
        epochs=epochs, lr=lr, batch_size=batch_size
    )

    comp_test_features = df_test[
        ['Close', 'MACD_Predicted', 'Signal_Predicted', "BB_Width","RSI"]].values
    comp_test_scaled = scaler_X_comp.transform(comp_test_features)
    comp_test_tensor = torch.tensor(comp_test_scaled, dtype=torch.float32)

    comp_predictor.eval()
    with torch.no_grad():
        comp_pred_scaled = comp_predictor(comp_test_tensor)
        comp_pred_unscaled = scaler_y_comp.inverse_transform(comp_pred_scaled.numpy())

    df_test['MACD_Compensated'] = comp_pred_unscaled
    df_test['Signal_Compensated'] = df_test['MACD_Compensated'].ewm(span=9, adjust=False).mean()

    final_cols = [
        'Date', 'Close', 'MACD_Vanilla', 'Signal_Vanilla',
        'MACD_Hindsight', 'Signal_Hindsight', 'MACD_Predicted', 'Signal_Predicted',
        'MACD_Compensated', 'Signal_Compensated'
    ]

    return df_test[final_cols]


# --- Optuna Objective Function ---

def objective(trial, X_train, y_train, X_val, y_val, device):
    params = {
        'lr': trial.suggest_float('lr', 1e-8, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 512]),
        'num_mfs': trial.suggest_int('num_mfs', 3, 7),
        'num_rules': trial.suggest_int('num_rules', 3, 512),
        'firing_conv_filters': trial.suggest_int('firing_conv_filters', 2, 512),
        'consequent_conv_filters': trial.suggest_int('consequent_conv_filters', 4, 16),
    }

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

    model_params = {
        'input_dim': X_train.shape[1],
        'num_mfs': params['num_mfs'],
        'num_rules': params['num_rules'],
        'firing_conv_filters': params['firing_conv_filters'],
        'consequent_conv_filters': params['consequent_conv_filters'],
        'device': device
    }
    model = HybridCnnAnfis(**model_params).to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    epochs = 30

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        validation_rmse = torch.sqrt(criterion(val_preds, y_val))
        validation_r2 = r2_score(y_val, val_preds)

    return validation_rmse.item(), validation_r2.item()


def objective_macd(trial, X_train, y_train, X_val, y_val, device):
    """
    Optuna objective function to optimize hyperparameters for predicting Hindsight MACD.
    """
    params = {
        'lr': trial.suggest_float('lr', 1e-8, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 512]),
        'num_mfs': trial.suggest_int('num_mfs', 3, 7),
        'num_rules': trial.suggest_int('num_rules', 3, 128),
        'firing_conv_filters': trial.suggest_int('firing_conv_filters', 2, 512),
        'consequent_conv_filters': trial.suggest_int('consequent_conv_filters', 4, 16),
    }

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

    model_params = {
        'input_dim': X_train.shape[1],
        'num_mfs': params['num_mfs'],
        'num_rules': params['num_rules'],
        'firing_conv_filters': params['firing_conv_filters'],
        'consequent_conv_filters': params['consequent_conv_filters'],
        'device': device
    }
    model = HybridCnnAnfis(**model_params).to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    epochs = 100  # Number of epochs for each trial

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)
        # Objective 1: Minimize RMSE
        validation_rmse = torch.sqrt(criterion(val_preds, y_val))
        # Objective 2: Maximize R2 Score
        validation_r2 = r2_score(y_val, val_preds)
        # Objective 3: Maximize Pearson Correlation
        validation_pearson = pearson_correlation(val_preds.squeeze(), y_val.squeeze())

    # Optuna will try to minimize the first value (RMSE) and maximize the second (R2)
    return validation_rmse.item(), validation_r2.item(), validation_pearson.item()


def run_validation():
    TICKER = 'AAPL'
    START_DATE = '2010-01-01'
    END_DATE = '2023-12-31'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    df = get_data(TICKER, START_DATE, END_DATE)
    df['MACD'], df['Signal'] = calculate_vanilla_macd(df['Close'])
    df.dropna(inplace=True)

    features_df = df[['Close', 'MACD', 'Signal']]
    target_series = df['Close']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features_df.values, target_series.values, test_size=0.2, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, shuffle=False
    )

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(DEVICE)

    # Create a multi-objective study
    study = optuna.create_study(directions=['minimize', 'maximize'])
    study.optimize(
        lambda trial: objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, DEVICE),
        n_trials=100, show_progress_bar=True
    )

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Pareto Front (Best Trials):")
    for i, trial in enumerate(study.best_trials):
        print(f"  --- Trial {trial.number} ---")
        print(f"  Metrics: RMSE={trial.values[0]:.6f}, R2={trial.values[1]:.6f}")
        print(f"  Params: {trial.params}")

    # Choose the absolute best trial based on RMSE for the final model
    best_trial = min(study.best_trials, key=lambda t: t.values[0])
    print(f"\nSelected best trial for final model (lowest RMSE): Trial {best_trial.number}")

    print("\n--- Training final model with best parameters... ---")
    X_train_val_tensor = torch.cat([X_train_tensor, X_val_tensor])
    y_train_val_tensor = torch.cat([y_train_tensor, y_val_tensor])

    final_model_params = {
        'input_dim': X_train_val_tensor.shape[1], 'device': DEVICE, **best_trial.params
    }
    final_model_params.pop('lr')
    final_model_params.pop('batch_size')

    final_model = HybridCnnAnfis(**final_model_params).to(DEVICE)
    final_train_dataset = TensorDataset(X_train_val_tensor, y_train_val_tensor)
    final_train_loader = DataLoader(
        final_train_dataset, batch_size=best_trial.params['batch_size'], shuffle=True, drop_last=True
    )

    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_trial.params['lr'], weight_decay=1e-5)

    for epoch in tqdm(range(100), desc="Final Training"):
        for batch_X, batch_y in final_train_loader:
            optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    final_model.eval()
    with torch.no_grad():
        test_preds_scaled = final_model(X_test_tensor)

        # Unscale for interpretation
        test_preds_unscaled = scaler_y.inverse_transform(test_preds_scaled.cpu().numpy())
        y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

        final_test_rmse = np.sqrt(np.mean((test_preds_unscaled - y_test_unscaled) ** 2))
        final_test_r2 = r2_score(torch.tensor(y_test_unscaled), torch.tensor(test_preds_unscaled))

    print(f"\nFinal Model Performance on Unseen Test Set:")
    print(f"  Test RMSE (unscaled): {final_test_rmse:.6f}")
    print(f"  Test R2 Score (unscaled): {final_test_r2:.6f}")
    return final_model_params, best_trial.params['batch_size'], best_trial.params['lr']


def run_validation_for_macd():
    """
    Prepares data and runs Optuna study to find the best hyperparameters
    for the Compensated MACD model (Model 4).
    """
    TICKER = ['AAPL']
    START_DATE = '2010-01-01'
    END_DATE = '2023-12-31'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- 1. Data Preparation ---
    df = get_data(TICKER, START_DATE, END_DATE)
    df = df['AAPL']

    # Calculate all necessary indicators
    df['MACD_Vanilla'], df['Signal_Vanilla'] = calculate_vanilla_macd(df['Close'])
    df['MACD_Hindsight'], _ = calculate_hindsight_macd(df['Close'])
    df['RSI'] = calculate_rsi(df, 14)
    df['BB_Width'] = calculate_bollinger_width(df['Close'], window=20)
    df.dropna(inplace=True)

    # Define features and the NEW target
    feature_cols = ['Close', 'MACD_Vanilla', 'Signal_Vanilla', 'RSI', 'BB_Width']
    target_col = 'MACD_Hindsight'

    features_df = df[feature_cols]
    target_series = df[target_col]

    # --- 2. Data Splitting and Scaling ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features_df.values, target_series.values, test_size=0.2, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, shuffle=False  # 0.2 of 0.8 is 0.16
    )

    # IMPORTANT: Scale features and target separately
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(DEVICE)

    study = optuna.create_study(directions=['minimize', 'maximize', 'maximize'])
    study.optimize(
        lambda trial: objective_macd(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, DEVICE),
        n_trials=100,
        show_progress_bar=True
    )

    print("\n--- Hyperparameter Tuning for MACD Prediction Complete ---")
    print("Pareto Front (Best Trials):")
    for i, trial in enumerate(study.best_trials):
        print(f"  --- Trial {trial.number} ---")
        # Access values by index: 0=RMSE, 1=R2, 2=Pearson
        print(f"  Metrics: RMSE={trial.values[0]:.6f}, R2={trial.values[1]:.6f}, Pearson={trial.values[2]:.6f}")
        print(f"  Params: {trial.params}")

    # Now your selection criteria can be more sophisticated
    best_pearson_trial = max(study.best_trials, key=lambda t: t.values[2])
    print(f"\nSelected best trial based on highest Pearson Correlation: Trial {best_pearson_trial.number}")

    # Or stick with the best RMSE trial
    best_rmse_trial = min(study.best_trials, key=lambda t: t.values[0])
    print(f"\nSelected best trial based on lowest RMSE: Trial {best_rmse_trial.number}")

    # Choose which trial's params to use for the final model
    best_trial = best_pearson_trial

    print(f"\nSelected best trial for final model (lowest RMSE): Trial {best_trial.number}")

    # --- 4. Prepare parameters for the main function ---
    final_model_params = {**best_trial.params}
    final_model_params.pop('lr')
    final_model_params.pop('batch_size')

    # Add necessary params that were not part of the search space
    final_model_params['input_dim'] = X_train_tensor.shape[1]
    final_model_params['device'] = DEVICE

    return final_model_params, best_trial.params['batch_size'], best_trial.params['lr']


if __name__ == '__main__':
    BATCH_SIZE = 32
    LR = 0.03951560476728942
    best_param = {
        'num_mfs': 3,
        'num_rules': 128,
        'firing_conv_filters': 79,
        'consequent_conv_filters': 7
    }
    # best_param, BATCH_SIZE, LR = run_validation_for_macd()
    # device = best_param['device']
    # best_param.pop("device")
    # best_param.pop('input_dim')
    TICKER = ['AAPL']
    START_DATE = '2010-01-01'
    END_DATE = '2023-12-31'

    results_df = run_all_models(ticker=TICKER, start_date=START_DATE, end_date=END_DATE, anfis_params=best_param,
                                batch_size=BATCH_SIZE, lr=LR, epochs=200)

    print(f"\n--- Comparison of MACD Models for {TICKER} (First 5 Rows of Test Set) ---")
    print(results_df.head())

    plot_predicted_comparison(results_df)
    plot_compensated_comparison(results_df)