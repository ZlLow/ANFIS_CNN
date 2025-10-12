from typing import List

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models.ANFIS.CNNANFIS import HybridCnnAnfis  # Import our existing model
from utilities.dataHandler import get_data, prepare_data


def pearson_correlation(preds, targets):
    """Calculates Pearson correlation coefficient using PyTorch tensors."""
    preds_mean = torch.mean(preds)
    targets_mean = torch.mean(targets)
    cov = torch.mean((preds - preds_mean) * (targets - targets_mean))
    preds_std = torch.std(preds)
    targets_std = torch.std(targets)
    return cov / (preds_std * targets_std + 1e-6)


def grid_search(X, y, model_params_base):
    """
    Performs Grid Search Cross-Validation to find the best hyperparameters.

    Args:
        X (np.array): The feature matrix.
        y (np.array): The target vector.
        model_params_base (dict): Base parameters for the model.

    Returns:
        GridSearchCV object: The fitted GridSearchCV object containing results.
    """
    # Skorch wrapper for our PyTorch model
    # We pass fixed parameters here. The ones we want to tune are defined in the grid.
    lr_scheduler = LRScheduler(policy=ReduceLROnPlateau,
                               monitor='valid_loss',
                               patience=10,
                               factor=0.5)

    net = NeuralNetRegressor(
        module=HybridCnnAnfis,
        module__input_dim=model_params_base['input_dim'],
        module__num_mfs=model_params_base['num_mfs'],
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        max_epochs=model_params_base['epochs'],
        callbacks=[lr_scheduler],  # <--- Add the callback here
        verbose=0
    )

    # --- UPDATE THIS: Add scheduler parameters to the grid search ---
    # Note the syntax: 'callbacks__<callback_name>__<param_name>'
    # The default name for the callback is its class name in snake_case.
    param_grid = {
        'lr': [0.1, 5e-3, 1e-4],
        'batch_size': [64, 128, 256],
        'module__num_filters': [16, 32, 64],
        'callbacks__lr_scheduler__patience': [5, 10],
        'callbacks__lr_scheduler__factor': [0.1, 0.5]
    }

    gs = GridSearchCV(
        estimator=net,
        param_grid=param_grid,
        scoring='r2',
        cv=3,
        refit=True,
        verbose=2
    )

    print("--- Starting Hyperparameter Grid Search ---")

    # Scale features before fitting
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Run the grid search
    gs.fit(X_scaled, y)

    print("--- Grid Search Complete ---")
    return gs


def bayesian_search(ticker: str, start_date: str, end_date: str, DEVICE, features_col: List[str], target_col: str):
    stock_data = get_data([ticker], start_date, end_date)
    df = stock_data[ticker]
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_X, scaler_y, x_test, y_test = prepare_data(df, feature_cols=features_col, target_col=target_col)

    # Create a multi-objective study
    study = optuna.create_study(directions=['minimize', 'maximize'])
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, DEVICE),
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
    best_pearson_trial = max(study.best_trials, key=lambda t: t.values[2])
    print(f"\nSelected best trial based on highest Pearson Correlation: Trial {best_pearson_trial.number}")

    # Or stick with the best RMSE trial
    best_rmse_trial = min(study.best_trials, key=lambda t: t.values[0])
    print(f"\nSelected best trial based on lowest RMSE: Trial {best_rmse_trial.number}")

    print("\n--- Training final model with best parameters... ---")
    x_train_final = X_train_scaled.append(X_val_scaled)
    y_train_final = y_train_scaled.append(y_val_scaled)

    final_model_params = {
        'input_dim': x_train_final.shape[1], 'device': DEVICE, **best_rmse_trial.params
    }
    final_model_params.pop('lr')
    final_model_params.pop('batch_size')

    final_model = HybridCnnAnfis(**final_model_params).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_rmse_trial.params['lr'], weight_decay=1e-5)
    final_model.fit(x_train_final, y_train_final, optimizer=optimizer, epochs=best_rmse_trial.params['epochs'], batch_size=best_rmse_trial.params['batch_size'])

    _, final_test_rmse, final_test_pearson , final_test_r2 = final_model.predict(x_test, y_test)

    print(f"\nFinal Model Performance on Unseen Test Set:")
    print(f"  Test RMSE (unscaled): {final_test_rmse:.6f}")
    print(f"  Test R2 Score (unscaled): {final_test_r2:.6f}")

    return final_model_params, best_rmse_trial.params['batch_size'], best_rmse_trial.params['lr']


def objective(trial, X_train, y_train, X_val, y_val, device, epochs=30):
    params = {
        'lr': trial.suggest_float('lr', 1e-8, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 512]),
        'num_mfs': trial.suggest_int('num_mfs', 3, 7),
        'num_rules': trial.suggest_int('num_rules', 3, 512),
        'firing_conv_filters': trial.suggest_int('firing_conv_filters', 2, 512),
        'consequent_conv_filters': trial.suggest_int('consequent_conv_filters', 4, 16),
    }

    model_params = {
        'input_dim': X_train.shape[1],
        'num_mfs': params['num_mfs'],
        'num_rules': params['num_rules'],
        'firing_conv_filters': params['firing_conv_filters'],
        'consequent_conv_filters': params['consequent_conv_filters'],
        'device': device
    }
    model = HybridCnnAnfis(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    model.fit(X_train, y_train, optimizer=optimizer, epochs=epochs, batch_size=params['batch_size'])
    _, validation_rmse, validation_pearson, validation_r2 = model.predict(X_val, y_val)
    return validation_rmse.item(), validation_r2.item(), validation_pearson.item()
