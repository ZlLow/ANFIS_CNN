# tuner.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from skorch import NeuralNetRegressor

from models.ANFIS.CNNANFIS import HybridCnnAnfis  # Import our existing model


def tune_hyperparameters(X, y, model_params_base):
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
    net = NeuralNetRegressor(
        module=HybridCnnAnfis,
        module__input_dim=model_params_base['input_dim'],
        module__num_mfs=model_params_base['num_mfs'],
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        max_epochs=model_params_base['epochs'],
        verbose=0  # Set to 1 to see skorch's training logs
    )

    # Define the grid of hyperparameters to search
    # Note the syntax: 'lr' for optimizer, 'batch_size' for data loading,
    # and 'module__<param>' for parameters of our HybridCnnAnfis model.
    param_grid = {
        'lr': [0.1, 0.005, 1e-4, 1e-8],
        'batch_size': [16, 64, 128, 256],
        'module__num_filters': [16, 32, 64, 128, 256]
    }

    # Set up GridSearchCV
    # We use 'neg_root_mean_squared_error'. Sklearn maximizes scores,
    # so minimizing RMSE is equivalent to maximizing negative RMSE.
    gs = GridSearchCV(
        estimator=net,
        param_grid=param_grid,
        scoring='r2',
        cv=3,  # Using 3 folds for a quicker demonstration
        refit=True,  # Refit the best model on the whole dataset
        verbose=2  # Show progress
    )

    print("--- Starting Hyperparameter Grid Search ---")

    # Scale features before fitting
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Run the grid search
    gs.fit(X_scaled, y)

    print("--- Grid Search Complete ---")
    return gs