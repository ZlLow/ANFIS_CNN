# tuner.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

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