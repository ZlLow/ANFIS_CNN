import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn


def k_fold(n_splits: int, X, y, model: nn.Module, optimizer: torch.optim.Optimizer, batch_size: int, epochs: int):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    r2_scores = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        X_train_fold, X_val_fold = X[train_ids], X[val_ids]
        y_train_fold, y_val_fold = y[train_ids], y[val_ids]

        model.fit(x_train_data=X_train_fold, y_train_data=y_train_fold, x_val_data=X_val_fold, y_val_data=y_val_fold,
                  optimizer=optimizer, fold=fold, batch_size=batch_size, epochs=epochs)
        val_loss, r2_score = model.predict(X_val_fold, y_val_fold, f"img/{fold + 1}_cnn_anfis_prediction.jpg")
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


