import torch
from sklearn.preprocessing import MinMaxScaler

from models.ANFIS.CNNANFIS import HybridCnnAnfis
from trade_utils.dataHandler import load_and_engineer_features
from trade_utils.plotter import plot_actual_vs_predicted, plot_table,plot_graph
from trade_utils.validation import k_fold, run_rolling_prediction

price_scaler = MinMaxScaler(feature_range=(0, 1))


def main():
    # --- Configuration ---
    FILEPATH = 'stock_data.csv'
    WINDOWS = [1, 5, 10, 15]

    # Training Hyperparameters
    K_FOLDS = 5
    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_FILTERS = 27

    # Model Hyperparameters
    model_params = {
        'input_dim': len(WINDOWS),
        'num_mfs': 3,
        'num_filters': NUM_FILTERS,
    }

    # --- Execution ---
    # 1. Load and prepare data (unscaled)
    X, y, dates = load_and_engineer_features(FILEPATH, WINDOWS)
    if X is None: return

    # -- Model Creation --
    model = HybridCnnAnfis(**model_params)
    k_fold_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rolling_pred_optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    # --- K-Fold Evaluation (Bias-Free) ---
    print("\n--- Starting K-Fold Cross-Validation ---")
    rmse_scores, r2_scores = k_fold(K_FOLDS, X, y, model, k_fold_optimizer, batch_size=BATCH_SIZE, epochs=EPOCHS, save_path="test")

    print("\n--- K-Fold Cross-Validation RMSE Results ---")
    plot_table(rmse_scores, title='RMSE', save_path="img/k_fold_rmse.png")

    print("\n--- K-Fold Cross-Validation R2 Results ---")
    plot_table(r2_scores,title="R2",save_path="img/k_fold_r2_score.png")

    # --- Rolling Forecast Evaluation (Bias-Free) ---
    predictions, actuals, test_dates, rmse, r2, historical_data = run_rolling_prediction(X, y, dates, model, rolling_pred_optimizer, batch_size=BATCH_SIZE,epochs=EPOCHS)

    print(f"\n--- Rolling Forecast Results ---")
    print(f"Final RMSE on Test Set: ${rmse:.2f}")
    print(f"Final R2 on Test Set: {r2:.4f}")

    plot_actual_vs_predicted(actuals, predictions, test_dates, title="Rolling Forecast: Predictions vs Actuals", save_path="img/rolling_price_prediction.jpg")
    plot_graph(historical_data,save_path="img/historical_data.jpg")

if __name__ == '__main__':
    main()