import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torchmetrics.functional import r2_score

from utilities.dataHandler import load_and_engineer_features
from utilities.plotter import plot_actual_vs_predicted, plot_comparison_graph, plot_r2_table
from evaluation.validation import run_rolling_prediction, print_r2_and_rmse

price_scaler = MinMaxScaler(feature_range=(0, 1))


def main():
    # --- Configuration ---
    FILEPATH = 'stock_data.csv'
    WINDOWS = [5,10,15,20]

    # Training Hyperparameters
    K_FOLDS = 5
    EPOCHS = 150
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-3

    # --- Model & Training Hyperparameters ---
    model_params = {
        'num_mfs': 3,
        'num_rules': 32,  # The number of fuzzy rules we want to generate
        'firing_conv_filters': 16,  # Filters for the firing strength head
        'consequent_conv_filters': 4  # Filters for the consequent generator head
    }

    # --- Execution ---
    # 1. Load and prepare data (unscaled)
    X, y, dates, shape = load_and_engineer_features(FILEPATH, WINDOWS)
    if X is None: return
    model_params['input_dim'] = shape[1]

    # --- K-Fold Evaluation (Bias-Free) ---
    # print("\n--- Starting K-Fold Cross-Validation ---")
    # rmse_scores, r2_scores = k_fold(K_FOLDS, X, y, model_params,batch_size=BATCH_SIZE, epochs=EPOCHS,lr=LEARNING_RATE)
    #
    # print("\n--- K-Fold Cross-Validation RMSE Results ---")
    # plot_table(rmse_scores, title='RMSE', save_path="img/k_fold_rmse.png")
    #
    # print("\n--- K-Fold Cross-Validation R2 Results ---")
    # plot_table(r2_scores,title="R2",save_path="img/k_fold_r2_score.png")

    # --- Rolling Forecast Evaluation (Bias-Free) 1 day ---
    predictions_1, actuals_1, test_dates, historical_data = run_rolling_prediction(X, y, dates, model_params,batch_size=BATCH_SIZE,epochs=EPOCHS,lr=LEARNING_RATE)

    plot_actual_vs_predicted(actuals_1, predictions_1, test_dates, title="Rolling Forecast 1 day: Predictions vs Actuals", save_path="img/1_day_rolling_price_prediction.jpg")

    predictions_3, actuals_3, test_dates, historical_data = run_rolling_prediction(X, y, dates, model_params,batch_size=BATCH_SIZE,epochs=EPOCHS,lr=LEARNING_RATE, forward_days=3)

    plot_actual_vs_predicted(actuals_3, predictions_3, test_dates, title="Rolling Forecast 3 day: Predictions vs Actuals", save_path="img/3_day_rolling_price_prediction.jpg")

    predictions_5, actuals_5, test_dates, historical_data = run_rolling_prediction(X, y, dates, model_params,batch_size=BATCH_SIZE,epochs=EPOCHS,lr=LEARNING_RATE, forward_days=5)

    plot_actual_vs_predicted(actuals_5, predictions_5, test_dates, title="Rolling Forecast 5 day: Predictions vs Actuals", save_path="img/5_day_rolling_price_prediction.jpg")

    predictions_10, actuals_10, test_dates, historical_data = run_rolling_prediction(X, y, dates, model_params,batch_size=BATCH_SIZE,epochs=EPOCHS,lr=LEARNING_RATE, forward_days=10)

    plot_actual_vs_predicted(actuals_10, predictions_10, test_dates, title="Rolling Forecast 10 day: Predictions vs Actuals", save_path="img/10_day_rolling_price_prediction.jpg")

    # --- Rolling Forecast Evaluation (Bias-Free) 13 day ---
    predictions_13, actuals_13, test_dates, historical_data = run_rolling_prediction(X, y, dates, model_params,batch_size=BATCH_SIZE,epochs=EPOCHS,lr=LEARNING_RATE, forward_days=13)

    plot_actual_vs_predicted(actuals_13, predictions_13, test_dates, title="Rolling Forecast 13 days: Predictions vs Actuals", save_path="img/13_days_rolling_price_prediction.jpg")
    predictions = [predictions_1, predictions_3, predictions_5, predictions_10, predictions_13]
    actuals = [actuals_1, actuals_3, actuals_5, actuals_10, actuals_13]
    results = []
    for i in range(len(predictions)):
        rmse = np.sqrt(np.mean((predictions[i] - actuals[i]) ** 2))
        r2 = r2_score(torch.tensor(actuals[i]), torch.tensor(predictions[i]))
        results.append([f"{rmse:.6f}, {r2:.6f}"])

    plot_r2_table(results,"img/r2_predictions_comparison.png")





if __name__ == '__main__':
    main()