import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
import warnings

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from constants import Y_HORIZON, DEVICE
from evaluation.validation import print_r2_and_rmse
from models.ANFIS.CNNANFIS import HybridCnnAnfis
from models.clustering.HDBScan import get_num_rules_with_hdbscan
from tuner.GA.GAHandler import genetic_algorithm_anfis
from utilities.dataHandler import get_data, prepare_data
from utilities.plotter import plot_actual_vs_predicted

warnings.filterwarnings('ignore')

# --- Helper Functions ---
def plot_predicted_comparison(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [2, 3]})
    fig.suptitle('Comparison: Predicted ANFIS MACD', fontsize=16)
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    ax1.set_title('Stock Price (Test Period)', fontsize=14)
    ax2.plot(df['Date'], df['MACD_Vanilla'], label='Vanilla MACD (Lagging)', color='orange', linestyle='--')
    ax2.plot(df['Date'], df['MACD_Hindsight'], label='Hindsight MACD (Benchmark)', color='black')
    ax2.plot(df['Date'], df['MACD_Predicted'], label='Predicted ANFIS MACD', color='green')
    ax2.set_title('MACD Indicator Comparison', fontsize=14)
    for ax in [ax1, ax2]: ax.legend(); ax.grid(True)
    img_dir = "img"
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, "predicted_macd.jpg"), bbox_inches='tight')
    plt.show()


def run_GA(gene_config_space, population_size, generations, mutation_rate, num_parents,num_elites,X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,scaler_X, scaler_y):
    print("\n --- Running GA to optimize hyper-parameters --- \n")
    num_rules_from_hdbscan = get_num_rules_with_hdbscan(X_train_scaled)
    anfis_fixed_params = {
        'input_dim': X_train_scaled.shape[1],
        'num_mfs': 3,
        'num_rules': num_rules_from_hdbscan
    }

    best_genome = genetic_algorithm_anfis(
        gene_config_space=gene_config_space,
        X_train=X_train_scaled, y_train=y_train_scaled,
        X_val=X_val_scaled, y_val=y_val_scaled,
        scaler_X=scaler_X,
        scaler_Y=scaler_y,
        DEVICE=DEVICE,
        anfis_fixed_params=anfis_fixed_params,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        num_parents_for_crossover=num_parents,
        num_elites=num_elites,
    )

    best_genome['num_rules'] = num_rules_from_hdbscan
    return best_genome

def main():
    TICKER = ['AAPL']
    START_DATE = '2010-01-01'
    END_DATE = '2023-12-31'
    GENE_CONFIG_SPACE = {
        'lr': {'type': 'float', 'min': 1e-8, 'max': 1e-4},
        'epochs': {'type': 'int', 'min': 50, 'max': 100, 'step': 10},
        'batch_size': {'type': 'categorical', 'choices': [16, 32]},
        'firing_conv_filters': {'type': 'categorical', 'choices': [32, 64, 128,256]},
        'consequent_conv_filters': {'type': 'int', 'min': 16, 'max': 64, 'step': 4},
    }
    GA_POPULATION_SIZE = 6
    GA_GENERATIONS = 3
    GA_MUTATION_RATE = 0.2
    GA_NUM_PARENTS = 2
    GA_NUM_ELITES = 2

    # Download data and create a copy of the dataframe before it's modified by prepare_data
    stock_data = get_data(TICKER, START_DATE, END_DATE)
    df_original = stock_data[TICKER[0]].copy()

    feature_names = ['Close','MACD_Vanilla', "Signal_Vanilla", "RSI", "BB_Width"]
    target_name = 'Close'

    # Prepare data (this function modifies the df by adding columns and dropping NaNs)
    X_train_s, y_train_s, X_val_s, y_val_s, scaler_X, scaler_y, X_test_s, y_test_s = prepare_data(
        df_original, feature_names, target_name
    )
    # After prepare_data, df_original has been processed. We use its state to get unscaled data.
    df_processed = df_original

    final_train_x_scaled = np.concatenate((X_train_s, X_val_s))
    final_train_y_scaled = np.concatenate((y_train_s, y_val_s))
    test_dates = df_processed.iloc[-len(X_test_s):]['Date']
    initial_unscaled_close_history = df_processed['Close'].iloc[:-len(X_test_s)]

    # --- Model Hyperparameters ---
    best_genome = run_GA(gene_config_space=GENE_CONFIG_SPACE,
                         population_size=GA_POPULATION_SIZE,
                         generations=GA_GENERATIONS,
                         mutation_rate=GA_MUTATION_RATE,
                         num_parents=GA_NUM_PARENTS,
                         num_elites=GA_NUM_ELITES,
                         X_train_scaled=final_train_x_scaled,
                         y_train_scaled=final_train_y_scaled,
                         X_val_scaled=X_test_s,
                         y_val_scaled=y_test_s,
                         scaler_X=scaler_X,
                         scaler_y=scaler_y)

    batch_size = best_genome['batch_size']
    epochs = best_genome['epochs']
    lr = best_genome['lr']
    anfis_params = {
        'input_dim': X_train_s.shape[1],
        'num_mfs': 3,
        'num_rules': best_genome['num_rules'],
        'firing_conv_filters': best_genome['firing_conv_filters'],
        'consequent_conv_filters': best_genome['consequent_conv_filters'],
        'feature_scaler': scaler_X,
        'target_scaler': scaler_y,
    }
    #
    # batch_size = 16
    # lr = 1e-4
    # epochs = 52
    # anfis_params = {
    #     'input_dim': X_train_s.shape[1],
    #     'num_mfs': 3,
    #     'num_rules': 3,
    #     'firing_conv_filters': 64,
    #     'consequent_conv_filters': 64,
    #     'feature_scaler': scaler_X,
    #     'target_scaler': scaler_y,
    # }

    print("\n--- Building and Training Model for Rolling Forecast ---")
    model = HybridCnnAnfis(**anfis_params, device=DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.fit(final_train_x_scaled, final_train_y_scaled, optimizer=optimizer, batch_size=batch_size, epochs=epochs)

    predictions, actuals, dates = model.rolling_prediction(
        initial_unscaled_close_history=initial_unscaled_close_history,
        train_X_scaled=final_train_x_scaled,
        train_y_scaled=final_train_y_scaled,
        X_test_scaled=X_test_s,
        y_test_scaled=y_test_s,
        test_dates=test_dates,
        look_forward_period=13
    )

    plot_actual_vs_predicted(actuals, predictions, dates,
                             title="Rolling Forecast (13-Day Steps): Predictions vs Actuals",
                             save_path="img/macd_5_days_rolling_price_prediction.jpg")
    print_r2_and_rmse(predictions, actuals)

if __name__ == '__main__':
    main()