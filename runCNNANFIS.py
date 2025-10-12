import os
import random
from abc import abstractmethod, ABC
from typing import List, Any, Dict, Optional

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import yfinance as yf
import warnings

from constants import Y_HORIZON, DEVICE
from models.ANFIS.CNNANFIS import HybridCnnAnfis
from models.clustering.HDBScan import get_num_rules_with_hdbscan
from trading.features import calculate_vanilla_macd
from tuner.GA.GAHandler import genetic_algorithm_anfis
from utilities.dataHandler import get_data, prepare_data
from utilities.plotter import plot_predicted_comparison

warnings.filterwarnings('ignore')

# --- Helper Functions ---
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
            current_day_index = len(full_history_df) - len(input_df) + i

            # Features are taken from the day *before* the current day
            current_features_np = full_history_df[['Close','MACD_Vanilla', 'Signal_Vanilla', 'RSI', 'BB_Width']].iloc[
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
def run_all_models(ticker, start_date, end_date):
    GENE_CONFIG_SPACE = {
        'lr': {'type': 'float', 'min': 1e-8, 'max': 1e-3},
        'epochs': {'type': 'int', 'min': 50, 'max': 100, 'step': 10},
        'batch_size': {'type': 'categorical', 'choices': [16,32,64]},
        'firing_conv_filters': {'type': 'categorical', 'choices': [32, 64, 128]},
        'consequent_conv_filters': {'type': 'int', 'min': 16, 'max': 64, 'step': 4},
    }
    GA_POPULATION_SIZE = 5
    GA_GENERATIONS = 3
    GA_MUTATION_RATE = 0.3
    GA_NUM_PARENTS = 2
    GA_NUM_ELITES = 2

    df = get_data(ticker, start_date, end_date)
    df = df[ticker[0]]
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_X, scaler_y, X_test_scaled, y_test_scaled = prepare_data(
        df, ['Close','MACD_Vanilla', "Signal_Vanilla", "RSI", "BB_Width"], 'Close')
    final_train_x = np.concatenate((X_train_scaled,X_val_scaled))
    final_train_y = np.concatenate((y_train_scaled,y_val_scaled))
    best_genome = run_GA(gene_config_space=GENE_CONFIG_SPACE,
                         population_size=GA_POPULATION_SIZE,
                         generations=GA_GENERATIONS,
                         mutation_rate=GA_MUTATION_RATE,
                         num_parents=GA_NUM_PARENTS,
                         num_elites=GA_NUM_ELITES,
                         X_train_scaled=X_train_scaled,
                         y_train_scaled=y_train_scaled,
                         X_val_scaled=X_val_scaled,
                         y_val_scaled=y_val_scaled,
                         scaler_X=scaler_X,
                         scaler_y=scaler_y)

    batch_size = best_genome['batch_size']
    epochs = best_genome['epochs']
    lr = best_genome['lr']
    anfis_params = {
        'input_dim': X_train_scaled.shape[1],
        'num_mfs': 3,
        'num_rules': best_genome['num_rules'],
        'firing_conv_filters': best_genome['firing_conv_filters'],
        'consequent_conv_filters': best_genome['consequent_conv_filters'],
        'feature_scaler': scaler_X,
        'target_scaler': scaler_y
    }

    # batch_size = 32
    # lr = 0.03951560476728942
    # epochs = 100
    # anfis_params = {
    #     'input_dim': X_train_scaled.shape[1],
    #     'num_mfs': 3,
    #     'num_rules': 128,
    #     'firing_conv_filters': 79,
    #     'consequent_conv_filters': 7,
    #     'feature_scaler': scaler_X,
    #     'target_scaler': scaler_y,
    #     'output_dim': Y_HORIZON
    # }
    print("\n--- Building Model 3: Multi-Step Price Predictor ---")
    model = HybridCnnAnfis(**anfis_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.fit(final_train_x, final_train_y, optimizer=optimizer, batch_size=batch_size, epochs=epochs)
    X_test = scaler_X.inverse_transform(X_test_scaled)
    predicted_macd_list, predicted_signal_list = generate_predicted_macd(X_test,df,"Generating Predicted MACD", model,scaler_X, scaler_y)
    tmp = df.copy()
    tmp = tmp.iloc[len(tmp)-len(X_test):]
    tmp['MACD_Predicted'] = predicted_macd_list
    tmp['Signal_Predicted'] = predicted_signal_list

    final_cols = [
        'Date', 'Close', 'MACD_Vanilla', 'Signal_Vanilla',
        'MACD_Hindsight', 'MACD_Predicted', 'Signal_Predicted'
    ]

    return tmp[final_cols]


def run_clustering(x_train):
    num_rules_from_hdbscan = get_num_rules_with_hdbscan(x_train)
    return num_rules_from_hdbscan


def run_GA(gene_config_space, population_size, generations, mutation_rate, num_parents,num_elites,X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,scaler_X, scaler_y):
    print("\n --- Running GA to optimize hyper-parameters --- \n")
    num_rules_from_hdbscan = run_clustering(x_train=X_train_scaled)
    anfis_fixed_params = {
        'input_dim': X_train_scaled.shape[1],
        'num_mfs': 3,
        'num_rules': num_rules_from_hdbscan,
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


    results_df = run_all_models(TICKER, START_DATE, END_DATE)

    print(f"\n--- Comparison of MACD Models for {TICKER} (First 5 Rows of Test Set) ---")
    print(results_df.head())

    plot_predicted_comparison(results_df)


if __name__ == '__main__':
    main()