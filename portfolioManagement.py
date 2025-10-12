import os
from typing import Dict

import hdbscan
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
import torch
import torch.nn as nn

import numpy as np

from portfolio.portfolioOptimizer import PortfolioOptimizer
from trading.features import calculate_rsi, calculate_bollinger_width
from tuner.GA import GAHandler
from constants import DEVICE, Y_HORIZON
from utilities.dataHandler import prepare_data


class MonteCarloSimulation:
    def __init__(self, returns, initial_investment=1, weights=None):
        self.returns = returns
        self.mean = returns.mean()
        self.covariance = returns.cov()
        self.initial_investment = initial_investment
        num_assets = len(self.mean)
        if weights is None:
            self.weights = np.ones(num_assets) / num_assets
        else:
            self.weights = np.array(weights)

    def run_simulation(self, num_simulations, time_horizon):
        all_cumulative_returns = np.zeros((time_horizon, num_simulations))
        final_portfolio_values = np.zeros(num_simulations)

        for sim in range(num_simulations):
            simulated_returns = np.random.multivariate_normal(
                self.mean, self.covariance, time_horizon
            )
            cumulative_returns = np.cumprod(1 + simulated_returns, axis=0)
            portfolio_cumulative_returns = cumulative_returns.dot(self.weights)
            all_cumulative_returns[:, sim] = portfolio_cumulative_returns * self.initial_investment
            final_portfolio_values[sim] = portfolio_cumulative_returns[-1] * self.initial_investment
        return all_cumulative_returns, final_portfolio_values


warnings.filterwarnings('ignore')


class GeneralizedBellMembershipFunc(nn.Module):
    def __init__(self, num_mfs, input_dim):
        super(GeneralizedBellMembershipFunc, self).__init__()
        self.a = nn.Parameter(torch.rand(num_mfs, input_dim) * 0.5 + 0.1)
        self.b = nn.Parameter(torch.rand(num_mfs, input_dim) * 2 + 0.5)
        self.c = nn.Parameter(torch.rand(num_mfs, input_dim))

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(2)
        a_exp, b_exp, c_exp = self.a.t().unsqueeze(0), self.b.t().unsqueeze(0), self.c.t().unsqueeze(0)
        b_clamped = torch.clamp(b_exp, min=0.01, max=10.0)
        base = torch.abs((x_unsqueezed - c_exp) / (a_exp + 1e-6))
        return 1. / (1. + base ** (2 * b_clamped))


class ConsequentGenerator(nn.Module):
    def __init__(self, input_dim, num_mfs, num_rules, num_conv_filters):
        super(ConsequentGenerator, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_conv_filters, kernel_size=2),
            nn.ReLU(), nn.Dropout(0.2), nn.Flatten(start_dim=1)
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, num_mfs)
            dummy_output = self.conv_net(dummy_input)
            linear_input_size = dummy_output.shape[1]
        self.fc_net = nn.Linear(linear_input_size, num_rules * (input_dim + 1))
        self.num_rules = num_rules
        self.output_dim_per_rule = input_dim + 1

    def forward(self, memberships):
        conv_features = self.conv_net(memberships)
        flat_params = self.fc_net(conv_features)
        dynamic_params = flat_params.view(-1, self.num_rules, self.output_dim_per_rule)
        return dynamic_params


class HybridCnnAnfis(nn.Module):
    def __init__(self, input_dim, num_mfs, num_rules, firing_conv_filters, consequent_conv_filters,
                 device=torch.device('cpu'), output_dim=1):
        super(HybridCnnAnfis, self).__init__()
        self.input_dim, self.num_mfs, self.num_rules = input_dim, num_mfs, num_rules
        self.device, self.output_dim = device, output_dim
        self.membership_funcs = GeneralizedBellMembershipFunc(num_mfs, input_dim)
        self.firing_strength_net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=firing_conv_filters, kernel_size=2),
            nn.ReLU(), nn.Dropout(0.2)).to(device)
        self.firing_fc = nn.Linear(firing_conv_filters, num_rules).to(device)
        self.batch_norm = nn.BatchNorm1d(num_rules).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.consequent_generator = ConsequentGenerator(input_dim, num_mfs, num_rules, consequent_conv_filters).to(
            device)
        self.output_projection = nn.Linear(1, self.output_dim).to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training and batch_size == 1: return self._forward_single(x)
        memberships = self.membership_funcs(x)
        firing_features = self.firing_strength_net(memberships).mean(dim=2)
        firing_strength_logits = self.firing_fc(firing_features)
        normalized_firing_strengths = self.softmax(self.batch_norm(firing_strength_logits))
        dynamic_consequent_params = self.consequent_generator(memberships)
        x_aug = torch.cat([x, torch.ones(batch_size, 1, device=self.device)], dim=1).unsqueeze(1)
        rule_outputs = (dynamic_consequent_params * x_aug).sum(dim=2)
        aggregated_output = (normalized_firing_strengths * rule_outputs).sum(dim=1, keepdim=True)
        final_output = self.output_projection(aggregated_output)
        return final_output

    def _forward_single(self, x):
        batch_size = x.shape[0]
        memberships = self.membership_funcs(x)
        firing_features = self.firing_strength_net(memberships).mean(dim=2)
        firing_strength_logits = self.firing_fc(firing_features)
        normalized_firing_strengths = self.softmax(firing_strength_logits)
        dynamic_consequent_params = self.consequent_generator(memberships)
        x_aug = torch.cat([x, torch.ones(batch_size, 1, device=self.device)], dim=1).unsqueeze(1)
        rule_outputs = (dynamic_consequent_params * x_aug).sum(dim=2)
        aggregated_output = (normalized_firing_strengths * rule_outputs).sum(dim=1, keepdim=True)
        return self.output_projection(aggregated_output)


def get_num_rules_with_hdbscan(df_train, features, min_cluster_size=15):
    """
    Uses HDBSCAN to find the optimal number of clusters (rules) from the training data.
    """
    print(f"Running HDBSCAN to determine num_rules with min_cluster_size={min_cluster_size}...")
    data_to_cluster = df_train[features].copy()
    data_to_cluster.dropna(inplace=True)

    # Scaling is crucial for distance-based algorithms like HDBSCAN
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_cluster)

    # Apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(scaled_data)

    # The number of clusters is the max label + 1 (labels are 0-indexed, -1 is noise)
    num_clusters = clusterer.labels_.max() + 1

    # Handle the case where no clusters are found
    if num_clusters == 0:
        print("Warning: HDBSCAN found 0 clusters. Defaulting to a small number of rules (e.g., 5).")
        return 5

    print(f"HDBSCAN identified {num_clusters} clusters (rules).")
    return num_clusters


def calculate_vanilla_macd(series, slow=26, fast=12, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_hindsight_macd(series, slow=26, fast=12, signal=9):
    return calculate_vanilla_macd(series.shift(-Y_HORIZON), fast, slow, signal)


def calculate_volatility(series, window=26): return series.pct_change().rolling(window=window).std()


def calculate_roc(series, window=13): return series.pct_change(periods=window)


def train_anfis_model(features_X, target_Y, model_params, epochs=50, lr=0.001, batch_size=32, ticker=''):
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_X.fit_transform(features_X)
    y_scaled = scaler_y.fit_transform(target_Y if len(target_Y.shape) > 1 else target_Y.reshape(-1, 1))
    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = HybridCnnAnfis(input_dim=features_X.shape[1], **model_params)
    criterion, optimizer = nn.MSELoss(reduction='mean'), torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    epoch_bar = tqdm(range(epochs), desc=f"Training {ticker} Model", leave=False)
    for _ in epoch_bar:
        epoch_loss, num_batches = 0.0, 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        epoch_bar.set_postfix(train_rmse=f"{epoch_loss / num_batches:.6f}")
    return model, scaler_X, scaler_y


def generate_predicted_macd(input_df, full_history_df, desc, price_predictor, scaler_X_price, scaler_y_price):
    predicted_macd_list, predicted_signal_list = [], []
    backward_days = 13
    price_predictor.eval()
    with torch.no_grad():
        for i in tqdm(range(len(input_df)), desc=desc, leave=False):
            current_day_index = input_df.index[i]
            current_features_np = full_history_df[['Close','Volume', 'MACD_Vanilla', 'Signal_Vanilla', 'RSI', 'BB_Width']].iloc[
                current_day_index - 1].values.reshape(1, -1)
            input_scaled = scaler_X_price.transform(current_features_np)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            pred_scaled = price_predictor(input_tensor)
            future_price_predictions = scaler_y_price.inverse_transform(pred_scaled.numpy()).flatten()
            past_prices = full_history_df['Close'].iloc[current_day_index - backward_days: current_day_index]
            hybrid_price_series = pd.concat([past_prices, pd.Series(future_price_predictions,
                                                                    index=range(len(past_prices),
                                                                                len(past_prices) + len(
                                                                                    future_price_predictions)))],
                                            ignore_index=True)
            hybrid_macd, hybrid_signal = calculate_vanilla_macd(hybrid_price_series)
            predicted_macd_list.append(hybrid_macd.iloc[backward_days - 1])
            predicted_signal_list.append(hybrid_signal.iloc[backward_days - 1])
    return predicted_macd_list, predicted_signal_list


def run_all_models(df, anfis_params, batch_size: int, epochs: int = 50, lr=0.001, ticker=''):
    df['MACD_Vanilla'], df['Signal_Vanilla'] = calculate_vanilla_macd(df['Close'])
    df['MACD_Hindsight'], df['Signal_Hindsight'] = calculate_hindsight_macd(df['Close'])
    df['RSI'] = calculate_rsi(df, 14)
    df['BB_Width'] = calculate_bollinger_width(df['Close'], window=20)
    target_cols = [f'Close_t+{i + 1}' for i in range(Y_HORIZON)]
    for i in range(Y_HORIZON): df[target_cols[i]] = df['Close'].shift(-(i + 1))
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    split_idx = int(len(df) * 0.8)
    df_train, df_test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    # --- HDBSCAN INTEGRATION ---
    clustering_features = ['Close', 'MACD_Vanilla', 'Signal_Vanilla', 'RSI', 'BB_Width']
    num_rules_from_hdbscan = get_num_rules_with_hdbscan(df_train, clustering_features)
    anfis_params['num_rules'] = num_rules_from_hdbscan
    # --- END HDBSCAN INTEGRATION ---

    price_model_features = df_train[['Close', 'Volume','MACD_Vanilla', 'Signal_Vanilla','RSI', 'BB_Width']].values
    price_model_target = df_train[target_cols].values
    anfis_params['output_dim'] = Y_HORIZON
    price_predictor, scaler_X_price, scaler_y_price = train_anfis_model(price_model_features, price_model_target,
                                                                        anfis_params, epochs=epochs, lr=lr,
                                                                        batch_size=batch_size, ticker=ticker)

    macd_pred_train, signal_pred_train = generate_predicted_macd(df_train, df, "Gen Predicted MACD (Train)",
                                                                 price_predictor, scaler_X_price, scaler_y_price)
    df_train['MACD_Predicted'], df_train['Signal_Predicted'] = macd_pred_train, signal_pred_train
    predicted_macd_list, predicted_signal_list = generate_predicted_macd(df_test, df, "Gen Predicted MACD (Test)",
                                                                         price_predictor, scaler_X_price,
                                                                         scaler_y_price)
    df_test['MACD_Predicted'], df_test['Signal_Predicted'] = predicted_macd_list, predicted_signal_list

    anfis_params.pop('output_dim', None)
    comp_model_features = df_train[
        ['Close', 'Volume','MACD_Predicted', 'Signal_Predicted', "RSI", "BB_Width"]].values
    comp_model_target = df_train['MACD_Hindsight'].values
    comp_predictor, scaler_X_comp, scaler_y_comp = train_anfis_model(comp_model_features, comp_model_target,
                                                                     anfis_params, epochs=epochs, lr=lr,
                                                                     batch_size=batch_size, ticker=ticker)

    comp_test_features = df_test[
        ['Close', 'Volume', 'MACD_Predicted', 'Signal_Predicted', "RSI", "BB_Width"]].values
    comp_test_scaled = scaler_X_comp.transform(comp_test_features)
    comp_test_tensor = torch.tensor(comp_test_scaled, dtype=torch.float32)
    comp_predictor.eval()
    with torch.no_grad():
        comp_pred_scaled = comp_predictor(comp_test_tensor)
        comp_pred_unscaled = scaler_y_comp.inverse_transform(comp_pred_scaled.numpy())
    df_test['MACD_Predicted'] = comp_pred_unscaled
    df_test['Signal_Predicted'] = df_test['MACD_Predicted'].ewm(span=9, adjust=False).mean()

    df_test['Signal_Hindsight'] = df_test['MACD_Hindsight'].ewm(span=9, adjust=False).mean()
    df_test['Signal_Vanilla'] = df_test['MACD_Vanilla'].ewm(span=9, adjust=False).mean()
    return df_test[['Date', 'Close',
                    'MACD_Vanilla', 'Signal_Vanilla',
                    'MACD_Hindsight', 'Signal_Hindsight',
                    'MACD_Predicted', 'Signal_Predicted']]


def get_multiple_data(tickers, start_date, end_date):
    stock_data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        print(f"Downloading and preprocessing data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No data downloaded for {ticker}. Check ticker symbol and date range.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"Successfully downloaded {len(df)} data points.")
        stock_data[ticker] = df
    return stock_data


def generate_trading_signals(results_df, macd_col, signal_col):
    """Generates trading signals based on a specified MACD crossover."""
    signals = pd.DataFrame(index=results_df.index)
    signals['signal'] = 0.0
    signals['signal'] = np.where(results_df[macd_col] > results_df[signal_col], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals


def calculate_strategy_performance(returns_df, weights_df, initial_investment, commission_rate=0.0):
    """Calculates the historical performance of the dynamic ANFIS strategy."""
    aligned_returns, aligned_weights = returns_df.align(weights_df, join='inner', axis=0)
    weight_changes = aligned_weights.diff().abs().sum(axis=1)
    trade_days = weight_changes > 1e-6
    if not trade_days.empty:
        trade_days.iloc[0] = False

    portfolio_history = pd.Series(index=aligned_returns.index, dtype=float)
    current_value = initial_investment
    if len(portfolio_history) > 0:
        portfolio_history.iloc[0] = current_value

    for i in range(1, len(aligned_returns)):
        daily_return = (aligned_returns.iloc[i] * aligned_weights.iloc[i - 1]).sum()
        current_value = portfolio_history.iloc[i - 1] * (1 + daily_return)
        if not trade_days.empty and trade_days.iloc[i]:
            current_value *= (1 - commission_rate)
        portfolio_history.iloc[i] = current_value
    return portfolio_history.dropna()


def calculate_benchmark_performance(returns_df, initial_investment):
    """Calculates the performance of an equal-weight, buy-and-hold strategy."""
    num_assets = len(returns_df.columns)
    weights = np.array([1 / num_assets] * num_assets)
    portfolio_returns = (returns_df * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return initial_investment * cumulative_returns


def calculate_dynamic_weights(all_signals, high_weight=0.6, low_weight=0.2):
    """Calculates dynamic portfolio weights based on trading signals."""
    tickers = list(all_signals.keys())
    num_assets = len(tickers)
    weights = pd.DataFrame(1 / num_assets, index=all_signals[tickers[0]].index, columns=tickers)

    for ticker in tickers:
        positions = all_signals[ticker]['positions']
        for i in range(1, len(positions)):
            current_weights = weights.iloc[i - 1].copy()
            if positions.iloc[i] == 1.0:  # Buy Signal
                current_weights[ticker] = high_weight
            elif positions.iloc[i] == -1.0:  # Sell Signal
                current_weights[ticker] = low_weight
            total_weight = np.sum(current_weights)
            if total_weight > 0:
                weights.iloc[i] = current_weights / total_weight
            else:
                weights.iloc[i] = 1 / num_assets
    return weights


# --- NEW: Function to detect MACD peaks and troughs ---
def detect_peaks_and_troughs(macd_series, prominence=0.1):
    """
    Detects peaks (local maxima) and troughs (local minima) in a MACD series.
    Returns a series with 1 for a peak, -1 for a trough, and 0 otherwise.
    """
    triggers = pd.Series(0, index=macd_series.index)

    # Find peaks (maxima)
    peaks, _ = find_peaks(macd_series, prominence=prominence)
    triggers.iloc[peaks] = 1

    # Find troughs (minima) by finding peaks in the inverted series
    troughs, _ = find_peaks(-macd_series, prominence=prominence)
    triggers.iloc[troughs] = -1

    return triggers


def plot_mc_simulation(simulated_data, final_values, tickers, initial_investment):
    """Plots the results of the Monte Carlo simulation."""
    plt.figure(figsize=(12, 8))
    mean_final_value = np.mean(final_values)
    percentile_5 = np.percentile(final_values, 5)
    percentile_95 = np.percentile(final_values, 95)
    mean_idx = np.argmin(np.abs(final_values - mean_final_value))
    p5_idx = np.argmin(np.abs(final_values - percentile_5))
    p95_idx = np.argmin(np.abs(final_values - percentile_95))
    special_indices = [mean_idx, p5_idx, p95_idx]

    num_simulations = simulated_data.shape[1]
    for i in range(num_simulations):
        if i not in special_indices:
            plt.plot(simulated_data[:, i], color='lightgray', alpha=0.25, linewidth=0.5)

    plt.plot(simulated_data[:, p95_idx], color='darkorange', linestyle='-', lw=2.5,
             label=f'95th Percentile Path: ${final_values[p95_idx]:,.2f}')
    plt.plot(simulated_data[:, mean_idx], color='red', linestyle='-', lw=2.5,
             label=f'Mean Path: ${final_values[mean_idx]:,.2f}')
    plt.plot(simulated_data[:, p5_idx], color='black', linestyle='-', lw=2.5,
             label=f'5th Percentile Path: ${final_values[p5_idx]:,.2f}')

    plt.title(f'Monte Carlo Simulation for Portfolio: {", ".join(tickers)} (1 Year Projection)')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    print("\n--- Monte Carlo Simulation Results ---")
    print(f"Initial Investment: ${initial_investment:,.2f}")
    print(f"Mean Expected Portfolio Value after 1 Year: ${mean_final_value:,.2f}")
    print(f"5% Worst Case Scenario: ${percentile_5:,.2f}")
    print(f"5% Best Case Scenario: ${percentile_95:,.2f}")
    if not os.path.exists("img"): os.makedirs("img")
    plt.savefig(os.path.join("img/mc_simulation_highlighted.jpg"), bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_performance_comparison(historical_performances, tickers):
    """Plots a comparison of historical backtesting and Monte Carlo projections."""
    plt.figure(figsize=(15, 10))
    colors = plt.cm.jet(np.linspace(0, 1, len(historical_performances)))
    color_map = {name: color for name, color in zip(historical_performances.keys(), colors)}

    for name, performance in historical_performances.items():
        plt.plot(performance.index, performance.values, label=f'{name} (Historical)', color=color_map[name],
                 linewidth=2)

    plt.title(f'Strategy Performance Comparison: {", ".join(tickers)}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    if not os.path.exists("img"): os.makedirs("img")
    plt.savefig(os.path.join("img/strategy_comparison.jpg"), bbox_inches='tight', pad_inches=0)
    print("\nSaved strategy comparison graph to img/strategy_comparison.jpg")
    plt.show()


if __name__ == '__main__':
    # --- Configuration ---
    PORTFOLIO_TICKERS = ['PPH', 'IYW', 'XLRE']
    START_DATE = '2010-01-01'
    END_DATE = '2023-12-31'
    INITIAL_INVESTMENT = 100000
    COMMISSION_RATE = 0.0125

    # --- GA Configuration ---
    # Defines the search space for the Genetic Algorithm
    GENE_CONFIG_SPACE = {
        'lr': {'type': 'float', 'min': 1e-8, 'max': 1e-4},
        'epochs': {'type': 'int', 'min': 50, 'max': 100, 'step': 10},
        'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
        'firing_conv_filters': {'type': 'categorical', 'choices': [32, 64, 128]},
        'consequent_conv_filters': {'type': 'int', 'min': 8, 'max': 32, 'step': 4},
    }
    GA_POPULATION_SIZE = 5
    GA_GENERATIONS = 3
    GA_MUTATION_RATE = 0.2
    GA_NUM_PARENTS = 2
    GA_NUM_ELITES = 2

    # --- 1. Data Retrieval ---
    portfolio_data = get_multiple_data(PORTFOLIO_TICKERS, START_DATE, END_DATE)

    # --- 2. Find Optimal Hyperparameters using GA for Each Ticker ---
    optimized_hyperparams = {}
    for ticker, df in portfolio_data.items():
        print(f"\n --- Running {ticker} ---")
        print("Clustering Data to generate rules")
        tmp = df.copy()
        tmp.reset_index(inplace=True)
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_X, scaler_y, X_test_scaled, y_test_scaled = prepare_data(
            tmp, ['Close', 'Volume','MACD_Vanilla', 'Signal_Vanilla', 'RSI','BB_Width'], 'Close', rolling_window=True)

        # --- HDBSCAN INTEGRATION ---
        clustering_features = ['Close', 'Volume', 'MACD_Vanilla', 'Signal_Vanilla', 'RSI', 'BB_Width']
        num_rules_from_hdbscan = get_num_rules_with_hdbscan(tmp, clustering_features)
        # anfis_fixed_params = {
        #     'input_dim': X_train_scaled.shape[1],
        #     'num_mfs': 3,
        #     'num_rules': num_rules_from_hdbscan,
        #     'firing_conv_filters': 128,
        #     'consequent_conv_filters': 64,
        #     'lr': 1e-6,
        #     'batch_size': 32,
        #     'epoch': 100
        # }
        anfis_fixed_params = {
            'input_dim': X_train_scaled.shape[1],
            'num_mfs': 3,
            'num_rules': num_rules_from_hdbscan,
        }
        optimized_hyperparams[ticker] = anfis_fixed_params
        # F. Run the Genetic Algorithm
        # best_genome = GAHandler.genetic_algorithm_anfis(
        #     gene_config_space=GENE_CONFIG_SPACE,
        #     X_train=X_train_scaled, y_train=y_train_scaled,
        #     X_val=X_val_scaled, y_val=y_val_scaled,
        #     scaler_X=scaler_X,
        #     scaler_Y=scaler_y,
        #     DEVICE=DEVICE,
        #     anfis_fixed_params=anfis_fixed_params,
        #     population_size=GA_POPULATION_SIZE,
        #     generations=GA_GENERATIONS,
        #     mutation_rate=GA_MUTATION_RATE,
        #     num_parents_for_crossover=GA_NUM_PARENTS,
        #     num_elites=GA_NUM_ELITES
        # )
        #
        # best_genome['num_rules'] = num_rules_from_hdbscan
        # optimized_hyperparams[ticker] = best_genome
    optimized_hyperparams['IYW'].update({
            'lr': 1e-5,
            'firing_conv_filters': 32,
            'consequent_conv_filters': 15,
            'epochs': 93,
            'batch_size': 16,
    })
    optimized_hyperparams['XLRE'].update({
        'lr': 16e-6,
        'firing_conv_filters': 32,
        'consequent_conv_filters': 27,
        'epochs': 77,
        'batch_size': 16,
    })
    optimized_hyperparams['PPH'].update({
        'lr': 6e-6,
        'firing_conv_filters': 64,
        'consequent_conv_filters': 29,
        'epochs': 96,
        'batch_size': 16,
    })
    # --- 3. Run ANFIS Model for Each Ticker using Optimized Params ---
    all_ticker_results = {}
    test_period_start_date = None

    for ticker, df in portfolio_data.items():
        print(f"\n--- Processing and training final model for {ticker} with optimized params ---")
        print(f"  Optimized Params: {optimized_hyperparams[ticker]}")
        tmp = df.copy()
        tmp.reset_index(inplace=True)

        # Extract optimized params
        best_params = optimized_hyperparams[ticker]
        batch_size = best_params['batch_size']
        epochs = best_params['epochs']
        lr = best_params['lr']
        anfis_params = {
            'num_mfs': 3,
            'num_rules': best_params['num_rules'],
            'firing_conv_filters': best_params['firing_conv_filters'],
            'consequent_conv_filters': best_params['consequent_conv_filters']
        }

        results_df = run_all_models(tmp, anfis_params, batch_size=batch_size, epochs=epochs, lr=lr, ticker=ticker)
        results_df.set_index('Date', inplace=True)
        all_ticker_results[ticker] = results_df

        if test_period_start_date is None:
            test_period_start_date = results_df.index[0]

    # --- 4. Define Strategies and Prepare Data for Backtesting ---
    adj_close_df = pd.concat({ticker: data['Close'] for ticker, data in portfolio_data.items()}, axis=1)
    portfolio_returns = adj_close_df.pct_change().dropna()
    test_returns = portfolio_returns[portfolio_returns.index >= test_period_start_date]

    strategies = {
        'Predicted': ('MACD_Predicted', 'Signal_Predicted'),  # MODIFIED: Changed from Compensated
        'Vanilla': ('MACD_Vanilla', 'Signal_Vanilla'),
        'Hindsight': ('MACD_Hindsight', 'Signal_Hindsight')
    }

    historical_performances = {}
    final_weights_per_strategy = {}

    print("\n--- Backtesting all strategies ---")

    # --- 5. Backtest Each Strategy ---

    # --- Part 1: Crossover-based strategies ---
    print("\n--- Part 1: Backtesting Simple Crossover Strategies ---")
    for strategy_name, (macd_col, signal_col) in strategies.items():
        print(f"  - Backtesting {strategy_name} crossover strategy...")
        all_ticker_signals = {
            ticker: generate_trading_signals(results_df, macd_col, signal_col)
            for ticker, results_df in all_ticker_results.items()
        }
        dynamic_weights = calculate_dynamic_weights(all_ticker_signals)
        final_weights_per_strategy[strategy_name] = dynamic_weights.iloc[-1].values
        strategy_performance = calculate_strategy_performance(
            test_returns, dynamic_weights, INITIAL_INVESTMENT, commission_rate=COMMISSION_RATE
        )
        historical_performances[strategy_name] = strategy_performance

    # --- Part 2: Dynamic Sharpe Ratio Optimization Strategies ---
    print("\n--- Part 2: Backtesting Dynamic Sharpe Optimization Strategies ---")
    for strategy_name, (macd_col, signal_col) in strategies.items():
        print(f"\n  - Backtesting Dynamic Sharpe Optimization triggered by '{strategy_name}' MACD...")

        # 1. Combine the relevant MACDs into one DataFrame
        macds_df = pd.concat(
            {ticker: res[macd_col] for ticker, res in all_ticker_results.items()},
            axis=1
        ).dropna()

        # 2. Detect peaks and troughs for each ticker's relevant MACD
        macd_triggers = pd.DataFrame(index=macds_df.index)
        for ticker in PORTFOLIO_TICKERS:
            macd_triggers[f'{ticker}_trigger'] = detect_peaks_and_troughs(macds_df[ticker])

        # 3. Determine the rebalance dates
        macd_triggers['rebalance_event'] = macd_triggers.abs().sum(axis=1) > 0
        rebalance_dates = macd_triggers[macd_triggers['rebalance_event']].index

        # 4. Run the backtest with dynamic re-optimization
        num_assets = len(PORTFOLIO_TICKERS)
        sharpe_weights = pd.DataFrame(index=test_returns.index, columns=PORTFOLIO_TICKERS)
        current_weights = np.array([1.0 / num_assets] * num_assets)
        lookback_period = 252

        for i, date in enumerate(tqdm(test_returns.index, desc=f"Dynamic Sharpe ({strategy_name})")):
            if date in rebalance_dates:
                # Get historical returns for optimization
                historical_data_end_index = portfolio_returns.index.get_loc(date)
                historical_slice = portfolio_returns.iloc[
                    max(0, historical_data_end_index - lookback_period):historical_data_end_index]

                if len(historical_slice) < 20:
                    sharpe_weights.loc[date] = current_weights
                    continue

                expected_returns = historical_slice.mean() * 252
                covariance_matrix = historical_slice.cov() * 252

                optimizer = PortfolioOptimizer(expected_returns, covariance_matrix, risk_free_rate=0.0125)
                try:
                    new_weights = optimizer.maximize_sharpe_ratio()
                    current_weights = new_weights
                except Exception as e:
                    # If optimization fails, hold previous weights
                    pass

            sharpe_weights.loc[date] = current_weights

        sharpe_weights.ffill(inplace=True)

        # Calculate performance of the new strategy
        sharpe_performance = calculate_strategy_performance(
            test_returns, sharpe_weights, INITIAL_INVESTMENT,commission_rate=COMMISSION_RATE
        )
        historical_performances[strategy_name] = sharpe_performance
        final_weights_per_strategy[strategy_name] = sharpe_weights.iloc[-1].values

    # --- 6. Calculate Benchmark Performance ---
    print("\n  - Calculating Benchmark performance...")
    benchmark_performance = calculate_benchmark_performance(test_returns, INITIAL_INVESTMENT)
    historical_performances['Benchmark'] = benchmark_performance

    # --- NEW: Enhanced Final Performance Reporting ---
    print(
        f"\n--- Historical Backtest Performance (Initial: ${INITIAL_INVESTMENT:,.2f} | Commission: {COMMISSION_RATE * 100:.3f}%) ---")

    # First, get the benchmark's final value for comparison
    benchmark_final_value = 0
    if 'Benchmark' in historical_performances and not historical_performances['Benchmark'].empty:
        benchmark_final_value = historical_performances['Benchmark'].iloc[-1]

    # Header for the results table
    print(f"{'Strategy':<30} | {'Final Value':>18} | {'% Gain (Initial)':>18} | {'% vs. Benchmark':>18}")
    print('-' * 90)

    # Sort strategies by final value for a ranked list
    sorted_performances = sorted(
        historical_performances.items(),
        key=lambda item: item[1].iloc[-1] if not item[1].empty else -np.inf,
        reverse=True
    )

    for name, performance in sorted_performances:
        if performance.empty:
            print(f"{name:<30} | {'N/A':>18} | {'N/A':>18} | {'N/A':>18}")
            continue

        final_value = performance.iloc[-1]

        # 1. Calculate the percentage increase from the initial investment
        pct_increase_from_initial = ((final_value - INITIAL_INVESTMENT) / INITIAL_INVESTMENT) * 100

        # 2. Calculate the percentage increase using the benchmark as the base
        if benchmark_final_value > 0 and name != 'Benchmark':
            pct_vs_benchmark = ((final_value - benchmark_final_value) / benchmark_final_value) * 100
            pct_vs_benchmark_str = f"{pct_vs_benchmark:+.2f}%"
        else:
            # The benchmark can't be compared to itself
            pct_vs_benchmark_str = "N/A"

        print(
            f"{name:<30} | ${final_value:>16,.2f} | {pct_increase_from_initial:>16,.2f}% | {pct_vs_benchmark_str:>18}")

    plot_performance_comparison(
        historical_performances,
        PORTFOLIO_TICKERS
    )
