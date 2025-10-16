import warnings

from scipy.signal import find_peaks
import pandas as pd
import torch
import yfinance as yf
import numpy as np
from tqdm import tqdm

from models.ANFIS.CNNANFIS import HybridCnnAnfis
from constants import DEVICE
from models.clustering.HDBScan import get_num_rules_with_hdbscan
from portfolio.portfolioOptimizer import PortfolioOptimizer
from trading.features import calculate_hindsight_macd
from tuner.GA.GAHandler import run_GA
from utilities.dataHandler import prepare_data, get_data
from utilities.plotter import plot_performance_comparison

warnings.filterwarnings('ignore')

def calculate_max_drawdown(performance_series: pd.Series) -> float:
    """Calculates the Maximum Drawdown for a strategy."""
    if performance_series.empty:
        return 0.0
    high_water_mark = performance_series.cummax()
    drawdown = (performance_series - high_water_mark) / high_water_mark
    max_drawdown = drawdown.min()
    return max_drawdown


def calculate_cagr(performance_series: pd.Series) -> float:
    """Calculates the Compound Annual Growth Rate (CAGR) for a strategy."""
    if performance_series.empty or len(performance_series) < 2:
        return 0.0
    start_value = performance_series.iloc[0]
    end_value = performance_series.iloc[-1]
    if start_value == 0:
        return 0.0
    num_days = (performance_series.index[-1] - performance_series.index[0]).days
    if num_days == 0:
        return 0.0
    num_years = num_days / 365.25
    cagr = (end_value / start_value) ** (1 / num_years) - 1
    return cagr

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


def generate_combined_triggers(results_df: pd.DataFrame, macd_col: str, rsi_col: str, bbw_col: str,
                               rsi_overbought: int = 70, rsi_oversold: int = 30,
                               bbw_squeeze_percentile: float = 0.05, bbw_lookback: int = 252) -> pd.Series:
    """
    Generates rebalancing triggers based on a combination of MACD, RSI, and Bollinger Band Width.

    A trigger is generated if:
    1. A MACD peak/trough coincides with an extreme RSI reading.
    2. The Bollinger Band Width is in a 'squeeze' (historically low).
    """
    triggers = pd.Series(False, index=results_df.index)

    # Condition 1: Confirmed Momentum Reversal (MACD + RSI)
    macd_events = detect_peaks_and_troughs(results_df[macd_col]) != 0
    rsi_extreme = (results_df[rsi_col] > rsi_overbought) | (results_df[rsi_col] < rsi_oversold)
    confirmed_reversal_triggers = macd_events & rsi_extreme

    # Condition 2: Volatility Breakout Anticipation (BBW Squeeze)
    rolling_bbw_low_threshold = results_df[bbw_col].rolling(window=bbw_lookback).quantile(bbw_squeeze_percentile)
    bbw_squeeze_triggers = results_df[bbw_col] < rolling_bbw_low_threshold

    # Combine all trigger conditions
    combined_triggers = confirmed_reversal_triggers | bbw_squeeze_triggers

    return combined_triggers


def mock_optimized_hyperparams():
    optimized_hyperparams = {'IYW': {
        'lr': 1e-5,
        'firing_conv_filters': 32,
        'consequent_conv_filters': 15,
        'epochs': 5,
        'batch_size': 8,
    }, 'XLRE': {
        'lr': 16e-6,
        'firing_conv_filters': 32,
        'consequent_conv_filters': 27,
        'epochs': 5,
        'batch_size': 16,
    }, 'PPH': {
        'lr': 6e-6,
        'firing_conv_filters': 64,
        'consequent_conv_filters': 29,
        'epochs': 5,
        'batch_size': 16,
    }}
    return optimized_hyperparams

def main():
    # --- Configuration ---
    PORTFOLIO_TICKERS = ['PPH', 'IYW', 'XLRE']

    # --- Benchmark ---
    BENCHMARK_TICKERS = {
        "S&P 500 (SPY)": ["SPY"],
        "JP MORGAN": ["JDOC", "JPRE"]
    }

    START_DATE = '2010-01-01'
    END_DATE = '2023-12-31'
    INITIAL_INVESTMENT = 100000
    COMMISSION_RATE = 0.0125

    # --- GA Configuration ---
    GENE_CONFIG_SPACE = {
        'lr': {'type': 'float', 'min': 1e-8, 'max': 1e-4},
        'epochs': {'type': 'int', 'min': 50, 'max': 150, 'step': 10},
        'batch_size': {'type': 'categorical', 'choices': [4, 8, 16, 32, 64]},
        'firing_conv_filters': {'type': 'categorical', 'choices': [16,32, 64, 128, 256, 512]},
        'consequent_conv_filters': {'type': 'int', 'min': 8, 'max': 128, 'step': 4},
    }
    GA_POPULATION_SIZE = 10
    GA_GENERATIONS = 5
    GA_MUTATION_RATE = 0.2
    GA_NUM_PARENTS = 2
    GA_NUM_ELITES = 2

    # --- 1. Data Retrieval ---
    portfolio_data = get_data(PORTFOLIO_TICKERS, START_DATE, END_DATE)
    ticker_results = {}
    test_period_start_date = None
    # optimized_hyperparams = mock_optimized_hyperparams()
    for ticker, df in portfolio_data.items():
        print(f"\n --- Running {ticker} ---")
        print("Clustering Data to generate rules")
        tmp = df.copy()
        tmp.reset_index(inplace=True)
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_X, scaler_y, X_test_scaled, y_test_scaled = prepare_data(
            tmp, ['Close', 'MACD_Vanilla', 'Signal_Vanilla', 'RSI','BB_Width'], 'Close', rolling_window=True)

        final_train_x_scaled = np.concatenate((X_train_scaled, X_val_scaled))
        final_train_y_scaled = np.concatenate((y_train_scaled, y_val_scaled))
        df_processed = tmp.copy()

        test_dates = df_processed.iloc[-len(X_test_scaled):]['Date']
        initial_unscaled_close_history = df_processed['Close'].iloc[:-len(X_test_scaled)]
        best_genome = run_GA(gene_config_space=GENE_CONFIG_SPACE,
                             population_size=GA_POPULATION_SIZE,
                             generations=GA_GENERATIONS,
                             mutation_rate=GA_MUTATION_RATE,
                             num_parents=GA_NUM_PARENTS,
                             num_elites=GA_NUM_ELITES,
                             X_train_scaled=final_train_x_scaled,
                             y_train_scaled=final_train_y_scaled,
                             X_val_scaled=X_val_scaled,
                             y_val_scaled=y_val_scaled,
                             scaler_X=scaler_X,
                             scaler_y=scaler_y)

        # num_rules_from_hdbscan = get_num_rules_with_hdbscan(final_train_x_scaled)
        # anfis_fixed_params = {
        #     'input_dim': X_train_scaled.shape[1],
        #     'num_mfs': 3,
        #     'num_rules': num_rules_from_hdbscan,
        # }
        # optimized_hyperparams[ticker].update(**anfis_fixed_params)
        # best_genome = optimized_hyperparams[ticker]
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
            'target_scaler': scaler_y,
        }

        model = HybridCnnAnfis(**anfis_params, device=DEVICE).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.fit(final_train_x_scaled, final_train_y_scaled, optimizer=optimizer, batch_size=batch_size, epochs=epochs)
        predictions, actuals, dates = model.rolling_prediction(
            initial_unscaled_close_history=initial_unscaled_close_history,
            train_X_scaled=final_train_x_scaled,
            train_y_scaled=final_train_y_scaled,
            X_test_scaled=X_test_scaled,
            y_test_scaled=y_test_scaled,
            test_dates=test_dates,
            look_forward_period=13
        )

        ticker_df = df_processed[df_processed['Date'].isin(dates)].copy()
        initial_unscaled_close_history.index = df_processed['Date'].iloc[:-len(X_test_scaled)]
        predicted_series = pd.Series(predictions, index=dates)

        combined_series = pd.concat([initial_unscaled_close_history, predicted_series])

        predicted_macd, predicted_macd_signal = calculate_hindsight_macd(combined_series)
        ticker_df['MACD_Predicted'] = predicted_macd.loc[dates]
        ticker_df['MACD_Signal'] = predicted_macd_signal.loc[dates]
        ticker_df.set_index('Date', inplace=True)
        ticker_results[ticker] = ticker_df
        if test_period_start_date is None:
            test_period_start_date = ticker_df.index[0]

    # --- Define Strategies and Prepare
    adj_close_df = pd.concat({ticker: data['Close'] for ticker, data in portfolio_data.items()}, axis=1)
    portfolio_returns = adj_close_df.pct_change().dropna()
    test_returns = portfolio_returns[portfolio_returns.index >= test_period_start_date]

    # --- 5. Backtest Each Strategy ---
    print("\n--- Backtesting all strategies ---")

    lookback_period = 252
    if len(test_returns) < lookback_period:
        raise ValueError("Test period is shorter than the lookback period. Cannot run backtest.")
    evaluation_start_date = test_returns.index[lookback_period]
    print(f"\nWarm-up period complete. All strategies will be evaluated from: {evaluation_start_date.date()}")
    evaluation_returns = test_returns[evaluation_start_date:]

    # Define the trigger methods and the MACD data sources separately
    trigger_methods = {
        "MACD Peak": "peak",
        "Combined": "combined"
    }

    # The base strategies defining which MACD data to use
    macd_sources = {
        'Predicted': 'MACD_Predicted',
        'Vanilla': 'MACD_Vanilla',
        'Hindsight': 'MACD_Hindsight'
    }

    historical_performances = {}
    final_weights_per_strategy = {}

    # Outer loop for trigger method, inner loop for MACD data source
    for trigger_name, trigger_type in trigger_methods.items():
        print(f"\n--- Backtesting Strategies with '{trigger_name}' Triggers ---")

        for source_name, macd_col in macd_sources.items():
            full_strategy_name = f"{source_name} ({trigger_name})"
            print(f"\n  - Backtesting '{full_strategy_name}'...")

            # 1. Generate triggers based on the selected method
            all_triggers = pd.DataFrame(index=evaluation_returns.index)
            for ticker in PORTFOLIO_TICKERS:
                ticker_results_df = ticker_results[ticker]

                if trigger_type == "peak":
                    # Method 1: Original MACD Peak/Trough detection
                    macd_events = detect_peaks_and_troughs(ticker_results_df[macd_col])
                    all_triggers[f'{ticker}_trigger'] = (macd_events != 0)

                elif trigger_type == "combined":
                    # Method 2: New Combined (MACD+RSI+BBW) detection
                    all_triggers[f'{ticker}_trigger'] = generate_combined_triggers(
                        results_df=ticker_results_df,
                        macd_col=macd_col,
                        rsi_col='RSI',
                        bbw_col='BB_Width'
                    )

            # 2. Determine rebalance dates
            rebalance_event = all_triggers.any(axis=1)
            rebalance_dates = evaluation_returns.index[rebalance_event]
            print(f"    Found {len(rebalance_dates)} rebalancing events.")

            # 3. Run the backtest with dynamic re-optimization
            num_assets = len(PORTFOLIO_TICKERS)
            sharpe_weights = pd.DataFrame(index=evaluation_returns.index, columns=PORTFOLIO_TICKERS)
            current_weights = np.array([1.0 / num_assets] * num_assets)

            for i, date in enumerate(tqdm(evaluation_returns.index, desc=f"Sharpe Opt ({full_strategy_name})")):
                if i == 0 or date in rebalance_dates:
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
                        pass

                sharpe_weights.loc[date] = current_weights

            sharpe_weights.ffill(inplace=True)

            # Calculate and store performance
            strategy_performance = calculate_strategy_performance(
                evaluation_returns, sharpe_weights, INITIAL_INVESTMENT
            )
            historical_performances[full_strategy_name] = strategy_performance
            final_weights_per_strategy[full_strategy_name] = sharpe_weights.iloc[-1].values

    # --- 6. Calculate Benchmark Performance ---
    print("\n--- Calculating Benchmark Performances ---")

    # First, the original equal-weight portfolio benchmark
    print("  - Calculating Equal-Weight Benchmark...")
    benchmark_performance = calculate_benchmark_performance(evaluation_returns, INITIAL_INVESTMENT)
    historical_performances['Equal-Weight Benchmark'] = benchmark_performance

    for name, ticker in BENCHMARK_TICKERS.items():
        print(f"  - Calculating '{name}' Benchmark...")
        try:
            # Download historical data for the benchmark ticker
            benchmark_df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

            # Get the returns for the specific evaluation period
            benchmark_test_returns = benchmark_df['Close'].pct_change().dropna()
            benchmark_evaluation_returns = benchmark_test_returns[evaluation_start_date:]

            if benchmark_evaluation_returns.empty:
                print(f"    Warning: No data available for {ticker} in the evaluation period. Skipping.")
                continue

            # Calculate the buy-and-hold performance
            ib_benchmark_performance = calculate_benchmark_performance(benchmark_evaluation_returns, INITIAL_INVESTMENT)
            historical_performances[name] = ib_benchmark_performance

        except Exception as e:
            print(f"    Could not download or process data for {ticker}: {e}")

    # --- 7. Enhanced Final Performance Reporting ---
    print(
        f"\n--- Historical Backtest Performance (Initial: ${INITIAL_INVESTMENT:,.2f} | Commission: {COMMISSION_RATE * 100:.3f}%) ---")

    # Use S&P 500 as the primary comparison point if available, otherwise use the equal-weight one
    primary_benchmark_name = 'S&P 500 (SPY)' if 'S&P 500 (SPY)' in historical_performances else 'Equal-Weight Benchmark'
    primary_benchmark_final_value = historical_performances.get(primary_benchmark_name, pd.Series([0])).iloc[-1]
    print(f"(Comparing against '{primary_benchmark_name}' as primary benchmark)")

    print(f"{'Strategy':<35} | {'Final Value':>18} | {'% Gain (Initial)':>18} | {'% vs. Benchmark':>18}")
    print('-' * 100)

    sorted_performances = sorted(
        historical_performances.items(),
        key=lambda item: item[1].iloc[-1] if not item[1].empty else -np.inf,
        reverse=True
    )

    for name, performance in sorted_performances:
        if performance.empty:
            print(f"{name:<35} | {'N/A':>18} | {'N/A':>18} | {'N/A':>18}")
            continue
        final_value = performance.iloc[-1]
        pct_increase_from_initial = ((final_value - INITIAL_INVESTMENT) / INITIAL_INVESTMENT) * 100

        if primary_benchmark_final_value > 0 and name != primary_benchmark_name:
            pct_vs_benchmark = ((final_value - primary_benchmark_final_value) / primary_benchmark_final_value) * 100
            pct_vs_benchmark_str = f"{pct_vs_benchmark:+.2f}%"
        else:
            pct_vs_benchmark_str = "N/A"

        print(
            f"{name:<35} | ${final_value:>16,.2f} | {pct_increase_from_initial:>16,.2f}% | {pct_vs_benchmark_str:>18}")

    benchmark_final_value = 0
    if 'Benchmark' in historical_performances and not historical_performances['Benchmark'].empty:
        benchmark_final_value = historical_performances['Benchmark'].iloc[-1]

    # Header for the results table
    print(f"{'Strategy':<20} | {'Final Value':>18} | {'% Gain (Initial)':>18} | {'CAGR':>10} | {'Max Drawdown':>15} | {'% vs. Benchmark':>18}")
    print('-' * 110)
    for name, performance in sorted_performances:
        if performance.empty:
            print(f"{name:<20} | {'N/A':>18} | {'N/A':>18} | {'N/A':>10} | {'N/A':>15} | {'N/A':>18}")
            continue

        final_value = performance.iloc[-1]
        pct_increase_from_initial = ((final_value - INITIAL_INVESTMENT) / INITIAL_INVESTMENT) * 100

        # Use the new functions to get CAGR and Drawdown
        cagr = calculate_cagr(performance)
        max_drawdown = calculate_max_drawdown(performance)

        if benchmark_final_value > 0 and name != 'Benchmark':
            pct_vs_benchmark = ((final_value - benchmark_final_value) / benchmark_final_value) * 100
            pct_vs_benchmark_str = f"{pct_vs_benchmark:+.2f}%"
        else:
            pct_vs_benchmark_str = "N/A"

        print(
            f"{name:<20} | ${final_value:>16,.2f} | {pct_increase_from_initial:>16,.2f}% | "
            f"{cagr:>9.2%} | {max_drawdown:>14.2%} | {pct_vs_benchmark_str:>18}"
        )

    hindsight_performance = {k:v for k,v in historical_performances.items() if "Combined" not in k}
    combined_performance = {k:v for k,v in historical_performances.items() if "MACD Peak" not in k}
    # --- 8. Plot Results using the subplot function ---
    # The existing plot function will automatically add the new benchmarks to the graphs.
    plot_performance_comparison(
        hindsight_performance,
        PORTFOLIO_TICKERS,
        "hindsight_macd_comparison_portfolio.jpg"
    )

    plot_performance_comparison(
        combined_performance,
        PORTFOLIO_TICKERS,
        "combined_indicator_comparison_portfolio.jpg"
    )

if __name__ == '__main__':
    main()