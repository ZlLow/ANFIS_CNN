import os

from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_actual_vs_predicted(actual, predicted, model: Optional[str] = "CNN-ANFIS", dates: Optional = None,title: Optional[str] = "Stock Price Prediction using Hybrid CNN-ANFIS", save_path: Optional[str] = None):
    plt.figure(figsize=(15, 7))
    plt.title(title)
    if dates is not None:
        plt.plot(dates,actual, label='Actual Price', color='blue', alpha=0.7)
        plt.plot(dates,predicted, label=f'{model} Predicted Price', color='red')
        plt.xlabel("Time step (Days)")
    else:
        plt.plot(actual, label='Actual Price', color='blue', alpha=0.7)
        plt.plot(predicted, label=f'{model} Predicted Price', color='red')
        plt.xlabel("Dates")
    plt.ylabel("Close Price")
    plt.legend()
    if save_path is not None:
        plt.savefig(os.path.join(f"img/{save_path}"),
                    bbox_inches='tight', pad_inches=0)
    plt.grid(True)
    plt.show()

def plot_learning_curves(train_losses, val_losses, title="Learning Curves", save_path: Optional[str] = None):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training RMSE')
    plt.plot(val_losses, label='Validation RMSE')
    plt.legend()
    plt.title(title)
    if save_path is not None:
        plt.savefig(os.path.join(f"img/{save_path}"),
                    bbox_inches='tight', pad_inches=0)
    plt.grid(True)

def plot_table(fold_results, title: str="RMSE", save_path: Optional[str] = None):
    table_data = []
    for i, score in enumerate(fold_results):
        table_data.append([f'Fold {i+1}', f'${score:.6f}'])

    column_headers = ["Metric", f"{title} (Unscaled)"]

    # 2. Create the plot
    fig, ax = plt.subplots(figsize=(6, 3)) # Adjust figsize as needed
    ax.axis('tight')
    ax.axis('off')

    # 3. Create the table and add it to the axes
    the_table = ax.table(cellText=table_data,
                         colLabels=column_headers,
                         loc='center',
                         cellLoc='center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 1.2) # Adjust scale to make it larger or smaller

    # 4. Add a title and show the plot
    plt.title(f"K-Fold {title} Table Score", fontsize=16, pad=20)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(f"img/{save_path}"),
                    bbox_inches='tight', pad_inches=0)


def plot_r2_table(result, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 3)) # Adjust figsize as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=result,
                     colLabels=['RMSE','R^2'],
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2) # Adjust scale to make it larger or smaller

    # 4. Add a title and show the plot
    plt.title(f"R^2 Table Score", fontsize=16, pad=20)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(f"img/{save_path}"),
                    bbox_inches='tight', pad_inches=0)


def plot_predicted_comparison(df, model: Optional[str] = "CNN-ANFIS", save_path: Optional[str] = None):
    """
    Generates a plot to compare the Predicted ANFIS model against the benchmarks.
    """
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [2, 3]})
    fig.suptitle(f'Comparison with Models: Predicted {model} MACD', fontsize=16)

    # Panel 1: Stock Closing Price for context
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue', alpha=0.8)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title('Stock Price (Test Period)', fontsize=14)
    ax1.legend()
    ax1.grid(True)

    # Panel 2: MACD Comparison
    ax2.plot(df['Date'], df['MACD_Vanilla'], label='Model 1: Vanilla (Lagging)', color='orange', linestyle='--',
             linewidth=1.5)
    ax2.plot(df['Date'], df['MACD_Hindsight'], label='Model 2: Hindsight (Ideal Benchmark)', color='black',
             linewidth=2.5)
    ax2.plot(df['Date'], df['MACD_Predicted'], label='Model 3: Predicted ANFIS', color='green', alpha=0.9,
             linewidth=1.5)

    ax2.axhline(0, color='grey', linestyle='--', linewidth=1)  # Zero line for crossovers
    ax2.set_ylabel('MACD Value', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('MACD Indicator Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True)
    if save_path is not None:
        plt.savefig(os.path.join(f"img/{save_path}"),
                    bbox_inches='tight', pad_inches=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def get_simulation_insights(sim_results, initial_investment):
    mean_return = np.mean(sim_results)
    median_return = np.median(sim_results)
    std_dev = np.std(sim_results)
    percentile_5 = np.percentile(sim_results, 5)
    var_95 = initial_investment - percentile_5  # VaR at 95% confidence
    cvar_95 = initial_investment - np.mean(sim_results[sim_results <= percentile_5])
    prob_loss = np.mean(sim_results < initial_investment) * 100
    sharpe_ratio = (mean_return - initial_investment) / std_dev  # Assuming risk-free rate is 0

    insights = {
        'Initial Investment': f"${initial_investment:,.2f}",
        'Expected Final Portfolio Value': f"${mean_return:,.2f}",
        'Median Final Portfolio Value': f"${median_return:,.2f}",
        'Standard Deviation of Final Portfolio Value': f"${std_dev:,.2f}",
        'Value at Risk (VaR 95%)': f"${var_95:,.2f}",
        'Conditional Value at Risk (CVaR 95%)': f"${cvar_95:,.2f}",
        'Probability of Loss': f"{prob_loss:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.4f}"
    }
    return insights


def plot_comparison_graph(strategy_perf, benchmark_perf, mc_mean_projection, tickers, model: Optional[str] = "CNN-ANFIS", save_path: Optional[str] = None):
    """Plots the historical strategy performance against the benchmark and adds the MC projection."""
    plt.figure(figsize=(14, 7))

    # Plot historical performance
    strategy_perf.plot(label=f'{model} Optimized Strategy', color='blue', lw=2)
    benchmark_perf.plot(label='Equal-Weight Buy & Hold Benchmark', color='gray', linestyle='--', lw=2)

    # Create future date index for the projection
    last_date = strategy_perf.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(mc_mean_projection) + 1, freq='B')[1:]

    # Combine last historical point with projection for a continuous line
    projection_series = pd.Series(
        np.concatenate(([strategy_perf.iloc[-1]], mc_mean_projection)),
        index=[last_date] + list(future_dates)
    )

    projection_series.plot(label='Mean Monte Carlo Projection', color='red', linestyle='-.', lw=2)

    plt.title(f'Strategy Performance vs. Benchmark for {", ".join(tickers)}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(os.path.join(f"img/{save_path}"),
                    bbox_inches='tight', pad_inches=0)
    plt.show()

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


def plot_performance_comparison(historical_performances, tickers, save_path: Optional[str] = None):
    """Plots a comparison of historical backtesting and Monte Carlo projections."""
    plt.figure(figsize=(15, 10))
    colors = plt.cm.jet(np.linspace(0, 1, len(historical_performances)))
    color_map = {name: color for name, color in zip(historical_performances.keys(), colors)}

    for name, performance in historical_performances.items():
        plt.plot(performance.index, performance.values, label=f'{name}', color=color_map[name],
                 linewidth=2)

    plt.title(f'Portfolio Comparison: {", ".join(tickers)}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    if not os.path.exists("img"): os.makedirs("img")
    if save_path is not None:
        plt.savefig(os.path.join(f"img/{save_path}"), bbox_inches='tight', pad_inches=0)
    plt.show()