import os

from typing import Optional
from matplotlib import pyplot as plt

def plot_actual_vs_predicted(actual, predicted, save_path: Optional[str] = None):
    plt.figure(figsize=(15, 7))
    plt.plot(actual, label='Actual Price', color='blue', alpha=0.7)
    plt.plot(predicted, label='Hybrid CNN-ANFIS Predicted Price', color='red')
    plt.legend()
    plt.title("Stock Price Prediction using Hybrid CNN-ANFIS")
    plt.xlabel("Time Steps (Days)")
    plt.ylabel("Close Price")
    if save_path is not None:
        plt.savefig(os.path.join(save_path),
                    bbox_inches='tight', pad_inches=0)
    plt.grid(True)
    plt.show()

def plt_learningcurves(train_losses, val_losses,title: Optional[str] = None, save_path: Optional[str] = None):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training RMSE (Scaled)')
    plt.plot(val_losses, label='Checking RMSE (Scaled)')
    plt.legend()
    if save_path is not None:
        fig = plt.plot.get_figure()
        fig.savefig(os.path.join(save_path),
                    bbox_inches='tight', pad_inches=0)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()

def plt_rmse_table(fold_results, mean_rmse, std_rmse, save_path: Optional[str] = None):
    table_data = []
    for i, score in enumerate(fold_results):
        table_data.append([f'Fold {i+1}', f'${score:.6f}'])
    table_data.append(['---', '---']) # Separator row
    table_data.append(['Mean', f'{mean_rmse:.6f}'])
    table_data.append(['Std Dev', f'{std_rmse:.6f}'])
    table_data.append(['---', '---'])

    column_headers = ["Metric", "RMSE (Unscaled)"]

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
    plt.title('K-Fold Cross-Validation Results', fontsize=16, pad=20)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(save_path),
                    bbox_inches='tight', pad_inches=0)
    plt.show()

def plt_r2_score_table(fold_results, mean_r2_score, std_r2_score, save_path: Optional[str] = None):
    table_data = []
    for i, score in enumerate(fold_results):
        table_data.append([f'Fold {i+1}', f'${score:.6f}'])
    table_data.append(['---', '---']) # Separator row
    table_data.append(['Mean', f'{mean_r2_score:.6f}'])
    table_data.append(['Std Dev', f'{std_r2_score:.6f}'])
    table_data.append(['---', '---'])

    column_headers = ["Metric", "R2 Score (Unscaled)"]

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
    plt.title('K-Fold Cross-Validation Results', fontsize=16, pad=20)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(save_path),
                    bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_graph(historical_data,save_path: Optional[str] = None):
    plt.figure(figsize=(15, 7))
    plt.plot(historical_data, label='Actual Price', color='blue', alpha=0.7)
    plt.legend()
    plt.title("Stock Price Prediction using Hybrid CNN-ANFIS")
    plt.xlabel("Time Steps (Days)")
    plt.ylabel("Close Price")
    if save_path is not None:
        plt.savefig(os.path.join(save_path),
                    bbox_inches='tight', pad_inches=0)
    plt.grid(True)
    plt.show()