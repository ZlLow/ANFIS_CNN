import os

from typing import Optional
from matplotlib import pyplot as plt

def plot_actual_vs_predicted(actual, predicted, dates: Optional = None,title: Optional[str] = "Stock Price Prediction using Hybrid CNN-ANFIS", save_path: Optional[str] = None):
    plt.figure(figsize=(15, 7))
    plt.title(title)
    if dates is not None:
        plt.plot(dates,actual, label='Actual Price', color='blue', alpha=0.7)
        plt.plot(dates,predicted, label='Hybrid CNN-ANFIS Predicted Price', color='red')
        plt.xlabel("Time step (Days)")
    else:
        plt.plot(actual, label='Actual Price', color='blue', alpha=0.7)
        plt.plot(predicted, label='Hybrid CNN-ANFIS Predicted Price', color='red')
        plt.xlabel("Dates")
    plt.ylabel("Close Price")
    plt.legend()
    if save_path is not None:
        plt.savefig(os.path.join(save_path),
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
        plt.savefig(os.path.join(save_path),
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
        fig.savefig(os.path.join(save_path),
                    bbox_inches='tight', pad_inches=0)

def plot_graph(historical_data,dates: Optional = None,save_path: Optional[str] = None):
    plt.figure(figsize=(15, 7))
    if dates is not None:
        plt.plot(dates,historical_data, label='Actual Price', color='blue', alpha=0.7)
    else:
        plt.plot(historical_data, label='Actual Price', color='blue', alpha=0.7)
    plt.legend()
    plt.title("Stock Price Prediction using Hybrid CNN-ANFIS")
    plt.xlabel("Dates")
    plt.ylabel("Close Price")
    if save_path is not None:
        plt.savefig(os.path.join(save_path),
                    bbox_inches='tight', pad_inches=0)
    plt.grid(True)