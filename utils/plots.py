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
