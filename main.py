import math

import torch
from torch.utils.data import TensorDataset

from DataHandler.dataHandler import get_tickers
from DataHandler.utils import plots, utils
from models.ANFIS.ANFIS import ANFIS
from models.clustering.HDBScan import HDBSCANHandler

# Global
y_horizon = 2

top100 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG',
          'LLY', 'JPM', 'AVGO', 'TSLA', 'UNH', 'XOM', 'V', 'PG', 'JNJ', 'MA',
          'COST', 'HD', 'ABBV', 'WMT', 'MRK', 'NFLX', 'KO', 'BAC', 'ADBE',
          'PEP', 'CVX', 'CRM', 'TMO', 'ORCL', 'LIN', 'AMD', 'ACN', 'MCD',
          'ABT', 'CSCO', 'PM', 'WFC', 'IBM', 'TXN', 'QCOM', 'GE', 'DHR',
          'VZ', 'INTU', 'AMGN', 'NOW', 'ISRG', 'NEE', 'SPGI', 'PFE', 'CAT',
          'DIS', 'RTX', 'CMCSA', 'GS', 'UNP', 'T', 'AMAT', 'PGR',
          'LOW', 'AXP', 'TJX', 'HON', 'BKNG', 'ELV', 'COP', 'SYK', 'MS',
          'LMT', 'VRTX', 'BLK', 'REGN', 'MDT', 'BSX', 'PLD', 'CB', 'ETN',
          'C', 'MMC', 'ADP', 'AMT', 'PANW', 'ADI', 'SBUX', 'MDLZ', 'CI',
          'TMUS', 'FI', 'BMY', 'DE', 'GILD', 'BX', 'NKE', 'SO', 'LRCX', 'MU', 'KLAC', 'SCHW']

top10 = top100[:10]

stationary_cols = ['Open', 'High', 'Low', 'Close', 'Volume']#, 'notional_traded']

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def eval_ANFIS():
    print("Evaluating ANFIS")
    train_X, train_y, test_X, test_y = get_tickers(top10, "1998-01-01", "2015-12-31", "2016-01-01",
                                                              "2024-01-01", y_horizon)

    cluster = HDBSCANHandler(train_X.drop(columns=stationary_cols + ['symbol']), train_y.drop(columns=['symbol']),
                             test_X.drop(columns=stationary_cols + ['symbol']), test_y.drop(columns=['symbol']),
                             )
    clustered_train_X, clustered_train_y, clustered_test_X, clustered_test_y = cluster.cluster()

    pred_col = test_y.columns[0]

    train_val_split = 0.7
    unique_train_dates = clustered_train_X.index.unique()
    split_date = unique_train_dates[int(len(unique_train_dates) * train_val_split)]

    X = clustered_train_X.loc[clustered_train_X.index < split_date].to_numpy()
    val_X = clustered_train_X.loc[clustered_train_X.index >= split_date].to_numpy()

    y = clustered_train_y[[c for c in clustered_train_y.columns if pred_col in c]].loc[
        clustered_train_y.index < split_date].to_numpy()
    val_y = clustered_train_y[[c for c in clustered_train_y.columns if pred_col in c]].loc[
        clustered_train_y.index >= split_date].to_numpy()
    crisp_val_y = train_y[pred_col].loc[train_y.index >= split_date].to_numpy()

    x_padded, x_test_padded = utils.padData(X, val_X, math.ceil(X.shape[1] / 10) * 10 - X.shape[1])

    MEMBFUNCS = [
        {'function': 'gaussian',
         'n_memb': 3,
         'params': {'mu': {'value': [-0.75,0.0, 0.75],
                           'trainable': True},
                    'sigma': {'value': [1.0, 1.0, 1.0],
                              'trainable': True}}},

        {'function': 'gaussian',
         'n_memb': 3,
         'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                           'trainable': True},
                    'sigma': {'value': [1.0, 1.0, 1.0],
                              'trainable': True}}},
    ]

    anfis = ANFIS(n_input=2, membfuncs=MEMBFUNCS, to_device=device)
    train_dataset = TensorDataset(torch.from_numpy(x_padded), torch.from_numpy(y))
    valid_dataset = TensorDataset(torch.from_numpy(x_test_padded), torch.from_numpy(val_y))
    optimizer = torch.optim.Adam(params=anfis.parameters())
    loss_functions = torch.nn.MSELoss(reduction='mean')
    history = anfis.fit(train_dataset, valid_dataset, optimizer=optimizer, loss_function=loss_functions, batch_size=4 * 512)

    y_pred = anfis.predict(X)
    # plot learning curves
    plots.plt_learningcurves(history, save_path='img/learning_curves.jpg')

    # plot prediction
    plots.plt_prediction(y, y_pred, save_path='img/mackey_prediction.jpg')


def main():
    eval_ANFIS()


if __name__ == "__main__":
    main()
