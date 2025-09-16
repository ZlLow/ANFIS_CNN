import math

import torch
from torch.utils.data import TensorDataset

from DataHandler.dataHandler import get_tickers
from DataHandler.utils import plots, utils
from models.ANFIS.ANFIS import ANFIS
from models.ANFIS.ClusteredANFIS import create_clustered_anfis_from_data, \
    prepare_clustered_data_for_training
from models.ANFIS.TSKCNN.ANFISCNN import create_clustered_cnn_anfis_from_data
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

stationary_cols = ['Open', 'High', 'Low', 'Close', 'Volume']  # , 'notional_traded']

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def eval_ANFIS_traditional():
    """Traditional ANFIS evaluation (for comparison)"""
    print("Evaluating Traditional ANFIS")
    train_X, train_y, test_X, test_y = get_tickers(top10, "2005-01-01", "2015-12-31", "2016-01-01",
                                                   "2018-01-01", y_horizon)

    pred_col = test_y.columns[0]

    train_val_split = 0.7
    unique_train_dates = train_X.index.unique()
    split_date = unique_train_dates[int(len(unique_train_dates) * train_val_split)]

    X = train_X.drop(columns=stationary_cols + ['symbol']).loc[train_X.index < split_date]
    val_X = train_X.drop(columns=stationary_cols + ['symbol']).loc[train_X.index >= split_date]
    X = X.dropna()
    val_X = val_X.dropna()

    y = train_y[pred_col].loc[train_y.index < split_date].to_numpy().reshape(-1, 1)
    val_y = train_y[pred_col].loc[train_y.index >= split_date].to_numpy().reshape(-1, 1)


    x_padded, x_test_padded = utils.padData(X, val_X, math.ceil(X.shape[1] / 10) * 10 - X.shape[1])

    MEMBFUNCS = [
        {'function': 'gaussian',
         'n_memb': 3,
         'params': {'mu': {'value': [-0.75, 0.0, 0.75],
                           'trainable': True},
                    'sigma': {'value': [1.0, 1.0, 1.0],
                              'trainable': True}}},

        {'function': 'gaussian',
         'n_memb': 3,
         'params': {'mu': {'value': [-0.5, 0.0, 0.5],
                           'trainable': True},
                    'sigma': {'value': [1.0, 1.0, 1.0],
                              'trainable': True}}},
        {'function': 'gaussian',
         "n_memb": 3,
         "params": {'mu': {'value': [-0.5, 0.0, 0.5],
                           'trainable': True},
                    'sigma': {'value': [1.0, 1.0, 1.0],
                              'trainable': True}}}
    ]

    anfis = ANFIS(n_input=x_padded.shape[1], membfuncs=MEMBFUNCS, to_device=device)
    train_dataset = TensorDataset(torch.from_numpy(x_padded).float(), torch.from_numpy(y).float())
    valid_dataset = TensorDataset(torch.from_numpy(x_test_padded).float(), torch.from_numpy(val_y).float())
    optimizer = torch.optim.Adam(params=anfis.parameters())
    loss_functions = torch.nn.MSELoss(reduction='mean')
    history = anfis.fit(train_dataset, valid_dataset, optimizer=optimizer,
                        loss_function=loss_functions, batch_size=4 * 512, epochs=1000)

    y_pred = anfis.predict([torch.from_numpy(x_padded).float()])

    # plot learning curves
    plots.plt_learningcurves(history, save_path='img/traditional_anfis_learning_curves.jpg')
    # plot prediction
    plots.plt_prediction(y, y_pred, save_path='img/traditional_anfis_prediction.jpg')

def eval_clustered_ANFIS():
    """Clustered ANFIS evaluation using pre-computed memberships"""
    print("Evaluating Clustered ANFIS")

    # Get original data
    train_X, train_y, test_X, test_y = get_tickers(top10, "2005-01-01", "2015-12-31", "2016-01-01",
                                                   "2018-01-01", y_horizon)

    # Perform clustering
    cluster = HDBSCANHandler(train_X.drop(columns=stationary_cols + ['symbol']),
                             train_y.drop(columns=['symbol']),
                             test_X.drop(columns=stationary_cols + ['symbol']),
                             test_y.drop(columns=['symbol']))
    clustered_train_X, clustered_train_y, clustered_test_X, clustered_test_y = cluster.cluster()

    pred_col = test_y.columns[0]

    # Split data
    train_val_split = 0.7
    unique_train_dates = clustered_train_X.index.unique()
    split_date = unique_train_dates[int(len(unique_train_dates) * train_val_split)]

    # Get original features for consequence layer
    original_train_features = train_X.drop(columns=stationary_cols + ['symbol']).loc[train_X.index < split_date]
    original_val_features = train_X.drop(columns=stationary_cols + ['symbol']).loc[train_X.index >= split_date]
    original_train_features = original_train_features.dropna()
    original_val_features = original_val_features.dropna()
    # Split clustered data
    train_clustered_X = clustered_train_X.loc[clustered_train_X.index < split_date]
    val_clustered_X = clustered_train_X.loc[clustered_train_X.index >= split_date]

    train_clustered_y = clustered_train_y.loc[clustered_train_y.index < split_date]
    val_clustered_y = clustered_train_y.loc[clustered_train_y.index >= split_date]

    # Get original targets for validation
    train_original_y = train_y[pred_col].loc[train_y.index < split_date]
    val_original_y = train_y[pred_col].loc[train_y.index >= split_date]

    print(f"Data shapes:")
    print(f"  Original train features: {original_train_features.shape}")
    print(f"  Clustered train features: {train_clustered_X.shape}")
    print(f"  Clustered train targets: {train_clustered_y.shape}")

    # Prepare data for ClusteredANFIS
    (train_memberships, train_original, train_targets,
     val_memberships, val_original, val_targets) = prepare_clustered_data_for_training(
        train_clustered_X, train_clustered_y, val_clustered_X, val_clustered_y,
        original_train_features, original_val_features
    )

    print(f"Prepared data shapes:")
    print(f"  Train memberships: {train_memberships.shape}")
    print(f"  Train original: {train_original.shape}")
    print(f"  Train targets: {train_targets.shape}")

    # Create ClusteredANFIS model
    clustered_anfis = create_clustered_anfis_from_data(
        train_clustered_X, original_train_features,
        expected_rules=train_memberships.shape[1]
    )

    print(f"ClusteredANFIS created with:")
    print(f"  Membership inputs: {clustered_anfis.n_membership_inputs}")
    print(f"  Original inputs: {clustered_anfis.n_original_inputs}")
    print(f"  Expected rules: {clustered_anfis.num_rules}")

    # Create datasets - ClusteredANFIS expects 3 tensors: memberships, features, targets
    train_dataset = TensorDataset(train_memberships, train_original, train_targets)
    valid_dataset = TensorDataset(val_memberships, val_original, val_targets)

    # Setup training
    optimizer = torch.optim.Adam(params=clustered_anfis.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss(reduction='mean')

    # Train model
    history = clustered_anfis.fit(
        train_dataset, valid_dataset,
        optimizer=optimizer,
        loss_function=loss_function,
        batch_size=4* 512,
        epochs=1000
    )

    # Make predictions
    y_pred_fuzzy = clustered_anfis.predict([train_memberships, train_original])


    # Plot results
    plots.plt_learningcurves(history, save_path='img/clustered_anfis_learning_curves.jpg')
    plots.plt_prediction(train_original_y.values.reshape(-1, 1),
                         y_pred_fuzzy.reshape(-1, 1),
                         save_path='img/clustered_anfis_prediction.jpg')

    print("Clustered ANFIS evaluation completed!")


def eval_clustered_ANFIS_CNN():
    print("Evaluating Clustered ANFIS")

    # Get original data
    train_X, train_y, test_X, test_y = get_tickers(top10, "2005-01-01", "2015-12-31", "2016-01-01",
                                                   "2018-01-01", y_horizon)

    # Perform clustering
    cluster = HDBSCANHandler(train_X.drop(columns=stationary_cols + ['symbol']),
                             train_y.drop(columns=['symbol']),
                             test_X.drop(columns=stationary_cols + ['symbol']),
                             test_y.drop(columns=['symbol']))
    clustered_train_X, clustered_train_y, clustered_test_X, clustered_test_y = cluster.cluster()

    pred_col = test_y.columns[0]

    # Split data
    train_val_split = 0.7
    unique_train_dates = clustered_train_X.index.unique()
    split_date = unique_train_dates[int(len(unique_train_dates) * train_val_split)]

    # Get original features for consequence layer
    original_train_features = train_X.drop(columns=stationary_cols + ['symbol']).loc[train_X.index < split_date]
    original_val_features = train_X.drop(columns=stationary_cols + ['symbol']).loc[train_X.index >= split_date]
    original_train_features = original_train_features.dropna()
    original_val_features = original_val_features.dropna()
    # Split clustered data
    train_clustered_X = clustered_train_X.loc[clustered_train_X.index < split_date]
    val_clustered_X = clustered_train_X.loc[clustered_train_X.index >= split_date]

    train_clustered_y = clustered_train_y.loc[clustered_train_y.index < split_date]
    val_clustered_y = clustered_train_y.loc[clustered_train_y.index >= split_date]

    # Get original targets for validation
    train_original_y = train_y[pred_col].loc[train_y.index < split_date]
    val_original_y = train_y[pred_col].loc[train_y.index >= split_date]

    print(f"Data shapes:")
    print(f"  Original train features: {original_train_features.shape}")
    print(f"  Clustered train features: {train_clustered_X.shape}")
    print(f"  Clustered train targets: {train_clustered_y.shape}")

    # Prepare data for ClusteredANFIS
    (train_memberships, train_original, train_targets,
     val_memberships, val_original, val_targets) = prepare_clustered_data_for_training(
        train_clustered_X, train_clustered_y, val_clustered_X, val_clustered_y,
        original_train_features, original_val_features
    )

    print(f"Prepared data shapes:")
    print(f"  Train memberships: {train_memberships.shape}")
    print(f"  Train original: {train_original.shape}")
    print(f"  Train targets: {train_targets.shape}")

    # Create ClusteredANFIS model
    clustered_anfis = create_clustered_anfis_from_data(
        train_clustered_X, original_train_features,
        expected_rules=train_memberships.shape[1]
    )

    print(f"ClusteredANFIS created with:")
    print(f"  Membership inputs: {clustered_anfis.n_membership_inputs}")
    print(f"  Original inputs: {clustered_anfis.n_original_inputs}")
    print(f"  Expected rules: {clustered_anfis.num_rules}")

    # Create datasets - ClusteredANFIS expects 3 tensors: memberships, features, targets
    train_dataset = TensorDataset(train_memberships, train_original, train_targets)
    valid_dataset = TensorDataset(val_memberships, val_original, val_targets)

    # Setup training
    optimizer = torch.optim.Adam(params=clustered_anfis.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss(reduction='mean')

    # Train model
    history = clustered_anfis.fit(
        train_dataset, valid_dataset,
        optimizer=optimizer,
        loss_function=loss_function,
        batch_size=512,
        epochs=100
    )

    # Make predictions
    y_pred_fuzzy = clustered_anfis.predict([train_memberships, train_original])

    # Plot results
    plots.plt_learningcurves(history, save_path='img/clustered_anfis_learning_curves.jpg')
    plots.plt_prediction(train_original_y.values.reshape(-1, 1),
                         y_pred_fuzzy.reshape(-1, 1),
                         save_path='img/clustered_anfis_prediction.jpg')

    print("Clustered ANFIS evaluation completed!")


def eval_clustered_CNN_ANFIS():
    """Clustered CNN-ANFIS evaluation using CNN-based fuzzy processing"""
    print("Evaluating Clustered CNN-ANFIS")

    # Get original data
    train_X, train_y, test_X, test_y = get_tickers(top10, "2005-01-01", "2015-12-31", "2016-01-01",
                                                   "2018-01-01", y_horizon)

    # Perform clustering
    cluster = HDBSCANHandler(train_X.drop(columns=stationary_cols + ['symbol']),
                             train_y.drop(columns=['symbol']),
                             test_X.drop(columns=stationary_cols + ['symbol']),
                             test_y.drop(columns=['symbol']))
    clustered_train_X, clustered_train_y, clustered_test_X, clustered_test_y = cluster.cluster()

    pred_col = test_y.columns[0]

    # Split data
    train_val_split = 0.7
    unique_train_dates = clustered_train_X.index.unique()
    split_date = unique_train_dates[int(len(unique_train_dates) * train_val_split)]

    # Get original features for consequence layer
    original_train_features = train_X.drop(columns=stationary_cols + ['symbol']).loc[train_X.index < split_date]
    original_val_features = train_X.drop(columns=stationary_cols + ['symbol']).loc[train_X.index >= split_date]
    original_train_features = original_train_features.dropna()
    original_val_features = original_val_features.dropna()

    # Split clustered data
    train_clustered_X = clustered_train_X.loc[clustered_train_X.index < split_date]
    val_clustered_X = clustered_train_X.loc[clustered_train_X.index >= split_date]

    train_clustered_y = clustered_train_y.loc[clustered_train_y.index < split_date]
    val_clustered_y = clustered_train_y.loc[clustered_train_y.index >= split_date]

    # Get original targets for validation
    train_original_y = train_y[pred_col].loc[train_y.index < split_date]
    val_original_y = train_y[pred_col].loc[train_y.index >= split_date]

    print(f"CNN-ANFIS Data shapes:")
    print(f"  Original train features: {original_train_features.shape}")
    print(f"  Clustered train features: {train_clustered_X.shape}")
    print(f"  Clustered train targets: {train_clustered_y.shape}")

    # Prepare data for ClusteredCNNANFIS
    (train_memberships, train_original, train_targets,
     val_memberships, val_original, val_targets) = prepare_clustered_data_for_training(
        train_clustered_X, train_clustered_y, val_clustered_X, val_clustered_y,
        original_train_features, original_val_features
    )

    print(f"Prepared CNN data shapes:")
    print(f"  Train memberships: {train_memberships.shape}")
    print(f"  Train original: {train_original.shape}")
    print(f"  Train targets: {train_targets.shape}")

    # Create ClusteredCNNANFIS model
    cnn_latent_dim = 27
    clustered_cnn_anfis = create_clustered_cnn_anfis_from_data(
        train_clustered_X, original_train_features,
        cnn_latent_dim=cnn_latent_dim,
        expected_rules=cnn_latent_dim
    )

    print(f"ClusteredCNNANFIS created with:")
    print(f"  Membership inputs: {clustered_cnn_anfis.n_membership_inputs}")
    print(f"  Original inputs: {clustered_cnn_anfis.n_original_inputs}")
    print(f"  CNN latent dimension: {clustered_cnn_anfis.cnn_latent_dim}")
    print(f"  Expected rules: {clustered_cnn_anfis.num_rules}")

    # Create datasets - ClusteredCNNANFIS expects 3 tensors: memberships, features, targets
    train_dataset = TensorDataset(train_memberships, train_original, train_targets)
    valid_dataset = TensorDataset(val_memberships, val_original, val_targets)

    # Setup training with lower learning rate for CNN
    optimizer = torch.optim.Adam(params=clustered_cnn_anfis.parameters(), lr=0.0001)
    loss_function = torch.nn.MSELoss(reduction='mean')

    # Train model
    history = clustered_cnn_anfis.fit(
        train_dataset, valid_dataset,
        optimizer=optimizer,
        loss_function=loss_function,
        batch_size=256,  # Smaller batch size for CNN
        epochs=150  # More epochs for CNN convergence
    )

    # Make predictions
    y_pred_fuzzy = clustered_cnn_anfis.predict([train_memberships, train_original])


    # Plot results
    plots.plt_learningcurves(history, save_path='img/clustered_cnn_anfis_learning_curves.jpg')
    plots.plt_prediction(train_original_y.values.reshape(-1, 1),
                         y_pred_fuzzy.reshape(-1, 1),
                         save_path='img/clustered_cnn_anfis_prediction.jpg')

    # Plot CNN latent space
    clustered_cnn_anfis.plot_latent_space(train_memberships, save_path='img/cnn_latent_space.jpg')

    print("Clustered CNN-ANFIS evaluation completed!")

def main():
    """Run both evaluations for comparison"""
    print("=" * 50)
    print("ANFIS Evaluation Comparison")
    print("=" * 50)
    #
    # # Run traditional ANFIS
    # eval_ANFIS_traditional()

    # print("\n" + "=" * 50)
    #
    # # Run clustered ANFIS
    # eval_clustered_ANFIS()
    #
    # print("\n" + "=" * 50)
    # print("Evaluation completed! Check img/ folder for results.")

    eval_clustered_CNN_ANFIS()
    print("\n" + "=" * 50)
    print("Evaluation completed! Check img/ folder for results.")


def convert_pickle_to_csv():
    train_X, train_y, test_X, test_y = get_tickers(top10, "2005-01-01", "2015-12-31", "2016-01-01",
                                                   "2018-01-01", y_horizon)


if __name__ == "__main__":
    main()