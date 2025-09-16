import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch import optim, nn

from models.ANFIS.ANFIS import ANFIS
from models.clustering.HDBScan import HDBSCANHandler
from models.ANFIS.ClusteredANFIS import create_clustered_anfis_from_data, \
    prepare_clustered_data_for_training


# --- Functions from the previous response ---

def load_stock_data(filepath: str) -> pd.DataFrame:
    """Loads stock data from a CSV file."""
    df = pd.read_excel(filepath, parse_dates=[0])
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df


def create_lagged_features_and_target(df: pd.DataFrame, feature_cols: list, target_col: str, n_lags: int) -> (
        pd.DataFrame, pd.DataFrame):
    """Creates lagged features and a corresponding target."""
    X_list, y_list, y_dates = [], [], []

    for i in range(len(df) - n_lags):
        feature_window = df[feature_cols].iloc[i: i + n_lags].values.flatten()
        X_list.append(feature_window)

        target_value = df[target_col].iloc[i + n_lags]
        target_date = df.index[i + n_lags]
        y_list.append(target_value)
        y_dates.append(target_date)

    lagged_feature_names = [f'{col}_lag_{lag + 1}' for lag in range(n_lags) for col in feature_cols]

    X = pd.DataFrame(X_list, columns=lagged_feature_names)
    y = pd.DataFrame(y_list, columns=[target_col], index=y_dates)
    return X, y


def perform_data_clustering(train_X, train_y, test_X, test_y) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                  pd.DataFrame, HDBSCANHandler):
    """Applies HDBSCAN clustering to the feature and target data."""
    hdbscan_handler = HDBSCANHandler(
        train_X, train_y, test_X, test_y,
        cluster_selection_method='eom',
        mergeCluster=True
    )
    clustered_train_X, clustered_train_y, clustered_test_X, clustered_test_y = hdbscan_handler.cluster()
    return clustered_train_X, clustered_train_y, clustered_test_X, clustered_test_y, hdbscan_handler


def define_membership_functions(n_inputs: int, n_membership_funcs: int = 3) -> list:
    """
    Defines a default set of Gaussian membership functions for ANFIS.
    """
    membfuncs = []
    # Define evenly spaced centers (mu) and a constant width (sigma)
    mus = np.linspace(0, 1, n_membership_funcs).tolist()
    sigmas = [1.0 / n_membership_funcs] * n_membership_funcs

    for i in range(n_inputs):
        membfuncs.append({
            'function': 'gaussian',
            'n_memb': n_membership_funcs,
            'params': {
                'mu': {'value': mus, 'trainable': True},
                'sigma': {'value': sigmas, 'trainable': True}
            },
        })
    print(f"Defined {len(membfuncs)} input features with {n_membership_funcs} Gaussian membership functions each.")
    return membfuncs


# --- New Main Function for Training and Plotting ANFIS ---

def train_predict_and_plot_anfis(train_path: str, test_path: str, n_lags: int = 5, epochs: int = 25):
    """
    Full pipeline to train the standard ANFIS model and plot its predictions.
    """
    # 1. Load and Prepare Data
    train_df = load_stock_data(train_path)
    test_df = load_stock_data(test_path)

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_col = 'Close'

    train_X, train_y = create_lagged_features_and_target(train_df, feature_cols, target_col, n_lags)
    test_X, test_y = create_lagged_features_and_target(test_df, feature_cols, target_col, n_lags)

    # 2. Scale the data
    print("Scaling the data...")
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # Fit scalers on training data and transform both sets
    train_X_scaled = x_scaler.fit_transform(train_X)
    test_X_scaled = x_scaler.transform(test_X)

    train_y_scaled = y_scaler.fit_transform(train_y)
    test_y_scaled = y_scaler.transform(test_y)

    # 3. Define ANFIS Membership Functions
    n_inputs = train_X.shape[1]
    membfuncs = define_membership_functions(n_inputs, n_membership_funcs=3)  # Using 2 MFs to keep rule base small

    # 4. Build and Train the ANFIS Model
    print("\nBuilding and training the standard ANFIS model...")
    model = ANFIS(n_input=n_inputs, membfuncs=membfuncs)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_function = nn.MSELoss()

    # Prepare TensorDatasets (for standard ANFIS, it's just X and y)
    train_dataset = TensorDataset(torch.from_numpy(train_X_scaled).float(), torch.from_numpy(train_y_scaled).float())
    validation_dataset = TensorDataset(torch.from_numpy(test_X_scaled).float(), torch.from_numpy(test_y_scaled).float())

    model.fit(
        train_data=train_dataset,
        valid_data=validation_dataset,
        optimizer=optimizer,
        loss_function=loss_function,
        epochs=epochs,
        batch_size=64
    )

    # 5. Make Predictions
    print("Making predictions on the test data...")
    # For prediction, ANFIS just needs the input features
    test_data_for_prediction = TensorDataset(torch.from_numpy(test_X_scaled).float())
    predictions_scaled = model.predict(test_data_for_prediction)

    # 6. Inverse-transform the Predictions
    print("Inverse-scaling the results...")
    final_predictions = y_scaler.inverse_transform(predictions_scaled.numpy())

    # 7. Plot the Results
    print("Plotting the results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    # Use the original (unscaled) test_y for plotting
    ax.plot(test_y.index, test_y.values, label='Actual Data (Supposed)', color='blue', linewidth=2)
    ax.plot(test_y.index, final_predictions, label='ANFIS Prediction', color='green', linestyle='--', linewidth=2)

    ax.set_title('Stock Close Price: Actual vs. Predicted (Standard ANFIS)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# --- New Function for Training and Plotting ---

def train_predict_and_plot(train_path: str, test_path: str, n_lags: int = 5, epochs: int = 1000):
    """
    Full pipeline to train the ClusteredANFIS model and plot its predictions against actual data.
    """
    # 1. Load and Prepare Data
    train_df = load_stock_data(train_path)
    test_df = load_stock_data(test_path)

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_col = 'Close'

    original_train_X, train_y = create_lagged_features_and_target(train_df, feature_cols, target_col, n_lags)
    original_test_X, test_y = create_lagged_features_and_target(test_df, feature_cols, target_col, n_lags)

    # 2. Perform Clustering
    clustered_train_X, clustered_train_y, clustered_test_X, clustered_test_y, hdbscan_handler = perform_data_clustering(
        original_train_X, train_y, original_test_X, test_y
    )

    # 3. Prepare Tensors for ANFIS
    (train_memberships, train_original, train_targets,
     test_memberships, test_original, test_targets) = prepare_clustered_data_for_training(
        clustered_train_X, clustered_train_y,
        clustered_test_X, clustered_test_y,
        original_train_X, original_test_X
    )

    train_dataset = TensorDataset(train_memberships, train_original, train_targets)
    # For prediction, we only need the input tensors, not the target
    test_data_for_prediction = TensorDataset(test_memberships, test_original)

    # 4. Build and Train the Model
    print("\nBuilding and training the ClusteredANFIS model...")
    # Use the helper function to create a model with the correct dimensions
    model = create_clustered_anfis_from_data(clustered_train_X, original_train_X)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.MSELoss()

    # We'll use the test set as validation for this example
    validation_dataset = TensorDataset(test_memberships, test_original, test_targets)

    model.fit(
        train_data=train_dataset,
        valid_data=validation_dataset,
        optimizer=optimizer,
        loss_function=loss_function,
        epochs=epochs,
        batch_size=2048
    )

    # 5. Make Predictions
    print("Making predictions on the test data...")
    fuzzy_predictions = model.predict(test_data_for_prediction)

    # 6. De-fuzzify the Predictions
    # This crucial step converts the fuzzy output back into a numerical value
    print("De-fuzzifying the results...")
    final_predictions = hdbscan_handler.de_fuzzify(fuzzy_predictions.numpy(), target_col=target_col)

    # 7. Plot the Results
    print("Plotting the results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(test_y.index, test_y.values, label='Actual Data (Supposed)', color='blue', linewidth=2)
    ax.plot(test_y.index, final_predictions, label='ClusteredANFIS Prediction', color='red', linestyle='--',
            linewidth=2)

    ax.set_title('Stock Close Price: Actual vs. Predicted', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    plt.show()



# --- Run the entire process ---
if __name__ == '__main__':
    train_file = 'train_data.xlsx'
    test_file = 'stock_data.xlsx'

    # Run the full pipeline with 5 lags and 25 training epochs
    #train_predict_and_plot(train_file, test_file, n_lags=2, epochs=1000)
    train_predict_and_plot_anfis(train_file, test_file, n_lags=2, epochs=1000)