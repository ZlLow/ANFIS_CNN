import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import HDBSCAN

from DataHandler.utils.logger import setup_logger
from models.clustering.AbstractClusterHandler import AbstractClusterHandler

logger = setup_logger(__name__)

class HDBSCANHandler(AbstractClusterHandler):
    """HDBSCAN clustering handler."""

    def __init__(self, train_X, train_y, test_X, test_y,
                 cluster_selection_method='eom',
                 mergeCluster=True, plt=False, pltNo=0):
        super().__init__(train_X, train_y, test_X, test_y, mergeCluster, plt, pltNo)
        self.cluster_selection_method = cluster_selection_method
        self.min_cluster_size = int(0.01 * len(train_X))

    def cluster(self):
        """Main clustering method that processes all columns."""
        train_X_res = []
        train_y_res = []
        test_X_res = []
        test_y_res = []

        assert self.train_X.shape[1] == self.test_X.shape[1], "Train and test data must have the same number of columns"
        assert self.train_y.shape[1] == self.test_y.shape[
            1], "Train_y and test_y data must have the same number of columns"
        assert self.train_X.columns.tolist() == self.test_X.columns.tolist(), "Train and test data must have the same columns"
        assert self.train_y.columns.tolist() == self.test_y.columns.tolist(), "Train_y and test_y data must have the same columns"

        # Cluster each feature column
        for column in self.train_X.columns:
            temp, test_temp = self._process_column(column, is_target=False)
            train_X_res.append(temp)
            test_X_res.append(test_temp)

        # Combine clustered results for features
        train_X_res = pd.concat(train_X_res, axis=1)
        test_X_res = pd.concat(test_X_res, axis=1)

        # Cluster each target variable
        for column in self.train_y.columns:
            temp, test_temp = self._process_column(column, is_target=True)
            train_y_res.append(temp)
            test_y_res.append(test_temp)

        # Combine clustered results for target variables
        train_y_res = pd.concat(train_y_res, axis=1)
        test_y_res = pd.concat(test_y_res, axis=1)

        return train_X_res, train_y_res, test_X_res, test_y_res

    def _process_column(self, column_name, is_target=False):
        """Process a single column for clustering."""
        temp, test_temp, cls_dic = self._cluster_column(column_name, is_target)

        # Remove temporary columns
        temp.drop([f"Cluster_{column_name}", "Data", "Cluster_Mean", "Cluster_Std"], axis=1, inplace=True)
        test_temp.drop(["Data"], axis=1, inplace=True)

        if is_target:
            temp.index = self.train_y.index
            test_temp.index = self.test_y.index
        else:
            test_temp.index = self.test_X.index
            temp.index = self.train_X.index

        # Store cluster statistics
        if is_target:
            self.y_cluster_stats[column_name] = cls_dic
        else:
            self.feature_cluster_stats[column_name] = cls_dic
        return temp, test_temp

    def _cluster_column(self, column_name, is_target=False):
        """Cluster a single column of data using HDBSCAN."""
        # Extract data
        spl = (self.train_y if is_target else self.train_X)[column_name].to_numpy().reshape(-1, 1)
        test_spl = (self.test_y if is_target else self.test_X)[column_name].to_numpy().reshape(-1, 1)

        # Perform HDBSCAN clustering
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size,
                            cluster_selection_method=self.cluster_selection_method,n_jobs=-1)
        cls1 = clusterer.fit_predict(spl)

        # Handle noise points (labeled as -1 by HDBSCAN)
        cls1[cls1 == -1] = max(cls1) + 1

        # Create temporary DataFrames
        temp = pd.DataFrame({'Data': spl.reshape(-1), f'Cluster_{column_name}': cls1})
        test_temp = pd.DataFrame({'Data': test_spl.reshape(-1)})

        # Calculate initial cluster statistics
        self._calculate_cluster_stats(temp, f'Cluster_{column_name}')

        # Merge small clusters if enabled
        if self.mergeCluster:
            self._merge_clusters(temp, f'Cluster_{column_name}')

        # Calculate membership functions
        cls_dic = self._calculate_memberships(temp, test_temp, f'Cluster_{column_name}', column_name)

        logger.info(f"{column_name} had {len(cls_dic)} clusters")
        return temp, test_temp, cls_dic

    def plot_condensed_tree(self, data, column_name):
        """Plot the condensed tree for HDBSCAN clustering."""
        spl = data[column_name].to_numpy().reshape(-1, 1)
        # scaler = StandardScaler()
        # spl_scaled = scaler.fit_transform(spl)

        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size,
                            cluster_selection_method=self.cluster_selection_method, n_jobs=-1)
        clusterer.fit(spl)

        clusterer.condensed_tree_.plot()
        plt.title(f'Condensed Tree for {column_name}')
        plt.show()