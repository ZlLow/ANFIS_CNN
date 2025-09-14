import numpy as np
from abc import ABC, abstractmethod
import plotly.graph_objects as go
import pandas as pd


class AbstractClusterHandler(ABC):
    def __init__(self, train_X, train_y, test_X, test_y, mergeCluster=True, plt=False, pltNo=0):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.mergeCluster = mergeCluster
        self.plt = plt
        self.pltNo = pltNo

        # Initialize dictionaries to store cluster statistics
        self.feature_cluster_stats = {}
        self.y_cluster_stats = {}

    @abstractmethod
    def cluster(self):
        """Main clustering method that processes all columns."""
        pass

    @abstractmethod
    def _process_column(self, column_name, is_target=False):
        """Process a single column for clustering."""
        pass

    @abstractmethod
    def _cluster_column(self, column_name, is_target=False):
        """Cluster a single column of data."""
        pass

    def _merge_clusters(self, df, cluster_col):
        """Merge small clusters."""
        cluster_no = sorted(df[cluster_col].unique())
        i, j = 0, 1
        while i < len(cluster_no) and j < len(cluster_no):
            lCls, rCls = cluster_no[i], cluster_no[j]
            if len(df[df[cluster_col] == lCls]) < 5 or len(df[df[cluster_col] == rCls]) < 5:
                df.loc[df[cluster_col] == rCls, cluster_col] = lCls
                new_mean = df[df[cluster_col] == lCls]['Data'].mean()
                new_std = df[df[cluster_col] == lCls]['Data'].std()
                if new_std == 0: new_std = 1e-6
                df.loc[df[cluster_col] == lCls, 'Cluster_Mean'] = new_mean
                df.loc[df[cluster_col] == lCls, 'Cluster_Std'] = new_std
                cluster_no.pop(j)
            else:
                i += 1
                j += 1

    def _calculate_memberships(self, temp, testTemp, cluster_col, column_name):
        """Calculate membership functions for each cluster."""
        clsDic = {}
        pdf_columns = []

        for clsNo, cluster_data in enumerate(temp.groupby(cluster_col)):
            _, data = cluster_data
            sampleMean = data['Data'].mean()
            sampleDevi = data['Data'].std(ddof=1)

            if sampleDevi == 0: sampleDevi = 1e-6

            # Handle prefix assignment
            prefix = column_name
            pdf_col = f'PDF_{prefix}_{clsNo}'
            pdf_columns.append(pdf_col)

            temp[pdf_col] = temp["Data"].apply(lambda x: self._membership(x, sampleMean, sampleDevi))
            testTemp[pdf_col] = testTemp["Data"].apply(lambda x: self._membership(x, sampleMean, sampleDevi))

            clsDic[clsNo] = {'mean': sampleMean, 'std': sampleDevi}

        # Combine all PDF columns at once
        temp = pd.concat([temp] + [temp[col] for col in pdf_columns], axis=1)
        testTemp = pd.concat([testTemp] + [testTemp[col] for col in pdf_columns], axis=1)

        return clsDic

    def _membership(self, x, sampleMean, sampleDevi):
        """Calculate Gaussian membership."""
        return np.exp(-((x - sampleMean) ** 2 / (2 * (max(sampleDevi, 1e-6) ** 2))))

    @staticmethod
    def cluster_graph(cluster_stats: dict):
        """Generate cluster distribution graphs."""

        means = [stats['mean'] for stats in cluster_stats.values()]
        stds = [stats['std'] for stats in cluster_stats.values()]
        x_min = min(means) - 3 * max(stds)
        x_max = max(means) + 3 * max(stds)
        x = np.linspace(x_min, x_max, 1000)

        def gaussian(x, mean, std):
            return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

        # Create traces for each cluster
        fig = go.Figure()

        for cluster, stats in cluster_stats.items():
            y = gaussian(x, stats['mean'], stats['std'])
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'Cluster {cluster}'))

        # Update layout for better visualization
        fig.update_layout(
            title="Gaussian Distributions for Clusters",
            xaxis_title="Value",
            yaxis_title="Density",
            showlegend=True
        )

        # Show the figure
        fig.show()

        return fig

    def de_fuzzify(self, pred, target_col: str) -> np.array:
        """
        defuzzification using weighted average technique
        transform fuzzy memberships -> log returns
        """
        centers = [dic['mean'] for dic in self.y_cluster_stats[target_col].values()]

        centers = np.tile(centers, (pred.shape[0], 1))
        pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)

        denominator = pred.sum(axis=1, keepdims=True)
        numerator = (pred * centers).sum(axis=1, keepdims=True)
        result = numerator / denominator
        result = np.squeeze(result)

        return result

    def _split_largest_cluster(self, splits):
        """Split the largest cluster."""
        differences = np.abs(np.diff(splits))
        max_diff_index = np.argmax(differences)
        middle_element = (splits[max_diff_index] + splits[max_diff_index + 1]) / 2
        return np.insert(splits, max_diff_index + 1, middle_element)

    def _calculate_cluster_stats(self, df, cluster_col):
        """Calculate mean and std for each cluster."""
        for cluster_id, cluster_data in df.groupby(cluster_col):
            df.loc[df[cluster_col] == cluster_id, 'Cluster_Mean'] = cluster_data['Data'].mean()

            std = cluster_data['Data'].std(ddof=1)
            if std == 0: std = 1e-6
            df.loc[df[cluster_col] == cluster_id, 'Cluster_Std'] = std