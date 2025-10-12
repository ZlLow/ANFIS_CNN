import hdbscan
from sklearn.preprocessing import StandardScaler


def get_num_rules_with_hdbscan(df_train, min_cluster_size=15):
    """
    Uses HDBSCAN to find the optimal number of clusters (rules) from the training data.
    """
    print(f"Running HDBSCAN to determine num_rules with min_cluster_size={min_cluster_size}...")

    # Scaling is crucial for distance-based algorithms like HDBSCAN
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_train)

    # Apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(scaled_data)

    # The number of clusters is the max label + 1 (labels are 0-indexed, -1 is noise)
    num_clusters = clusterer.labels_.max() + 1

    # Handle the case where no clusters are found
    if num_clusters == 0:
        print("Warning: HDBSCAN found 0 clusters. Defaulting to a small number of rules (e.g., 5).")
        return 5

    print(f"HDBSCAN identified {num_clusters} clusters (rules).")
    return num_clusters
