import torch

from typing import Optional
from torch import nn
from models.ANFIS.AbstractANFIS import AbstractANFIS, _ConsequenceLayer


class ClusteredANFIS(AbstractANFIS):
    """
    Hybrid ANFIS that accepts pre-computed membership values from clustering
    while maintaining ANFIS architecture for rule generation and consequence layers.
    """

    def __init__(self, n_membership_inputs: int, n_original_inputs: int,
                 expected_rules: Optional[int] = None, to_device: Optional[str] = None,
                 drop_out_rate: Optional[int] = 0.2):
        """
        Initialize ClusteredANFIS

        Args:
            n_membership_inputs: Number of pre-computed membership values from clustering
            n_original_inputs: Number of original input features for consequence layer
            expected_rules: Expected number of rules (if None, will be inferred)
            to_device: Device to run model on
            drop_out_rate: Dropout rate
        """
        # Initialize parent with minimal membfuncs (won't be used)
        super().__init__(n_input=n_original_inputs, membfuncs=[], to_device=to_device,
                         drop_out_rate=drop_out_rate)

        self.n_membership_inputs = n_membership_inputs
        self.n_original_inputs = n_original_inputs
        self._rules = expected_rules if expected_rules else n_membership_inputs

        # Build layers without fuzzy layer
        self.layers = nn.ModuleDict({
            # Skip Layer 1 - fuzzification (use pre-computed memberships)

            # Layer 2 - Rule layer (modified to handle pre-computed memberships)
            'rules': _ClusteredRuleLayer(n_membership_inputs),

            # Layer 4 - Consequence layer
            'consequence': _ConsequenceLayer(self.n_original_inputs, self._rules)
        })

    def _reset_model_parameter(self):
        optlclass = self.optimizer.__class__
        self.optimizer = optlclass(self.parameters(), lr=self.optimizer.__dict__['param_groups'][0]['lr'])

        with torch.no_grad():
            for layer in self.layers.values():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, membership_batch: torch.Tensor, X_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using pre-computed memberships

        Args:
            membership_batch: Pre-computed membership values from clustering [batch_size, n_membership_inputs]
            X_batch: Original input features [batch_size, n_original_inputs]
        """
        # Skip Layer 1 - use pre-computed memberships directly

        # Layer 2 - Rule layer (treat memberships as firing strengths)
        firing_strength = self.layers['rules'](membership_batch)

        # Layer 3 - Normalization
        normalized = torch.nn.functional.normalize(firing_strength, p=1, dim=1)

        # Layer 4 - Consequence layer
        consequences = self.layers['consequence'](X_batch, normalized)

        # Layer 5 - Summation
        output = consequences.sum(axis=1).reshape(-1, 1)

        return output

    @property
    def premise(self):
        # No premise parameters since we use pre-computed memberships
        return []

    @premise.setter
    def premise(self, new_memberships: list):
        # No-op since we don't have premise parameters
        pass

    @property
    def consequence(self):
        return self.layers['consequence'].coeffs

    @consequence.setter
    def consequence(self, new_consequence: dict):
        self.layers['consequence'].coeffs = new_consequence

    def plotmfs(self, *args, **kwargs):
        print("Membership function plotting not available for ClusteredANFIS.")
        print("Membership functions are computed by the clustering algorithm.")


class _ClusteredRuleLayer(nn.Module):
    """
    Modified rule layer that handles pre-computed membership values
    """

    def __init__(self, n_membership_inputs: int):
        super(_ClusteredRuleLayer, self).__init__()
        self.n_membership_inputs = n_membership_inputs

    def forward(self, membership_batch: torch.Tensor) -> torch.Tensor:
        """
        Treat pre-computed memberships as firing strengths

        Args:
            membership_batch: [batch_size, n_membership_inputs]

        Returns:
            firing_strength: [batch_size, n_membership_inputs] (pass-through)
        """
        # For clustered data, memberships are already computed
        # We can either pass them through directly or apply some transformation
        return membership_batch


def create_clustered_anfis_from_data(clustered_X, original_X, expected_rules=None):
    """
    Helper function to create ClusteredANFIS with correct dimensions from clustered data

    Args:
        clustered_X: DataFrame with clustered features (includes PDF columns)
        original_X: DataFrame with original features
        expected_rules: Optional expected number of rules

    Returns:
        ClusteredANFIS instance with correct dimensions
    """
    # Count PDF columns (membership values)
    pdf_columns = [col for col in clustered_X.columns if col.startswith('PDF_')]
    n_membership_inputs = len(pdf_columns)
    n_original_inputs = original_X.shape[1]

    if expected_rules is None:
        # Estimate rules from membership functions
        # This is a heuristic - you might want to adjust based on your clustering setup
        expected_rules = n_membership_inputs

    return ClusteredANFIS(
        n_membership_inputs=n_membership_inputs,
        n_original_inputs=n_original_inputs,
        expected_rules=expected_rules
    )


def prepare_clustered_data_for_training(clustered_train_X, clustered_train_y,
                                        clustered_test_X, clustered_test_y,
                                        original_train_X, original_test_X):
    """
    Prepare clustered data for ClusteredANFIS training

    Args:
        clustered_train_X: Clustered training features
        clustered_train_y: Clustered training targets
        clustered_test_X: Clustered test features
        clustered_test_y: Clustered test targets
        original_train_X: Original training features
        original_test_X: Original test features

    Returns:
        Tuple of tensors ready for ClusteredANFIS
    """
    # Extract PDF columns (membership values)
    train_pdf_cols = [col for col in clustered_train_X.columns if col.startswith('PDF_')]
    test_pdf_cols = [col for col in clustered_test_X.columns if col.startswith('PDF_')]

    # Extract membership values
    train_memberships = torch.from_numpy(clustered_train_X[train_pdf_cols].values).float()
    test_memberships = torch.from_numpy(clustered_test_X[test_pdf_cols].values).float()

    # Extract original features
    train_original = torch.from_numpy(original_train_X.values).float()
    test_original = torch.from_numpy(original_test_X.values).float()

    # Extract target PDF columns
    target_pdf_cols = [col for col in clustered_train_y.columns if col.startswith('PDF_')]
    train_targets = torch.from_numpy(clustered_train_y[target_pdf_cols].values).float()
    test_targets = torch.from_numpy(clustered_test_y[target_pdf_cols].values).float()

    return (train_memberships, train_original, train_targets,
            test_memberships, test_original, test_targets)
