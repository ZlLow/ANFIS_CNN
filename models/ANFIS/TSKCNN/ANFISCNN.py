import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from models.ANFIS.AbstractANFIS import AbstractANFIS, _ConsequenceLayer
from models.CNN.Encoder import Encoder
from models.CNN.fuzzy.DefuzzyLayer import NormalizationLayer
from models.CNN.fuzzy.FuzzyLayer import FuzzyLayer


class ClusteredANFISCNN(AbstractANFIS):
    """
    Hybrid ANFIS-CNN that integrates fuzzy CNN layers for fire strength computation and normalization
    Uses pre-computed membership values from clustering as input to the CNN layers
    """

    def __init__(self, n_membership_inputs: int, n_original_inputs: int,
                 cnn_config: dict, expected_rules: Optional[int] = None,
                 to_device: Optional[str] = None, drop_out_rate: Optional[float] = 0.2):
        """
        Initialize ClusteredANFIS with CNN integration

        Args:
            n_membership_inputs: Number of pre-computed membership values from clustering
            n_original_inputs: Number of original input features for consequence layer
            cnn_config: Configuration for CNN layers with keys:
                       - 'fuzzy_dims': Fuzzy dimensions for CNN
                       - 'latent_dims': Latent space dimensions
                       - 'output_dims': Output dimensions after CNN processing
            expected_rules: Expected number of rules (if None, will be inferred)
            to_device: Device to run model on
            drop_out_rate: Dropout rate
        """
        # Initialize parent with minimal membfuncs (won't be used)
        super().__init__(n_input=n_original_inputs, membfuncs=[],
                         to_device=to_device, drop_out_rate=drop_out_rate)

        self.n_membership_inputs = n_membership_inputs
        self.n_original_inputs = n_original_inputs
        self.cnn_config = cnn_config
        self._rules = expected_rules if expected_rules else cnn_config.get('output_dims', n_membership_inputs)

        # Build layers with CNN integration
        self.layers = nn.ModuleDict({
            # Skip Layer 1 - fuzzification (use pre-computed memberships)

            # Layer 2 & 3 - CNN-based fuzzy processing
            'cnn_fuzzy': _FuzzyCNNLayer(
                membership_dims=n_membership_inputs,
                fuzzy_dims=cnn_config['fuzzy_dims'],
                latent_dims=cnn_config['latent_dims'],
                output_dims=cnn_config['output_dims'],
                drop_out_rate=drop_out_rate
            ),

            # Layer 4 - Consequence layer
            'consequence': _ConsequenceLayer(self._rules, self.n_original_inputs)
        })

    def _reset_model_parameter(self):
        optlclass = self.optimizer.__class__
        self.optimizer = optlclass(self.parameters(),
                                   lr=self.optimizer.__dict__['param_groups'][0]['lr'])

        with torch.no_grad():
            for layer in self.layers.values():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, membership_batch: torch.Tensor, X_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using CNN-enhanced fuzzy processing

        Args:
            membership_batch: Pre-computed membership values [batch_size, n_membership_inputs]
            X_batch: Original input features [batch_size, n_original_inputs]
        """
        batch_size = membership_batch.shape[0]

        # Skip Layer 1 - use pre-computed memberships directly

        # Layer 2 & 3 - CNN-based fuzzy processing
        # Reshape memberships for CNN processing if needed
        if len(membership_batch.shape) == 2:
            # Reshape for CNN: [batch, channels, height, width]
            # Assuming square arrangement, adjust as needed
            h = w = int(np.sqrt(self.n_membership_inputs))
            if h * w < self.n_membership_inputs:
                h = w = int(np.sqrt(self.n_membership_inputs)) + 1
                # Pad if necessary
                padding_needed = h * w - self.n_membership_inputs
                membership_batch = F.pad(membership_batch, (0, padding_needed))
            membership_batch = membership_batch.view(batch_size, 1, h, w)

        normalized_firing, mu, logvar, z = self.layers['cnn_fuzzy'](membership_batch)

        # Layer 4 - Consequence layer
        consequences = self.layers['consequence'](X_batch, normalized_firing)

        # Layer 5 - Summation
        output = consequences.sum(axis=1).reshape(-1, 1)

        return output, mu, logvar, z  # Return VAE parameters for potential loss calculation

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

    def get_latent_representation(self, membership_batch: torch.Tensor):
        """Extract latent representation from CNN encoder"""
        batch_size = membership_batch.shape[0]

        if len(membership_batch.shape) == 2:
            h = w = int(np.sqrt(self.n_membership_inputs))
            if h * w < self.n_membership_inputs:
                h = w = int(np.sqrt(self.n_membership_inputs)) + 1
                padding_needed = h * w - self.n_membership_inputs
                membership_batch = F.pad(membership_batch, (0, padding_needed))
            membership_batch = membership_batch.view(batch_size, 1, h, w)

        with torch.no_grad():
            _, mu, logvar, z = self.layers['cnn_fuzzy'](membership_batch)

        return z, mu, logvar

    def plotmfs(self, *args, **kwargs):
        print("Membership function plotting not available for ClusteredANFIS-CNN.")
        print("Membership functions are computed by the clustering algorithm and processed by CNN.")


class _FuzzyCNNLayer(nn.Module):
    """
    CNN-based fuzzy processing layer that combines encoder and fuzzy operations
    """

    def __init__(self, membership_dims: int, fuzzy_dims: int, latent_dims: int,
                 output_dims: int, drop_out_rate: float = 0.2):
        super(_FuzzyCNNLayer, self).__init__()

        self.membership_dims = membership_dims
        self.fuzzy_dims = fuzzy_dims
        self.latent_dims = latent_dims
        self.output_dims = output_dims

        # CNN Encoder for processing membership values
        self.encoder = Encoder(
            in_channels=1,  # Single channel for membership values
            out_channels=latent_dims,
            drop_out_rate=drop_out_rate
        )

        # Fuzzy processing layers
        self.fuzzy_layer = FuzzyLayer.from_dimensions(
            size_in=latent_dims,
            size_out=fuzzy_dims,
            trainable=True
        )

        self.normalization_layer = NormalizationLayer.from_dimensions(
            size_in=fuzzy_dims,
            size_out=output_dims,
            trainable=True,
            with_norm=True
        )

        # Additional processing layers
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, membership_input: torch.Tensor):
        """
        Process membership values through CNN and fuzzy layers

        Args:
            membership_input: [batch_size, channels, height, width] or [batch_size, membership_dims]

        Returns:
            normalized_output: Normalized firing strengths
            mu: Mean from VAE encoder
            logvar: Log variance from VAE encoder
            z: Latent sample from VAE encoder
        """
        batch_size = membership_input.shape[0]

        # CNN encoding with VAE capabilities
        encoded, mu, logvar, z = self.encoder(membership_input)

        # Fuzzy processing
        fuzzy_output = self.fuzzy_layer(z)
        fuzzy_output = self.activation(fuzzy_output)
        fuzzy_output = self.dropout(fuzzy_output)

        # Normalization (this acts as both firing strength computation and normalization)
        normalized_output = self.normalization_layer(fuzzy_output)

        return normalized_output, mu, logvar, z

    def reset_parameters(self):
        """Reset layer parameters"""
        # Reset fuzzy layer parameters if available
        if hasattr(self.fuzzy_layer, 'centroids'):
            nn.init.normal_(self.fuzzy_layer.centroids)
        if hasattr(self.fuzzy_layer, 'scales'):
            nn.init.ones_(self.fuzzy_layer.scales)

        # Reset normalization layer parameters
        if hasattr(self.normalization_layer, 'Z'):
            nn.init.normal_(self.normalization_layer.Z)


def create_clustered_anfis_cnn_from_data(clustered_X, original_X, cnn_config=None):
    """
    Helper function to create ClusteredANFIS-CNN with correct dimensions from clustered data

    Args:
        clustered_X: DataFrame with clustered features (includes PDF columns)
        original_X: DataFrame with original features
        cnn_config: CNN configuration dictionary

    Returns:
        ClusteredANFISCNN instance with correct dimensions
    """
    # Count PDF columns (membership values)
    pdf_columns = [col for col in clustered_X.columns if col.startswith('PDF_')]
    n_membership_inputs = len(pdf_columns)
    n_original_inputs = original_X.shape[1]

    # Default CNN configuration if not provided
    if cnn_config is None:
        cnn_config = {
            'fuzzy_dims': max(8, n_membership_inputs // 2),
            'latent_dims': max(16, n_membership_inputs),
            'output_dims': max(4, n_membership_inputs // 4)
        }

    return ClusteredANFISCNN(
        n_membership_inputs=n_membership_inputs,
        n_original_inputs=n_original_inputs,
        cnn_config=cnn_config,
        expected_rules=cnn_config['output_dims']
    )


class VAELoss(nn.Module):
    """
    Combined loss function for ANFIS prediction and VAE regularization
    """

    def __init__(self, prediction_weight=1.0, kl_weight=0.1):
        super(VAELoss, self).__init__()
        self.prediction_weight = prediction_weight
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true, mu, logvar):
        # Prediction loss
        pred_loss = self.mse_loss(y_pred, y_true)

        # KL divergence loss for VAE regularization
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / y_pred.size(0)  # Average over batch

        # Combined loss
        total_loss = self.prediction_weight * pred_loss + self.kl_weight * kl_loss

        return total_loss, pred_loss, kl_loss

class _CNNLayer(nn.Module):
    def __init__(self, fuzzy_dims, in_channel, out_channel, drop_out_rate=0.2):
        super(_CNNLayer, self).__init__()
        self.encoder = Encoder(fuzzy_dims, in_channel, drop_out_rate=drop_out_rate)
        self.fuzzy = nn.Sequential(
            FuzzyLayer.from_dimensions(fuzzy_dims, in_channel),
            NormalizationLayer.from_dimensions(in_channel, out_channel),
        )

    def forward(self, input_):
        batch_size = input_.shape[0]
        x, mu, logvar, z = self.encoder(input_)
        output = self.fuzzy(x)
        output = output.reshape(batch_size, self._out_channel)
        return output, mu, logvar, z


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.ANFIS.AbstractANFIS import AbstractANFIS, _ConsequenceLayer
from models.CNN.Encoder import Encoder
from models.CNN.fuzzy.DefuzzyLayer import NormalizationLayer
from models.CNN.fuzzy.FuzzyLayer import FuzzyLayer


class ClusteredCNNANFIS(AbstractANFIS):
    """
    Hybrid ANFIS that combines:
    - Pre-computed cluster memberships (input)
    - CNN-based fuzzy processing (layers 2-3)
    - Traditional ANFIS consequence layer (layer 4)
    """

    def __init__(self, n_membership_inputs: int, n_original_inputs: int,
                 cnn_latent_dim: int = 64, expected_rules: Optional[int] = None,
                 to_device: Optional[str] = None, drop_out_rate: Optional[float] = 0.2):
        """
        Initialize ClusteredCNNANFIS

        Args:
            n_membership_inputs: Number of pre-computed membership values from clustering
            n_original_inputs: Number of original input features for consequence layer
            cnn_latent_dim: Dimension of CNN latent space
            expected_rules: Expected number of rules (if None, will be inferred)
            to_device: Device to run model on
            drop_out_rate: Dropout rate
        """
        # Initialize parent with minimal membfuncs (won't be used)
        super().__init__(n_input=n_original_inputs, membfuncs=[], to_device=to_device,
                         drop_out_rate=drop_out_rate)

        self.n_membership_inputs = n_membership_inputs
        self.n_original_inputs = n_original_inputs
        self.cnn_latent_dim = cnn_latent_dim
        self._rules = expected_rules if expected_rules else cnn_latent_dim

        # Calculate CNN input dimensions - reshape memberships to 2D for Conv2d
        # We'll treat memberships as a square-ish image
        self.membership_height = int(n_membership_inputs ** 0.5) + 1
        self.membership_width = int(n_membership_inputs / self.membership_height) + 1
        self.padded_size = self.membership_height * self.membership_width

        print(f"CNN input will be reshaped to: {self.membership_height}x{self.membership_width}")

        # Build layers with CNN-based fuzzy processing
        self.layers = nn.ModuleDict({
            # Skip Layer 1 - fuzzification (use pre-computed memberships)

            # Layer 2-3 - CNN-based fuzzy processing
            'cnn_fuzzy': _CNNFuzzyLayer(
                membership_inputs=n_membership_inputs,
                membership_height=self.membership_height,
                membership_width=self.membership_width,
                latent_dim=cnn_latent_dim,
                output_rules=self._rules,
                drop_out_rate=drop_out_rate
            ),

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
        Forward pass using pre-computed memberships with CNN processing

        Args:
            membership_batch: Pre-computed membership values [batch_size, n_membership_inputs]
            X_batch: Original input features [batch_size, n_original_inputs]
        """
        # Layer 2-3: CNN-based fuzzy processing
        normalized_firing, latent_features = self.layers['cnn_fuzzy'](membership_batch)

        # Layer 4: Consequence layer using original features
        consequences = self.layers['consequence'](X_batch, normalized_firing)

        # Layer 5: Summation
        output = consequences.sum(axis=1).reshape(-1, 1)

        return output

    def forward_with_latent(self, membership_batch: torch.Tensor, X_batch: torch.Tensor):
        """
        Forward pass that also returns latent representations for analysis
        """
        normalized_firing, latent_features = self.layers['cnn_fuzzy'](membership_batch)
        consequences = self.layers['consequence'](X_batch, normalized_firing)
        output = consequences.sum(axis=1).reshape(-1, 1)

        return output, latent_features, normalized_firing

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
        print("Membership function plotting not available for ClusteredCNNANFIS.")
        print("Membership functions are computed by the clustering algorithm.")
        print("Use plot_latent_space() to visualize CNN latent representations.")

    def plot_latent_space(self, membership_data, save_path=None):
        """Plot CNN latent space representations"""
        with torch.no_grad():
            self.eval()
            _, latent_features = self.layers['cnn_fuzzy'](membership_data)

            # Use PCA or t-SNE for visualization if latent_dim > 2
            if latent_features.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                latent_2d = pca.fit_transform(latent_features.cpu().numpy())
            else:
                latent_2d = latent_features.cpu().numpy()

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title('CNN Latent Space Representation')
            plt.grid(True)

            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()


class _CNNFuzzyLayer(nn.Module):
    """
    Combined CNN encoder and fuzzy processing layer
    Processes pre-computed memberships through CNN and applies fuzzy transformations
    """

    def __init__(self, membership_inputs: int, membership_height: int, membership_width: int,
                 latent_dim: int, output_rules: int, drop_out_rate: float = 0.2):
        super().__init__()

        self.membership_inputs = membership_inputs
        self.membership_height = membership_height
        self.membership_width = membership_width
        self.latent_dim = latent_dim
        self.output_rules = output_rules
        self.padded_size = membership_height * membership_width

        # Input preprocessing: reshape memberships to 2D image format
        self.input_reshape = _MembershipReshapeLayer(
            membership_inputs, membership_height, membership_width
        )

        # CNN Encoder for processing membership spatial patterns
        self.encoder = Encoder(
            in_channels=1,  # Single channel for membership "image"
            out_channels=latent_dim,
            input_height=membership_height,
            input_width=membership_width,
            drop_out_rate=drop_out_rate
        )

        # Fuzzy processing in latent space
        self.fuzzy_processor = nn.Sequential(
            FuzzyLayer.from_dimensions(latent_dim, output_rules, trainable=True),
            NormalizationLayer.from_dimensions(output_rules, output_rules, trainable=True)
        )

    def forward(self, membership_batch: torch.Tensor):
        """
        Process membership values through CNN and fuzzy layers

        Args:
            membership_batch: [batch_size, membership_inputs]

        Returns:
            normalized_firing: [batch_size, output_rules]
            latent_features: [batch_size, latent_dim] - for analysis
        """
        batch_size = membership_batch.shape[0]

        # Reshape memberships to image format for CNN
        membership_images = self.input_reshape(membership_batch)

        # CNN encoding
        latent_features = self.encoder(membership_images)

        # Fuzzy processing in latent space
        normalized_firing = self.fuzzy_processor(latent_features)

        return normalized_firing, latent_features


class _MembershipReshapeLayer(nn.Module):
    """
    Reshapes 1D membership vectors into 2D images for CNN processing
    """

    def __init__(self, membership_inputs: int, target_height: int, target_width: int):
        super().__init__()
        self.membership_inputs = membership_inputs
        self.target_height = target_height
        self.target_width = target_width
        self.padded_size = target_height * target_width

        # Padding layer if needed
        padding_needed = self.padded_size - membership_inputs
        if padding_needed > 0:
            self.padding = nn.ConstantPad1d((0, padding_needed), 0)
        else:
            self.padding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, membership_inputs]
        Returns:
            reshaped: [batch_size, 1, target_height, target_width]
        """
        batch_size = x.shape[0]

        # Apply padding if needed
        if self.padding:
            x = self.padding(x)

        # Reshape to 2D image format
        x = x.view(batch_size, 1, self.target_height, self.target_width)

        return x

def create_clustered_cnn_anfis_from_data(clustered_X, original_X, cnn_latent_dim=64, expected_rules=None):
    """
    Helper function to create ClusteredCNNANFIS with correct dimensions from clustered data

    Args:
        clustered_X: DataFrame with clustered features (includes PDF columns)
        original_X: DataFrame with original features
        cnn_latent_dim: Dimension of CNN latent space
        expected_rules: Optional expected number of rules

    Returns:
        ClusteredCNNANFIS instance with correct dimensions
    """
    # Count PDF columns (membership values)
    pdf_columns = [col for col in clustered_X.columns if col.startswith('PDF_')]
    n_membership_inputs = len(pdf_columns)
    n_original_inputs = original_X.shape[1]

    if expected_rules is None:
        expected_rules = cnn_latent_dim

    return ClusteredCNNANFIS(
        n_membership_inputs=n_membership_inputs,
        n_original_inputs=n_original_inputs,
        cnn_latent_dim=cnn_latent_dim,
        expected_rules=expected_rules
    )