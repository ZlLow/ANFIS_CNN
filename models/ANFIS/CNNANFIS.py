import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class GeneralizedBellMembershipFunc(nn.Module):
    def __init__(self, num_mfs, input_dim):
        super(GeneralizedBellMembershipFunc, self).__init__()
        self.a = nn.Parameter(torch.rand(num_mfs, input_dim) * 0.5 + 0.1)
        self.b = nn.Parameter(torch.rand(num_mfs, input_dim) * 2 + 0.5)
        self.c = nn.Parameter(torch.rand(num_mfs, input_dim))

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(2)
        a_exp, b_exp, c_exp = self.a.t().unsqueeze(0), self.b.t().unsqueeze(0), self.c.t().unsqueeze(0)
        b_clamped = torch.clamp(b_exp, min=0.01, max=10.0)
        base = torch.abs((x_unsqueezed - c_exp) / (a_exp + 1e-6))
        return 1. / (1. + base ** (2 * b_clamped))


class ConsequentGenerator(nn.Module):
    def __init__(self, input_dim, num_mfs, num_rules, num_conv_filters):
        super(ConsequentGenerator, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_conv_filters, kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(start_dim=1)  # Flatten the output for the linear layer
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, num_mfs)
            dummy_output = self.conv_net(dummy_input)
            linear_input_size = dummy_output.shape[1]

        self.fc_net = nn.Linear(linear_input_size, num_rules * (input_dim + 1))

        # Store these for reshaping in the forward pass
        self.num_rules = num_rules
        self.output_dim_per_rule = input_dim + 1

    def forward(self, memberships):
        # memberships shape: (batch_size, input_dim, num_mfs)
        conv_features = self.conv_net(memberships)

        # fc_net outputs a flat vector of all parameters for all rules
        # Shape: (batch_size, num_rules * (input_dim + 1))
        flat_params = self.fc_net(conv_features)

        # Reshape to get the parameters for each rule separately
        # Shape: (batch_size, num_rules, input_dim + 1)
        dynamic_params = flat_params.view(-1, self.num_rules, self.output_dim_per_rule)

        return dynamic_params


class HybridCnnAnfis(nn.Module):
    def __init__(self, input_dim, num_mfs, num_rules, firing_conv_filters, consequent_conv_filters,
                 device=torch.device('cpu'), output_dim=1):
        super(HybridCnnAnfis, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        self.num_rules = num_rules
        self.device = device
        self.output_dim = output_dim

        # Layer 1: Fuzzification
        self.membership_funcs = GeneralizedBellMembershipFunc(num_mfs, input_dim)

        # Head 1: Firing Strength Network
        self.firing_strength_net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=firing_conv_filters, kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(device)
        self.firing_fc = nn.Linear(firing_conv_filters, num_rules).to(device)
        self.batch_norm = nn.BatchNorm1d(num_rules).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

        # Head 2: Dynamic Consequent Parameter Network
        self.consequent_generator = ConsequentGenerator(
            input_dim, num_mfs, num_rules, consequent_conv_filters
        ).to(device)

        # Final projection layer for multi-step output
        self.output_projection = nn.Linear(1, self.output_dim).to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training and batch_size == 1:
            return self._forward_single(x)

        # Layer 1: Fuzzification
        memberships = self.membership_funcs(x)

        # Head 1: Calculate Firing Strengths (w̄ᵢ)
        firing_features = self.firing_strength_net(memberships).mean(dim=2)
        firing_strength_logits = self.firing_fc(firing_features)
        normalized_firing_strengths = self.softmax(self.batch_norm(firing_strength_logits))

        # Head 2: Generate Consequent Parameters (a, b, c...)
        dynamic_consequent_params = self.consequent_generator(memberships)

        # Layer 4 & 5: Consequent Calculation and Aggregation
        x_aug = torch.cat([x, torch.ones(batch_size, 1, device=self.device)], dim=1).unsqueeze(1)

        rule_outputs = (dynamic_consequent_params * x_aug).sum(dim=2)
        aggregated_output = (normalized_firing_strengths * rule_outputs).sum(dim=1, keepdim=True)

        # Project to the desired output dimension (for multi-step forecasting)
        final_output = self.output_projection(aggregated_output)

        return final_output

    def _forward_single(self, x):
        batch_size = x.shape[0]
        memberships = self.membership_funcs(x)
        firing_features = self.firing_strength_net(memberships).mean(dim=2)
        firing_strength_logits = self.firing_fc(firing_features)
        normalized_firing_strengths = self.softmax(firing_strength_logits)
        dynamic_consequent_params = self.consequent_generator(memberships)
        x_aug = torch.cat([x, torch.ones(batch_size, 1, device=self.device)], dim=1).unsqueeze(1)
        rule_outputs = (dynamic_consequent_params * x_aug).sum(dim=2)
        aggregated_output = (normalized_firing_strengths * rule_outputs).sum(dim=1, keepdim=True)
        final_output = self.output_projection(aggregated_output)
        return final_output


def train_anfis_model(features_X, target_Y, model_params, epochs=50, lr=0.001, batch_size=32):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(features_X)
    # The target_Y can be 1D or 2D, MinMaxScaler handles both
    y_scaled = scaler_y.fit_transform(target_Y if len(target_Y.shape) > 1 else target_Y.reshape(-1, 1))

    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32),
                            torch.tensor(y_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = HybridCnnAnfis(input_dim=features_X.shape[1], **model_params)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    epoch_bar = tqdm(range(epochs), desc="Training Multi-Step Model", leave=False)

    for epoch in epoch_bar:
        epoch_loss = 0.0
        num_batches = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        epoch_bar.set_postfix(train_rmse=f"{epoch_loss / num_batches:.6f}")

    return model, scaler_X, scaler_y