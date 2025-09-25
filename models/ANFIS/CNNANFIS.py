import torch
import torch.nn as nn

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
    def __init__(self, input_dim, num_mfs, num_rules, firing_conv_filters, consequent_conv_filters, device = torch.device('cpu')):
        super(HybridCnnAnfis, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        self.num_rules = num_rules
        self.device = device

        # Layer 1: Fuzzification
        self.membership_funcs = GeneralizedBellMembershipFunc(num_mfs, input_dim)

        # Head 1: Firing Strength Network
        self.firing_strength_net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=firing_conv_filters, kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(device)
        self.firing_fc = nn.Linear(firing_conv_filters, num_rules).to(device)
        self.batch_norm = nn.BatchNorm1d(num_rules).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

        # Head 2: Dynamic Consequent Parameter Network
        self.consequent_generator = ConsequentGenerator(
            input_dim, num_mfs, num_rules, consequent_conv_filters
        ).to(device)

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

        # Calculate rule outputs using the dynamically generated parameters
        # (batch_size, num_rules, input_dim + 1) * (batch_size, 1, input_dim + 1) -> sum over last dim
        rule_outputs = (dynamic_consequent_params * x_aug).sum(dim=2)
        final_output = (normalized_firing_strengths * rule_outputs).sum(dim=1, keepdim=True)
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
        final_output = (normalized_firing_strengths * rule_outputs).sum(dim=1, keepdim=True)
        return final_output


class MultiOutputConsequentGenerator(nn.Module):
    def __init__(self, input_dim, num_mfs, num_rules, num_conv_filters, output_dim):
        super(MultiOutputConsequentGenerator, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_conv_filters, kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(start_dim=1)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, num_mfs)
            dummy_output = self.conv_net(dummy_input)
            linear_input_size = dummy_output.shape[1]

        # The output size is now num_rules * y_horizon (one output per rule for each future day)
        self.fc_net = nn.Linear(linear_input_size, num_rules * output_dim)
        self.num_rules = num_rules
        self.output_dim = output_dim

    def forward(self, memberships):
        conv_features = self.conv_net(memberships)
        flat_params = self.fc_net(conv_features)
        # Reshape to (batch_size, num_rules, y_horizon)
        dynamic_params = flat_params.view(-1, self.num_rules, self.output_dim)
        return dynamic_params


class MultiOutputHybridCnnAnfis(nn.Module):
    def __init__(self, input_dim, num_mfs, num_rules, firing_conv_filters, consequent_conv_filters, output_dim,
                 device=torch.device('cpu')):
        super(MultiOutputHybridCnnAnfis, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        self.num_rules = num_rules
        self.output_dim = output_dim
        self.device = device

        self.membership_funcs = GeneralizedBellMembershipFunc(num_mfs, input_dim)

        self.firing_strength_net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=firing_conv_filters, kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(device)
        self.firing_fc = nn.Linear(firing_conv_filters, num_rules).to(device)
        self.batch_norm = nn.BatchNorm1d(num_rules).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

        # Use the new multi-output consequent generator
        self.consequent_generator = MultiOutputConsequentGenerator(
            input_dim, num_mfs, num_rules, consequent_conv_filters, output_dim
        ).to(device)

    def forward(self, x):
        # Layer 1: Fuzzification
        memberships = self.membership_funcs(x)

        # Head 1: Calculate Firing Strengths (w̄ᵢ)
        firing_features = self.firing_strength_net(memberships).mean(dim=2)
        firing_strength_logits = self.firing_fc(firing_features)

        # Handle batch size of 1 during training/eval
        if x.shape[0] > 1:
            normalized_firing_strengths = self.softmax(self.batch_norm(firing_strength_logits))
        else:
            normalized_firing_strengths = self.softmax(firing_strength_logits)

        # Head 2: Generate Consequent Parameters (now rule_outputs)
        # Shape: (batch_size, num_rules, y_horizon)
        rule_outputs = self.consequent_generator(memberships)

        # Layer 5: Aggregation
        # Multiply firing strengths across each of the y_horizon outputs
        # (batch_size, num_rules, 1) * (batch_size, num_rules, y_horizon)
        weighted_outputs = normalized_firing_strengths.unsqueeze(2) * rule_outputs

        # Sum across the rules to get the final output
        # Shape: (batch_size, y_horizon)
        final_output = weighted_outputs.sum(dim=1)
        return final_output