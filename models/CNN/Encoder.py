import torch
from torch import nn


class Encoder(nn.Module):
    """
    Modified CNN encoder specifically for membership data processing
    """

    def __init__(self, in_channels: int, out_channels: int,
                 input_height: int, input_width: int, drop_out_rate: float = 0.2):
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width

        conv1_out_h = input_height
        conv1_out_w = input_width
        pool1_out_h = conv1_out_h // 2
        pool1_out_w = conv1_out_w // 2
        conv2_out_h = pool1_out_h
        conv2_out_w = pool1_out_w
        pool2_out_h = conv2_out_h // 2
        pool2_out_w = conv2_out_w // 2

        self.final_h = pool2_out_h
        self.final_w = pool2_out_w

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=drop_out_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=drop_out_rate)

        # Flatten and linear layers
        self.flatten = nn.Flatten()
        linear_input_size = 64 * self.final_h * self.final_w
        self.fc1 = nn.Linear(linear_input_size, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, height, width]
        Returns:
            features: [batch_size, out_channels]
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten and linear
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        return x