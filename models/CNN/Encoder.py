import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out_rate=0.2):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding='valid'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=drop_out_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(out_channels * 4 * 4, out_channels)
        )

        self.softplus = nn.Softplus()


    def forward(self, x, eps: float = 1e-8):
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, chunks=2, dim=1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        z = dist.rsample()
        return x, mu, logvar, z



    def __del__(self):
        del self.conv1
        del self.norm1
        del self.dropout
        del self.pool1
        del self.conv2
        del self.norm2
        del self.pool2
        del self.flatten
        del self.fc1
        pass
