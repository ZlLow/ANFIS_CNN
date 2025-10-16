from torch import nn


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(output_dim, output_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(output_dim * 2, output_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv1d(output_dim * 2, output_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv1d(output_dim * 2, output_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim * 4),
            nn.MaxPool1d(2))
        self.layer8 = nn.Sequential(
            nn.Conv1d(output_dim, output_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim * 4),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv1d(output_dim * 4, output_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim * 4),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer10 = nn.Sequential(
            nn.Conv1d(output_dim * 4, output_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim * 4),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.MaxPool1d(2))
        # self.fc = nn.Linear(4*4*256, 3)
        self.fc = nn.Linear(4 * 4 * output_dim * 4, 1)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = self.layer6(out4)
        out = self.layer7(out)
        # re_layer_0 = self.layer
        res_layer_1 = self.layer5(self.layer8(out1))
        # print(res_layer.size())
        # print(out.size())
        out5 = res_layer_1 + out
        # out5 = out4
        out = self.layer9(out5)
        out = self.layer10(out)
        # out = self.layer10(out)
        out6 = out.view(out.size(0), -1)
        # print(out6.size())
        out7 = self.fc(out6)
        return out7