import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, dilation_rate, n_filters, kernel_size=5, dropout_rate=0.15):
        super().__init__()
        self.res = nn.Conv1d(in_channels, n_filters, kernel_size=1, padding="same")
        self.conv_1 = nn.Conv1d(in_channels, n_filters, kernel_size=kernel_size, dilation=dilation_rate, padding="same")
        self.conv_2 = nn.Conv1d(in_channels, n_filters, kernel_size=kernel_size, dilation=dilation_rate * 2, padding="same")
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv_3 = nn.Conv1d(2 * n_filters, n_filters, kernel_size=1, padding="same")

    def forward(self, x):
        res_x = self.res(x)
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(x)
        concat = torch.cat([conv_1, conv_2], dim=1)
        out = self.elu(concat)
        out = self.dropout(out)
        out = self.conv_3(out)
        return res_x + out, out


class TCN(nn.Module):
    def __init__(self, in_channels, n_filters=20, kernel_size=5, n_dilations=11, dropout_rate=0.15):
        super().__init__()
        dilations = [2**i for i in range(n_dilations)]
        self.tcn_layers = nn.ModuleDict({})
        for idx, d in enumerate(dilations):
            self.tcn_layers[f"tcn_{idx}"] = ResBlock(in_channels if idx == 0 else n_filters, d, n_filters, kernel_size, dropout_rate)
        self.activation = nn.ELU()

    def forward(self, x):
        skip_connections = []
        for tcn_i in self.tcn_layers:
            x, skip_out = self.tcn_layers[tcn_i](x)
            skip_connections.append(skip_out)
        x = self.activation(x)
        skip = torch.stack(skip_connections, dim=-1).sum(dim=-1)
        return x, skip
