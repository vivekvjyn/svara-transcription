from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


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


class CarnaticAudioEncoder(nn.Module):
    def __init__(
        self,
        *,
        bands: int = 81,
        filters: int = 20,
        kernel_size: int = 5,
        n_dilations: int = 11,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.bands = bands
        self.out_channels = filters
        self.conv_1 = nn.Conv2d(1, filters, kernel_size=(3, 3), stride=1, padding="valid")
        self.elu_1 = nn.ELU()
        self.mp_1 = nn.MaxPool2d((1, 3))
        self.dropout_1 = nn.Dropout(dropout)
        self.conv_2 = nn.Conv2d(filters, filters, kernel_size=(1, 10), stride=1, padding="valid")
        self.elu_2 = nn.ELU()
        self.mp_2 = nn.MaxPool2d((1, 3))
        self.dropout_2 = nn.Dropout(dropout)
        self.conv_3 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=1, padding="valid")
        self.elu_3 = nn.ELU()
        self.mp_3 = nn.MaxPool2d((1, 3))
        self.dropout_3 = nn.Dropout(dropout)
        self.tcn = TCN(
            filters,
            filters,
            kernel_size=kernel_size,
            n_dilations=n_dilations,
            dropout_rate=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.bands:
            raise ValueError(f"expected {self.bands} filter bands, got {features.shape[-1]}")
        x = features.unsqueeze(1)
        x = F.pad(x, (0, 0, 2, 2), mode="replicate")
        x = self.dropout_1(self.mp_1(self.elu_1(self.conv_1(x))))
        x = self.dropout_2(self.mp_2(self.elu_2(self.conv_2(x))))
        x = self.dropout_3(self.mp_3(self.elu_3(self.conv_3(x))))
        x = x.squeeze(-1)
        hidden, _ = self.tcn(x)
        return hidden

    def load_pretrained(self, checkpoint: Path | str, device: torch.device | str = "cpu") -> None:
        state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        heads = ("beats_", "downbeats_", "tempo_")
        trunk = {key: value for key, value in state.items() if not key.startswith(heads)}
        self.load_state_dict(trunk)
