from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import torch
from torch import nn


def branch_widths(out_channels: int, kernel_sizes: tuple[int, ...]) -> list[int]:
    base, remainder = divmod(out_channels, len(kernel_sizes))
    return [base + (1 if index < remainder else 0) for index in range(len(kernel_sizes))]


class _LegacyBranchNames:
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for index in range(len(self.branches)):
            legacy = f"branch{index + 1}."
            for key in [k for k in state_dict if k.startswith(prefix + legacy)]:
                renamed = key.replace(prefix + legacy, f"{prefix}branches.{index}.", 1)
                state_dict[renamed] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class InceptionBlock(_LegacyBranchNames, nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple[int, ...] = (9, 19, 39)) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.branches = nn.ModuleList(
            nn.Conv1d(in_channels, width, kernel_size=size, padding=size // 2)
            for size, width in zip(kernel_sizes, branch_widths(out_channels, kernel_sizes), strict=True)
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(inputs) for branch in self.branches], dim=1) + self.residual(inputs)


class FrameEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_features: int = 2,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
    ) -> None:
        super().__init__()
        widths = [embed_dim * 2**exponent for exponent in range(depth - 1, 0, -1)]
        channels = [num_features, *widths, embed_dim]

        blocks = [
            nn.Sequential(
                InceptionBlock(in_channels, out_channels, kernel_sizes),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
            )
            for in_channels, out_channels in pairwise(channels[:-1])
        ]
        blocks.append(InceptionBlock(channels[-2], channels[-1], kernel_sizes))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, contours: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            contours = block(contours)
        return contours

    def load_pretrained(self, state_dict: dict[str, torch.Tensor] | str | Path, device: torch.device | str = "cpu") -> None:
        if isinstance(state_dict, (str, Path)):
            state_dict = torch.load(Path(state_dict), map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
