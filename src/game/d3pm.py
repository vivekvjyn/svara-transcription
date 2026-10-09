from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tcn import TCN


def d3pm_time_schedule(t: torch.Tensor) -> torch.Tensor:
    return (1 + torch.cos(t * torch.pi)) / 2


def remove_boundaries(boundaries: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    keep = (1 - p).unsqueeze(-1)
    return boundaries & (torch.rand_like(boundaries, dtype=torch.float32) < keep)


def boundaries_to_regions(boundaries: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(boundaries.long(), dim=-1) + 1


def first_frame_index(regions: torch.Tensor) -> torch.Tensor:
    batch, frames = regions.shape
    first = torch.full((batch, int(regions.max()) + 1), frames, dtype=torch.long, device=regions.device)
    positions = torch.arange(frames, device=regions.device).unsqueeze(0).expand(batch, -1)
    return first.scatter_reduce(1, regions, positions, reduce="amin", include_self=True)


class CyclicRegionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, cycle: int = 3) -> None:
        super().__init__()
        self.cycle = cycle
        self.embedding = nn.Embedding(cycle, embedding_dim)

    def forward(self, regions: torch.Tensor) -> torch.Tensor:
        if self.training:
            shift = torch.randint(0, self.cycle, (*regions.shape[:-1], 1), device=regions.device)
            regions = regions + shift
        return self.embedding(regions % self.cycle)


class LocalDownsample(nn.Module):
    def forward(self, features: torch.Tensor, regions: torch.Tensor) -> torch.Tensor:
        count = int(regions.max())
        batch = (1,) * (features.ndim - 2)
        index = torch.arange(count + 1, dtype=torch.long, device=regions.device).reshape(*batch, -1, 1)
        in_region = (index == regions.unsqueeze(-2)).float()
        pooled = in_region @ features / in_region.sum(dim=-1, keepdim=True).clamp_min(1)
        return pooled[..., 1:, :]


class GaussianSoftBoundaryLoss(nn.Module):
    def __init__(self, std: float = 1.0) -> None:
        super().__init__()
        self.std = std

    def forward(self, logits: torch.Tensor, boundaries: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        targets = self._soften(boundaries)
        mass = targets.sum().detach()
        total = mask.float().sum().detach() if mask is not None else targets.numel()
        weight = ((total - mass) / mass.clamp_min(1e-2)).clamp(1.0, 100.0)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weight.reshape(1), reduction="none"
        )
        if mask is None:
            return loss.mean()
        mask = mask.float()
        return (loss * mask).sum() / mask.sum().clamp_min(1)

    def _soften(self, boundaries: torch.Tensor) -> torch.Tensor:
        padded = F.pad(boundaries.to(torch.long), (1, 1), value=0).to(torch.bool)
        index = torch.arange(padded.shape[-1], device=padded.device)
        marked = torch.where(padded, index, torch.full_like(index, -1))
        far = padded.shape[-1]
        before = marked.cummax(dim=-1).values
        after = marked.flip(-1).cummax(dim=-1).values.flip(-1)
        distance = torch.minimum(
            torch.where(before >= 0, index - before, far),
            torch.where(after >= 0, after - index, far),
        )[..., 1:-1].clamp(0, far)
        return torch.exp(-0.5 * (distance.to(torch.float32) / self.std) ** 2)


def find_local_extremum(values: torch.Tensor, threshold: float, radius: int = 2, maxima: bool = True) -> torch.Tensor:
    infinity = float("-inf") if maxima else float("+inf")
    padded = F.pad(values, (radius, radius), mode="constant", value=infinity)
    window = padded.unfold(-1, 2 * radius + 1, 1)
    if maxima:
        points = window.argmax(dim=-1) == radius
        return points & (values >= threshold)
    points = window.argmin(dim=-1) == radius
    return points & (values <= threshold)


def decode_soft_boundaries(
    probabilities: torch.Tensor, threshold: float = 0.2, radius: int = 2, mask: torch.Tensor | None = None
) -> torch.Tensor:
    points = find_local_extremum(probabilities, threshold=threshold, radius=radius, maxima=True)
    if mask is not None:
        points = points & mask
    return points


class Segmenter(nn.Module):
    def __init__(self, channels: int, *, depth: int = 6, kernel: int = 5, dropout: float = 0.15) -> None:
        super().__init__()
        self.region_embedding = CyclicRegionEmbedding(channels)
        self.time_embedding = nn.Sequential(nn.Linear(1, 4 * channels), nn.ELU(), nn.Linear(4 * channels, channels))
        self.context = TCN(channels, channels, kernel_size=kernel, n_dilations=depth, dropout_rate=dropout)
        self.readout = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, embedding: torch.Tensor, regions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        context = embedding + self.region_embedding(regions).transpose(1, 2)
        context = context + self.time_embedding(t.unsqueeze(-1)).unsqueeze(-1)
        hidden, _ = self.context(context)
        return self.readout(hidden).squeeze(1)
