from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from game import (
    GaussianSoftBoundaryLoss,
    LocalDownsample,
    Segmenter,
    boundaries_to_regions,
    decode_soft_boundaries,
    d3pm_time_schedule,
    first_frame_index,
    remove_boundaries,
)
from rise import FrameEncoder, apply_lora
from tcn import CarnaticAudioEncoder, TCN


def load_encoder_weights(path: Path | str, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return torch.load(Path(path), map_location=device, weights_only=True)


class TranscriptionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        embed_dim: int,
        depth: int,
        tcn_filters: int,
        tcn_kernel: int,
        tcn_dilations: int,
        dropout: float,
        segmenter_depth: int,
        boundary_std: float,
        boundary_threshold: float,
        boundary_radius: int,
        d3pm_steps: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.boundary_std = boundary_std
        self.boundary_threshold = boundary_threshold
        self.boundary_radius = boundary_radius
        self.d3pm_steps = d3pm_steps

        self.encoder = FrameEncoder(embed_dim, depth)
        self.embedding = TCN(embed_dim, tcn_filters, kernel_size=tcn_kernel, n_dilations=tcn_dilations, dropout_rate=dropout)
        self.audio_encoder = CarnaticAudioEncoder(dropout=dropout)
        self.segmenter = Segmenter(
            tcn_filters + self.audio_encoder.out_channels,
            depth=segmenter_depth,
            kernel=tcn_kernel,
            dropout=dropout,
        )
        self.pool = LocalDownsample()
        self.classifier = nn.Sequential(
            nn.LayerNorm(tcn_filters),
            nn.Linear(tcn_filters, tcn_filters),
            nn.ELU(),
            nn.Linear(tcn_filters, num_classes),
        )
        self.boundary_loss = GaussianSoftBoundaryLoss(boundary_std)
        for parameter in self.audio_encoder.parameters():
            parameter.requires_grad = False

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.embedding(self.encoder(inputs))
        return hidden

    def encode_audio(self, features: torch.Tensor) -> torch.Tensor:
        self.audio_encoder.eval()
        return self.audio_encoder(features)

    def fuse(self, embedding: torch.Tensor, audio_embedding: torch.Tensor) -> torch.Tensor:
        return torch.cat([embedding, audio_embedding], dim=1)

    def classifier_outputs(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedding = self.encode(inputs)
        regions = boundaries_to_regions(boundaries)
        region_logits = self.classifier(self.pool(embedding.transpose(1, 2), regions))
        present = torch.arange(region_logits.shape[1], device=regions.device) <= regions.max(dim=1).values.unsqueeze(1)
        return {
            "region_logits": region_logits,
            "region_labels": region_label_counts(targets, regions, self.num_classes).argmax(dim=-1)[:, 1:],
            "region_mask": present,
            "regions": regions,
        }

    def segmenter_outputs(
        self,
        inputs: torch.Tensor,
        audio_features: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedding = self.fuse(self.encode(inputs), self.encode_audio(audio_features))
        batch = boundaries.shape[0]
        t = torch.rand(batch, device=boundaries.device)
        corrupted = remove_boundaries(boundaries, d3pm_time_schedule(t))
        return {"boundary_logits": self.segmenter(embedding, boundaries_to_regions(corrupted), t)}

    @torch.no_grad()
    def transcribe(self, inputs: torch.Tensor, audio_features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.transcribe_embedding(self.encode(inputs), self.encode_audio(audio_features))

    @torch.no_grad()
    def transcribe_embedding(
        self,
        embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        fused = self.fuse(embedding, audio_embedding)
        boundaries = torch.zeros(embedding.shape[0], embedding.shape[2], dtype=torch.bool, device=embedding.device)
        step = torch.full((1,), 1 / self.d3pm_steps, device=embedding.device)
        for index in range(self.d3pm_steps):
            t = index * step.expand(embedding.shape[0])
            boundaries = remove_boundaries(boundaries, d3pm_time_schedule(t))
            logits = self.segmenter(fused, boundaries_to_regions(boundaries), t)
            boundaries = decode_soft_boundaries(
                logits.sigmoid(), threshold=self.boundary_threshold, radius=self.boundary_radius
            )

        regions = boundaries_to_regions(boundaries)
        region_logits = self.classifier(self.pool(embedding.transpose(1, 2), regions))
        if was_training:
            self.train()
        return {
            "boundaries": boundaries,
            "regions": regions,
            "region_logits": region_logits,
            "region_first_frame": first_frame_index(regions),
        }

    def load_pretrained_encoder(self, checkpoint: Path | str, device: torch.device | str = "cpu") -> None:
        self.encoder.load_state_dict(load_encoder_weights(checkpoint, device))

    def load_pretrained_audio_encoder(self, checkpoint: Path | str, device: torch.device | str = "cpu") -> None:
        self.audio_encoder.load_pretrained(checkpoint, device)

    def apply_lora(self, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        self.encoder = apply_lora(self.encoder, rank, alpha, dropout)

    def set_encoders_trainable(self, trainable: bool = True) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def region_label_counts(targets: torch.Tensor, regions: torch.Tensor, num_classes: int) -> torch.Tensor:
    batch, _ = regions.shape
    count = int(regions.max()) + 1
    offset = torch.arange(batch, device=regions.device).unsqueeze(1) * count * num_classes
    index = (regions * num_classes + targets + offset).reshape(-1)
    counts = torch.zeros(batch * count * num_classes, dtype=torch.float32, device=targets.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float32))
    return counts.reshape(batch, count, num_classes)
