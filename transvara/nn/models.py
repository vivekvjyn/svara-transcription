"""The contextual svara classifier, vendored from RISE (submodule removed).

Three InceptionTime encoders read the preceding, current and succeeding contours;
co-attention joins them with respect to the final hidden state of the current
stream, and the concatenation is classified. The body is copied verbatim from
``RISE/src/rise/nn/models.py`` so the checkpoints in ``model_weights/`` load
unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import CoAttention
from .inception import InceptionEncoder
from .lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK, apply_lora

#: Pitch plus its binary silence mask.
NUM_INPUT_FEATURES = 2

#: The preceding, current and succeeding streams of the contextual classifier.
CONTEXT_STREAMS = ("prec", "curr", "succ")


class ContextualSvaraClassifier(nn.Module):
    """Classify a *svara* from its own contour and those of its two neighbours.

    Each stream has its own encoder and its own GRU; co-attention then summarises
    all three with respect to the final hidden state of the current stream, and the
    concatenation of the three context vectors is classified. Attaching the
    attention to the current stream is what makes the context *relative*: the model
    looks at the neighbours for whatever bears on the *svara* in hand.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_classes: int,
        num_features: int = NUM_INPUT_FEATURES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.prec_encoder = InceptionEncoder(embed_dim, depth, num_features)
        self.curr_encoder = InceptionEncoder(embed_dim, depth, num_features)
        self.succ_encoder = InceptionEncoder(embed_dim, depth, num_features)

        self.prec_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.curr_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.succ_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

        self.attention = CoAttention(embed_dim)
        self.head = nn.Sequential(
            nn.BatchNorm1d(embed_dim * len(CONTEXT_STREAMS)),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * len(CONTEXT_STREAMS), embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    @property
    def encoder(self) -> InceptionEncoder:
        """The current-stream encoder, i.e. the one that sees the *svara* itself."""
        return self.curr_encoder

    @property
    def encoders(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return self.prec_encoder, self.curr_encoder, self.succ_encoder

    def forward(self, prec: torch.Tensor, curr: torch.Tensor, succ: torch.Tensor) -> torch.Tensor:
        """Return the ``(batch, num_classes)`` logits."""
        return self.head(self.encode(prec, curr, succ))

    def encode(self, prec: torch.Tensor, curr: torch.Tensor, succ: torch.Tensor) -> torch.Tensor:
        """Return the ``(batch, 3 * embed_dim)`` representation the head classifies."""
        prec_states = self._run(prec, self.prec_encoder, self.prec_gru)
        curr_states = self._run(curr, self.curr_encoder, self.curr_gru)
        succ_states = self._run(succ, self.succ_encoder, self.succ_gru)
        reference = curr_states[:, -1:]

        return torch.cat(
            [
                self.attention(prec_states, reference),
                self.attention(curr_states, reference),
                self.attention(succ_states, reference),
            ],
            dim=1,
        )

    @staticmethod
    def _run(contours: torch.Tensor, encoder: nn.Module, gru: nn.GRU) -> torch.Tensor:
        hidden_states, _ = gru(encoder(contours).permute(0, 2, 1))
        return hidden_states

    def apply_lora(
        self,
        rank: int = DEFAULT_RANK,
        alpha: int = DEFAULT_ALPHA,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Replace the three encoders in place with LoRA-adapted versions."""
        self.prec_encoder = apply_lora(self.prec_encoder, rank, alpha, dropout)
        self.curr_encoder = apply_lora(self.curr_encoder, rank, alpha, dropout)
        self.succ_encoder = apply_lora(self.succ_encoder, rank, alpha, dropout)

    def load_pretrained_encoders(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialise all three encoders from one pretrained backbone checkpoint."""
        for encoder in self.encoders:
            encoder.load_state_dict(state_dict)

    def set_encoders_trainable(self, trainable: bool = True) -> None:
        """Freeze or unfreeze the three encoders, leaving the head unaffected."""
        for encoder in self.encoders:
            for parameter in encoder.parameters():
                parameter.requires_grad = trainable

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
