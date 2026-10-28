from __future__ import annotations

from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

DEFAULT_RANK = 8

DEFAULT_ALPHA = 16

DEFAULT_DROPOUT = 0.05


def apply_lora(
    module: nn.Module,
    rank: int = DEFAULT_RANK,
    alpha: int = DEFAULT_ALPHA,
    dropout: float = DEFAULT_DROPOUT,
) -> PeftModel:
    targets = [name for name, child in module.named_modules() if isinstance(child, nn.Conv1d)]
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=targets)
    return get_peft_model(module, config)
