from __future__ import annotations

from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn


def apply_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> PeftModel:
    targets = [name for name, child in module.named_modules() if isinstance(child, nn.Conv1d)]
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=targets)
    return get_peft_model(module, config)
