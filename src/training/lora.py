"""Shared LoRA configuration for training entry points."""

from peft import LoraConfig, TaskType


def build_lora_config(config, model):
    if not config.enabled or getattr(model, "peft_config", None):
        return None

    layers_to_transform = None
    if config.num_layers != -1:
        total_layers = model.config.get_text_config().num_hidden_layers
        layers_to_transform = list(
            range(total_layers - config.num_layers, total_layers)
        )

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.r),
        lora_alpha=int(config.alpha),
        lora_dropout=float(config.dropout),
        bias=config.bias,
        target_modules=config.target_modules,
        layers_to_transform=layers_to_transform,
    )
