# On-policy distillation

**Status: Work in progress (WIP).**

> [!WARNING]
> This training path is work in progress. The TRL trainer can generate tool
> calls, but environment-scoped online tool execution, checkpoint configuration,
> and early stopping are not fully integrated yet.

[`train_opd.py`](../train_opd.py) uses TRL's `DistillationTrainer` to train a
student model from completions and token distributions produced by a larger
teacher. Hydra loads its defaults from
[`config/train_opd.yaml`](../config/train_opd.yaml).

```bash
python generate_data.py
python train_opd.py
```

A typical second-stage run starts from the adapter produced by SFT:

```bash
python train_opd.py \
  train.model_name=outputs/gemma-4-E2B-it-sft/final_model \
  train.teacher_model=google/gemma-4-12B-it \
  train.run_name=gemma-4-E2B-it-sft-opd
```

## Input preparation

OPD reads `opd_train` and `opd_validation`. They use different seeded scenarios
from SFT while sharing its template ranges: 1–10 for training and 11–15 for
validation. These splits contain only the initial user message.
`prepare_opd_source()` exposes that message under `prompt`, which is the field
expected by `DistillationTrainer` for online generation.

The generation template receives:

```yaml
enable_thinking: false
```

This explicitly disables model-specific thinking tokens for the current
experiments.

## Stateful tool execution

The trainer receives the Python procurement tool functions, but the environment
is stateful: quotes and delivery options belong to one seeded episode. Correct
online training therefore requires an adapter that assigns each rollout its own
`ProcurementEnvironment` and preserves that instance across tool turns.

That routing layer is not implemented yet. The current OPD path should be
treated as an experimental generation pipeline, not as fully
environment-verified online distillation.

## LoRA

OPD shares [`config/lora.yaml`](../config/lora.yaml) with SFT:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `enabled` | `true` | Train PEFT adapter parameters |
| `r` | `16` | Adapter rank |
| `alpha` | `32` | LoRA scaling parameter |
| `dropout` | `0.05` | Adapter dropout |
| `target_modules` | `all-linear` | Target linear transformer modules |
| `num_layers` | `-1` | Apply LoRA to every transformer layer |

When the student path already contains a PEFT adapter, the setup continues that
adapter rather than intentionally stacking a second one.

## Main hyperparameters

These defaults come from `config/train_opd.yaml`:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `train.learning_rate` | `2e-5` | Student optimizer learning rate |
| `train.per_device_train_batch_size` | `32` | Physical training batch per device |
| `train.gradient_accumulation_steps` | `1` | Steps used to build the effective batch |
| `train.max_steps` | `50` | Training optimizer steps |
| `train.temperature` | `1.0` | Rollout sampling temperature |
| `train.top_p` | `1.0` | Nucleus-sampling threshold |
| `train.top_k` | `0` | Top-k filtering; zero disables it |
| `train.max_completion_length` | `256` | Maximum generated tokens per completion |
| `train.max_tool_calling_iterations` | `5` | Maximum online tool-loop iterations |
| `train.bf16` | `true` | Use bfloat16 training |

The student and teacher checkpoints are selected through `train.model_name` and
`train.teacher_model` respectively. Completion logging is controlled by
`log_completions` and `num_completions_to_print`.

## Early stopping and evaluation

Early stopping is not currently implemented for OPD. Although
`config/train_opd.yaml` contains `eval_strategy`, `eval_steps`, and checkpoint
settings, `train_opd.py` does not yet pass them to `DistillationConfig` or
register an early-stopping callback. They should therefore not be assumed to be
active.

Before using OPD for a full experiment, wire evaluation and checkpoint saving
into `DistillationConfig`, select a meaningful validation metric, and align the
evaluation and save intervals.

## Outputs and experiment tracking

At startup, the entry point writes the resolved Hydra configuration to:

```text
outputs/<run-name>/config/train.yaml
```

When `train.final_model.save=true`, it saves the final adapter and tokenizer to:

```text
outputs/<run-name>/final_model/
```

Weights & Biases uses the project named by `WANDB_PROJECT`. The intended
`outputs/<run-name>/checkpoints/` directory is constructed by the entry point,
but the current checkpoint settings are not passed to `DistillationConfig`; this
is part of the remaining OPD work.
