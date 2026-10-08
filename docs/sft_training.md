# Supervised fine-tuning

[`train_sft.py`](../train_sft.py) fine-tunes a model on complete, verified
procurement tool trajectories with TRL's `SFTTrainer`. Hydra loads the defaults
from [`config/train_sft.yaml`](../config/train_sft.yaml).

Generate the dataset first, then start training from the repository root:

```bash
python generate_data.py
python train_sft.py
```

Hydra values can be overridden from the command line:

```bash
python train_sft.py \
  train.model_name=google/gemma-4-E2B-it \
  train.run_name=gemma-4-E2B-it-sft
```

## Input preparation

SFT reads `sft_train` and `sft_validation` from the Hugging Face `DatasetDict`.
The configured procurement system prompt is prepended in memory, so the stored
dataset remains independent of a particular training prompt.

Before constructing `SFTTrainer`, the preprocessor applies the selected model's
chat template and materializes `input_ids` and assistant-only `labels`. Tool
arguments remain JSON strings in Arrow storage and are converted to Python
mappings immediately before tokenization. This is required by templates such as
Gemma's and prevents Arrow from merging arguments from different tools into a
single struct containing unrelated null fields.

The tool schemas and these chat-template arguments accompany every prepared
conversation:

```yaml
enable_thinking: false
reasoning_effort: none
```

The supervised trajectories contain actions but no reasoning traces, so
thinking is disabled by default.

## Loss, masks, and sequence handling

The preprocessor implements assistant-only loss directly in the materialized
`labels`: system, user, and tool-response tokens remain visible through the
attention mask but receive label `-100`; only the assistant's tool calls
contribute to loss. `SFTConfig.assistant_only_loss` stays disabled because TRL
correctly sees this pretokenized dataset as non-conversational; enabling that
option would reject the already prepared `input_ids` and `labels`.

Gemma 4's inference template does not contain TRL generation markers, so the
preprocessor adds them around assistant tool calls. The full call—including the
tool-call closing token—is supervised, while the environment response is not.
This can produce TRL's generic warning about an end-of-turn token outside the
loss mask.

The resulting tokenized dataset is passed with `skip_prepare_dataset=True`.
This avoids storing Python mappings through Arrow and avoids the unsupported
combination of `SFTTrainer` with `Dataset.with_transform()`.

The maximum sequence length is 8,192 tokens. Packing and padding-free training
are disabled because they require a compatible Flash Attention implementation
to preserve sample boundaries reliably.

## LoRA

LoRA is configured in [`config/lora.yaml`](../config/lora.yaml) and built with
PEFT. The current defaults are:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `enabled` | `true` | Train adapters instead of all model weights |
| `r` | `16` | Adapter rank |
| `alpha` | `32` | LoRA scaling parameter |
| `dropout` | `0.05` | Adapter dropout |
| `bias` | `none` | Do not train bias parameters |
| `target_modules` | `all-linear` | Target linear transformer modules |
| `num_layers` | `-1` | Apply LoRA to every transformer layer |

A positive `num_layers` value targets only the final N transformer layers.

```bash
python train_sft.py lora.r=32 lora.num_layers=12
```

## Main hyperparameters

These defaults come from `config/train_sft.yaml`:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `train.learning_rate` | `2e-5` | Optimizer learning rate |
| `train.per_device_train_batch_size` | `8` | Physical training batch per device |
| `train.gradient_accumulation_steps` | `1` | Steps used to build the effective batch |
| `train.per_device_eval_batch_size` | `16` | Validation-loss batch per device |
| `train.max_steps` | `48` | Training optimizer steps |
| `train.max_seq_length` | `8192` | Maximum rendered trajectory length |
| `train.gradient_checkpointing` | `true` | Trade additional compute for activation memory |
| `train.bf16` | `true` | Use bfloat16 training |
| `train.eval_steps` | `4` | Validation-loss interval |

The effective batch size is
`per_device_train_batch_size × gradient_accumulation_steps × devices`.

## Environment validation

Validation loss does not prove that generated tool calls solve the task. A
training callback therefore runs a fixed, seeded, task-stratified set of
environment episodes using the current in-memory model. Active episodes are
batched together at each turn.

```yaml
validation_rollout_scenarios: 10
validation_rollout_steps: 16
```

The callback reports environment success rate and average return to Weights &
Biases. Full held-out model comparison remains the responsibility of the
evaluation entry points.

## Early stopping

SFT enables Transformers' `EarlyStoppingCallback` by default:

```yaml
early_stopping:
  enabled: true
  patience: 3
  threshold: 0.01
  metric: eval_loss
  greater_is_better: false
```

Patience counts evaluation calls. Checkpoint saving and evaluation are both
configured every four steps so the best checkpoint can be restored before the
final adapter is saved.

## Resume from a checkpoint

Set `train.model_name` to a local `checkpoint-*` directory:

```yaml
train:
  model_name: outputs/my-run/checkpoints/checkpoint-24
```

The script detects `trainer_state.json`, loads the LoRA adapter as trainable,
and restores the optimizer, scheduler, random state, and completed step count.
`train.max_steps` remains the total target step count, not the number of extra
steps. A local adapter or final model without `trainer_state.json` starts a new
training run from those weights instead.

## Outputs and experiment tracking

Each run writes to `outputs/<run-name>/`:

```text
outputs/<run-name>/
├── config/train.yaml
├── checkpoints/
└── final_model/
```

The configuration is copied at startup. Checkpoints respect `save_steps` and
`save_total_limit`; `final_model/` receives the best loaded LoRA adapter and
tokenizer when `train.final_model.save=true`.

Weights & Biases logs to the project named by `WANDB_PROJECT`. Set
`train.run_name` when another job needs a predictable output path.

Publish a final adapter and add it to the project collection with:

```bash
python scripts/publish-model.py outputs/<run-name>/final_model
```
