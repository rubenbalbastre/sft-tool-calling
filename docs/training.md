# Training

Training configuration is managed with Hydra:

- [`train_sft.py`](../train_sft.py) and
  [`config/train_sft.yaml`](../config/train_sft.yaml) run supervised fine-tuning.
- [`train_opd.py`](../train_opd.py) and
  [`config/train_opd.yaml`](../config/train_opd.yaml) run on-policy distillation.

Generate the dataset before starting either job.

```bash
python generate_data.py
python train_sft.py
python train_opd.py
```

## LoRA

Both training entry points use the shared
[`config/lora.yaml`](../config/lora.yaml). LoRA is enabled by default and
targets all linear modules. `num_layers: -1` applies it to every transformer
layer; a positive value applies it to the last N transformer layers.

```bash
python train_sft.py lora.r=32 lora.num_layers=12
python train_opd.py lora.enabled=false
```

When OPD starts from an existing PEFT model, it continues the loaded adapter
instead of creating a second adapter.

SFT rendering prepends the same procurement system prompt used during
evaluation. The stored dataset remains prompt-agnostic; the instruction is
added when the model-specific chat template is applied.

The default SFT sequence limit is 8,192 tokens so complete multi-tool
trajectories are retained. Padding tokens receive an attention mask of zero;
system, user, assistant, and tool tokens that fit in the sequence receive an
attention mask of one. Keep the physical batch small enough for the selected
model and use gradient accumulation when a larger effective batch is needed.

SFT loss is restricted to assistant output. For Gemma 4, preprocessing adds
generation markers around assistant tool calls because its inference chat
template does not provide assistant masks. Tool responses remain visible
through the attention mask but receive label `-100`, so they do not contribute
to the training or validation loss.

## Publishing a model

Upload a saved model or LoRA adapter to its own Hugging Face model repository
and register it in the project collection:

```bash
python scripts/publish-model.py \
  outputs/my-run/final_model
```

The script reads `HF_TOKEN` or `HUGGINGFACE_API_KEY` from the environment or
`.env`. By default, this example publishes to
`rubenbalbastre/procurement-function-calling-my-run`. A different repository ID
can be passed as the second positional argument. The script creates the
repository if needed, adds a basic model card when no `README.md` exists,
supports repeat uploads, and keeps the local directory unchanged. Use
`--private` for a private repository and `--note` to attach a short description
to the collection entry.

## OPD generation

The procurement tools are stateful: quotes and delivery options belong to one
seeded environment episode. The tool schemas are registered with the current
TRL trainer, but correct online tool execution requires an adapter that routes
each rollout to its own `ProcurementEnvironment`. Until that adapter is added,
use the OPD prompt splits for rollout experiments rather than treating the
plain callable tool loop as environment-verified training.

`train.enable_thinking` in [`config/train_opd.yaml`](../config/train_opd.yaml)
is passed to the model chat template for rollout generation. It defaults to
`false` so reasoning tokens are disabled explicitly instead of relying on each
model template's default.

SFT consumes `sft_train` and `sft_validation`. OPD consumes `opd_train` and
`opd_validation`. These names can be changed under `dataset` in each training
configuration.

The formatter passed to `SFTTrainer` renders each conversation with the selected
model's chat template. It converts stored JSON argument strings to mappings for
templates such as Gemma's and includes the tool definitions. The source column
is renamed before trainer preparation so TRL tokenizes the rendered text instead
of detecting and rendering the original messages a second time.

`train.enable_thinking` and `train.reasoning_effort` are also passed directly to
that chat-template call. They default to `false` and `none` because the
supervised trajectories contain tool actions but no reasoning traces. These
arguments are model-template-specific and are ignored by templates that do not
support them.

## Outputs and experiment tracking

Each run writes to an `outputs/<wandb-run-name>/` directory containing:

```text
config/train.yaml
checkpoints/
final_model/
```

Weights & Biases uses the project named by `WANDB_PROJECT`. The final model and
tokenizer are saved when `train.final_model.save` is enabled.

Set `train.run_name` to make this path deterministic for a downstream job. The
experiment runner uses `gemma-4-E2B-it-sft` and
`gemma-4-E2B-it-sft-opd`, producing:

```text
outputs/gemma-4-E2B-it-sft/final_model/
outputs/gemma-4-E2B-it-sft-opd/final_model/
```

## SFT early stopping

Early stopping is controlled by `train.early_stopping`:

```yaml
early_stopping:
  enabled: true
  patience: 3
  threshold: 0.0
  metric: eval_loss
  greater_is_better: false
```

`patience` counts evaluation calls, not epochs. A threshold of `0.0` accepts any
loss reduction as an improvement; use a small positive value such as `0.001`
when negligible fluctuations should not reset patience.

When early stopping is enabled, the best checkpoint is restored before
`final_model/` is saved. Keep `checkpointing.save_steps` aligned with
`eval_steps` for predictable behavior.

## Tokenizer integration check

Before a large run, verify that the selected tokenizer renders tool calls:

```python
from datasets import load_from_disk
from transformers import AutoTokenizer

dataset = load_from_disk("data/pipeline/hf_dataset")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
print(tokenizer.apply_chat_template(
    dataset["sft_train"][0]["messages"],
    tools=YOUR_TOOL_SCHEMAS,
    tokenize=False,
))
```
