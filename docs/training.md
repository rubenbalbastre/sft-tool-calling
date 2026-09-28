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

SFT consumes `sft_train` and `sft_validation`. OPD consumes `opd_train` and
`opd_validation`. These names can be changed under `dataset` in each training
configuration.

## Outputs and experiment tracking

Each run writes to an `outputs/<wandb-run-name>/` directory containing:

```text
config/train.yaml
checkpoints/
final_model/
```

Weights & Biases uses the project named by `WANDB_PROJECT`. The final model and
tokenizer are saved when `train.final_model.save` is enabled.

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
