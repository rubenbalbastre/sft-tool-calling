# Dataset generation

The root entry point [`generate_data.py`](../generate_data.py) reads
[`config/data_generation.yaml`](../config/data_generation.yaml), creates a
Hugging Face `DatasetDict`, and saves it to `data/pipeline/hf_dataset` by
default.

```bash
python generate_data.py
```

## Splits

- `sft_train` and `sft_validation`: complete supervised trajectories.
- `opd_train` and `opd_validation`: initial user message plus hidden scenario
  state for online rollouts.
- `test`: held-out prompt-only scenarios.

All splits have the same columns:

- `messages`: full SFT conversation or initial OPD/test prompt;
- `scenario_json`: hidden state for the environment and verifier;
- `scenario_id`, `stage`, `language`, `trajectory_type`, and `difficulty`;
- `tool_sequence`: expected action sequence for analysis.

Only `messages` is model input. Never include `scenario_json` or
`tool_sequence` in the model prompt.

Generation is seeded and uses 30 request phrases per language: ten at each of
the simple, medium, and hard levels. Tool arguments are JSON strings under
`assistant.tool_calls[].function.arguments`.

## Hydra overrides

```bash
python generate_data.py \
  splits.sft_train=40 \
  splits.sft_validation=5 \
  splits.opd_train=20 \
  splits.opd_validation=5 \
  splits.test=10 \
  output_dir=data/pilot/hf_dataset
```

## Load locally

```python
from datasets import load_from_disk

dataset = load_from_disk("data/pipeline/hf_dataset")
example = dataset["sft_train"][0]
```

## Publish to the Hugging Face Hub

Set `HF_TOKEN` in the shell or the repository's ignored `.env`, then run:

```bash
python generate_data.py \
  hub.push=true \
  hub.repo_id=your-account/supply-chain-tool-calling \
  hub.config_name=pipeline-v1
```

Load the named configuration with:

```python
from datasets import load_dataset

dataset = load_dataset(
    "your-account/supply-chain-tool-calling",
    "pipeline-v1",
)
```
