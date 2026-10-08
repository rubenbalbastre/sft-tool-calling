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
- `test`: held-out prompt-only scenarios.

All splits have the same columns:

- `messages`: full SFT conversation or initial test prompt;
- `scenario_json`: hidden state for the environment and verifier;
- `scenario_id`, `stage`, `language`, `trajectory_type`, and `difficulty`;
- `prompt_variant`: language and template provenance for the surface form;
- `tool_sequence`: reference trajectory sequence for analysis, not a required
  sequence enforced by the verifier.

Only `messages` is model input. Never include `scenario_json` or
`tool_sequence` in the model prompt.

Generation is seeded and cycles through direct-supplier, open-search,
compliance, preferred-with-fallback, and no-feasible-option tasks. Each semantic
scenario is assigned to one split before prompt expansion, preventing semantic
scenario leakage. Prompt templates are also held out by stage in every
language:

- training: templates 1–10;
- validation: templates 11–15;
- test: templates 16–25.

The values under `splits` are semantic scenario counts, not final row counts.
With four languages, each training scenario produces 40 rows, each validation
scenario produces 20, and each test scenario produces 40. Complete SFT
conversations are produced by running a reference policy once through the same
environment used for evaluation and reusing that verified action trace across
its prompt variants. Tool arguments are JSON strings under
`assistant.tool_calls[].function.arguments`.

Reference routes follow the user instruction: preferred suppliers are quoted
before searching for fallbacks, and compliance-first trajectories inspect all
profiles but request quotes only from suppliers whose observed certifications
satisfy the requirement.

## Hydra overrides

```bash
python generate_data.py \
  splits.sft_train=40 \
  splits.sft_validation=5 \
  splits.test=10 \
  output_dir=data/pilot/hf_dataset
```

Override the template allocation with:

```bash
python generate_data.py \
  prompt_generation.languages='[English,Spanish]' \
  prompt_generation.templates.train='[1,2,3]' \
  prompt_generation.templates.validation='[11]' \
  prompt_generation.templates.test='[16,17]'
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
python generate_data.py hub.push=true
```

To publish a dataset that is already saved locally without regenerating it,
use:

```bash
python scripts/publish-dataset.py
```

The default source is `data/pipeline/hf_dataset`. Pass another saved dataset
directory as the first argument when needed:

```bash
python scripts/publish-dataset.py \
  runpod-artifacts/data/pipeline/hf_dataset
```

The script also publishes `src/data_generation/README.md` as the Hugging Face
dataset card. It uploads only the current `sft_train`, `sft_validation`, and
`test` splits, ignoring unrelated legacy splits in older saved artifacts.

By default, this updates
[`rubenbalbastre/supply-chain-tool-calling`](https://huggingface.co/datasets/rubenbalbastre/supply-chain-tool-calling)
under the `default` configuration. Override `hub.repo_id` and
`hub.config_name` to publish elsewhere.

Load the named configuration with:

```python
from datasets import load_dataset

dataset = load_dataset(
    "rubenbalbastre/supply-chain-tool-calling",
    "default",
)
```
