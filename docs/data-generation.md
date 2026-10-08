# Dataset generation

The root entry point [`generate_data.py`](../generate_data.py) reads
[`config/data_generation.yaml`](../config/data_generation.yaml), creates a
Hugging Face `DatasetDict`, and saves it to `data/pipeline/hf_dataset` by
default.

```bash
python generate_data.py
```

## End-to-end process

Dataset creation is programmatic, but it does not write an assumed sequence of
tool calls directly. It constructs a hidden scenario, executes a reference
policy against the environment, and retains the conversation only after the
same verifier used during evaluation accepts the terminal action.

```text
SQLite master data
  → seeded semantic scenario
  → route-validity checks
  → verified reference rollout
  → multilingual prompt expansion
  → normalized Hugging Face rows
  → DatasetDict saved to disk
```

### 1. Build the deterministic market

The environment creates `data/environment/procurement.db` on first use. It
contains fixed materials, suppliers, countries, certifications, reliability,
material coverage, base prices, capacities, and preparation times. A scenario
seed deterministically derives the episode-specific quote and delivery values:
price, availability, ready date, transport cost, arrival date, delivery
reliability, and carbon emissions.

The market derivation is keyed by scenario, supplier, quote, and transport
mode. It is therefore independent of tool-call order: asking for supplier A
before supplier B does not change either supplier's result.

### 2. Generate semantic scenarios

Each split is generated with a distinct seed derived from the configured base
seed: `seed + 1` for `sft_train`, `seed + 2` for `sft_validation`, and
`seed + 3` for `test`. A scenario samples:

- one of three materials and its required unit;
- quantity, destination, order date, and deadline;
- allowed countries, budget, and minimum delivery reliability;
- route-specific supplier, certification, or fallback requirements;
- preference weights and the acceptable utility tolerance.

Task types cycle evenly by scenario index. Generation retries a candidate, up
to 200 attempts, until the intended route is meaningful:

| Task type | Construction condition |
| --- | --- |
| `direct_supplier` | Names a supplier that has a feasible option and exposes only that supplier initially. |
| `open_search` | Requires at least two feasible suppliers so comparison is necessary. |
| `compliance_first` | Adds `ISO-14001` as a hard requirement and keeps only scenarios with a compliant feasible option. |
| `preferred_with_fallback` | Names an infeasible preferred supplier while ensuring that a feasible fallback exists. |
| `no_feasible_option` | Lowers the budget to EUR 100 and verifies that the oracle finds no feasible option. |

See [the environment documentation](environment.md#hard-constraints-and-evidence)
for the complete constraint and evidence rules.

### 3. Execute and verify the reference trajectory

`build_reference_trajectory` interacts with `ProcurementEnvironment` through
the same tools available to evaluated models:

1. Follow the route: use the named supplier, search the market, or try the
   preferred supplier first.
2. For compliance-first tasks, inspect supplier profiles and retain only
   suppliers with all required certifications.
3. Request a quote and delivery options for every relevant supplier.
4. Assemble feasible options exclusively from observed tool results.
5. Submit the observed option with the highest utility, or report no feasible
   option when none exists.
6. Keep the trajectory only if the terminal environment result reports
   `success=true`; otherwise generation raises an error.

Every action is stored as an assistant tool call followed by the corresponding
tool observation. IDs returned by one call are carried into later calls, so the
result is a genuine multi-turn trajectory rather than isolated function-call
examples. The reference sequence is useful supervision, but evaluation remains
outcome-based and may accept other valid research routes.

### 4. Expand wording without leaking scenarios

Only after a semantic scenario and its reference trajectory have been created
does generation render prompt variants. The first user message is replaced by
each selected language/template combination while the verified tool trace and
hidden market state remain unchanged.

Scenario assignment happens before this expansion. Consequently, every prompt
variant sharing a `scenario_id` remains in one split. Template indices are also
disjoint by stage:

- training: templates 1–10;
- validation: templates 11–15;
- test: templates 16–25.

All three stages keep English, Spanish, German, and French. The default row
counts are therefore:

| Split | Semantic scenarios | Variants per scenario | Rows | Stored messages |
| --- | ---: | ---: | ---: | --- |
| `sft_train` | 100 | 4 languages × 10 templates | 4,000 | Complete verified trajectory |
| `sft_validation` | 10 | 4 languages × 5 templates | 200 | Complete verified trajectory |
| `test` | 10 | 4 languages × 10 templates | 400 | User prompt only |

### 5. Normalize and save the DatasetDict

All messages are normalized to one Arrow schema with `role`, `content`,
`tool_calls`, `name`, and `tool_call_id`. Assistant function arguments are
stored as JSON strings for portable dataset serialization; training
preprocessing deserializes them before applying model chat templates that
require argument mappings.

SFT splits retain the full verified conversation. The test split deliberately
retains only the initial user message so evaluation must discover the market
through fresh tool calls. `scenario_json` is saved alongside each row solely so
the environment and verifier can reconstruct hidden state; it must never be
included in the model prompt.

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

The values under `splits` are semantic scenario counts, not final row counts.

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

Set `HF_TOKEN` in the shell or the repository's ignored `.env`, then publish an
existing locally saved dataset without regenerating it:

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
under the `default` configuration. Pass a repository ID as the second
positional argument to publish elsewhere.

Load the named configuration with:

```python
from datasets import load_dataset

dataset = load_dataset(
    "rubenbalbastre/supply-chain-tool-calling",
    "default",
)
```
