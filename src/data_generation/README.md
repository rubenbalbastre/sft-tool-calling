---
pretty_name: Multilingual Procurement Tool Calling
language:
  - en
  - es
  - de
  - fr
task_categories:
  - text-generation
tags:
  - tool-calling
  - function-calling
  - synthetic
  - multilingual
  - procurement
---

# Multilingual Procurement Tool Calling

This dataset contains multilingual, multi-turn tool-calling trajectories for a
simulated procurement environment. Models must gather supplier information,
request quotes, inspect delivery options and finish with either a valid
procurement plan or a verified no-feasible-option decision.

The trajectories are generated and executed against the same deterministic
environment used for evaluation. Reference traces are retained only after the
environment verifies that they satisfy the scenario's constraints.

## Languages

- English
- Spanish
- German
- French

Each semantic scenario is expanded into multiple independently written prompt
templates. Scenario IDs remain within a single split to prevent semantic
leakage, and the test split uses prompt templates excluded from training.

## Splits

- `sft_train`: complete supervised trajectories for SFT.
- `sft_validation`: held-out supervised trajectories for validation.
- `test`: prompt-only held-out scenarios for closed-loop environment
  evaluation.

## Schema

| Column | Description |
| --- | --- |
| `messages` | Chat messages. SFT rows contain full trajectories; test rows contain the initial user request. |
| `scenario_json` | Serialized hidden environment state used by the simulator and verifier. |
| `scenario_id` | Stable identifier shared by prompt variants of one semantic scenario. |
| `stage` | Intended pipeline stage: `sft` or `evaluation`. |
| `language` | Prompt language. |
| `prompt_variant` | Language and template provenance for the user request. |
| `trajectory_type` | Procurement route required by the scenario. |
| `difficulty` | Scenario difficulty label. |
| `tool_sequence` | Tool names in the verified reference trajectory. |

Assistant tool arguments are stored as JSON strings under
`messages[].tool_calls[].function.arguments`. Deserialize them into mappings
before applying chat templates that require JSON objects.

Only `messages` should be provided as model input. `scenario_json` contains
hidden state and must not be exposed to the model. `tool_sequence` is available
for analysis but is not the only valid sequence accepted by the environment.

## Task families

- Direct-supplier requests
- Open supplier search
- Compliance-first selection
- Preferred supplier with fallback
- No feasible option

Correctness is based on executed outcomes rather than exact trace matching. The
verifier checks observed evidence, hard constraints and the quality of the
submitted decision.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    "rubenbalbastre/supply-chain-tool-calling",
    "default",
)
```

## Source

Generation code, environment logic and evaluation tools are available in the
[`sft-tool-calling`](https://github.com/rubenbalbastre/sft-tool-calling)
repository.
