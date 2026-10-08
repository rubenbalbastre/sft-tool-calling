# Multilingual procurement tool calling

> [!NOTE]
> **Work in progress.** The executable environment, multilingual dataset
> pipeline, SFT workflow, and model evaluation are functional. On-policy
> distillation and larger-scale experiments are still under active development.

Published models and experiment artifacts are collected on
[Hugging Face](https://huggingface.co/collections/rubenbalbastre/2b-tool-calling-using-sft).

## Project description

This project explores how supervised fine-tuning and on-policy distillation can
improve a small language model's ability to use tools reliably. The target task
is stateful, multi-turn procurement: a model must decide which information it
needs, call the appropriate functions, use returned values in later calls, and
finish with a valid purchasing decision.

The purpose of fine-tuning is not to teach one fixed workflow. It is to improve
the model's ability to select different tool routes from the user's constraints,
preserve arguments across turns, recover from tool errors, and make a grounded
decision from the evidence it has collected. Task success is measured by the
executable environment alongside token-level validation loss.

The project connects the complete experimentation loop:

```text
seeded scenarios → multilingual trajectories → LoRA SFT / OPD
       ↓                                         ↓
SQLite environment ← tool calls ← model evaluation and verification
```

## Fine-tuned models

The current main experiment fine-tunes
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
with LoRA supervised fine-tuning on the generated multilingual tool
trajectories.

| Experiment | Base model | Method | Status |
| --- | --- | --- | --- |
| [`gemma-4-E2B-it-sft`](https://huggingface.co/rubenbalbastre/procurement-function-calling-gemma-4-E2B-it-sft) | `google/gemma-4-E2B-it` | LoRA SFT | Experimental |

Published checkpoints are listed in the
[project's Hugging Face collection](https://huggingface.co/collections/rubenbalbastre/2b-tool-calling-using-sft).

## Results

Results use the same seeded sample of 50 scenario–prompt pairs from the
held-out `test` split and the executable environment evaluator. Repeated runs
are averaged; the range beside success shows observed run-to-run variation.

These preliminary runs predate the template-level holdout. The next experiment
will train on templates 1–10, validate on templates 11–15, and report final
results on unseen templates 16–25.

| Model | Reasoning | Training tokens | Task success | Average return | Mean episode latency |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | Disabled | — | 20% (18–22%) | 0.262 | 15.1 s |
| Base | Enabled | — | 36% (34–38%) | 0.474 | 100.6 s |
| LoRA SFT (48 steps) | Disabled | — | 33% (30–36%) | 0.542 | 21.6 s |

The 48-step checkpoint's Trainer state reports no input-token count, so that
value is not included.

Task-level cells report `success rate / average return`.

| Model | Reasoning | Compliance first | Direct supplier | No feasible option | Open search | Preferred with fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | Disabled | 46.2% / 0.549 | 37.5% / 0.283 | 0% / -0.028 | 12.5% / 0.492 | 0% / 0.000 |
| Base | Enabled | 38.5% / 0.716 | 81.3% / 0.663 | 5.6% / 0.056 | 18.8% / 0.303 | 37.5% / 0.512 |
| LoRA SFT (48 steps) | Disabled | 34.6% / 0.890 | 68.8% / 0.550 | 50.0% / 0.500 | 25.0% / 0.506 | 0% / 0.214 |

The 48-step SFT checkpoint improves non-thinking average success by 13
percentage points and average return by 0.280. It nearly matches the
thinking-enabled base model's success and exceeds its average return, while its
mean episode latency is less than one quarter as large. Its positive return on
preferred-supplier fallback tasks, despite no complete successes, reflects
intermediate rewards for useful actions.

| Model | Reasoning | Input tokens | Output tokens | Total tokens |
| --- | --- | ---: | ---: | ---: |
| Base | Disabled | 807,913 | 19,439 | 827,351 |
| Base | Enabled | 1,117,125 | 132,650 | 1,249,774 |
| LoRA SFT (48 steps) | Disabled | 1,044,139 | 22,521 | 1,066,660 |

Thinking increases output tokens substantially and raises mean latency by more
than six times over the non-thinking base. The 48-step SFT model averages 10.07
steps per episode, so its stronger behavior also carries more tool context and
slightly higher latency than the non-thinking base.

## Environment description

The environment simulates supplier selection for constrained material requests.
A model can search suppliers, inspect profiles, request quotes, retrieve delivery
options, submit a procurement plan, or report that no feasible option exists.
Tasks vary between direct supplier requests, open searches, compliance-first
decisions, preferred-supplier fallback, and infeasible cases.

Unlike a static function-calling benchmark, correctness is determined by
executing the model's actions. The verifier checks observed evidence,
constraint satisfaction, and decision quality instead of requiring one exact
gold sequence.

Supplier master data lives in SQLite. Episode-specific price, availability,
readiness, transport cost, arrival date, reliability, and emissions are derived
deterministically from a seed. This makes experiments reproducible while still
requiring the model to discover the state through tool calls. The interface
follows the familiar Gymnasium `reset`/`step` shape and uses structured tool-call
dictionaries as actions.

![Procurement tool-calling environment](docs/assets/environment-overview.svg)

## Techniques used

- **Data generation:** leakage-safe Hugging Face dataset splits, verified
  reference trajectories, and multiple prompt templates in English, Spanish,
  German, and French.
- **Post-training:** assistant-only supervised loss, LoRA adapters through PEFT,
  early stopping, and an experimental TRL on-policy distillation stage.
- **Evaluation:** closed-loop environment rollouts with task success, return,
  full traces, token usage, and latency—not only validation loss.
- **Inference:** local evaluation with Transformers or vLLM and remote evaluation
  with OpenAI models through the same task semantics.
- **Experiment management:** Hydra configuration, deterministic seeds, Weights &
  Biases tracking, checkpointing, and reproducible RunPod execution.
- **Tool integration:** model-specific chat templates, generated JSON schemas,
  multi-turn tool state, dynamic validation batching, and native LoRA serving in
  vLLM.

The generated training and evaluation splits are published as the
[`rubenbalbastre/supply-chain-tool-calling`](https://huggingface.co/datasets/rubenbalbastre/supply-chain-tool-calling)
dataset.

## Quick start

Run commands from the repository root with the project environment activated.

```bash
python generate_data.py
python -m src.evaluation.evaluate_local episodes=100 seed=1234
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Use `./scripts/create-env.sh` to create `.venv` and install the declared
dependencies first.

## Entry points

| Entry point | Configuration | Purpose |
| --- | --- | --- |
| `generate_data.py` | `config/data_generation.yaml` | Create the procurement SFT, OPD, and test dataset |
| `train_sft.py` | `config/train_sft.yaml` | Supervised fine-tuning |
| `train_opd.py` | `config/train_opd.yaml` | On-policy distillation |
| `python -m src.evaluation.evaluate_local` | `config/eval.yaml` | Evaluate with Transformers or an automatically managed vLLM server |
| `python -m src.evaluation.evaluate_openai` | CLI arguments | Evaluate an OpenAI model |

Hydra entry points accept command-line overrides, for example:

```bash
python generate_data.py splits.sft_train=40 splits.opd_train=40
python -m src.evaluation.evaluate_local backend=vllm episodes=100
```

## Documentation

- [Environment and verifiable trajectories](docs/environment.md)
- [Dataset generation and Hugging Face publishing](docs/data-generation.md)
- [Supervised fine-tuning](docs/sft_training.md)
- [On-policy distillation](docs/opd_training.md)
- [Local and OpenAI evaluation](docs/evaluation.md)
- [RunPod experiment environment](docs/runpod.md)

The procurement environment lives in
[`src/environment/procurement`](src/environment/procurement), and its SQLite
database is created automatically below `data/environment/`. Generated
artifacts are written below `data/` or `outputs/`.
