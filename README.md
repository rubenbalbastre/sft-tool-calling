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
| `gemma-4-E2B-it-sft` | `google/gemma-4-E2B-it` | LoRA SFT | Experimental |

Published checkpoints are listed in the
[project's Hugging Face collection](https://huggingface.co/collections/rubenbalbastre/2b-tool-calling-using-sft).

## Results

Results use 50 episodes from the held-out `test` split, with 10 episodes from
each task type, using the executable environment evaluator.

| Model | Training stage | Episodes | Task success | Average return |
| --- | --- | ---: | ---: | ---: |
| `google/gemma-4-E2B-it` | Baseline | 50 | 34% | 0.325 |
| `gemma-4-E2B-it-sft` | LoRA SFT, final checkpoint | 50 | 40% | 0.452 |
| `gemma-4-E2B-it-sft-opd` | LoRA OPD | — | — | WIP |

| Task type | Baseline success | SFT success | Baseline return | SFT return |
| --- | ---: | ---: | ---: | ---: |
| Compliance first | 40% | 50% | 0.352 | 0.630 |
| Direct supplier | 60% | 60% | 0.524 | 0.524 |
| No feasible option | 0% | 20% | -0.050 | 0.180 |
| Open search | 70% | 70% | 0.799 | 0.697 |
| Preferred with fallback | 0% | 0% | 0.000 | 0.226 |

The SFT checkpoint improves overall success by 6 percentage points and average
return by 0.127. Its positive return on preferred-supplier fallback tasks,
despite no complete successes, reflects intermediate rewards for useful actions.

| Model | Input tokens | Output tokens | Total tokens | Latency |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 487,802 | 13,934 | 501,736 | 841.6 s |
| LoRA SFT | 753,206 | 17,410 | 770,616 | 1,322.5 s |

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
| `generate_data.py` | `config/data_generation.yaml` | Create the procurement SFT/OPD pilot dataset |
| `train_sft.py` | `config/train_sft.yaml` | Supervised fine-tuning |
| `train_opd.py` | `config/train_opd.yaml` | On-policy distillation |
| `python -m src.evaluation.evaluate_local` | `config/eval.yaml` | Evaluate with Transformers or an automatically managed vLLM server |
| `python -m src.evaluation.evaluate_openai` | CLI arguments | Evaluate an OpenAI model |

Hydra entry points accept command-line overrides, for example:

```bash
python generate_data.py splits.sft_train=40 splits.opd_train=20
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
