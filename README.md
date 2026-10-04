# Procurement tool-calling environment

A small executable environment for evaluating models on procurement option
selection. Models research suppliers, request seeded quotes, inspect delivery
options, and submit a feasible near-optimal plan. Prompts require different
routes rather than one fixed sequence of tool calls.

The SQLite master data is fixed while quote and delivery conditions are
deterministically derived from each episode seed. The verifier scores final
feasibility, utility, evidence, and route-specific requirements without
comparing against a gold trajectory.

The dataset generator uses the same scenarios, tools, and verifier. It currently
creates deterministic reference trajectories; these can later be replaced or
augmented with filtered teacher rollouts without changing the dataset schema.

## Quick start

Run commands from the repository root with the project environment activated.

```bash
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
- [SFT and on-policy training](docs/training.md)
- [Local and OpenAI evaluation](docs/evaluation.md)

The procurement environment lives in
[`src/environment/procurement`](src/environment/procurement), and its SQLite
database is created automatically below `data/environment/`. Generated
artifacts are written below `data/` or `outputs/`.
