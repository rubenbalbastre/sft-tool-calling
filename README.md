# Multilingual supply-chain tool calling

A small environment for training and evaluating models on multilingual,
multi-turn tool calling. The model resolves a requested city to a plant, asks
for structured clarification when necessary, and submits a final material
fulfilment request.

The same deterministic rules support two stages:

- supervised fine-tuning (SFT) on complete trajectories;
- on-policy distillation on prompt-only scenarios.

## Quick start

Run commands from the repository root with the project environment activated.

```bash
python generate_data.py
python train_sft.py
python train_opd.py
```

Evaluate a saved model:

```bash
python -m src.evaluation.evaluate_local
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Entry points

| Entry point | Configuration | Purpose |
| --- | --- | --- |
| `generate_data.py` | `config/data_generation.yaml` | Create and optionally publish the Hugging Face dataset |
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

The plant master data is in
[`src/environment/master_data.csv`](src/environment/master_data.csv), and all
generated artifacts are written below `data/` or `outputs/`.
