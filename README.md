# Multilingual supply-chain tool-calling SFT data

This project generates complete, deterministic conversations for fine-tuning SmolLM3-3B to route material-fulfillment requests. It covers direct calls with explicit plant IDs, unique/multiple/zero city matches, explicit-ID precedence, distractors, clarification, and state retention in English, Spanish, German, and French.

The project separates the task from dataset construction:

- [`environment/`](environment) contains the plant master table and deterministic tool implementations.
- [`data_generation/`](data_generation) contains scenario sampling, request templates, conversation construction, and validation.

`environment/master_data.csv` currently contains 37 plants across 26 European cities; every record has `name`, `plant_id`, and `city`. A natural-language city produces `check_location({"city": "..."})`; the environment filters the CSV data by that field and returns all matching records.

## Generate the 50-example pilot

```bash
python3 -m data_generation.generate_sft_data
```

This writes 40 training examples and 10 evaluation examples under `data/pilot/`. Evaluation scenarios use a separate random seed and are created before training scenarios. Use a new output directory when scaling so the pilot remains frozen:

```bash
python3 -m data_generation.generate_sft_data --train-size 400 --eval-size 100 --output-dir data/sanity_500
python3 -m data_generation.generate_sft_data --train-size 2700 --eval-size 300 --output-dir data/full_3000
```

Python owns all IDs, records, arguments, routing, match results, and selected plants. The small built-in phrase set is deliberately just a pipeline pilot. When scaling, replace `verbalize()` with a teacher-model call while keeping the scenario and conversation builder deterministic.

## Data format

Each JSONL row contains:

- `scenario`: the semantic source of truth and split/family metadata;
- `messages`: the full conversation in Hugging Face/OpenAI-style chat form.

Tool arguments are JSON strings inside `assistant.tool_calls[].function.arguments`; tool results use `role: "tool"` and the matching `tool_call_id`. Multiple-match rows preserve the original request across a lookup, tool result, assistant question, and short user clarification before the final fulfillment call.

## Training integration check

Before scaling, load roughly 50 rows and render each with the exact tokenizer version used for training:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("json", data_files={"train": "data/pilot/train.jsonl", "eval": "data/pilot/eval.jsonl"})
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
row = ds["train"][0]
rendered = tok.apply_chat_template(row["messages"], tools=YOUR_TOOL_SCHEMAS, tokenize=False)
print(rendered)
```

Pin the model/tokenizer revision after this succeeds. In TRL, keep each conversation intact and use conversational examples rather than flattening turns. Verify assistant-only loss masking against rendered tokens: the desired labels are assistant clarification text and assistant tool calls, not user or tool-result tokens. Run a small overfit/sanity job on the pilot, inspect generations for every trajectory type, then generate the 500-example run.

The generator validates lookup counts, zero-match behavior, multi-match clarification, and exact final fulfillment arguments.
