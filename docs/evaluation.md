# Evaluation

Both local and OpenAI evaluation load held-out scenario variants from the
Hugging Face dataset configured under `dataset` in `config/eval.yaml`. The
default evaluates the complete `test` split at `data/pipeline/hf_dataset/`.
`episodes` can limit the number of scenario–variant pairs; `seed`
deterministically controls their order and subset.

The default is equivalent to:

```yaml
dataset:
  split: test
episodes: null
```

With the default generated dataset, this evaluates 50 scenarios across 8
held-out multilingual prompt variants, for 400 episodes. This is substantially
more expensive than evaluating all 50 scenarios with one fixed prompt variant.

Both evaluators run fresh seeded procurement scenarios through the same
environment. Results are written to numbered directories:

```text
data/evals/eval-0001/
├── config.json
├── results.json
├── results.jsonl
└── vllm.log          # vLLM runs only
```

`results.jsonl` contains prompts, complete episode traces, terminal feasibility
and utility metrics. `results.json` adds aggregate and per-task-type metrics,
average steps, token totals, and mean, median, p95, and aggregate episode
latency.

## OpenAI models

Set `OPENAI_API_KEY` in the shell or `.env`, then run:

```bash
python -m src.evaluation.evaluate_openai \
  --model gpt-5.4-nano \
  --reasoning-effort none \
  --temperature 0 \
  --episodes 100
```

This performs paid API calls. Start with a small episode count. Use the same
seed and episode count when comparing models or prompts.

The OpenAI evaluator reads its defaults from `config/eval.yaml`. It shares
`openai_model`, `episodes`, `seed`, `max_steps`, `reasoning_effort`,
`temperature`, and `output_root` with that configuration; command-line options
override them. Local-only backend, device, concurrency, quantization, and chat
template settings are ignored.

## Local models

[`evaluate_local.py`](../src/evaluation/evaluate_local.py) reads
[`config/eval.yaml`](../config/eval.yaml) and supports two backends.

### Transformers

The model is loaded directly into the evaluator. This is useful for CPU runs
and debugging and processes one inference request at a time.

```bash
python -m src.evaluation.evaluate_local \
  backend=transformers \
  model=outputs/my-run/final_model \
  device=auto \
  episodes=100
```

### vLLM

When `backend=vllm`, the evaluator automatically starts a local server, waits
for it, evaluates the model through its OpenAI-compatible HTTP API, and stops
the server afterward.

```bash
python -m src.evaluation.evaluate_local \
  backend=vllm \
  model=outputs/my-run/final_model \
  episodes=100
```

[`config/vllm.yaml`](../config/vllm.yaml) enables prefix caching and automatic
tool choice. With `tool_call_parser=auto`, the launcher selects Gemma 4's native
parser for Gemma 4 models and Hermes otherwise. Complete episodes run
concurrently up to `concurrency`; turns remain ordered within each episode. An
episode's latency timer starts only after it enters this pool, so waiting behind
earlier episodes is excluded while its model calls and tool loop are included.
When the selected local path contains `adapter_config.json`, the evaluator
serves its recorded base model and mounts the directory as a native vLLM LoRA
adapter. A merged checkpoint is not required.

## Thinking and quantization

`enable_thinking` and `reasoning_effort` are passed to the tokenizer chat
template by both local backends. Their exact effect is model-specific. The
defaults explicitly disable thinking and select `none` reasoning effort:

```yaml
enable_thinking: false
reasoning_effort: "none"
temperature: 0.0
```

`temperature` controls decoding randomness for both local backends. At `0.0`,
Transformers uses greedy decoding and vLLM receives zero temperature. A positive
value enables sampling in Transformers.

The OpenAI evaluator exposes reasoning effort through its existing
`--reasoning-effort` argument; `enable_thinking` is a local chat-template
option and is not sent to the OpenAI Responses API.

For vLLM, `quantization=bnb_4bit` applies BitsAndBytes 4-bit quantization while
loading an ordinary checkpoint. `quantization=none` applies no override.
Checkpoints that already declare their quantization in `config.json` should be
loaded with `quantization=none`; vLLM detects their stored format.
