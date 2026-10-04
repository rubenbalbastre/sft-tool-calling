# Evaluation

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
token totals, and latency.

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

[`config/vllm.yaml`](../config/vllm.yaml) enables prefix caching, automatic tool
choice, and the `hermes` tool-call parser. Requests from different episodes run
concurrently up to `concurrency`; turns remain ordered within each episode.

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
