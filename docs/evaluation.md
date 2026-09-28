# Evaluation

Both evaluators run fresh hidden scenarios through the same multi-turn
environment. Results are written to numbered directories:

```text
data/evals/eval-0001/
├── config.json
├── results.json
├── results.jsonl
└── vllm.log          # vLLM runs only
```

`results.jsonl` contains complete episode traces. `results.json` adds aggregate
and per-trajectory metrics, token totals, and latency.

## OpenAI models

Set `OPENAI_API_KEY` in the shell or `.env`, then run:

```bash
python -m src.evaluation.evaluate_openai \
  --model gpt-5.4-nano \
  --reasoning-effort none \
  --episodes 20
```

This performs paid API calls. Start with a small episode count. Use the same
seed and episode count when comparing models or prompts.

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
  episodes=20
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

`enable_thinking: false` disables reasoning output through the tokenizer chat
template or vLLM `chat_template_kwargs`.

For vLLM, `quantization=bnb_4bit` applies BitsAndBytes 4-bit quantization while
loading an ordinary checkpoint. `quantization=none` applies no override.
Checkpoints that already declare their quantization in `config.json` should be
loaded with `quantization=none`; vLLM detects their stored format.
