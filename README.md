# Multilingual supply-chain tool-calling environment

This project provides a small supply-chain task for supervised fine-tuning (SFT) and, eventually, reinforcement learning with verifiable rewards (RLVR). A model must route a material request through the correct tools, resolve cities to internal plants, ask for clarification when a city has multiple plants, and retain the original request fields across turns.

## Project structure

```text
src/environment/
├── master_data.csv       # Plant master data
└── tools.py              # Deterministic task tools

src/data_generation/
└── generate_sft_data.py  # Scenarios, conversations, and validation

data/
└── pilot/hf_dataset/     # Saved Hugging Face DatasetDict
```

The separation is intentional. `src/environment/` defines what actions mean and what results they produce. `src/data_generation/` decides which situations to sample and turns them into complete training conversations.

## Environment

[`src/environment/master_data.csv`](src/environment/master_data.csv) contains 37 plants across 26 European cities. Each record has the same three fields:

```csv
name,plant_id,city
Valencia Manufacturing,ES-03,Valencia
Valencia Distribution Centre,ES-08,Valencia
```

[`src/environment/tools.py`](src/environment/tools.py) exposes four deterministic tools.

### `check_location`

```python
check_location(city: str)
```

It filters the master data by city. An unknown city naturally returns no records; the environment does not maintain a separate list of missing cities.

```json
{
  "matches": [
    {"name": "Valencia Manufacturing", "plant_id": "ES-03", "city": "Valencia"},
    {"name": "Valencia Distribution Centre", "plant_id": "ES-08", "city": "Valencia"}
  ]
}
```

### `ask_for_clarification`

```python
ask_for_clarification(candidate_plant_ids: list[str])
```

This is a structured assistant action for ambiguous results. A free-form clarification question would be difficult to verify reliably in RLVR. The tool makes the decision checkable: the candidate IDs must be distinct, exist in the master data, and correspond to the records returned by `check_location`.

The tool acknowledges the request and exposes the candidates:

```json
{
  "status": "clarification_requested",
  "candidates": [
    {"name": "Valencia Manufacturing", "plant_id": "ES-03", "city": "Valencia"},
    {"name": "Valencia Distribution Centre", "plant_id": "ES-08", "city": "Valencia"}
  ]
}
```

The simulated user then selects one plant by name. The assistant must map that selection to its plant ID in the final call.

### `can_fulfill_material_request`

```python
can_fulfill_material_request(
    material_id: str,
    quantity: float,
    unit: str,
    required_date: str,
    plant_id: str,
)
```

This is the terminal task action. Its arguments must preserve the material, quantity, unit, and date from the original request and use the plant selected by the routing process.

### `request_new_location`

```python
request_new_location()
```

This structured action asks the user for another city after a lookup has no matching records. Like structured clarification, it makes the behavior directly verifiable without judging free-form assistant wording. The simulated user then provides a replacement city and the normal lookup process starts again.

## Valid trajectories

The correct trajectory follows task invariants rather than a stored reference conversation.

### Explicit plant ID

If the target plant ID is explicit, the assistant calls fulfillment directly. It must not call `check_location`.

```text
user request with ES-03
→ can_fulfill_material_request(plant_id="ES-03", ...)
```

An explicit target ID still takes precedence when a city is also mentioned.

### City with one match

```text
user request for Bilbao
→ check_location(city="Bilbao")
→ one master-data record
→ can_fulfill_material_request(plant_id="ES-09", ...)
```

No clarification is needed.

### City with multiple matches

```text
user request for Valencia
→ check_location(city="Valencia")
→ ES-03 and ES-08
→ ask_for_clarification(candidate_plant_ids=["ES-03", "ES-08"])
→ user selects "Valencia Manufacturing"
→ can_fulfill_material_request(plant_id="ES-03", ...)
```

This is a genuine multi-turn trajectory. The user's selection contains no repeated material details, so the assistant must retain them from the initial request.

### City with no matches

```text
user request for an unknown city
→ check_location(city=<city>)
→ no records
→ request_new_location()
→ user provides a replacement city
→ check_location(city=<replacement city>)
→ resolve one or multiple matches
→ can_fulfill_material_request(plant_id=<resolved plant>, ...)
```

The assistant must not invent a plant ID or call fulfillment before resolving the replacement city. If the replacement city has multiple plants, the clarification trajectory is used before fulfillment.

### Distractor plant ID

An ID mentioned only as historical or unrelated context is not an explicit target. The assistant resolves the actual requested city and must not use the distractor ID.

## Programmatic verification

The generator does not compare a rollout with exact assistant wording. It verifies executable properties of the trajectory:

- `check_location` is called exactly when the request lacks an explicit target plant ID.
- The lookup argument contains the requested city.
- Lookup results equal the records obtained from the master-data CSV.
- `ask_for_clarification` occurs exactly when more than one plant matches.
- Clarification arguments contain exactly the plant IDs returned by the lookup.
- The simulated user's selection is one of those records.
- The final fulfillment plant ID corresponds to the user's selection.
- Material ID, quantity, unit, and required date are unchanged.
- A zero-match lookup is followed by `request_new_location`, a user-provided replacement city, and another lookup before fulfillment.

These invariants allow multiple valid linguistic realizations and make the task suitable for RLVR: reward can be based on whether the actions and state transitions are correct, not whether the output matches a specific training example.

The same rules are used both to build SFT conversations and to evaluate online RLVR episodes through `reset()` and `step(action)`.

## RLVR environment API

[`environment/rlvr.py`](environment/rlvr.py) provides that online abstraction as `SupplyChainEnvironment`. The scenario is hidden task state; only the initial user request and subsequent observations should be shown to the model.

```python
from environment import SupplyChainEnvironment

env = SupplyChainEnvironment(scenario, user_request)
observation = env.reset()

observation, reward, done, info = env.step({
    "name": "check_location",
    "arguments": {"city": "Valencia"},
})
```

Intermediate rewards are configurable when the environment is initialized:

```python
env = SupplyChainEnvironment(
    scenario,
    user_request,
    reward_weights={
        "lookup": 0.25,
        "clarification": 0.25,
        "new_location": 0.1,
    },
    success_reward=1.0,
    failure_reward=0.0,
)
```

The default intermediate weights are `0.2` for a valid city lookup, `0.2` for valid clarification, and `0.2` for requesting a replacement city. On successful completion, the terminal action receives the unallocated portion of `success_reward`. Therefore all successful paths have the same total return even though longer paths receive intermediate feedback:

```text
direct:     1.0
unique:     0.2 lookup + 0.8 fulfillment = 1.0
ambiguous:  0.2 lookup + 0.2 clarification + 0.6 fulfillment = 1.0
no match:   0.2 failed lookup + 0.2 replacement request
            + 0.2 replacement lookup + 0.4 fulfillment = 1.0
```

Weights must be non-negative. Their maximum possible total—two lookups, a replacement-city request, and clarification—cannot exceed `success_reward`. If a later action fails, the final transition claws back previously issued intermediate rewards so the complete episode return equals `failure_reward`. This prevents a model from retaining lookup credit for an ultimately invalid trajectory. `failure_reward` can be negative when failed episodes should receive an additional penalty.

`step()` also accepts an OpenAI-style function call whose `arguments` value is a JSON string. The environment tracks four expected-action states:

```text
expect_lookup
expect_clarification
expect_new_location_request
expect_fulfillment
```

Correct intermediate transitions return their configured reward and the next observation. A complete valid trajectory receives the remaining success reward. A wrong tool, wrong arguments, invented plant, premature fulfillment, or invalid transition terminates with `failure_reward` and a reason in `info`.

## SFT data generation

Generate the small pilot from the repository root:

```bash
python3 -m src.data_generation.generate_sft_data
```

The default configuration saves a native Hugging Face `DatasetDict` under `data/pilot/hf_dataset/` with `train`, `validation`, and `test` splits. Larger runs can be generated explicitly:

```bash
python3 -m src.data_generation.generate_sft_data \
  --train-size 400 \
  --validation-size 50 \
  --test-size 50 \
  --output-dir data/sanity_500/hf_dataset
```

Python owns the scenarios, IDs, routing decisions, tool results, user selections, and expected final arguments. The built-in multilingual `REQUESTS` collection contains 30 phrases per language: 10 simple, 10 medium, and 10 hard. Each scenario stores a seeded `request_variant`, making generation reproducible while exercising different phrasing. These templates are an offline verbalization mechanism and can later be replaced by a teacher LLM without changing environment behavior or target tool calls.

Each row contains six columns:

- `messages`: the complete Hugging Face/OpenAI-style conversation used for SFT;
- `scenario_id`: a stable identifier for debugging;
- `language`: the conversation language;
- `trajectory_type`: the routing pattern;
- `difficulty`: simple, medium, or hard;
- `tool_sequence`: the expected ordered tool names.

Training should consume only `messages`; the remaining lightweight metadata supports filtering and evaluation. Detailed hidden scenario state remains in the generation and RLVR environment rather than being published in the SFT dataset.

Tool arguments are JSON strings under `assistant.tool_calls[].function.arguments`. Tool results use `role: "tool"` and the corresponding `tool_call_id`.

## Training integration check

Before scaling, render pilot conversations with the exact tokenizer revision used for training:

```python
from datasets import load_from_disk
from transformers import AutoTokenizer

dataset = load_from_disk("data/pilot/hf_dataset")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
row = dataset["train"][0]
rendered = tokenizer.apply_chat_template(
    row["messages"],
    tools=YOUR_TOOL_SCHEMAS,
    tokenize=False,
)
print(rendered)
```

To share the saved dataset on the Hugging Face Hub:

```python
from datasets import load_from_disk

dataset = load_from_disk("data/pilot/hf_dataset")
dataset.push_to_hub("your-account/supply-chain-tool-calling")
```

Keep every conversation intact during training. For assistant-only loss, train on assistant tool calls and assistant responses while masking user messages and tool results. Verify the pilot end to end before producing the larger dataset.

## Evaluate an existing model and prompt

[`src/evaluation/evaluate_openai.py`](src/evaluation/evaluate_openai.py) runs an OpenAI model online against freshly generated hidden scenarios. It does not show the model the scenario object or generated target conversation. For every episode it:

1. Sends the initial user request, prompt, and tool schemas to the model.
2. Executes the model's function call through `SupplyChainEnvironment.step()`.
3. Sends the resulting tool observation or simulated user selection back to the model.
4. Continues until the environment succeeds, rejects an action, or reaches the step limit.
5. Stores the action trace, rewards, failure reason, token usage, and latency.

The project virtual environment contains the OpenAI SDK and `python-dotenv`. Put the API key in the repository's ignored `.env` file:

```dotenv
OPENAI_API_KEY=your-key
```

The evaluator calls `load_dotenv()` before creating the OpenAI client. Existing process environment variables take precedence over values in `.env`. Then try GPT-5.4 nano with reasoning effort `none`:

```bash
.venv/bin/python -m src.evaluation.evaluate_openai \
  --model gpt-5.4-nano \
  --reasoning-effort none \
  --episodes 20
```

Evaluation makes paid API calls. Start with a small number of episodes before increasing the sample size.

Each invocation creates the next numbered directory under `data/evals/`:

```text
data/evals/eval-0001/
├── config.json
├── results.json
└── results.jsonl
```

`config.json` records the model, reasoning effort, prompt, tool schemas, seed, episode count, step limit, and UTC creation time. `results.jsonl` contains one complete trace per episode. `results.json` contains the same episode results together with overall and per-trajectory summaries, token totals, and latency. Use the same seed and episode count when comparing prompts or models.

Use `--output-root` only when the numbered runs should be stored somewhere other than `data/evals/`.

### Evaluate a local model

[`src/evaluation/evaluate_local.py`](src/evaluation/evaluate_local.py) supports two local inference backends configured through [`config/eval.yaml`](config/eval.yaml):

- `transformers` loads a model directly in the evaluation process. Use it for CPU runs, debugging, and small evaluations.
- `vllm` calls an already-running OpenAI-compatible vLLM server. Use it for higher-throughput GPU evaluation.

Set the defaults in `config/eval.yaml` or override them with Hydra. To evaluate a saved model directly:

```bash
.venv/bin/python -m src.evaluation.evaluate_local \
  backend=transformers \
  model=outputs/breezy-surf-7/final_model \
  device=auto \
  episodes=20
```

For vLLM, the evaluator starts and stops the server automatically by default.
[`config/vllm.yaml`](config/vllm.yaml) enables automatic tool choice and
selects the `hermes` parser explicitly:

```bash
.venv/bin/python -m src.evaluation.evaluate_local \
  backend=vllm \
  model=outputs/breezy-surf-7/final_model \
  episodes=100
```

Selecting `backend=vllm` always manages the local server: it starts vLLM,
waits for its health endpoint, performs the evaluation, and shuts it down. The
server output is saved as `vllm.log` in the numbered evaluation directory.
The evaluator submits all episodes concurrently while allowing at most
`concurrency` simultaneous model requests (default: `8`). The permit is held
only during inference, so another episode can use the GPU while an earlier one
processes a tool result. Turns remain ordered inside each episode. The
Transformers backend always uses concurrency `1` because it performs inference
directly in the evaluator process.

Both backends execute the same multi-turn environment and write the same `config.json`, `results.json`, and `results.jsonl` files under the next `data/evals/eval-NNNN/` directory. Direct Transformers inference uses the tokenizer's chat and response templates to format tools and parse generated calls. A model without tool-aware templates fails the episode explicitly instead of silently treating free text as a valid action.

SmolLM3 supports vLLM's `hermes` tool-call parser. The evaluator sets
`enable_thinking: false` by default in `config/eval.yaml`. For direct
Transformers inference this is passed to `apply_chat_template`; for vLLM it is
sent as `chat_template_kwargs`. Set it to `true` when reasoning traces are
desired, at the cost of additional output tokens and latency.
