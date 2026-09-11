# Multilingual supply-chain tool-calling environment

This project provides a small supply-chain task for supervised fine-tuning (SFT) and, eventually, reinforcement learning with verifiable rewards (RLVR). A model must route a material request through the correct tools, resolve cities to internal plants, ask for clarification when a city has multiple plants, and retain the original request fields across turns.

## Project structure

```text
environment/
├── master_data.csv       # Plant master data
└── tools.py              # Deterministic task tools

data_generation/
└── generate_sft_data.py  # Scenarios, conversations, and validation

data/
└── pilot/                # Generated train/eval JSONL files
```

The separation is intentional. `environment/` defines what actions mean and what results they produce. `data_generation/` decides which situations to sample and turns them into complete training conversations.

## Environment

[`environment/master_data.csv`](environment/master_data.csv) contains 37 plants across 26 European cities. Each record has the same three fields:

```csv
name,plant_id,city
Valencia Manufacturing,ES-03,Valencia
Valencia Distribution Centre,ES-08,Valencia
```

[`environment/tools.py`](environment/tools.py) exposes four deterministic tools.

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
python3 -m data_generation.generate_sft_data
```

The default configuration writes training and evaluation JSONL files under `data/pilot/`. Larger runs can be generated explicitly:

```bash
python3 -m data_generation.generate_sft_data \
  --train-size 400 \
  --eval-size 100 \
  --output-dir data/sanity_500
```

Python owns the scenarios, IDs, routing decisions, tool results, user selections, and expected final arguments. The built-in multilingual `REQUESTS` templates are only an offline verbalization mechanism for pipeline testing. They can later be replaced by a teacher LLM without changing environment behavior or target tool calls.

Each JSONL row contains:

- `scenario`: the structured source of truth;
- `messages`: the complete conversation in Hugging Face/OpenAI-style chat format.

Tool arguments are JSON strings under `assistant.tool_calls[].function.arguments`. Tool results use `role: "tool"` and the corresponding `tool_call_id`.

## Training integration check

Before scaling, render pilot conversations with the exact tokenizer revision used for training:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/pilot/train.jsonl",
        "eval": "data/pilot/eval.jsonl",
    },
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
row = dataset["train"][0]
rendered = tokenizer.apply_chat_template(
    row["messages"],
    tools=YOUR_TOOL_SCHEMAS,
    tokenize=False,
)
print(rendered)
```

Keep every conversation intact during training. For assistant-only loss, train on assistant tool calls and assistant responses while masking user messages and tool results. Verify the pilot end to end before producing the larger dataset.

## Evaluate an existing model and prompt

[`evaluation/evaluate_openai.py`](evaluation/evaluate_openai.py) runs an OpenAI model online against freshly generated hidden scenarios. It does not show the model the scenario object or generated target conversation. For every episode it:

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
.venv/bin/python -m evaluation.evaluate_openai \
  --model gpt-5.4-nano \
  --reasoning-effort none \
  --episodes 20 \
  --output evaluation/gpt-5.4-nano.jsonl
```

Evaluation makes paid API calls. Start with a small number of episodes before increasing the sample size.

To evaluate a different prompt without changing source code:

```bash
.venv/bin/python -m evaluation.evaluate_openai \
  --prompt-file prompts/my_prompt.txt \
  --episodes 50
```

The JSONL output contains one complete result per episode. A sibling `.summary.json` file reports overall success rate, average return, token totals, total latency, and results grouped by trajectory kind. Use the same seed and episode count when comparing prompts or models.
