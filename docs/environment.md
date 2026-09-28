# Environment and verifiable trajectories

The environment separates deterministic task behavior from data generation.
Plant records live in [`master_data.csv`](../src/environment/master_data.csv),
tools in [`tools.py`](../src/environment/tools.py), and the stateful verifier in
[`env.py`](../src/environment/env.py).

## Tools

- `check_location(city)` returns matching `{name, plant_id, city}` records. An
  unknown city returns an empty list.
- `ask_for_clarification(candidate_plant_ids)` makes ambiguity handling a
  structured, verifiable action. The simulated user then selects a plant.
- `request_new_location()` requests another city after a lookup returns no
  records.
- `can_fulfill_material_request(...)` is the terminal action. It must preserve
  the material, quantity, unit, and required date and use the resolved plant ID.

## Valid trajectories

```text
explicit plant ID
→ can_fulfill_material_request

city with one match
→ check_location
→ can_fulfill_material_request

city with multiple matches
→ check_location
→ ask_for_clarification
→ user selects a plant
→ can_fulfill_material_request

city with no matches
→ check_location
→ request_new_location
→ user provides another city
→ check_location
→ optional clarification
→ can_fulfill_material_request
```

A plant ID mentioned only as unrelated history is a distractor and must not be
used as the target.

## Verification and rewards

[`SupplyChainEnvironment`](../src/environment/env.py) validates actions from
task state rather than comparing them with one reference response. It checks
the tool order, city, candidate IDs, user selection, final plant, and preservation
of all original request fields. Invalid actions terminate the episode with a
reason in `info`.

```python
from src.environment.env import SupplyChainEnvironment

env = SupplyChainEnvironment(scenario, user_request)
observation = env.reset()
observation, reward, done, info = env.step({
    "name": "check_location",
    "arguments": {"city": "Valencia"},
})
```

Intermediate reward weights are configurable:

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

The terminal reward is the unallocated part of `success_reward`, so every valid
trajectory has the same total return. If a later action fails, prior intermediate
rewards are clawed back and the episode return becomes `failure_reward`.

The executable verifier accepts different linguistic trajectories as long as
their structured actions are correct.
