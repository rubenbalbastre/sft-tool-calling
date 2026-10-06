"""Build procurement datasets from trajectories verified by the environment."""

import json
from copy import deepcopy

from src.environment.procurement import (
    LANGUAGES,
    ProcurementEnvironment,
    TEMPLATES_PER_LANGUAGE,
    generate_scenarios,
    prompt_variants,
)


def _tool_call(call_id, action):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": action["name"],
            "arguments": json.dumps(action["arguments"], ensure_ascii=False),
        },
    }


def _take_action(env, messages, action):
    """Execute one action and append its assistant/tool message pair."""
    call_id = f"call_{len(env.actions) + 1}"
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [_tool_call(call_id, action)],
    })
    observation, _, terminated, truncated, info = env.step(action)
    messages.append({
        "role": "tool",
        "name": action["name"],
        "tool_call_id": call_id,
        "content": json.dumps(observation["content"], ensure_ascii=False),
    })
    return observation, terminated or truncated, info


def _quote_and_deliver(env, messages, scenario, supplier_id):
    """Observe one supplier quote and all of its delivery options."""
    quote_observation, _, _ = _take_action(env, messages, {
        "name": "request_quote",
        "arguments": {
            "supplier_id": supplier_id,
            "material_id": scenario["material_id"],
            "quantity": scenario["quantity"],
            "unit": scenario["unit"],
            "required_date": scenario["required_date"],
        },
    })
    quote_id = quote_observation["content"]["quote_id"]
    _take_action(env, messages, {
        "name": "get_delivery_options",
        "arguments": {
            "quote_id": quote_id,
            "destination": scenario["destination"],
        },
    })


def build_reference_trajectory(scenario):
    """Research all relevant options and submit the best verified result."""
    env = ProcurementEnvironment(scenario)
    messages = [{"role": "user", "content": scenario["user_request"]}]
    kind = scenario["task_type"]

    if kind == "direct_supplier":
        supplier_ids = [scenario["requested_supplier_id"]]
    elif kind == "preferred_with_fallback":
        preferred = scenario["preferred_supplier_id"]
        _quote_and_deliver(env, messages, scenario, preferred)
        if env.observed_feasible_options():
            supplier_ids = []
        else:
            _take_action(env, messages, {
                "name": "search_suppliers",
                "arguments": {
                    "material_id": scenario["material_id"],
                    "countries": scenario["allowed_countries"],
                },
            })
            supplier_ids = sorted(env.known_suppliers - {preferred})
    else:
        _take_action(env, messages, {
            "name": "search_suppliers",
            "arguments": {
                "material_id": scenario["material_id"],
                "countries": scenario["allowed_countries"],
            },
        })
        supplier_ids = sorted(env.known_suppliers)

    if kind == "compliance_first":
        certified_supplier_ids = []
        for supplier_id in supplier_ids:
            profile_observation, _, _ = _take_action(env, messages, {
                "name": "get_supplier_profile",
                "arguments": {"supplier_id": supplier_id},
            })
            certifications = profile_observation["content"]["certifications"]
            if all(
                required in certifications
                for required in scenario["required_certifications"]
            ):
                certified_supplier_ids.append(supplier_id)
        supplier_ids = certified_supplier_ids

    for supplier_id in supplier_ids:
        _quote_and_deliver(env, messages, scenario, supplier_id)

    observed = env.observed_feasible_options()

    if observed:
        best = max(observed, key=lambda option: option["utility"])
        final_action = {
            "name": "submit_procurement_plan",
            "arguments": {
                "quote_id": best["quote_id"],
                "delivery_option_id": best["delivery_option_id"],
            },
        }
    else:
        final_action = {"name": "report_no_feasible_option", "arguments": {}}

    _, done, info = _take_action(env, messages, final_action)
    if not done or not info["success"]:
        raise RuntimeError(f"Reference trajectory failed for {scenario['scenario_id']}")
    return messages


def generate(
    count,
    split,
    seed,
    languages=LANGUAGES,
    templates_per_language=TEMPLATES_PER_LANGUAGE,
):
    """Generate scenarios, then expand them into grouped prompt variants."""
    rows = []
    for scenario in generate_scenarios(count, split, seed):
        reference_messages = build_reference_trajectory(scenario)
        for variant in prompt_variants(
            scenario, languages, templates_per_language
        ):
            messages = deepcopy(reference_messages)
            messages[0]["content"] = variant["user_request"]
            rows.append({"scenario": variant, "messages": messages})
    return rows


def _normalise_messages(messages):
    """Give every split the same Arrow schema for chat messages."""
    return [
        {
            "role": message["role"],
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls", []),
            "name": message.get("name", ""),
            "tool_call_id": message.get("tool_call_id", ""),
        }
        for message in messages
    ]


def to_pipeline_row(row, stage):
    """Create an SFT conversation or prompt-only online-training example."""
    scenario = row["scenario"]
    full_messages = row["messages"]
    messages = full_messages if stage == "sft" else full_messages[:1]
    return {
        "messages": _normalise_messages(messages),
        "scenario_json": json.dumps(scenario, ensure_ascii=False, sort_keys=True),
        "scenario_id": scenario["scenario_id"],
        "stage": stage,
        "language": scenario["language"],
        "prompt_variant": scenario["prompt_variant"],
        "trajectory_type": scenario["task_type"],
        "difficulty": scenario["difficulty"],
        "tool_sequence": [
            call["function"]["name"]
            for message in full_messages
            for call in message.get("tool_calls", [])
        ],
    }


def build_pipeline_dataset(
    split_sizes,
    seed,
    languages=LANGUAGES,
    templates_per_language=TEMPLATES_PER_LANGUAGE,
):
    """Build split-safe prompt variants for SFT, online training, and evaluation."""
    from datasets import Dataset, DatasetDict, Features, List, Value

    expected_splits = (
        "sft_train",
        "sft_validation",
        "opd_train",
        "opd_validation",
        "test",
    )
    unknown = set(split_sizes) - set(expected_splits)
    missing = set(expected_splits) - set(split_sizes)
    if unknown or missing:
        raise ValueError(
            f"split_sizes must contain exactly {expected_splits}; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    features = Features({
        "messages": List({
            "role": Value("string"),
            "content": Value("string"),
            "tool_calls": List({
                "id": Value("string"),
                "type": Value("string"),
                "function": {
                    "name": Value("string"),
                    "arguments": Value("string"),
                },
            }),
            "name": Value("string"),
            "tool_call_id": Value("string"),
        }),
        "scenario_json": Value("string"),
        "scenario_id": Value("string"),
        "stage": Value("string"),
        "language": Value("string"),
        "prompt_variant": Value("string"),
        "trajectory_type": Value("string"),
        "difficulty": Value("string"),
        "tool_sequence": List(Value("string")),
    })

    datasets = {}
    for offset, split in enumerate(expected_splits, start=1):
        stage = "sft" if split.startswith("sft_") else (
            "opd" if split.startswith("opd_") else "evaluation"
        )
        rows = generate(
            int(split_sizes[split]),
            split,
            seed + offset,
            languages=languages,
            templates_per_language=templates_per_language,
        )
        datasets[split] = Dataset.from_list(
            [to_pipeline_row(row, stage) for row in rows], features=features
        )
    return DatasetDict(datasets)
