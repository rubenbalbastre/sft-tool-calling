"""Small deterministic generator for multilingual tool-calling conversations."""

import json
import random
from datetime import date, timedelta

from src.environment.tools import (
    MASTER_DATA,
    ask_for_clarification,
    check_location,
    request_new_location,
)
from src.data_generation.constants import (
    DIFFICULTIES,
    DISTRACTOR_PREFIXES,
    KINDS,
    LANGUAGES,
    REQUESTS,
)


def balanced_sample(count, weighted_values, rng):
    """Take an approximately proportional sample, including small pilot sets."""
    labels = []
    for index in range(count):
        labels.append(weighted_values[index * len(weighted_values) // count])
    rng.shuffle(labels)
    return labels


def make_scenario(index, split, kind, difficulty, rng):
    language = LANGUAGES[index % len(LANGUAGES)]
    scenario = {
        "scenario_id": f"{split}_{index:05d}",
        "language": language,
        "kind": kind,
        "difficulty": difficulty,
        "request_variant": rng.randrange(10),
        "material_id": f"MAT-{rng.randint(1000, 9999)}",
        "quantity": rng.randint(10, 5000),
        "unit": rng.choice(["kg", "units"]),
        "required_date": (date(2026, 10, 1) + timedelta(days=rng.randint(1, 365))).isoformat(),
        "city": None,
        "matches": [],
        "replacement_city": None,
        "replacement_matches": [],
        "explicit_plant_id": None,
        "selected_plant_id": None,
        "selected_plant_name": None,
        "distractor_plant_id": None,
    }

    if kind in ("explicit", "explicit_with_city"):
        plant = rng.choice(MASTER_DATA)
        scenario["explicit_plant_id"] = plant["plant_id"]
        scenario["selected_plant_id"] = plant["plant_id"]
        scenario["selected_plant_name"] = plant["name"]
        if kind == "explicit_with_city":
            scenario["city"] = plant["city"]
    elif kind == "missing":
        scenario["city"] = rng.choice(["Albor", "Monteluz", "Belle-Rive"])
        scenario["matches"] = check_location(scenario["city"])["matches"]
        cities = sorted({plant["city"] for plant in MASTER_DATA})
        scenario["replacement_city"] = rng.choice(cities)
        replacement_matches = check_location(scenario["replacement_city"])["matches"]
        selected = rng.choice(replacement_matches)
        scenario["replacement_matches"] = replacement_matches
        scenario["selected_plant_id"] = selected["plant_id"]
        scenario["selected_plant_name"] = selected["name"]
    else:
        cities = sorted({plant["city"] for plant in MASTER_DATA})
        choices = [city for city in cities if len(check_location(city)["matches"]) == (1 if kind == "unique" else 2)]
        if kind == "distractor":
            choices = cities
        city = rng.choice(choices)
        matches = check_location(city)["matches"]
        plant = rng.choice(matches)
        scenario["city"] = city
        scenario["matches"] = matches
        scenario["selected_plant_id"] = plant["plant_id"]
        scenario["selected_plant_name"] = plant["name"]
        if kind == "distractor":
            valid_ids = {p["plant_id"] for p in matches}
            other_ids = [p["plant_id"] for p in MASTER_DATA if p["plant_id"] not in valid_ids]
            scenario["distractor_plant_id"] = rng.choice(other_ids)
    return scenario


def verbalize(scenario):
    level = {"simple": 0, "medium": 1, "hard": 2}[scenario["difficulty"]]
    template_index = level * 10 + scenario["request_variant"]
    if scenario["kind"] == "explicit":
        target = scenario["explicit_plant_id"]
    elif scenario["kind"] == "explicit_with_city":
        target = f'{scenario["city"]} ({scenario["explicit_plant_id"]})'
    else:
        target = scenario["city"]

    request = REQUESTS[scenario["language"]][template_index].format(
        target=target,
        quantity=scenario["quantity"],
        unit=scenario["unit"],
        material=scenario["material_id"],
        date=scenario["required_date"],
    )
    if scenario["distractor_plant_id"]:
        request = DISTRACTOR_PREFIXES[scenario["language"]].format(
            plant_id=scenario["distractor_plant_id"]
        ) + request
    return request


def call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def build_conversation(scenario):
    messages = [{"role": "user", "content": verbalize(scenario)}]
    direct = scenario["explicit_plant_id"] is not None

    if not direct:
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [call("location_1", "check_location", {"city": scenario["city"]})],
        })
        messages.append({
            "role": "tool",
            "name": "check_location",
            "tool_call_id": "location_1",
            "content": json.dumps({"matches": scenario["matches"]}, ensure_ascii=False),
        })

        if not scenario["matches"]:
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [call(
                    "new_location_1", "request_new_location", {}
                )],
            })
            messages.append({
                "role": "tool",
                "name": "request_new_location",
                "tool_call_id": "new_location_1",
                "content": json.dumps(request_new_location()),
            })
            messages.append({
                "role": "user",
                "content": scenario["replacement_city"],
            })
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [call(
                    "location_2", "check_location",
                    {"city": scenario["replacement_city"]},
                )],
            })
            messages.append({
                "role": "tool",
                "name": "check_location",
                "tool_call_id": "location_2",
                "content": json.dumps(
                    {"matches": scenario["replacement_matches"]},
                    ensure_ascii=False,
                ),
            })

        final_matches = scenario["replacement_matches"] or scenario["matches"]
        if len(final_matches) > 1:
            candidate_ids = [plant["plant_id"] for plant in final_matches]
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [call(
                    "clarification_1",
                    "ask_for_clarification",
                    {"candidate_plant_ids": candidate_ids},
                )],
            })
            messages.append({
                "role": "tool",
                "name": "ask_for_clarification",
                "tool_call_id": "clarification_1",
                "content": json.dumps(
                    ask_for_clarification(candidate_ids), ensure_ascii=False
                ),
            })
            messages.append({"role": "user", "content": scenario["selected_plant_name"]})

    arguments = {
        "material_id": scenario["material_id"],
        "quantity": scenario["quantity"],
        "unit": scenario["unit"],
        "required_date": scenario["required_date"],
        "plant_id": scenario["selected_plant_id"],
    }
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [call("fulfillment_1", "can_fulfill_material_request", arguments)],
    })
    return messages


def validate(scenario, messages):
    calls = [c for message in messages for c in message.get("tool_calls", [])]
    names = [c["function"]["name"] for c in calls]
    direct = scenario["explicit_plant_id"] is not None
    expected_lookups = 0 if direct else (2 if scenario["kind"] == "missing" else 1)
    assert names.count("check_location") == expected_lookups
    if not direct:
        lookups = [c for c in calls if c["function"]["name"] == "check_location"]
        assert json.loads(lookups[0]["function"]["arguments"]) == {"city": scenario["city"]}
        assert scenario["matches"] == check_location(scenario["city"])["matches"]
        assert all(set(plant) == {"name", "plant_id", "city"} for plant in scenario["matches"])
        if scenario["kind"] == "missing":
            assert json.loads(lookups[1]["function"]["arguments"]) == {
                "city": scenario["replacement_city"]
            }
            assert scenario["replacement_matches"] == check_location(
                scenario["replacement_city"]
            )["matches"]

    final_matches = scenario["replacement_matches"] or scenario["matches"]
    clarification_calls = [
        c for c in calls if c["function"]["name"] == "ask_for_clarification"
    ]
    if len(final_matches) > 1:
        assert len(clarification_calls) == 1
        clarification_args = json.loads(
            clarification_calls[0]["function"]["arguments"]
        )
        expected_candidates = [p["plant_id"] for p in final_matches]
        assert clarification_args == {"candidate_plant_ids": expected_candidates}
    else:
        assert not clarification_calls

    new_location_calls = [
        c for c in calls if c["function"]["name"] == "request_new_location"
    ]
    assert len(new_location_calls) == (1 if scenario["kind"] == "missing" else 0)
    if new_location_calls:
        assert json.loads(new_location_calls[0]["function"]["arguments"]) == {}

    fulfillment = [json.loads(c["function"]["arguments"]) for c in calls if c["function"]["name"] == "can_fulfill_material_request"]
    assert len(fulfillment) == 1
    expected = {
        "material_id": scenario["material_id"],
        "quantity": scenario["quantity"],
        "unit": scenario["unit"],
        "required_date": scenario["required_date"],
        "plant_id": scenario["selected_plant_id"],
    }
    assert fulfillment[0] == expected
    if len(final_matches) > 1:
        assert [m["role"] for m in messages][-2:] == ["user", "assistant"]
        assert messages[-2]["content"] == scenario["selected_plant_name"]
        selected = next(
            plant
            for plant in final_matches
            if plant["name"] == messages[-2]["content"]
        )
        assert fulfillment[0]["plant_id"] == selected["plant_id"]


def generate(count, split, seed):
    rng = random.Random(seed)
    rows = []
    kinds = balanced_sample(count, KINDS, rng)
    difficulties = balanced_sample(count, DIFFICULTIES, rng)
    for index, (kind, difficulty) in enumerate(zip(kinds, difficulties)):
        scenario = make_scenario(index, split, kind, difficulty, rng)
        messages = build_conversation(scenario)
        validate(scenario, messages)
        rows.append({"scenario": scenario, "messages": messages})
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
    """Create an SFT example or a prompt-only online-rollout example."""
    scenario = row["scenario"]
    messages = row["messages"] if stage == "sft" else row["messages"][:1]
    tool_sequence = [
        tool_call["function"]["name"]
        for message in row["messages"]
        for tool_call in message.get("tool_calls", [])
    ]
    return {
        "messages": _normalise_messages(messages),
        "scenario_json": json.dumps(scenario, ensure_ascii=False, sort_keys=True),
        "scenario_id": scenario["scenario_id"],
        "stage": stage,
        "language": scenario["language"],
        "trajectory_type": scenario["kind"],
        "difficulty": scenario["difficulty"],
        "tool_sequence": tool_sequence,
    }


def build_pipeline_dataset(split_sizes, seed):
    """Build the five splits used by the SFT -> on-policy pipeline."""
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
        "trajectory_type": Value("string"),
        "difficulty": Value("string"),
        "tool_sequence": List(Value("string")),
    })

    datasets = {}
    for offset, split in enumerate(expected_splits, start=1):
        stage = "sft" if split.startswith("sft_") else (
            "opd" if split.startswith("opd_") else "evaluation"
        )
        rows = generate(int(split_sizes[split]), split, seed + offset)
        datasets[split] = Dataset.from_list(
            [to_pipeline_row(row, stage) for row in rows],
            features=features,
        )
    return DatasetDict(datasets)
