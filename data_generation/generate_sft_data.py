"""Small deterministic generator for multilingual tool-calling conversations."""

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from environment.tools import (
    MASTER_DATA,
    ask_for_clarification,
    check_location,
    request_new_location,
)


LANGUAGES = ["English", "Spanish", "German", "French"]
KINDS = (
    ["explicit"] * 30
    + ["unique"] * 25
    + ["ambiguous"] * 20
    + ["missing"] * 10
    + ["explicit_with_city"] * 10
    + ["distractor"] * 5
)
DIFFICULTIES = ["simple"] * 30 + ["medium"] * 40 + ["hard"] * 30

REQUESTS = {
    "English": [
        "Can {target} provide {quantity} {unit} of {material} by {date}?",
        "For {target}, we still need {quantity} {unit} of {material} on {date}. Can we cover it?",
        "The later requirement is covered. At {target}, the {date} run is short {quantity} {unit} of {material}. Please check that shortfall.",
    ],
    "Spanish": [
        "¿Puede {target} suministrar {quantity} {unit} de {material} para el {date}?",
        "Para {target} aún necesitamos {quantity} {unit} de {material} el {date}. ¿Podemos cubrirlo?",
        "El requisito posterior ya está cubierto. En {target}, para el {date} faltan {quantity} {unit} de {material}. Comprueba ese faltante.",
    ],
    "German": [
        "Kann {target} bis zum {date} {quantity} {unit} {material} liefern?",
        "Für {target} fehlen am {date} noch {quantity} {unit} {material}. Können wir das decken?",
        "Der spätere Bedarf ist gedeckt. In {target} fehlen für den Lauf am {date} noch {quantity} {unit} {material}. Bitte prüfe diesen Fehlbestand.",
    ],
    "French": [
        "Est-ce que {target} peut fournir {quantity} {unit} de {material} pour le {date} ?",
        "Pour {target}, il manque encore {quantity} {unit} de {material} le {date}. Peut-on couvrir ce besoin ?",
        "Le besoin ultérieur est couvert. À {target}, il manque {quantity} {unit} de {material} pour le {date}. Merci de vérifier ce manque.",
    ],
}
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
        "material_id": f"MAT-{rng.randint(1000, 9999)}",
        "quantity": rng.randint(10, 5000),
        "unit": rng.choice(["kg", "units"]),
        "required_date": (date(2026, 10, 1) + timedelta(days=rng.randint(1, 365))).isoformat(),
        "city": None,
        "matches": [],
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
    if scenario["kind"] == "explicit":
        target = scenario["explicit_plant_id"]
    elif scenario["kind"] == "explicit_with_city":
        target = f'{scenario["city"]} ({scenario["explicit_plant_id"]})'
    else:
        target = scenario["city"]

    request = REQUESTS[scenario["language"]][level].format(
        target=target,
        quantity=scenario["quantity"],
        unit=scenario["unit"],
        material=scenario["material_id"],
        date=scenario["required_date"],
    )
    if scenario["distractor_plant_id"]:
        request = f'The previous shipment used {scenario["distractor_plant_id"]}, but this request is different. {request}'
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
            return messages

        if len(scenario["matches"]) > 1:
            candidate_ids = [plant["plant_id"] for plant in scenario["matches"]]
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
    assert names.count("check_location") == (0 if direct else 1)
    if not direct:
        lookup = next(c for c in calls if c["function"]["name"] == "check_location")
        assert json.loads(lookup["function"]["arguments"]) == {"city": scenario["city"]}
        assert scenario["matches"] == check_location(scenario["city"])["matches"]
        assert all(set(plant) == {"name", "plant_id", "city"} for plant in scenario["matches"])

    clarification_calls = [
        c for c in calls if c["function"]["name"] == "ask_for_clarification"
    ]
    if len(scenario["matches"]) > 1:
        assert len(clarification_calls) == 1
        clarification_args = json.loads(
            clarification_calls[0]["function"]["arguments"]
        )
        expected_candidates = [p["plant_id"] for p in scenario["matches"]]
        assert clarification_args == {"candidate_plant_ids": expected_candidates}
    else:
        assert not clarification_calls

    new_location_calls = [
        c for c in calls if c["function"]["name"] == "request_new_location"
    ]
    assert len(new_location_calls) == (1 if not direct and not scenario["matches"] else 0)
    if new_location_calls:
        assert json.loads(new_location_calls[0]["function"]["arguments"]) == {}

    fulfillment = [json.loads(c["function"]["arguments"]) for c in calls if c["function"]["name"] == "can_fulfill_material_request"]
    if not direct and not scenario["matches"]:
        assert not fulfillment
        return

    assert len(fulfillment) == 1
    expected = {
        "material_id": scenario["material_id"],
        "quantity": scenario["quantity"],
        "unit": scenario["unit"],
        "required_date": scenario["required_date"],
        "plant_id": scenario["selected_plant_id"],
    }
    assert fulfillment[0] == expected
    if len(scenario["matches"]) > 1:
        assert [m["role"] for m in messages][-2:] == ["user", "assistant"]
        assert messages[-2]["content"] == scenario["selected_plant_name"]
        selected = next(
            plant
            for plant in scenario["matches"]
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


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pilot"))
    args = parser.parse_args()

    # Freeze evaluation separately before producing training data.
    eval_rows = generate(args.eval_size, "eval", args.seed + 1)
    train_rows = generate(args.train_size, "train", args.seed + 2)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    print(f"Wrote {len(train_rows)} train and {len(eval_rows)} eval conversations")


if __name__ == "__main__":
    main()
