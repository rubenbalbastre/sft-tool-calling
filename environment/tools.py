"""Tool implementations backed by the plant master data."""

import csv
from pathlib import Path


MASTER_DATA_PATH = Path(__file__).parent / "master_data.csv"

with MASTER_DATA_PATH.open(encoding="utf-8", newline="") as file:
    MASTER_DATA = list(csv.DictReader(file))

TOOLS = [
    {
        "type": "function",
        "name": "check_location",
        "description": "Find all plants in a city.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "ask_for_clarification",
        "description": "Ask the user to select one of multiple matching plants.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_plant_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["candidate_plant_ids"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "request_new_location",
        "description": "Ask for another city when location lookup has no matches.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "can_fulfill_material_request",
        "description": "Check fulfillment at one resolved plant.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "material_id": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string", "enum": ["kg", "units"]},
                "required_date": {"type": "string"},
                "plant_id": {"type": "string"},
            },
            "required": [
                "material_id", "quantity", "unit", "required_date", "plant_id"
            ],
            "additionalProperties": False,
        },
    },
]


def check_location(city):
    """Return every plant whose city matches the supplied city."""
    matches = [
        plant
        for plant in MASTER_DATA
        if plant["city"].casefold() == city.strip().casefold()
    ]
    return {"matches": matches}


def ask_for_clarification(candidate_plant_ids):
    """Request that the user choose one of the candidate plants."""
    if len(candidate_plant_ids) < 2 or len(candidate_plant_ids) != len(
        set(candidate_plant_ids)
    ):
        raise ValueError("Clarification requires at least two distinct plant IDs")
    candidates = [
        plant for plant in MASTER_DATA if plant["plant_id"] in candidate_plant_ids
    ]
    if len(candidates) != len(candidate_plant_ids):
        raise ValueError("All candidate plant IDs must exist in master data")
    return {
        "status": "clarification_requested",
        "candidates": candidates,
    }


def request_new_location():
    """Request a different city after a location lookup returns no records."""
    return {"status": "new_location_requested"}


def can_fulfill_material_request(
    material_id, quantity, unit, required_date, plant_id
):
    """Represent the fulfillment tool exposed by the task environment."""
    return {
        "material_id": material_id,
        "quantity": quantity,
        "unit": unit,
        "required_date": required_date,
        "plant_id": plant_id,
    }
