"""Tool implementations backed by the plant master data."""

import csv
from pathlib import Path
from transformers.utils import get_json_schema


MASTER_DATA_PATH = Path(__file__).parent / "master_data.csv"

with MASTER_DATA_PATH.open(encoding="utf-8", newline="") as file:
    MASTER_DATA = list(csv.DictReader(file))

def check_location(city: str):
    """
    Return every plant whose city matches the supplied city.
    
    Args:
        city: The name of the city to search for.
    
    Returns:
        dict: A dictionary containing the matching plants.
    """
    matches = [
        plant
        for plant in MASTER_DATA
        if plant["city"].casefold() == city.strip().casefold()
    ]
    return {"matches": matches}


def ask_for_clarification(candidate_plant_ids: list[str]):
    """
    Request that the user choose one of the candidate plants.

    Args:
        candidate_plant_ids: A list of candidate plant IDs.

    Returns:
        dict: A dictionary containing the clarification status and candidate plants.
    """
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
    """
    Request a different city after a location lookup returns no records.
    
    Args:
        None
    
    Returns:
        dict: A dictionary indicating that a new location is requested.
    """
    return {"status": "new_location_requested"}


def can_fulfill_material_request(
    material_id: str,
    quantity: float,
    unit: str,
    required_date: str,
    plant_id: str,
):
    """
    Represent the fulfillment tool exposed by the task environment.
    
    Args:
        material_id: The ID of the material to check.
        quantity: The quantity of the material requested.
        unit: The unit of measurement. (choices: ["kg", "units"])
        required_date: The date by which the material is required.
        plant_id: The ID of the plant to check for fulfillment.

    Returns:
        dict: A dictionary containing the material request details.
    """
    return {
        "material_id": material_id,
        "quantity": quantity,
        "unit": unit,
        "required_date": required_date,
        "plant_id": plant_id,
    }


TOOL_FUNCTIONS = [
    check_location,
    ask_for_clarification,
    request_new_location,
    can_fulfill_material_request,
]

CHAT_TOOLS = [get_json_schema(function) for function in TOOL_FUNCTIONS]
for tool in CHAT_TOOLS:
    parameters = tool["function"]["parameters"]
    parameters.setdefault("required", [])
    parameters["additionalProperties"] = False

TOOLS = [
    {"type": "function", **tool["function"], "strict": True}
    for tool in CHAT_TOOLS
]
