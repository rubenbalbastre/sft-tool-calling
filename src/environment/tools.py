"""Tool implementations backed by the plant master data."""

import csv
from pathlib import Path
from transformers.utils import get_json_schema


MASTER_DATA_PATH = Path(__file__).parent / "master_data.csv"

with MASTER_DATA_PATH.open(encoding="utf-8", newline="") as file:
    MASTER_DATA = list(csv.DictReader(file))


def check_location(city: str):
    """Look up plants in the master data by city.

    Use this before fulfillment when the user provides a city instead of an
    explicit target plant ID. Matching ignores case and surrounding whitespace.

    Args:
        city: City requested by the user.

    Returns:
        A mapping whose ``matches`` list contains the matching plant records.
        The list is empty when the city is unavailable.
    """
    matches = [
        plant
        for plant in MASTER_DATA
        if plant["city"].casefold() == city.strip().casefold()
    ]
    return {"matches": matches}


def ask_for_clarification(candidate_plant_ids: list[str]):
    """Ask the user to select one of several matching plants.

    Use this only after a city lookup returns multiple plants.

    Args:
        candidate_plant_ids: At least two distinct plant IDs returned by the
            preceding city lookup.

    Returns:
        A clarification status and the corresponding plant records.

    Raises:
        ValueError: If IDs are duplicated, unknown, or fewer than two.
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
    """Ask the user for another city after a lookup returns no plants.

    Returns:
        A status indicating that a new location was requested.
    """
    return {"status": "new_location_requested"}


def can_fulfill_material_request(
    material_id: str,
    quantity: float,
    unit: str,
    required_date: str,
    plant_id: str,
):
    """Submit the material request to one resolved plant.

    Call this only after the target plant is explicit or has been resolved by
    location lookup and, when necessary, user clarification.

    Args:
        material_id: Material identifier from the original request.
        quantity: Requested material quantity.
        unit: The unit of measurement. (choices: ["kg", "units"])
        required_date: Required delivery date from the original request.
        plant_id: Explicit or resolved target plant identifier.

    Returns:
        The normalized material request submitted for fulfillment.
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
