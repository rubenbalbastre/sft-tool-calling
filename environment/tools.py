"""Tool implementations backed by the plant master data."""

import csv
from pathlib import Path


MASTER_DATA_PATH = Path(__file__).parent / "master_data.csv"

with MASTER_DATA_PATH.open(encoding="utf-8", newline="") as file:
    MASTER_DATA = list(csv.DictReader(file))


def check_location(city):
    """Return every plant whose city matches the supplied city."""
    matches = [
        plant
        for plant in MASTER_DATA
        if plant["city"].casefold() == city.strip().casefold()
    ]
    return {"matches": matches}


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
