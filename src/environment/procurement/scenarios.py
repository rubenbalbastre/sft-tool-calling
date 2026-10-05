"""Deterministic scenario generation for procurement evaluation."""

import random
from datetime import date, timedelta

from .database import ProcurementRepository
from .environment import ProcurementEnvironment
from .prompts import format_prompt


TASK_TYPES = (
    "direct_supplier",
    "open_search",
    "compliance_first",
    "preferred_with_fallback",
    "no_feasible_option",
)


def _base_scenario(index, split, seed, attempt):
    rng = random.Random(seed * 100_003 + index * 997 + attempt)
    material_id, unit = rng.choice([
        ("MAT-1042", "kg"),
        ("MAT-2031", "kg"),
        ("MAT-3305", "units"),
    ])
    order_date = date(2027, 1, 5) + timedelta(days=rng.randint(0, 120))
    return {
        "scenario_id": f"{split}_{index:05d}",
        "world_seed": seed * 1_000_003 + index * 101 + attempt,
        "kind": TASK_TYPES[index % len(TASK_TYPES)],
        "task_type": TASK_TYPES[index % len(TASK_TYPES)],
        "language": "English",
        "difficulty": "hard" if index % 3 == 2 else "medium",
        "material_id": material_id,
        "quantity": rng.choice([600, 900, 1200]),
        "unit": unit,
        "destination": rng.choice(["Zaragoza", "Lyon", "Munich", "Milan"]),
        "order_date": order_date.isoformat(),
        "required_date": (order_date + timedelta(days=rng.randint(9, 15))).isoformat(),
        "allowed_countries": ["Spain", "France", "Germany", "Italy"],
        "maximum_total_cost": 25_000.0,
        "minimum_delivery_reliability": 0.94,
        "required_certifications": [],
        "excluded_supplier_ids": [],
        "requested_supplier_id": None,
        "preferred_supplier_id": None,
        "initial_supplier_ids": [],
        "preferences": {"cost": 0.3, "carbon": 0.6, "reliability": 0.1},
        "utility_tolerance": 0.05,
        "user_request": "",
    }


def generate_scenarios(count, split="evaluation", seed=1234):
    """Generate valid seeded tasks with several naturally different routes."""
    repository = ProcurementRepository()
    scenarios = []
    try:
        for index in range(count):
            for attempt in range(200):
                scenario = _base_scenario(index, split, seed, attempt)
                env = ProcurementEnvironment(scenario, repository=repository)
                feasible = env.oracle_options()
                feasible_ids = {row["supplier_id"] for row in feasible}
                all_suppliers = repository.search_suppliers(scenario["material_id"])
                all_ids = [row["supplier_id"] for row in all_suppliers]
                kind = scenario["task_type"]

                if kind == "direct_supplier" and feasible_ids:
                    supplier_id = sorted(feasible_ids)[0]
                    scenario["requested_supplier_id"] = supplier_id
                    scenario["initial_supplier_ids"] = [supplier_id]
                elif kind == "open_search" and len(feasible_ids) >= 2:
                    pass
                elif kind == "compliance_first":
                    scenario["required_certifications"] = ["ISO-14001"]
                    env = ProcurementEnvironment(scenario, repository=repository)
                    if not env.oracle_options():
                        continue
                elif kind == "preferred_with_fallback":
                    infeasible_ids = sorted(set(all_ids) - feasible_ids)
                    if not feasible_ids or not infeasible_ids:
                        continue
                    scenario["preferred_supplier_id"] = infeasible_ids[0]
                    scenario["initial_supplier_ids"] = [infeasible_ids[0]]
                elif kind == "no_feasible_option":
                    scenario["maximum_total_cost"] = 100.0
                    env = ProcurementEnvironment(scenario, repository=repository)
                    if env.oracle_options():
                        continue
                else:
                    continue

                scenario["user_request"] = format_prompt(scenario)
                scenarios.append(scenario)
                break
            else:
                raise RuntimeError(f"Could not construct {kind} scenario {index}")
    finally:
        repository.close()
    return scenarios
