# Procurement environment

The environment evaluates constrained procurement option selection. Fixed
materials and supplier capabilities are stored in a small SQLite database.
Episode-specific prices, availability, readiness, transport cost, arrival,
reliability, and emissions are derived deterministically from the episode seed.

## Tools

- `search_suppliers` discovers suppliers for open requests.
- `get_supplier_profile` exposes certifications, risk, and reliability.
- `request_quote` returns seeded price, availability, and readiness.
- `get_delivery_options` returns three seeded transport options.
- `submit_procurement_plan` terminates with an observed quote and delivery.
- `report_no_feasible_option` terminates impossible tasks after sufficient
  evidence has been collected.

## Routes

Seeded scenarios cycle through several task types:

- `direct_supplier`: check only the supplier named by the user;
- `open_search`: discover and compare supplier options;
- `compliance_first`: observe certification evidence before selection;
- `preferred_with_fallback`: try the named supplier before searching;
- `no_feasible_option`: research candidates and prove none satisfies the task.

The executable scenario is separate from its wording. Dataset generation
renders each scenario with ten templates in English, Spanish, German, and
French, while evaluation can continue to select a single deterministic prompt.
Prompt variants never change the seeded market state or verifier.

The verifier does not store an expected tool list. It checks whether submitted
IDs were observed, all hard constraints hold, required evidence was collected,
and the selected utility is within the scenario tolerance of the oracle option.

## Interface

```python
from src.environment.procurement import ProcurementEnvironment, generate_scenarios

scenario = generate_scenarios(1, seed=42)[0]
env = ProcurementEnvironment(scenario)

observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step({
    "name": "search_suppliers",
    "arguments": {
        "material_id": scenario["material_id"],
        "countries": scenario["allowed_countries"],
    },
})
```

This follows the Gymnasium reset/step result shape while retaining structured
tool-call dictionaries for LLM runners. Invalid calls return recoverable tool
errors and a small penalty. Success, explicit failure, or the maximum step
limit ends an episode.

## Reproducibility

The fixed database is created automatically at:

```text
data/environment/procurement.db
```

The database records schema version and master-data seed. A stable SHA-256
derived random stream produces each supplier quote and transport mode, so tool
call order cannot change the market. Identical scenario seeds reconstruct the
same prompt, quotes, delivery options, oracle utility, and reward.
