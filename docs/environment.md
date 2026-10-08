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

The executable scenario is separate from its wording. In each language,
dataset generation uses ten training templates, five validation templates, and
ten test templates. Evaluation can use every held-out variant or select one
specific variant. Prompt wording never changes the seeded market state or
verifier.

The verifier does not store an expected tool list. It checks whether submitted
IDs were observed, all hard constraints hold, required evidence was collected,
and the selected utility is within the scenario tolerance of the oracle option.

## Hard constraints and evidence

A submitted quote and delivery option are feasible only when every applicable
check below passes. Preferences never compensate for a failed hard constraint.

| Check | Requirement | Evidence used by the verifier |
| --- | --- | --- |
| Requested supplier | A `direct_supplier` task must use exactly the named supplier. | Supplier ID on the submitted quote |
| Quantity | Available quantity must be at least the requested quantity. | `available_quantity` from `request_quote` |
| Unit | The quote unit must equal the requested unit. | Quote and scenario units |
| Deadline | Arrival must be on or before the required date. | `arrival_date` from `get_delivery_options` |
| Budget | Material plus shipping cost must not exceed the total budget. | Computed `total_cost` |
| Delivery reliability | The selected transport option must meet the reliability floor. | Delivery-option reliability |
| Allowed country | The supplier must belong to an allowed country. | Supplier master data |
| Exclusions | An excluded supplier cannot be selected. | Scenario exclusions and supplier ID |
| Certifications | The supplier must hold every required certification. | Supplier profile/master data |

The model must also collect the evidence required by the route:

- Certification-constrained tasks require a prior `get_supplier_profile` call
  for the selected supplier. Knowing the hidden database value is not enough.
- Preferred-supplier fallback tasks require evidence that the preferred
  supplier was actually quoted before a fallback is accepted.
- `report_no_feasible_option` succeeds only after every supplier returned for
  the requested material and allowed countries has been discovered and quoted,
  and every resulting quote has delivery options.
- Quote and delivery IDs must have been returned by earlier tools, and the
  selected delivery option must belong to the selected quote.

Tool calls preserve user-controlled request fields. In particular,
`request_quote` must use the original material, quantity, unit, and required
date, while `get_delivery_options` must use the original destination. Invalid
or fabricated arguments produce a recoverable tool error and a `-0.05` reward.

## Preferences, utility, and success

After filtering by the hard constraints, feasible options are ranked by a
weighted utility:

```text
utility = 0.3 × cost score + 0.6 × carbon score + 0.1 × reliability
```

Cost and carbon scores are normalized to `[0, 1]`; the default scenarios
therefore prioritize lower emissions, then lower cost, with reliability as a
smaller preference after its hard minimum has been met. The environment also
computes an oracle over the same deterministic market state.

A submitted plan counts as successful when it is feasible and its utility is
within the scenario's `0.05` tolerance of the best oracle option. A feasible
submission receives `0.5 + 0.5 × utility`, which provides partial credit even
when its regret is too high for task success. A correctly evidenced
no-feasible-option decision receives `1.0`.

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
