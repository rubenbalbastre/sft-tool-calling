"""Gymnasium-style procurement option-selection environment."""

import hashlib
import json
import random
from datetime import date, timedelta

from .database import ProcurementRepository


TRANSPORT_MODES = {
    "express_road": {"days": 2, "cost": 0.48, "carbon": 0.24, "reliability": 0.985},
    "standard_road": {"days": 4, "cost": 0.29, "carbon": 0.17, "reliability": 0.965},
    "rail": {"days": 6, "cost": 0.18, "carbon": 0.07, "reliability": 0.945},
}


def derived_rng(seed, *parts):
    payload = ":".join(map(str, (seed, *parts)))
    number = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
    return random.Random(number)


def stable_id(prefix, *parts):
    payload = ":".join(map(str, parts))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


class ToolError(ValueError):
    """A recoverable invalid tool call."""


class ProcurementEnvironment:
    """Verify procurement research and selection without a gold trajectory."""

    def __init__(self, scenario, repository=None, max_steps=20):
        self.scenario = scenario
        self.repository = repository or ProcurementRepository()
        self.max_steps = max_steps
        self.reset()

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.steps = 0
        self.episode_return = 0.0
        self.terminated = False
        self.state = "research"
        self.known_suppliers = set(self.scenario.get("initial_supplier_ids", []))
        self.inspected_suppliers = set()
        self.quotes = {}
        self.delivery_options = {}
        self.actions = []
        self.terminal_metrics = {}
        observation = {"role": "user", "content": self.scenario["user_request"]}
        return observation, {"scenario_id": self.scenario["scenario_id"]}

    def step(self, action):
        if self.terminated:
            raise RuntimeError("Episode is finished; call reset()")

        self.steps += 1
        terminated = False
        reward = 0.0
        try:
            name, arguments = self._parse_action(action)
            if name == "invalid_model_output":
                return self._terminal_failure("Invalid model output")
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                raise ToolError(f"Unknown tool: {name}")
            content, terminated, reward = handler(**arguments)
            observation = {"role": "tool", "name": name, "content": content}
            self.actions.append({"name": name, "arguments": arguments})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            name = action.get("name", "invalid_action") if isinstance(action, dict) else "invalid_action"
            observation = {"role": "tool", "name": name, "content": {"error": str(error)}}
            reward = -0.05

        self.episode_return += reward
        truncated = self.steps >= self.max_steps and not terminated
        if terminated or truncated:
            self.terminated = True
        info = self._info(success=terminated and self.state == "success")
        if truncated:
            info["reason"] = "Maximum environment steps reached"
        return observation, reward, terminated, truncated, info

    def _tool_search_suppliers(self, material_id, countries=None):
        if material_id != self.scenario["material_id"]:
            raise ToolError("Search must use the requested material_id")
        suppliers = self.repository.search_suppliers(material_id, countries)
        self.known_suppliers.update(row["supplier_id"] for row in suppliers)
        return {"suppliers": suppliers}, False, 0.0

    def _tool_get_supplier_profile(self, supplier_id):
        self._require_known_supplier(supplier_id)
        supplier = self.repository.supplier(supplier_id)
        if not supplier:
            raise ToolError("Unknown supplier_id")
        self.inspected_suppliers.add(supplier_id)
        return supplier, False, 0.0

    def _tool_request_quote(self, supplier_id, material_id, quantity, unit, required_date):
        self._require_known_supplier(supplier_id)
        expected = {
            "material_id": self.scenario["material_id"],
            "quantity": self.scenario["quantity"],
            "unit": self.scenario["unit"],
            "required_date": self.scenario["required_date"],
        }
        actual = {
            "material_id": material_id,
            "quantity": quantity,
            "unit": unit,
            "required_date": required_date,
        }
        if actual != expected:
            raise ToolError(f"Quote request must preserve the user request: {expected}")
        quote = self._generate_quote(supplier_id)
        if not quote:
            raise ToolError("Supplier does not offer the requested material")
        self.quotes[quote["quote_id"]] = quote
        return quote, False, 0.0

    def _tool_get_delivery_options(self, quote_id, destination):
        if quote_id not in self.quotes:
            raise ToolError("Delivery options require a previously observed quote")
        if destination != self.scenario["destination"]:
            raise ToolError("Destination must match the user request")
        options = self._generate_delivery_options(self.quotes[quote_id])
        self.delivery_options.update({row["delivery_option_id"]: row for row in options})
        return {"delivery_options": options}, False, 0.0

    def _tool_submit_procurement_plan(self, quote_id, delivery_option_id):
        if quote_id not in self.quotes:
            raise ToolError("Selected quote was not observed")
        delivery = self.delivery_options.get(delivery_option_id)
        if not delivery:
            raise ToolError("Selected delivery option was not observed")
        if delivery["quote_id"] != quote_id:
            raise ToolError("Delivery option does not belong to the selected quote")

        option = self._combine(self.quotes[quote_id], delivery)
        feasible, checks = self._feasibility(option, require_evidence=True)
        oracle = self.oracle_options()
        optimal_utility = max((row["utility"] for row in oracle), default=0.0)
        regret = optimal_utility - option["utility"] if feasible else None
        success = feasible and regret <= self.scenario["utility_tolerance"]
        reward = (0.5 + 0.5 * option["utility"]) if feasible else 0.0
        self.state = "success" if success else "failure"
        result = {
            "success": success,
            "feasible": feasible,
            "checks": checks,
            "selected": option,
            "optimal_utility": round(optimal_utility, 4),
            "regret": round(regret, 4) if regret is not None else None,
        }
        self.terminal_metrics = result
        return result, True, reward

    def _tool_report_no_feasible_option(self):
        candidates = {
            row["supplier_id"]
            for row in self.repository.search_suppliers(
                self.scenario["material_id"], self.scenario["allowed_countries"]
            )
        }
        quoted = {row["supplier_id"] for row in self.quotes.values()}
        delivered_quotes = {row["quote_id"] for row in self.delivery_options.values()}
        evidence_complete = (
            candidates <= self.known_suppliers
            and candidates <= quoted
            and set(self.quotes) <= delivered_quotes
        )
        success = not self.oracle_options() and evidence_complete
        self.state = "success" if success else "failure"
        result = {
            "success": success,
            "no_feasible_option": not self.oracle_options(),
            "evidence_complete": evidence_complete,
        }
        self.terminal_metrics = result
        return result, True, 1.0 if success else 0.0

    def _generate_quote(self, supplier_id):
        base = self.repository.supplier_material(supplier_id, self.scenario["material_id"])
        if not base:
            return None
        rng = derived_rng(self.scenario["world_seed"], "quote", supplier_id)
        quantity = self.scenario["quantity"]
        unit_price = round(base["base_unit_price"] * rng.uniform(0.9, 1.15), 2)
        available = round(base["normal_capacity"] * rng.uniform(0.55, 1.1), 0)
        preparation_days = max(1, base["preparation_days"] + rng.randint(-1, 3))
        ready_date = date.fromisoformat(self.scenario["order_date"]) + timedelta(days=preparation_days)
        return {
            "quote_id": stable_id("QUOTE", self.scenario["world_seed"], supplier_id),
            "supplier_id": supplier_id,
            "material_id": self.scenario["material_id"],
            "quantity": quantity,
            "unit": base["unit"],
            "unit_price": unit_price,
            "material_cost": round(unit_price * quantity, 2),
            "available_quantity": available,
            "ready_date": ready_date.isoformat(),
        }

    def _generate_delivery_options(self, quote):
        supplier = self.repository.supplier(quote["supplier_id"])
        country_factor = {"Spain": 1.0, "France": 1.15, "Germany": 1.35, "Italy": 1.25}[supplier["country"]]
        options = []
        for mode, base in TRANSPORT_MODES.items():
            rng = derived_rng(self.scenario["world_seed"], "delivery", quote["quote_id"], mode)
            days = base["days"] + rng.randint(0, 2)
            pickup = date.fromisoformat(quote["ready_date"])
            options.append({
                "delivery_option_id": stable_id("DEL", quote["quote_id"], mode),
                "quote_id": quote["quote_id"],
                "mode": mode,
                "pickup_date": pickup.isoformat(),
                "arrival_date": (pickup + timedelta(days=days)).isoformat(),
                "shipping_cost": round(self.scenario["quantity"] * base["cost"] * country_factor * rng.uniform(0.92, 1.08), 2),
                "reliability": round(min(0.999, base["reliability"] * rng.uniform(0.99, 1.01)), 4),
                "carbon_kg": round(self.scenario["quantity"] * base["carbon"] * country_factor * rng.uniform(0.95, 1.05), 2),
            })
        return options

    def oracle_options(self):
        options = []
        suppliers = self.repository.search_suppliers(
            self.scenario["material_id"], self.scenario["allowed_countries"]
        )
        for supplier in suppliers:
            quote = self._generate_quote(supplier["supplier_id"])
            for delivery in self._generate_delivery_options(quote):
                option = self._combine(quote, delivery)
                feasible, _ = self._feasibility(option, require_evidence=False)
                if feasible:
                    options.append(option)
        return options

    def observed_feasible_options(self):
        """Return feasible options assembled only from observed tool results."""
        options = []
        for delivery in self.delivery_options.values():
            option = self._combine(self.quotes[delivery["quote_id"]], delivery)
            feasible, _ = self._feasibility(option, require_evidence=True)
            if feasible:
                options.append(option)
        return options

    def _combine(self, quote, delivery):
        supplier = self.repository.supplier(quote["supplier_id"])
        option = {
            **quote,
            **delivery,
            "supplier_country": supplier["country"],
            "supplier_reliability": supplier["reliability"],
            "certifications": supplier["certifications"],
            "total_cost": round(quote["material_cost"] + delivery["shipping_cost"], 2),
        }
        option["utility"] = self._utility(option)
        return option

    def _feasibility(self, option, require_evidence):
        scenario = self.scenario
        requested_supplier = scenario.get("requested_supplier_id")
        checks = {
            "requested_supplier": not requested_supplier or option["supplier_id"] == requested_supplier,
            "quantity": option["available_quantity"] >= scenario["quantity"],
            "unit": option["unit"] == scenario["unit"],
            "on_time": option["arrival_date"] <= scenario["required_date"],
            "budget": option["total_cost"] <= scenario["maximum_total_cost"],
            "delivery_reliability": option["reliability"] >= scenario["minimum_delivery_reliability"],
            "country": option["supplier_country"] in scenario["allowed_countries"],
            "not_excluded": option["supplier_id"] not in scenario["excluded_supplier_ids"],
            "certifications": all(item in option["certifications"] for item in scenario["required_certifications"]),
        }
        if require_evidence and scenario["required_certifications"]:
            checks["compliance_observed"] = option["supplier_id"] in self.inspected_suppliers
        if require_evidence and scenario["task_type"] == "preferred_with_fallback":
            preferred = scenario["preferred_supplier_id"]
            checks["preferred_supplier_tried"] = any(
                quote["supplier_id"] == preferred for quote in self.quotes.values()
            )
        return all(checks.values()), checks

    def _utility(self, option):
        weights = self.scenario["preferences"]
        budget = max(self.scenario["maximum_total_cost"], 1)
        cost_score = max(0.0, min(1.0, 1.0 - option["total_cost"] / budget))
        carbon_score = max(0.0, min(1.0, 1.0 - option["carbon_kg"] / 600.0))
        reliability_score = option["reliability"]
        return round(
            weights["cost"] * cost_score
            + weights["carbon"] * carbon_score
            + weights["reliability"] * reliability_score,
            4,
        )

    def _require_known_supplier(self, supplier_id):
        if supplier_id not in self.known_suppliers:
            raise ToolError("Supplier must be mentioned by the user or discovered by search")

    def _terminal_failure(self, reason):
        self.terminated = True
        self.state = "failure"
        info = self._info(success=False)
        info["reason"] = reason
        return {"role": "tool", "name": "invalid_model_output", "content": {"error": reason}}, 0.0, True, False, info

    def _info(self, success):
        info = {
            "success": success,
            "state": self.state,
            "episode_return": self.episode_return,
        }
        if self.terminal_metrics:
            info["metrics"] = self.terminal_metrics
        return info

    @staticmethod
    def _parse_action(action):
        if not isinstance(action, dict):
            raise TypeError("Action must be a mapping")
        if "function" in action:
            action = action["function"]
        name = action["name"]
        arguments = action.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a JSON object")
        return name, arguments
