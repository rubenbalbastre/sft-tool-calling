import unittest

from src.environment.procurement import (
    LANGUAGES,
    ProcurementEnvironment,
    TEMPLATES_PER_LANGUAGE,
    generate_scenarios,
    prompt_variants,
)


class ProcurementEnvironmentTest(unittest.TestCase):
    def test_every_scenario_has_all_multilingual_prompt_variants(self):
        for scenario in generate_scenarios(5, "prompt_test", 17):
            variants = list(prompt_variants(scenario))
            self.assertEqual(
                len(variants), len(LANGUAGES) * TEMPLATES_PER_LANGUAGE
            )
            self.assertEqual(
                {variant["scenario_id"] for variant in variants},
                {scenario["scenario_id"]},
            )
            self.assertEqual(
                {variant["language"] for variant in variants}, set(LANGUAGES)
            )
            self.assertTrue(all(variant["user_request"] for variant in variants))

    def test_seeded_scenarios_are_reproducible(self):
        first = generate_scenarios(5, seed=1234)
        second = generate_scenarios(5, seed=1234)
        self.assertEqual(first, second)

    def test_direct_supplier_accepts_best_observed_option(self):
        scenario = generate_scenarios(1, seed=55)[0]
        env = ProcurementEnvironment(scenario)
        supplier_id = scenario["requested_supplier_id"]
        quote_args = {
            "supplier_id": supplier_id,
            "material_id": scenario["material_id"],
            "quantity": scenario["quantity"],
            "unit": scenario["unit"],
            "required_date": scenario["required_date"],
        }
        quote_observation, *_ = env.step({"name": "request_quote", "arguments": quote_args})
        quote_id = quote_observation["content"]["quote_id"]
        env.step({"name": "get_delivery_options", "arguments": {
            "quote_id": quote_id,
            "destination": scenario["destination"],
        }})
        best = max(env.oracle_options(), key=lambda option: option["utility"])
        _, _, terminated, truncated, info = env.step({
            "name": "submit_procurement_plan",
            "arguments": {
                "quote_id": best["quote_id"],
                "delivery_option_id": best["delivery_option_id"],
            },
        })
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])

    def test_no_feasible_report_requires_evidence(self):
        scenario = generate_scenarios(5, seed=91)[4]
        env = ProcurementEnvironment(scenario)
        _, _, terminated, _, info = env.step({
            "name": "report_no_feasible_option",
            "arguments": {},
        })
        self.assertTrue(terminated)
        self.assertFalse(info["success"])
        self.assertFalse(info["metrics"]["evidence_complete"])


if __name__ == "__main__":
    unittest.main()
