import json
import unittest

from src.data_generation.generate_sft_data import build_pipeline_dataset
from src.environment.procurement.database import ProcurementRepository


class DataGenerationTest(unittest.TestCase):
    @staticmethod
    def actions(row):
        return [
            {
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            }
            for message in row["messages"]
            for call in message["tool_calls"]
        ]

    def test_pipeline_splits_and_views(self):
        sizes = {
            "sft_train": 5,
            "sft_validation": 1,
            "opd_train": 5,
            "opd_validation": 1,
            "test": 1,
        }
        dataset = build_pipeline_dataset(
            sizes,
            seed=42,
            languages=["English", "Spanish"],
            templates_per_language=2,
        )

        self.assertEqual(set(dataset), set(sizes))
        self.assertEqual(
            {name: len(split) for name, split in dataset.items()},
            {name: count * 4 for name, count in sizes.items()},
        )
        self.assertGreater(len(dataset["sft_train"][0]["messages"]), 1)
        self.assertEqual(len(dataset["opd_train"][0]["messages"]), 1)
        self.assertEqual(len(dataset["test"][0]["messages"]), 1)
        self.assertEqual(dataset["sft_train"][0]["stage"], "sft")
        self.assertEqual(dataset["opd_train"][0]["stage"], "opd")
        self.assertEqual(dataset["test"][0]["stage"], "evaluation")
        self.assertIn("material_id", json.loads(dataset["test"][0]["scenario_json"]))
        tool_names = {
            call["function"]["name"]
            for message in dataset["sft_train"][0]["messages"]
            for call in message["tool_calls"]
        }
        self.assertTrue(tool_names <= {
            "search_suppliers",
            "get_supplier_profile",
            "request_quote",
            "get_delivery_options",
            "submit_procurement_plan",
            "report_no_feasible_option",
        })
        self.assertTrue(
            {"submit_procurement_plan", "report_no_feasible_option"} & tool_names
        )
        for split in dataset.values():
            self.assertEqual(set(split["language"]), {"English", "Spanish"})
            scenario_counts = {
                scenario_id: split["scenario_id"].count(scenario_id)
                for scenario_id in set(split["scenario_id"])
            }
            self.assertTrue(all(count == 4 for count in scenario_counts.values()))

        split_ids = [set(split["scenario_id"]) for split in dataset.values()]
        for index, scenario_ids in enumerate(split_ids):
            for other_ids in split_ids[index + 1:]:
                self.assertTrue(scenario_ids.isdisjoint(other_ids))

        rows_by_type = {}
        for row in dataset["sft_train"]:
            rows_by_type.setdefault(row["trajectory_type"], row)

        preferred_actions = self.actions(
            rows_by_type["preferred_with_fallback"]
        )
        self.assertEqual(
            [action["name"] for action in preferred_actions[:3]],
            [
                "request_quote",
                "get_delivery_options",
                "search_suppliers",
            ],
        )

        compliance_actions = self.actions(rows_by_type["compliance_first"])
        quoted_supplier_ids = {
            action["arguments"]["supplier_id"]
            for action in compliance_actions
            if action["name"] == "request_quote"
        }
        repository = ProcurementRepository()
        try:
            self.assertTrue(quoted_supplier_ids)
            self.assertTrue(all(
                "ISO-14001" in repository.supplier(supplier_id)["certifications"]
                for supplier_id in quoted_supplier_ids
            ))
        finally:
            repository.close()


if __name__ == "__main__":
    unittest.main()
