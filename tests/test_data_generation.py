import json
import unittest

from src.data_generation.generate_sft_data import build_pipeline_dataset


class DataGenerationTest(unittest.TestCase):
    def test_pipeline_splits_and_views(self):
        sizes = {
            "sft_train": 6,
            "sft_validation": 2,
            "opd_train": 6,
            "opd_validation": 2,
            "test": 2,
        }
        dataset = build_pipeline_dataset(sizes, seed=42)

        self.assertEqual(set(dataset), set(sizes))
        self.assertEqual({name: len(split) for name, split in dataset.items()}, sizes)
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


if __name__ == "__main__":
    unittest.main()
