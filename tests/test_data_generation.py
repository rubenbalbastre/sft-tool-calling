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


if __name__ == "__main__":
    unittest.main()
