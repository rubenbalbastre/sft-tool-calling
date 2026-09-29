import json
import unittest

from src.data_generation.generate_sft_data import build_pipeline_dataset
from src.training.preprocessing import (
    deserialize_tool_arguments,
    format_sft_example,
)


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
        if not isinstance(arguments, dict):
            raise TypeError("arguments must be a mapping")
        return "rendered conversation"


class TrainingPreprocessingTest(unittest.TestCase):
    def test_deserializes_tool_call_arguments(self):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "check_location",
                    "arguments": json.dumps({"city": "Bilbao"}),
                },
            }],
        }]

        normalized = deserialize_tool_arguments(messages)

        self.assertEqual(
            normalized[0]["tool_calls"][0]["function"]["arguments"],
            {"city": "Bilbao"},
        )
        self.assertIsInstance(
            messages[0]["tool_calls"][0]["function"]["arguments"], str
        )

    def test_formats_sft_row_for_trainer(self):
        sizes = {
            "sft_train": 1,
            "sft_validation": 1,
            "opd_train": 1,
            "opd_validation": 1,
            "test": 1,
        }
        source = build_pipeline_dataset(sizes, seed=42)["sft_train"]
        tokenizer = RecordingTokenizer()

        rendered = format_sft_example(source[0], tokenizer)

        self.assertEqual(rendered, "rendered conversation")
        self.assertEqual(tokenizer.calls[0][1]["tools"][0]["type"], "function")
        self.assertIn("function", tokenizer.calls[0][1]["tools"][0])


if __name__ == "__main__":
    unittest.main()
