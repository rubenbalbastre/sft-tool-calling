import unittest

from src.training.preprocessing import format_sft_example


class RecordingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered"


class TrainingPreprocessingTest(unittest.TestCase):
    def test_sft_formatter_disables_thinking_explicitly(self):
        tokenizer = RecordingTokenizer()
        example = {
            "source_messages": [{"role": "user", "content": "Choose an option"}]
        }

        rendered = format_sft_example(
            example,
            tokenizer=tokenizer,
            enable_thinking=False,
            reasoning_effort="none",
        )

        self.assertEqual(rendered, "rendered")
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertEqual(tokenizer.kwargs["reasoning_effort"], "none")
        self.assertFalse(tokenizer.kwargs["add_generation_prompt"])
        self.assertFalse(tokenizer.kwargs["tokenize"])


if __name__ == "__main__":
    unittest.main()
