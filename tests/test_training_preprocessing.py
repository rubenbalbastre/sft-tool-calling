import unittest

from src.environment.procurement import SYSTEM_PROMPT
from src.training.preprocessing import (
    GEMMA_TOOL_CALL_END,
    GEMMA_TOOL_CALL_START,
    enable_assistant_tool_call_mask,
    format_sft_example,
)


class RecordingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered"


class TrainingPreprocessingTest(unittest.TestCase):
    def test_gemma_tool_call_mask_excludes_following_tool_response(self):
        tokenizer = RecordingTokenizer()
        tokenizer.chat_template = (
            GEMMA_TOOL_CALL_START + "\n"
            "                tool call rendering\n"
            + GEMMA_TOOL_CALL_END
        )

        enable_assistant_tool_call_mask(tokenizer)

        self.assertIn("{%- generation -%}", tokenizer.chat_template)
        self.assertIn("{%- endgeneration -%}", tokenizer.chat_template)
        self.assertLess(
            tokenizer.chat_template.index("{%- endgeneration -%}"),
            tokenizer.chat_template.index("{%- set ns_tr_out"),
        )

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
        self.assertEqual(
            tokenizer.messages[0],
            {"role": "system", "content": SYSTEM_PROMPT},
        )
        self.assertEqual(tokenizer.messages[1]["role"], "user")
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertEqual(tokenizer.kwargs["reasoning_effort"], "none")
        self.assertFalse(tokenizer.kwargs["add_generation_prompt"])
        self.assertFalse(tokenizer.kwargs["tokenize"])

    def test_sft_formatter_does_not_duplicate_existing_system_prompt(self):
        tokenizer = RecordingTokenizer()
        messages = [
            {"role": "system", "content": "Custom instructions"},
            {"role": "user", "content": "Choose an option"},
        ]

        format_sft_example(
            {"source_messages": messages},
            tokenizer=tokenizer,
            enable_thinking=False,
            reasoning_effort="none",
        )

        self.assertEqual(len(tokenizer.messages), 2)
        self.assertEqual(tokenizer.messages[0]["role"], "system")
        self.assertEqual(tokenizer.messages[0]["content"], "Custom instructions")
        self.assertEqual(tokenizer.messages[1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
