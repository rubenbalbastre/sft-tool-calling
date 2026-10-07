import unittest

from datasets import Dataset

from src.training.preprocessing import (
    GEMMA_TOOL_CALL_END,
    GEMMA_TOOL_CALL_STARTS,
    enable_assistant_tool_call_mask,
    prepare_sft_source,
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
            GEMMA_TOOL_CALL_STARTS[0] + "\n"
            "                tool call rendering\n"
            + GEMMA_TOOL_CALL_END
            + "\n            {%- endif -%}\n"
            + "            {%- set ns_tr_out = namespace(flag=false) -%}"
        )

        enable_assistant_tool_call_mask(tokenizer)

        self.assertIn("{%- generation -%}", tokenizer.chat_template)
        self.assertIn("{%- endgeneration -%}", tokenizer.chat_template)
        self.assertLess(
            tokenizer.chat_template.index("{%- endgeneration -%}"),
            tokenizer.chat_template.index("{%- set ns_tr_out"),
        )

    def test_prepared_tool_calls_keep_only_their_own_arguments(self):
        messages = [
            {"role": "user", "content": "Find an option"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "request_quote",
                        "arguments": '{"supplier_id": "SUP-001"}',
                    },
                }],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_delivery_options",
                        "arguments": '{"quote_id": "QUOTE-001"}',
                    },
                }],
            },
        ]
        dataset = Dataset.from_list([{"messages": messages}])

        prepared = prepare_sft_source(dataset, False, "none")[0]["messages"]
        calls = [
            message["tool_calls"][0]["function"]
            for message in prepared
            if message["role"] == "assistant"
        ]

        self.assertEqual(calls[0]["arguments"], {"supplier_id": "SUP-001"})
        self.assertEqual(calls[1]["arguments"], {"quote_id": "QUOTE-001"})


if __name__ == "__main__":
    unittest.main()
