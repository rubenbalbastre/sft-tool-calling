import unittest

from src.training.preprocessing import (
    GEMMA_TOOL_CALL_END,
    GEMMA_TOOL_CALL_STARTS,
    enable_assistant_tool_call_mask,
)


class RecordingTokenizer:
    pass


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

if __name__ == "__main__":
    unittest.main()
