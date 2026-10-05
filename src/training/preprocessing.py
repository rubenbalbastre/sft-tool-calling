"""Model-specific rendering for supervised conversations."""

import json

from src.environment.procurement import CHAT_TOOLS, SYSTEM_PROMPT


def deserialize_tool_arguments(messages):
    """Return messages whose tool-call arguments are Python mappings."""
    normalized = []
    for message in messages:
        normalized_message = dict(message)
        normalized_calls = []
        for tool_call in message.get("tool_calls") or []:
            normalized_call = dict(tool_call)
            function = dict(tool_call["function"])
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
            normalized_call["function"] = function
            normalized_calls.append(normalized_call)
        normalized_message["tool_calls"] = normalized_calls
        normalized.append(normalized_message)
    return normalized


def format_sft_example(
    example, tokenizer, enable_thinking, reasoning_effort
):
    """Render one stored conversation with model-compatible tool arguments."""
    messages = deserialize_tool_arguments(example["source_messages"])
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    return tokenizer.apply_chat_template(
        messages,
        tools=CHAT_TOOLS,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )


def prepare_sft_source(dataset):
    """Hide the conversational column so TRL tokenizes formatted text once."""
    return dataset.select_columns(["messages"]).rename_column(
        "messages", "source_messages"
    )


def prepare_opd_source(dataset):
    """Expose prompt-only messages under the column required by TRL OPD."""
    return dataset.select_columns(["messages"]).rename_column("messages", "prompt")
