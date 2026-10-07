"""Model-specific rendering for supervised conversations."""

import json

from datasets import Dataset

from src.environment.procurement import CHAT_TOOLS, SYSTEM_PROMPT


GEMMA_TOOL_CALL_STARTS = (
    "{%- if message.get('tool_calls') -%}",
    "{%- if message['tool_calls'] -%}",
)
GEMMA_TOOL_CALL_END = "{%- set ns.prev_message_type = 'tool_call' -%}"


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


def enable_assistant_tool_call_mask(tokenizer):
    """Mark Gemma 4 assistant tool calls for TRL assistant-only loss."""
    template = tokenizer.chat_template
    if "{% generation" in template or "{%- generation" in template:
        return

    starts = [marker for marker in GEMMA_TOOL_CALL_STARTS if marker in template]
    if len(starts) != 1 or template.count(GEMMA_TOOL_CALL_END) != 1:
        raise ValueError(
            "The model chat template cannot produce assistant masks. "
            "Use a template with {% generation %} markers."
        )

    start = starts[0]
    template = template.replace(
        start,
        start + "\n                {%- generation -%}",
        1,
    )
    template = template.replace(
        GEMMA_TOOL_CALL_END,
        GEMMA_TOOL_CALL_END + "\n                {%- endgeneration -%}",
        1,
    )
    tokenizer.chat_template = template


def prepare_sft_source(dataset, enable_thinking, reasoning_effort):
    """Return structured conversations so TRL can build assistant masks."""
    rows = []
    for example in dataset:
        messages = [dict(message) for message in example["messages"]]
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        rows.append({
            "messages": messages,
            "tools": json.dumps(CHAT_TOOLS),
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort,
            },
        })

    prepared = Dataset.from_list(rows)

    def deserialize_batch(batch):
        batch["messages"] = [
            deserialize_tool_arguments(messages)
            for messages in batch["messages"]
        ]
        return batch

    return prepared.with_transform(deserialize_batch)


def prepare_opd_source(dataset):
    """Expose prompt-only messages under the column required by TRL OPD."""
    return dataset.select_columns(["messages"]).rename_column("messages", "prompt")
