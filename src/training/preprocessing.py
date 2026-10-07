"""Model-specific rendering for supervised conversations."""

import json

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


def prepare_sft_dataset(
    dataset,
    tokenizer,
    enable_thinking,
    reasoning_effort,
    max_length,
):
    """Materialize tokenized SFT examples with assistant-only labels."""
    def tokenize_example(example):
        messages = [dict(message) for message in example["messages"]]
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        tokenized = tokenizer.apply_chat_template(
            deserialize_tool_arguments(messages),
            tools=CHAT_TOOLS,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        input_ids = tokenized["input_ids"][:max_length]
        assistant_mask = tokenized["assistant_masks"][:max_length]
        labels = [
            token_id if is_assistant else -100
            for token_id, is_assistant in zip(input_ids, assistant_mask)
        ]
        return {"input_ids": input_ids, "labels": labels}

    tokenized = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Tokenizing SFT dataset",
    )
    return tokenized.filter(
        lambda example: any(label != -100 for label in example["labels"]),
        desc="Dropping fully masked examples",
    )


def prepare_opd_source(dataset):
    """Expose prompt-only messages under the column required by TRL OPD."""
    return dataset.select_columns(["messages"]).rename_column("messages", "prompt")
