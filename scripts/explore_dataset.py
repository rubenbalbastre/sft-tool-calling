import argparse
from pathlib import Path
import sys

from datasets import load_from_disk
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.procurement import CHAT_TOOLS, SYSTEM_PROMPT
from src.training.preprocessing import (
    deserialize_tool_arguments,
    enable_assistant_tool_call_mask,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="google/gemma-4-E2B-it",
    help="Tokenizer name or local model path.",
)
args = parser.parse_args()


dataset = load_from_disk(PROJECT_ROOT / "data/pipeline/hf_dataset")["sft_train"]

example = dataset[0]

# for index, message in enumerate(example["messages"]):
#     print(
#         index,
#         f"role: {message['role']}",
#         f"content: {message['content']}",
#         f"tool_calls: {message.get('tool_calls', [])}",
#         f"tool_call_id: {message.get('tool_call_id', '')}",
#         sep="\n",
#         end="\n\n",
#     )

messages = [dict(message) for message in example["messages"]]
if not messages or messages[0]["role"] != "system":
    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
messages = deserialize_tool_arguments(messages)
template_kwargs = {"enable_thinking": True, "reasoning_effort": "low"}

for index, message in enumerate(messages):
    print(
        index,
        f"role: {message['role']}",
        f"content: {message['content']}",
        f"tool_calls: {message.get('tool_calls', [])}",
        f"tool_call_id: {message.get('tool_call_id', '')}",
        sep="\n",
        end="\n\n",
    )

tokenizer = AutoTokenizer.from_pretrained(args.model)
enable_assistant_tool_call_mask(tokenizer)
print(f"Template kwargs: {template_kwargs}")

rendered_prompt = tokenizer.apply_chat_template(
    messages,
    tools=CHAT_TOOLS,
    tokenize=False,
    add_generation_prompt=False,
    **template_kwargs,
)
tokenized = tokenizer.apply_chat_template(
    messages,
    tools=CHAT_TOOLS,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
    return_assistant_tokens_mask=True,
    **template_kwargs,
)
input_ids = tokenized["input_ids"]
attention_mask = tokenized["attention_mask"]
assistant_mask = tokenized["assistant_masks"]
loss_mask = [
    attended and assistant
    for attended, assistant in zip(attention_mask, assistant_mask)
]
print("Attention mask:", attention_mask)
print("Loss mask:", [int(value) for value in loss_mask])
token_count = (
    len(input_ids[0])
    if input_ids and isinstance(input_ids[0], list)
    else len(input_ids)
)


print("=" * 80)
print(f"Rendered prompt for {args.model} ({token_count} tokens)")
print("=" * 80)
print(rendered_prompt)
