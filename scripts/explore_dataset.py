from pathlib import Path
import sys

from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.preprocessing import prepare_sft_source


dataset = load_from_disk(PROJECT_ROOT / "data/pipeline/hf_dataset")["sft_train"]

examples = dataset.select(range(2))
example = examples[0]

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

prepared = prepare_sft_source(
    examples,
    enable_thinking=False,
    reasoning_effort="none",
)
prepared_example = prepared[0]
for index, message in enumerate(prepared_example["messages"]):
    print(
        index,
        f"role: {message['role']}",
        f"content: {message['content']}",
        f"tool_calls: {message.get('tool_calls', [])}",
        f"tool_call_id: {message.get('tool_call_id', '')}",
        sep="\n",
        end="\n\n",
    )
