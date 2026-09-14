"""Evaluate a local model with Transformers or a running vLLM server."""

import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import OmegaConf

# Direct file execution adds src/evaluation, not the repository root, to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.generate_sft_data import generate
from src.environment.env import SupplyChainEnvironment
from src.environment.tools import TOOLS
from src.evaluation.common import (
    DEFAULT_PROMPT,
    create_run_directory,
    episode_result,
    save_config,
    save_results,
)
from src.evaluation.vllm import VLLMServer


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

def chat_tools():
    """Convert Responses API tool schemas to Chat Completions schemas."""
    return [
        {
            "type": "function",
            "function": {
                key: value
                for key, value in tool.items()
                if key not in {"type", "strict"}
            },
        }
        for tool in TOOLS
    ]


def normalize_call(call):
    function = call.get("function", call)
    return {
        "name": function["name"],
        "arguments": function.get("arguments", {}),
    }


def parse_transformers_response(tokenizer, generated_ids):
    """Parse with the tokenizer, falling back to SmolLM3 XML tool calls."""
    try:
        return tokenizer.parse_response(generated_ids, tools=TOOLS)
    except AttributeError as error:
        if "response_template" not in str(error):
            raise

    text = (
        generated_ids
        if isinstance(generated_ids, str)
        else tokenizer.decode(generated_ids, skip_special_tokens=False)
    )
    matches = TOOL_CALL_PATTERN.findall(text)
    calls = []
    for index, match in enumerate(matches, 1):
        payload = json.loads(match)
        calls.append({
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": payload["name"],
                "arguments": payload.get("arguments", {}),
            },
        })

    content = TOOL_CALL_PATTERN.sub("", text).strip()
    return {"role": "assistant", "content": content, "tool_calls": calls}


def continue_conversation(messages, assistant_message, call, observation):
    messages.append(assistant_message)
    call_id = call.get("id", f"call_{len(messages)}")
    tool_name = normalize_call(call)["name"]

    if observation["role"] == "tool":
        messages.append({
            "role": "tool",
            "name": observation["name"],
            "tool_call_id": call_id,
            "content": json.dumps(observation["content"], ensure_ascii=False),
        })
        return

    messages.extend([
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": json.dumps(
                observation["tool_result"], ensure_ascii=False
            ),
        },
        {"role": "user", "content": observation["content"]},
    ])


class TransformersBackend:
    def __init__(self, model_path, max_new_tokens, device, enable_thinking):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device,
        )
        self.model.eval()

    async def generate(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            enable_thinking=self.enable_thinking,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0, input_length:]
        parsed = parse_transformers_response(self.tokenizer, generated_ids)
        calls = parsed.get("tool_calls") or []
        usage = {
            "input_tokens": input_length,
            "output_tokens": len(generated_ids),
            "total_tokens": input_length + len(generated_ids),
        }
        return parsed, calls, usage


class VLLMBackend:
    def __init__(
        self, model, base_url, api_key, max_new_tokens, enable_thinking
    ):
        from openai import AsyncOpenAI

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def generate(self, messages):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=chat_tools(),
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0,
            max_tokens=self.max_new_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                }
            },
        )
        message = response.choices[0].message
        assistant_message = message.model_dump(exclude_none=True)
        calls = [call.model_dump(exclude_none=True) for call in message.tool_calls or []]
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return assistant_message, calls, usage


async def run_episode(
    backend, row, prompt, max_steps, inference_semaphore=None
):
    """Run one ordered episode while allowing other episodes to make progress."""
    scenario = row["scenario"]
    user_request = row["messages"][0]["content"]
    env = SupplyChainEnvironment(scenario, user_request)
    env.reset()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_request},
    ]
    trace = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    started = time.perf_counter()

    for step_number in range(1, max_steps + 1):
        output_error = None
        try:
            if inference_semaphore:
                async with inference_semaphore:
                    assistant_message, calls, step_usage = await backend.generate(
                        messages
                    )
            else:
                assistant_message, calls, step_usage = await backend.generate(messages)
            for key in usage:
                usage[key] += step_usage[key]
            if len(calls) != 1:
                output_error = (
                    f"Model produced {len(calls)} tool calls; expected exactly one"
                )
                raise ValueError(output_error)

            call = calls[0]
            action = normalize_call(call)
            observation, reward, done, info = env.step(action)
            trace.append({
                "step": step_number,
                "action": action,
                "reward": reward,
                "state": env.state,
            })
        except Exception as error:
            output_error = output_error or f"Inference error: {error}"
            _, _, done, info = env.step(
                {"name": "invalid_model_output", "arguments": {}}
            )
            trace.append({"step": step_number, "error": output_error})

        if done:
            return episode_result(
                scenario, info, step_number, started, usage, trace, output_error
            )

        continue_conversation(messages, assistant_message, call, observation)

    _, _, _, info = env.step({"name": "invalid_model_output", "arguments": {}})
    return episode_result(
        scenario,
        info,
        max_steps,
        started,
        usage,
        trace,
        "Maximum steps reached",
    )


async def run_concurrent_episodes(backend, rows, prompt, max_steps, concurrency):
    """Keep up to `concurrency` model inference requests active at once."""
    semaphore = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*(
        run_episode(backend, row, prompt, max_steps, semaphore)
        for row in rows
    ))


def resolve_model_path(model):
    local_path = PROJECT_ROOT / model
    return str(local_path) if local_path.exists() else model


@hydra.main(config_path="../../config", config_name="eval", version_base=None)
def main(args):
    if args.backend not in {"transformers", "vllm"}:
        raise ValueError("backend must be 'transformers' or 'vllm'")

    model = resolve_model_path(args.model)
    output_root = PROJECT_ROOT / args.output_root
    run_directory = create_run_directory(output_root)

    config = {
        **OmegaConf.to_container(args, resolve=True),
        "prompt": DEFAULT_PROMPT,
        "tools": TOOLS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_config(run_directory, config)

    server = None

    if args.backend == "transformers":
        backend = TransformersBackend(
            model,
            args.max_new_tokens,
            args.device,
            args.enable_thinking,
        )
    else:
        server = VLLMServer(
            model=model,
            config_path=PROJECT_ROOT / args.vllm_server_config,
            base_url=args.base_url,
            timeout=args.vllm_startup_timeout,
            log_path=run_directory / "vllm.log",
            quantization=args.quantization,
        )
        server.start()
        try:
            backend = VLLMBackend(
                args.served_model_name,
                args.base_url,
                args.api_key,
                args.max_new_tokens,
                args.enable_thinking,
            )
        except Exception:
            server.stop()
            raise

    rows = generate(args.episodes, "evaluation", args.seed)
    try:
        concurrency = args.concurrency if args.backend == "vllm" else 1
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        results = asyncio.run(run_concurrent_episodes(
            backend,
            rows,
            DEFAULT_PROMPT,
            args.max_steps,
            concurrency,
        ))

        for index, result in enumerate(results, 1):
            print(
                f"[{index}/{args.episodes}] {result['kind']}: "
                f"{'PASS' if result['success'] else 'FAIL'}"
            )

        summary = save_results(run_directory, results)
        print(f"Saved evaluation run to {run_directory}")
        print(json.dumps(summary, indent=2))
    finally:
        if server:
            server.stop()


if __name__ == "__main__":
    main()
