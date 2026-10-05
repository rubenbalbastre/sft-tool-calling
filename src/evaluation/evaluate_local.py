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

from src.environment.procurement import (
    CHAT_TOOLS,
    TOOLS,
    ProcurementEnvironment,
    generate_scenarios,
)
from src.evaluation.common import (
    DEFAULT_PROMPT,
    create_run_directory,
    episode_result,
    save_config,
    save_results,
)
from src.evaluation.vllm import VLLMServer, resolve_model_and_adapter


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

def chat_tools():
    """Convert Responses API tool schemas to Chat Completions schemas."""
    return CHAT_TOOLS


def normalize_call(call):
    function = call.get("function", call)
    return {
        "name": function["name"],
        "arguments": function.get("arguments", {}),
    }


def parse_transformers_response(tokenizer, generated_ids, prefix_ids):
    """Parse with the tokenizer, falling back to SmolLM3 XML tool calls."""
    try:
        return tokenizer.parse_response(
            generated_ids,
            prefix=prefix_ids,
            tools=CHAT_TOOLS,
        )
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
    def __init__(
        self, model_path, max_new_tokens, device,
        enable_thinking, reasoning_effort, temperature,
    ):
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_class = (
            AutoPeftModelForCausalLM
            if (Path(model_path) / "adapter_config.json").is_file()
            else AutoModelForCausalLM
        )
        self.model = model_class.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device,
        )
        self.model.eval()

    async def generate(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=CHAT_TOOLS,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]

        with self.torch.inference_mode():
            generation_config = {
                "do_sample": self.temperature > 0,
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if self.temperature > 0:
                generation_config["temperature"] = self.temperature
            output = self.model.generate(
                **inputs,
                **generation_config,
            )

        generated_ids = output[0, input_length:]
        parsed = parse_transformers_response(
            self.tokenizer,
            generated_ids,
            inputs["input_ids"][0],
        )
        calls = parsed.get("tool_calls") or []
        usage = {
            "input_tokens": input_length,
            "output_tokens": len(generated_ids),
            "total_tokens": input_length + len(generated_ids),
        }
        return parsed, calls, usage


class VLLMBackend:
    def __init__(
        self, model, base_url, api_key, max_new_tokens,
        enable_thinking, reasoning_effort, temperature,
    ):
        from openai import AsyncOpenAI

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def generate(self, messages):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=chat_tools(),
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                    "reasoning_effort": self.reasoning_effort,
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
    user_request = scenario["user_request"]
    env = ProcurementEnvironment(scenario, max_steps=max_steps)
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
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            trace.append({
                "step": step_number,
                "action": action,
                "observation": observation,
                "reward": reward,
                "state": env.state,
            })
        except Exception as error:
            output_error = output_error or f"Inference error: {error}"
            _, _, terminated, truncated, info = env.step(
                {"name": "invalid_model_output", "arguments": {}}
            )
            done = terminated or truncated
            trace.append({"step": step_number, "error": output_error})

        if done:
            return episode_result(
                scenario,
                info,
                step_number,
                started,
                usage,
                trace,
                output_error,
                user_prompt=user_request,
            )

        continue_conversation(messages, assistant_message, call, observation)

    _, _, _, _, info = env.step({"name": "invalid_model_output", "arguments": {}})
    return episode_result(
        scenario,
        info,
        max_steps,
        started,
        usage,
        trace,
        "Maximum steps reached",
        user_prompt=user_request,
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
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")

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
            args.reasoning_effort,
            args.temperature,
        )
    else:
        base_model, adapter_path, adapter_rank = resolve_model_and_adapter(model)
        server = VLLMServer(
            model=base_model,
            config_path=PROJECT_ROOT / args.vllm_server_config,
            base_url=args.base_url,
            timeout=args.vllm_startup_timeout,
            log_path=run_directory / "vllm.log",
            served_model_name=args.served_model_name,
            quantization=args.quantization,
            adapter_path=adapter_path,
            adapter_rank=adapter_rank,
        )
        server.start()
        try:
            backend = VLLMBackend(
                args.served_model_name,
                args.base_url,
                args.api_key,
                args.max_new_tokens,
                args.enable_thinking,
                args.reasoning_effort,
                args.temperature,
            )
        except Exception:
            server.stop()
            raise

    rows = [
        {"scenario": scenario}
        for scenario in generate_scenarios(args.episodes, "evaluation", args.seed)
    ]
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
