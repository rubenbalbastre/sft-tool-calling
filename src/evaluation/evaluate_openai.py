"""Evaluate an OpenAI model and prompt on the supply-chain environment."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Direct execution adds evaluation/ to sys.path, not the repository root.
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
    summarize,
)


def response_call(response):
    """Return the single function call, or an error describing model output."""
    calls = [item for item in response.output if item.type == "function_call"]
    if len(calls) != 1:
        return None, f"Model produced {len(calls)} tool calls; expected exactly one"
    return calls[0], None


def next_input(call, observation):
    """Convert an environment observation into Responses API input items."""
    if observation["role"] == "tool":
        return [{
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(observation["content"], ensure_ascii=False),
        }]

    # Clarification produces both the tool acknowledgement and a user selection.
    return [
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(observation["tool_result"], ensure_ascii=False),
        },
        {"role": "user", "content": observation["content"]},
    ]


def add_usage(total, response):
    usage = response.usage
    total["input_tokens"] += usage.input_tokens
    total["output_tokens"] += usage.output_tokens
    total["total_tokens"] += usage.total_tokens


def run_episode(client, row, model, prompt, reasoning_effort, max_steps):
    scenario = row["scenario"]
    user_request = row["messages"][0]["content"]
    env = SupplyChainEnvironment(scenario, user_request)
    env.reset()
    trace = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    previous_response_id = None
    model_input = user_request
    started = time.perf_counter()

    for step_number in range(1, max_steps + 1):
        request = {
            "model": model,
            "instructions": prompt,
            "input": model_input,
            "tools": TOOLS,
            "parallel_tool_calls": False,
            "reasoning": {"effort": reasoning_effort},
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        try:
            response = client.responses.create(**request)
        except Exception as error:
            _, _, _, info = env.step({
                "name": "invalid_model_output", "arguments": {}
            })
            return episode_result(
                scenario,
                info,
                step_number,
                started,
                usage,
                trace,
                f"API error: {error}",
            )
        add_usage(usage, response)
        call, output_error = response_call(response)

        if output_error:
            observation, reward, done, info = env.step({
                "name": "invalid_model_output", "arguments": {}
            })
            trace.append({"step": step_number, "error": output_error})
        else:
            action = {"name": call.name, "arguments": call.arguments}
            observation, reward, done, info = env.step(action)
            trace.append({
                "step": step_number,
                "action": action,
                "reward": reward,
                "state": env.state,
            })

        if done:
            return episode_result(
                scenario,
                info,
                step_number,
                started,
                usage,
                trace,
                output_error,
            )

        previous_response_id = response.id
        model_input = next_input(call, observation)

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "evals",
    )
    args = parser.parse_args()

    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    prompt = DEFAULT_PROMPT
    run_directory = create_run_directory(args.output_root)
    config = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "prompt": prompt,
        "tools": TOOLS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_config(run_directory, config)
    rows = generate(args.episodes, "evaluation", args.seed)
    client = OpenAI()
    results = []
    for index, row in enumerate(rows, 1):
        result = run_episode(
            client, row, args.model, prompt, args.reasoning_effort, args.max_steps
        )
        results.append(result)
        print(
            f"[{index}/{args.episodes}] {result['kind']}: "
            f"{'PASS' if result['success'] else 'FAIL'}"
        )

    summary = save_results(run_directory, results)
    print(f"Saved evaluation run to {run_directory}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
