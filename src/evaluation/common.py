"""Backend-neutral helpers for model evaluation."""

import json
import math
import random
import statistics
import time
from collections import defaultdict

from datasets import load_from_disk

from src.environment.procurement import SYSTEM_PROMPT


DEFAULT_PROMPT = SYSTEM_PROMPT


def load_evaluation_rows(
    dataset_path, split, prompt_variant, episodes, seed
):
    """Load unique held-out scenarios from a Hugging Face dataset split."""
    dataset = load_from_disk(str(dataset_path))
    if split not in dataset:
        raise ValueError(
            f"Evaluation split {split!r} is not present in {dataset_path}"
        )

    rows = []
    seen = set()
    for row in dataset[split]:
        if prompt_variant and row["prompt_variant"] != prompt_variant:
            continue
        scenario = json.loads(row["scenario_json"])
        row_id = (scenario["scenario_id"], row["prompt_variant"])
        if row_id not in seen:
            seen.add(row_id)
            rows.append({"scenario": scenario})

    if episodes is not None and episodes > len(rows):
        raise ValueError(
            f"Requested {episodes} episodes, but split {split!r} contains "
            f"only {len(rows)} unique scenario variants for prompt variant "
            f"{prompt_variant!r}"
        )

    random.Random(seed).shuffle(rows)
    return rows if episodes is None else rows[:episodes]


def create_run_directory(output_root):
    """Atomically create the next eval-NNNN directory."""
    output_root.mkdir(parents=True, exist_ok=True)
    run_number = 1
    while True:
        run_directory = output_root / f"eval-{run_number:04d}"
        try:
            run_directory.mkdir()
            return run_directory
        except FileExistsError:
            run_number += 1


def episode_result(
    scenario, info, steps, started, usage, trace, reason=None, user_prompt=None
):
    """Build the common result record returned by every backend."""
    return {
        "scenario_id": scenario["scenario_id"],
        "kind": scenario["kind"],
        "language": scenario["language"],
        "difficulty": scenario["difficulty"],
        "prompt": user_prompt,
        "success": info["success"],
        "episode_return": info["episode_return"],
        "steps": steps,
        "reason": info.get("reason") or reason,
        "metrics": info.get("metrics", {}),
        "latency_seconds": time.perf_counter() - started,
        "usage": usage,
        "trace": trace,
    }


def summarize(results):
    groups = defaultdict(list)
    for result in results:
        groups[result["kind"]].append(result)

    def metrics(rows):
        return {
            "episodes": len(rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "average_return": (
                sum(row["episode_return"] for row in rows) / len(rows)
            ),
            "average_steps": sum(row["steps"] for row in rows) / len(rows),
        }

    episode_latencies = sorted(row["latency_seconds"] for row in results)
    return {
        "overall": metrics(results),
        "by_kind": {
            kind: metrics(rows) for kind, rows in sorted(groups.items())
        },
        "tokens": {
            key: sum(row["usage"][key] for row in results)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        },
        "mean_episode_latency_seconds": (
            sum(episode_latencies) / len(episode_latencies)
        ),
        "median_episode_latency_seconds": statistics.median(episode_latencies),
        "p95_episode_latency_seconds": episode_latencies[
            math.ceil(0.95 * len(episode_latencies)) - 1
        ],
        "aggregate_episode_latency_seconds": sum(episode_latencies),
    }


def save_config(run_directory, config):
    (run_directory / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_results(run_directory, results):
    with (run_directory / "results.jsonl").open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = summarize(results)
    (run_directory / "results.json").write_text(
        json.dumps(
            {"summary": summary, "results": results},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
