"""Backend-neutral helpers for model evaluation."""

import json
import time
from collections import defaultdict


DEFAULT_PROMPT = """You handle supply-chain material requests using tools.
If the user gives an explicit target plant ID (which is in the from CC-NN), call fulfillment directly.
Otherwise resolve the requested city. Clarify when multiple plants match, and
request a new location when none match. Preserve the original material,
quantity, unit, and date. Never invent a plant ID. Use only tool calls."""


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


def episode_result(scenario, info, steps, started, usage, trace, reason=None):
    """Build the common result record returned by every backend."""
    return {
        "scenario_id": scenario["scenario_id"],
        "kind": scenario["kind"],
        "language": scenario["language"],
        "difficulty": scenario["difficulty"],
        "success": info["success"],
        "episode_return": info["episode_return"],
        "steps": steps,
        "reason": info.get("reason") or reason,
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
        }

    return {
        "overall": metrics(results),
        "by_kind": {
            kind: metrics(rows) for kind, rows in sorted(groups.items())
        },
        "tokens": {
            key: sum(row["usage"][key] for row in results)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        },
        "latency_seconds": sum(row["latency_seconds"] for row in results),
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
