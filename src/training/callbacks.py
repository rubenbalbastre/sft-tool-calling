"""Training callbacks for executable environment validation."""

import asyncio
import json
import random

from transformers import TrainerCallback

from src.environment.procurement import SYSTEM_PROMPT
from src.evaluation.common import summarize
from src.evaluation.evaluate_local import (
    TransformersBackend,
    run_concurrent_episodes,
)


def validation_rollout_rows(dataset, count, seed):
    """Select a fixed, task-stratified set of unique scenarios."""
    rng = random.Random(seed)
    by_type = {}
    for row in dataset:
        scenario = json.loads(row["scenario_json"])
        by_type.setdefault(scenario["task_type"], {})[
            scenario["scenario_id"]
        ] = scenario

    groups = [list(scenarios.values()) for scenarios in by_type.values()]
    for group in groups:
        rng.shuffle(group)

    rows = []
    while len(rows) < count:
        added = False
        for group in groups:
            if group and len(rows) < count:
                rows.append({"scenario": group.pop()})
                added = True
        if not added:
            break
    rng.shuffle(rows)
    return rows


class EnvironmentValidationCallback(TrainerCallback):
    """Measure closed-loop task success after regular validation."""

    def __init__(
        self, rows, tokenizer, enable_thinking, reasoning_effort,
        evaluation_steps, max_steps=20, max_new_tokens=256,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.evaluation_steps = evaluation_steps
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if state.global_step % self.evaluation_steps:
            return control

        was_training = model.training
        model.eval()
        backend = TransformersBackend.from_model(
            model,
            self.tokenizer,
            self.max_new_tokens,
            self.enable_thinking,
            self.reasoning_effort,
        )
        try:
            results = asyncio.run(run_concurrent_episodes(
                backend,
                self.rows,
                SYSTEM_PROMPT,
                self.max_steps,
                concurrency=1,
            ))
        finally:
            if was_training:
                model.train()

        summary = summarize(results)["overall"]
        metrics["eval_environment_success_rate"] = summary["success_rate"]
        metrics["eval_environment_return"] = summary["average_return"]
        print(
            "Environment validation: "
            f"success={summary['success_rate']:.3f}, "
            f"return={summary['average_return']:.3f}"
        )
        return control
