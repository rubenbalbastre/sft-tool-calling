#!/usr/bin/env python3
"""Download SFT losses from W&B and plot them by optimizer step."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv


DEFAULT_RUN = "ruben-balbastre-uv/supply-chain-tool-calling/t505qhfc"
DEFAULT_OUTPUT_DIR = Path("docs/assets/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help="W&B run path in entity/project/run-id form.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the downloaded CSV and generated figures.",
    )
    return parser.parse_args()


def download_losses(run_path: str) -> list[dict[str, float | str]]:
    run = wandb.Api().run(run_path)
    rows = []
    history = run.scan_history(
        keys=["train/global_step", "train/loss", "eval/loss"],
        page_size=100,
    )

    for record in history:
        step = record.get("train/global_step")
        if step is None:
            continue
        if record.get("train/loss") is not None:
            rows.append(
                {
                    "optimizer_step": int(step),
                    "split": "Training",
                    "loss": record["train/loss"],
                }
            )
        if record.get("eval/loss") is not None:
            rows.append(
                {
                    "optimizer_step": int(step),
                    "split": "Validation",
                    "loss": record["eval/loss"],
                }
            )

    if not rows:
        raise RuntimeError(f"No training or validation losses found in {run_path}")
    return rows


def save_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["optimizer_step", "split", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def plot_losses(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    colors = {"Training": "#3B5BA7", "Validation": "#D55E00"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axis = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    axis.set_facecolor("white")
    for split in ("Training", "Validation"):
        points = [row for row in rows if row["split"] == split]
        axis.plot(
            [row["optimizer_step"] for row in points],
            [row["loss"] for row in points],
            color=colors[split],
            marker="o" if split == "Training" else "s",
            markersize=2.8 if split == "Training" else 4.5,
            markeredgewidth=0,
            linewidth=1.25 if split == "Training" else 2.0,
            alpha=0.72 if split == "Training" else 1.0,
            label=f"{split} loss",
            zorder=2 if split == "Training" else 3,
        )

    axis.set_title("Optimization dynamics", loc="left", pad=22)
    axis.text(
        0,
        1.025,
        "Gemma 4 E2B · LoRA SFT · assistant-only objective",
        transform=axis.transAxes,
        color="#5B6472",
        fontsize=9,
        va="bottom",
    )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_xlim(0, 49)
    axis.set_ylim(0, 0.145)
    axis.set_xticks(range(0, 49, 8))
    axis.grid(axis="y", color="#D8DCE3", linewidth=0.7, linestyle="--")
    axis.tick_params(axis="both", colors="#3F4650", length=3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#7A828E")
    axis.spines[["left", "bottom"]].set_linewidth(0.8)
    axis.legend(frameon=False, loc="upper right", handlelength=2.5)

    for split, offset in (("Training", -13), ("Validation", 9)):
        last = [row for row in rows if row["split"] == split][-1]
        axis.annotate(
            f"{float(last['loss']):.3f}",
            (last["optimizer_step"], last["loss"]),
            xytext=(-4, offset),
            textcoords="offset points",
            ha="right",
            va="center",
            color=colors[split],
            fontsize=8.5,
            fontweight="bold",
        )

    axis.text(
        1,
        -0.19,
        "Validation evaluated every 4 optimizer steps",
        transform=axis.transAxes,
        ha="right",
        color="#6B7280",
        fontsize=8,
    )
    fig.tight_layout()

    for extension in ("svg", "png"):
        fig.savefig(
            output_dir / f"sft-training-loss.{extension}",
            dpi=220,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    load_dotenv()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = download_losses(args.run)
    save_csv(rows, args.output_dir / "sft-training-loss.csv")
    plot_losses(rows, args.output_dir)
    print(f"Saved loss data and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
