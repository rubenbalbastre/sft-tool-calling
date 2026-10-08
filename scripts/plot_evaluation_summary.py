#!/usr/bin/env python3
"""Plot the three-run full-test evaluation summary used in the README."""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("docs/assets/figures")
CONFIGURATIONS = [
    {
        "label": "Base\nNo thinking",
        "color": "#4263A8",
        "success": (16.58, 0.29),
        "return": (0.230, 0.010),
        "latency": (15.75, 0.14),
    },
    {
        "label": "Base\nThinking",
        "color": "#D66B27",
        "success": (40.33, 1.66),
        "return": (0.476, 0.017),
        "latency": (95.36, 2.04),
    },
    {
        "label": "LoRA SFT\nNo thinking",
        "color": "#16866F",
        "success": (42.50, 0.35),
        "return": (0.567, 0.005),
        "latency": (20.31, 0.19),
    },
]


def plot_panel(axis, metric, title, ylabel, ylim, decimals, takeaway):
    for index, configuration in enumerate(CONFIGURATIONS):
        mean, deviation = configuration[metric]
        axis.bar(
            index,
            mean,
            width=0.58,
            color=configuration["color"],
            alpha=0.9,
            yerr=deviation,
            error_kw={
                "ecolor": "#252A31",
                "elinewidth": 1.15,
                "capsize": 3,
                "capthick": 1.15,
            },
            zorder=3,
        )
        axis.annotate(
            f"{mean:.{decimals}f} ± {deviation:.{decimals}f}",
            (index, mean),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="#252A31",
            fontsize=7.8,
            fontweight="bold",
        )

    axis.set_title(title, loc="left", pad=34)
    axis.text(
        0,
        1.02,
        takeaway,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="#16866F",
        fontsize=8,
        fontweight="bold",
    )
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    axis.set_xlim(-0.45, len(CONFIGURATIONS) - 0.55)
    axis.set_xticks(
        range(len(CONFIGURATIONS)),
        [configuration["label"] for configuration in CONFIGURATIONS],
    )
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.7)
    axis.tick_params(axis="both", colors="#3F4650", length=3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#7A828E")
    axis.spines[["left", "bottom"]].set_linewidth(0.8)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.4), facecolor="white")
    for axis in axes:
        axis.set_facecolor("white")

    plot_panel(
        axes[0],
        "success",
        "A   Task success",
        "Success rate (%)",
        (0, 52),
        2,
        "+25.9 pp vs base",
    )
    plot_panel(
        axes[1],
        "return",
        "B   Environment return",
        "Average return",
        (0, 0.66),
        3,
        "+0.337 vs base",
    )
    plot_panel(
        axes[2],
        "latency",
        "C   Episode latency",
        "Mean latency (s)",
        (0, 112),
        2,
        "4.7× faster than thinking",
    )

    figure.suptitle(
        "LoRA SFT recovers thinking-level quality without the latency cost",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.89,
        "Gemma 4 E2B · mean ± sample SD · 3 full-test runs · 400 episodes per run",
        ha="left",
        color="#5B6472",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.18,
        top=0.64,
        wspace=0.34,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        figure.savefig(
            OUTPUT_DIR / f"evaluation-summary.{extension}",
            dpi=220,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(figure)
    print(f"Saved evaluation figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
