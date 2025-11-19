#!/usr/bin/env python
"""
Visualize calibration metrics per SNR as paired violin plots with overlaid points.

This script filters benchmark results to a single inverse estimator and a fixed
set of configuration parameters, then creates a 3x2 grid of subplots showing
the distribution of pre- and post-calibration metrics for each SNR level.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

SNR_COLUMN = "alpha_SNR"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str

    @property
    def pre_column(self) -> str:
        return f"pre_cal_{self.name}"

    @property
    def post_column(self) -> str:
        return f"post_cal_{self.name}"


METRICS: List[MetricSpec] = [
    MetricSpec("mean_posterior_std", "Mean Posterior Std"),
    MetricSpec("mean_calibration_error", "Mean Calibration Error"),
    MetricSpec("mean_signed_deviation", "Mean Signed Deviation"),
    MetricSpec("mean_absolute_deviation", "Mean Absolute Deviation"),
    MetricSpec("max_overconfidence_deviation", "Max Overconfidence Deviation"),
    MetricSpec("emd", "Earth Mover's Distance"),
]

PRE_COLOR = "#1f77b4"
POST_COLOR = "#d62728"


def _coerce_value(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def filter_dataframe(df: pd.DataFrame, solver: str, filters: Dict[str, object]) -> pd.DataFrame:
    missing_cols = [
        col for col in [SNR_COLUMN, "solver", *(f.pre_column for f in METRICS), *(f.post_column for f in METRICS)]
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"The CSV is missing required columns: {missing_cols}")

    subset = df[df["solver"] == solver]
    if subset.empty:
        raise ValueError(f"No rows found for solver '{solver}'.")

    for key, value in filters.items():
        if key not in subset.columns:
            raise ValueError(f"Filter column '{key}' is not present in the CSV.")
        subset = subset[subset[key] == value]
        if subset.empty:
            raise ValueError(f"No rows left after applying filter {key}={value}.")
    return subset


def plot_metric_violin(ax: plt.Axes, df: pd.DataFrame, metric: MetricSpec) -> None:
    snr_values = np.sort(df[SNR_COLUMN].unique())
    if snr_values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    rng = np.random.default_rng(42)
    width = 0.35

    for idx, snr_value in enumerate(snr_values):
        snr_mask = df[SNR_COLUMN] == snr_value
        pre_vals = df.loc[snr_mask, metric.pre_column].dropna().to_numpy()
        post_vals = df.loc[snr_mask, metric.post_column].dropna().to_numpy()
        x_center = idx + 1  # positions start at 1 for nicer ticks
        pre_pos = x_center - width / 2
        post_pos = x_center + width / 2

        if pre_vals.size:
            viol = ax.violinplot(pre_vals, positions=[pre_pos], widths=width * 0.9, showextrema=False)
            for body in viol["bodies"]:
                body.set_facecolor(PRE_COLOR)
                body.set_edgecolor(PRE_COLOR)
                body.set_alpha(0.35)
            jitter = rng.normal(scale=0.02, size=pre_vals.size)
            ax.scatter(np.full(pre_vals.size, pre_pos) + jitter, pre_vals, color=PRE_COLOR, s=12, alpha=0.7)
        if post_vals.size:
            viol = ax.violinplot(post_vals, positions=[post_pos], widths=width * 0.9, showextrema=False)
            for body in viol["bodies"]:
                body.set_facecolor(POST_COLOR)
                body.set_edgecolor(POST_COLOR)
                body.set_alpha(0.35)
            jitter = rng.normal(scale=0.02, size=post_vals.size)
            ax.scatter(np.full(post_vals.size, post_pos) + jitter, post_vals, color=POST_COLOR, s=12, alpha=0.7)
        transform = ax.get_xaxis_transform()
        ax.text(pre_pos, 0.02, f"n_runs={pre_vals.size}", color=PRE_COLOR, fontsize=8, ha="center", va="bottom", transform=transform)
        ax.text(post_pos, 0.02, f"n_runs={post_vals.size}", color=POST_COLOR, fontsize=8, ha="center", va="bottom", transform=transform)

    ax.set_xticks(np.arange(1, len(snr_values) + 1))
    ax.set_xticklabels([f"{snr:g}" for snr in snr_values])


def plot_metrics(
    df: pd.DataFrame,
    metrics: Iterable[MetricSpec],
    output_path: Path,
    show: bool = False,
    title_context: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=False)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        plot_metric_violin(ax, df, metric)
        ax.set_title(metric.label)
        ax.set_xlabel("SNR (alpha_SNR)")
        ax.set_ylabel(metric.label)
        ax.grid(True, axis="y", alpha=0.2)

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    legend_handles = [
        Patch(facecolor=PRE_COLOR, edgecolor=PRE_COLOR, alpha=0.35, label="Pre-calibration"),
        Patch(facecolor=POST_COLOR, edgecolor=POST_COLOR, alpha=0.35, label="Post-calibration"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))

    subtitle = title_context or ""
    fig.suptitle(f"Calibration metric distributions per SNR\n{subtitle}".strip(), fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    # --- User-configurable section -------------------------------------------------
    path_to_csv = Path(
        "analysis_results/benchmark_results/benchmark_results_20251118_201651.csv"
    )
    solver_name = "BMN"
    output_path = Path("analysis_results/benchmark_results/figures/benchmark_violin_statistics.png")
    show_figure = False

    filter_dict: Dict[str, object] = {
        "nnz": 3,
        "orientation_type": "fixed",
        "noise_type": "baseline",
        "subject": "CC120166",
    }
    extra_filters: List[str] = []
    # -------------------------------------------------------------------------------

    if not path_to_csv.exists():
        raise FileNotFoundError(f"CSV file '{path_to_csv}' was not found.")

    for clause in extra_filters:
        if "=" not in clause:
            raise ValueError(f"Invalid filter expression '{clause}'. Expected format key=value.")
        key, value = clause.split("=", 1)
        filter_dict[key.strip()] = _coerce_value(value.strip())

    df = pd.read_csv(path_to_csv)
    subset = filter_dataframe(df, solver=solver_name, filters=filter_dict)

    filter_text = ", ".join(f"{key}={value}" for key, value in filter_dict.items()) if filter_dict else "No filters"
    context = f"solver={solver_name}; {filter_text}"

    plot_metrics(subset, METRICS, output_path=output_path, show=show_figure, title_context=context)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
