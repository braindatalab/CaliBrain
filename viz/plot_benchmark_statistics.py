#!/usr/bin/env python
"""
Generate calibration metric summaries from benchmark CSV results.

The script filters the experiments to a single inverse estimator and a fixed
set of configuration parameters, then aggregates the selected runs by their
SNR level. For each SNR, it plots the mean and standard deviation for the
pre- and post-calibration variants of several metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

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


def summarize_metric(df: pd.DataFrame, metric: MetricSpec) -> pd.DataFrame:
    grouped = df.groupby(SNR_COLUMN)
    rows: List[Dict[str, float]] = []
    for snr_value, group in grouped:
        rows.append(
            {
                "snr": snr_value,
                "pre_mean": group[metric.pre_column].mean(),
                "pre_std": group[metric.pre_column].std(ddof=0),
                "post_mean": group[metric.post_column].mean(),
                "post_std": group[metric.post_column].std(ddof=0),
                "count": len(group),
            }
        )
    stats = pd.DataFrame(rows).sort_values("snr")
    if not stats.empty:
        stats = stats.fillna(0.0)
    return stats


def plot_metrics(
    df: pd.DataFrame,
    metrics: Iterable[MetricSpec],
    output_path: Path,
    show: bool = False,
    title_context: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        summary = summarize_metric(df, metric)
        if summary.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(metric.label)
            continue

        x_values = summary["snr"].to_numpy()

        ax.errorbar(
            x_values,
            summary["pre_mean"],
            yerr=summary["pre_std"],
            label="Pre-calibration",
            marker="o",
            linestyle="-",
            capsize=4,
            color="#1f77b4",
        )
        ax.errorbar(
            x_values,
            summary["post_mean"],
            yerr=summary["post_std"],
            label="Post-calibration",
            marker="s",
            linestyle="-",
            capsize=4,
            color="#d62728",
        )

        ax.set_title(metric.label)
        ax.set_xlabel("SNR (alpha_SNR)")
        ax.set_ylabel(metric.label)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend()

    # Handle potential unused axes if metrics list shorter than grid.
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    subtitle = title_context or ""
    fig.suptitle(f"Calibration metrics per SNR\n{subtitle}".strip(), fontsize=16)
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
    output_path = Path("analysis_results/benchmark_results/figures/benchmark_statistics.png")
    show_figure = False

    # Filters keyed by column name, e.g. {"nnz": 3, "orientation_type": "fixed"}
    filter_dict: Dict[str, object] = {
        "nnz": 3,
        "orientation_type": "fixed",
        "noise_type": "baseline",
        "subject": "CC120166",
    }
    # Add extra filters as string expressions "key=value" if desired.
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
