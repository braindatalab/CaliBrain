#!/usr/bin/env python
from __future__ import annotations

def plot_metric_violin_box(ax: plt.Axes, df: pd.DataFrame, metric: MetricSpec) -> int:
    snr_values = np.sort(df[SNR_COLUMN].unique())
    if metric.has_pre_post:
        counts_df = df[[metric.pre_column, metric.post_column]].dropna(how="all")
    else:
        counts_df = df[[metric.value_column]].dropna()
    total_samples = int(len(counts_df))
    if snr_values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return total_samples

    rng = np.random.default_rng(42)
    width = 0.35

    for idx, snr_value in enumerate(snr_values):
        snr_mask = df[SNR_COLUMN] == snr_value
        if metric.has_pre_post:
            pre_vals = df.loc[snr_mask, metric.pre_column].dropna().to_numpy()
            post_vals = df.loc[snr_mask, metric.post_column].dropna().to_numpy()
            x_center = idx + 1
            pre_pos = x_center - width / 2
            post_pos = x_center + width / 2

            if pre_vals.size:
                viol = ax.violinplot(pre_vals, positions=[pre_pos], widths=width * 0.9, showextrema=False)
                for body in viol["bodies"]:
                    body.set_facecolor(PRE_COLOR)
                    body.set_edgecolor(PRE_COLOR)
                    body.set_alpha(0.35)
                # Overlay boxplot
                bp = ax.boxplot(pre_vals, positions=[pre_pos], widths=width * 0.5, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(PRE_COLOR)
                    patch.set_alpha(0.7)
                for whisker in bp['whiskers']:
                    whisker.set_color(PRE_COLOR)
                for cap in bp['caps']:
                    cap.set_color(PRE_COLOR)
                for median in bp['medians']:
                    median.set_color('black')
            if post_vals.size:
                viol = ax.violinplot(post_vals, positions=[post_pos], widths=width * 0.9, showextrema=False)
                for body in viol["bodies"]:
                    body.set_facecolor(POST_COLOR)
                    body.set_edgecolor(POST_COLOR)
                    body.set_alpha(0.35)
                bp = ax.boxplot(post_vals, positions=[post_pos], widths=width * 0.5, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(POST_COLOR)
                    patch.set_alpha(0.7)
                for whisker in bp['whiskers']:
                    whisker.set_color(POST_COLOR)
                for cap in bp['caps']:
                    cap.set_color(POST_COLOR)
                for median in bp['medians']:
                    median.set_color('black')
        else:
            values = df.loc[snr_mask, metric.value_column].dropna().to_numpy()
            x_center = idx + 1
            if values.size:
                viol = ax.violinplot(values, positions=[x_center], widths=width * 0.9, showextrema=False)
                for body in viol["bodies"]:
                    body.set_facecolor(PRE_COLOR)
                    body.set_edgecolor(PRE_COLOR)
                    body.set_alpha(0.35)
                bp = ax.boxplot(values, positions=[x_center], widths=width * 0.5, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(PRE_COLOR)
                    patch.set_alpha(0.7)
                for whisker in bp['whiskers']:
                    whisker.set_color(PRE_COLOR)
                for cap in bp['caps']:
                    cap.set_color(PRE_COLOR)
                for median in bp['medians']:
                    median.set_color('black')

    ax.set_xticks(np.arange(1, len(snr_values) + 1))
    ax.set_xticklabels([f"{snr:g}" for snr in snr_values])
    return total_samples
def plot_violin_box_metrics(
    df: pd.DataFrame,
    metrics: Iterable[MetricSpec],
    output_path: Path,
    show: bool = False,
    title_context: Optional[str] = None,
) -> None:
    # Remove inf and -inf values globally before plotting
    df = _sanitize_dataframe(df)
    # Filter out metrics that are all-NaN for this solver
    filtered_metrics = []
    for metric in metrics:
        if metric.has_pre_post:
            col_pre = metric.pre_column
            col_post = metric.post_column
            all_nan = df[col_pre].isna().all() and df[col_post].isna().all()
        else:
            col_val = metric.value_column
            all_nan = df[col_val].isna().all()
        if not all_nan:
            filtered_metrics.append(metric)
    metrics = filtered_metrics
    n_metrics = len(metrics)
    nrows = 2
    ncols = int(np.ceil(n_metrics / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    n_subjects, seeds_per_subject, subject_seed_sample_count = _compute_subject_seed_stats(df)
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        fallback_count = plot_metric_violin_box(ax, df, metric)
        sample_count = subject_seed_sample_count if subject_seed_sample_count is not None else fallback_count
        label_suffix = f" (n={sample_count})"
        ax.set_xlabel("SNR")
        ax.set_ylabel(f"{metric.label}{label_suffix}")
        ax.grid(True, axis="y", alpha=0.2)

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    if any(metric.has_pre_post for metric in metrics):
        legend_handles = [
            Patch(facecolor=PRE_COLOR, edgecolor=PRE_COLOR, alpha=0.35, label="Pre-calibration"),
            Patch(facecolor=POST_COLOR, edgecolor=POST_COLOR, alpha=0.35, label="Post-calibration"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.89), fontsize=16)

    subtitle = title_context or ""
    extra_info = (
        f"Subjects: {n_subjects if n_subjects is not None else 'N/A'}, Seeds/Subject: {seeds_per_subject:.1f}"
        if n_subjects and seeds_per_subject is not None
        else ""
    )
    if extra_info:
        subtitle = f"{subtitle}\n{extra_info}" if subtitle else extra_info
    main_title = "Calibration and Evaluation Metrics by SNR"
    fig.suptitle(main_title, fontsize=18, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(0.5, 0.96, subtitle.strip(), ha="center", va="top", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.85))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_metric_boxplot(ax: plt.Axes, df: pd.DataFrame, metric: MetricSpec) -> int:
    snr_values = np.sort(df[SNR_COLUMN].unique())
    if metric.has_pre_post:
        counts_df = df[[metric.pre_column, metric.post_column]].dropna(how="all")
    else:
        counts_df = df[[metric.value_column]].dropna()
    total_samples = int(len(counts_df))
    if snr_values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return total_samples

    width = 0.35
    for idx, snr_value in enumerate(snr_values):
        snr_mask = df[SNR_COLUMN] == snr_value
        if metric.has_pre_post:
            pre_vals = df.loc[snr_mask, metric.pre_column].dropna().to_numpy()
            post_vals = df.loc[snr_mask, metric.post_column].dropna().to_numpy()
            x_center = idx + 1
            pre_pos = x_center - width / 2
            post_pos = x_center + width / 2

            if pre_vals.size:
                bp = ax.boxplot(pre_vals, positions=[pre_pos], widths=width * 0.9, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(PRE_COLOR)
                    patch.set_alpha(0.35)
                for whisker in bp['whiskers']:
                    whisker.set_color(PRE_COLOR)
                for cap in bp['caps']:
                    cap.set_color(PRE_COLOR)
                # Leave median line as default color for boxplot-only
            if post_vals.size:
                bp = ax.boxplot(post_vals, positions=[post_pos], widths=width * 0.9, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(POST_COLOR)
                    patch.set_alpha(0.35)
                for whisker in bp['whiskers']:
                    whisker.set_color(POST_COLOR)
                for cap in bp['caps']:
                    cap.set_color(POST_COLOR)
                for median in bp['medians']:
                    median.set_color('black')
        else:
            values = df.loc[snr_mask, metric.value_column].dropna().to_numpy()
            x_center = idx + 1
            if values.size:
                bp = ax.boxplot(values, positions=[x_center], widths=width * 0.9, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(PRE_COLOR)
                    patch.set_alpha(0.35)
                for whisker in bp['whiskers']:
                    whisker.set_color(PRE_COLOR)
                for cap in bp['caps']:
                    cap.set_color(PRE_COLOR)
                for median in bp['medians']:
                    median.set_color('black')

    ax.set_xticks(np.arange(1, len(snr_values) + 1))
    ax.set_xticklabels([f"{snr:g}" for snr in snr_values])
    return total_samples
def plot_boxplot_metrics(
    df: pd.DataFrame,
    metrics: Iterable[MetricSpec],
    output_path: Path,
    show: bool = False,
    title_context: Optional[str] = None,
) -> None:
    # Remove inf and -inf values globally before plotting
    df = _sanitize_dataframe(df)
    # Filter out metrics that are all-NaN for this solver
    filtered_metrics = []
    for metric in metrics:
        if metric.has_pre_post:
            col_pre = metric.pre_column
            col_post = metric.post_column
            all_nan = df[col_pre].isna().all() and df[col_post].isna().all()
        else:
            col_val = metric.value_column
            all_nan = df[col_val].isna().all()
        if not all_nan:
            filtered_metrics.append(metric)
    metrics = filtered_metrics
    n_metrics = len(metrics)
    nrows = 2
    ncols = int(np.ceil(n_metrics / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    n_subjects, seeds_per_subject, subject_seed_sample_count = _compute_subject_seed_stats(df)
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        fallback_count = plot_metric_boxplot(ax, df, metric)
        sample_count = subject_seed_sample_count if subject_seed_sample_count is not None else fallback_count
        label_suffix = f" (n={sample_count})"
        ax.set_xlabel("SNR")
        ax.set_ylabel(f"{metric.label}{label_suffix}")
        ax.grid(True, axis="y", alpha=0.2)

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    if any(metric.has_pre_post for metric in metrics):
        legend_handles = [
            Patch(facecolor=PRE_COLOR, edgecolor=PRE_COLOR, alpha=0.35, label="Pre-calibration"),
            Patch(facecolor=POST_COLOR, edgecolor=POST_COLOR, alpha=0.35, label="Post-calibration"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.89), fontsize=16)

    subtitle = title_context or ""
    extra_info = (
        f"Subjects: {n_subjects if n_subjects is not None else 'N/A'}, Seeds/Subject: {seeds_per_subject:.1f}"
        if n_subjects and seeds_per_subject is not None
        else ""
    )
    if extra_info:
        subtitle = f"{subtitle}\n{extra_info}" if subtitle else extra_info
    main_title = "Calibration and Evaluation Metrics by SNR"
    fig.suptitle(main_title, fontsize=18, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(0.5, 0.96, subtitle.strip(), ha="center", va="top", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.85))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


"""
Generate violin and summary calibration plots from benchmark CSV results using
a shared configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

pd.set_option("future.no_silent_downcasting", True)

SNR_COLUMN = "alpha_SNR"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    metric_type: str  # "evaluation" or "calibration"

    @property
    def has_pre_post(self) -> bool:
        return self.metric_type == "calibration"

    @property
    def value_column(self) -> str:
        return self.name if self.metric_type == "evaluation" else self.pre_column

    @property
    def pre_column(self) -> str:
        if self.metric_type == "calibration":
            return f"pre_cal_{self.name}"
        return self.name

    @property
    def post_column(self) -> str:
        if self.metric_type == "calibration":
            return f"post_cal_{self.name}"
        return None


METRICS: List[MetricSpec] = [
    # MetricSpec("jaccard_error", "Jaccard Error", "evaluation"),
    # MetricSpec("mse", "Mean Squared Error", "evaluation"),
    # MetricSpec("euclidean_distance", "Euclidean Distance", "evaluation"),
    # MetricSpec("f1", "F1 Score", "evaluation"),
    # MetricSpec("accuracy", "Accuracy", "evaluation"),
    # MetricSpec("mean_calibration_error", "Mean Calibration Error", "calibration"),
    MetricSpec("mean_signed_deviation", "Mean Signed Deviation", "calibration"),
    MetricSpec("mean_absolute_deviation", "Mean Absolute Deviation", "calibration"),
    MetricSpec("max_underconfidence_deviation", "Max Underconfidence Deviation", "calibration"),
    MetricSpec("max_overconfidence_deviation", "Max Overconfidence Deviation", "calibration"),
    
    MetricSpec("mean_posterior_std", "Mean Posterior Std", "evaluation"),
    MetricSpec("emd", "Earth Mover's Distance", "evaluation"),
    MetricSpec("gamma", "Gamma Norm", "evaluation"),
]

PRE_COLOR = "#1f77b4"
POST_COLOR = "#d62728"


def _compute_subject_seed_stats(df: pd.DataFrame) -> Tuple[Optional[int], Optional[float], Optional[int]]:
    n_subjects = df['subject'].nunique() if 'subject' in df.columns else None
    seeds_per_subject: Optional[float] = None
    sample_count: Optional[int] = None
    if n_subjects and 'run_id' in df.columns:
        seeds_per_subject = df.groupby('subject')['run_id'].nunique().mean()
        if seeds_per_subject is not None and not pd.isna(seeds_per_subject):
            sample_count = int(round(n_subjects * seeds_per_subject))
    return n_subjects, seeds_per_subject, sample_count


def _count_metric_samples(df: pd.DataFrame, metric: MetricSpec) -> int:
    if metric.has_pre_post:
        counts_df = df[[metric.pre_column, metric.post_column]].dropna(how="all")
    else:
        counts_df = df[[metric.value_column]].dropna()
    return int(len(counts_df))


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/- inf with NaN and keep object dtypes consistent."""
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    try:
        cleaned = cleaned.infer_objects(copy=False)
    except AttributeError:
        pass
    return cleaned


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


def _format_filter_text(filters: Dict[str, object]) -> str:
    if not filters:
        return "No filters"
    parts = []
    for key, value in filters.items():
        parts.append(f"{key}={value if value is not None else 'ALL'}")
    return ", ".join(parts)


def filter_dataframe(df: pd.DataFrame, solver: Optional[str], filters: Dict[str, object]) -> pd.DataFrame:
    missing_cols = [
        col
        for col in {
            SNR_COLUMN,
            "solver",
            *(
                f.pre_column
                for f in METRICS
                if f.has_pre_post
            ),
            *(
                f.post_column
                for f in METRICS
                if f.has_pre_post
            ),
            *(
                f.value_column
                for f in METRICS
                if not f.has_pre_post
            ),
        }
        if col and col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"The CSV is missing required columns: {missing_cols}")

    subset = df
    if solver is not None:
        subset = subset[subset["solver"] == solver]
        if subset.empty:
            raise ValueError(f"No rows found for solver '{solver}'.")

    for key, value in filters.items():
        if value is None:
            continue
        if key not in subset.columns:
            raise ValueError(f"Filter column '{key}' is not present in the CSV.")
        subset = subset[subset[key] == value]
        if subset.empty:
            raise ValueError(f"No rows left after applying filter {key}={value}.")
    return subset


def _resolve_filters(filter_dict: Dict[str, object], extra_filters: Sequence[str]) -> Dict[str, object]:
    resolved = dict(filter_dict)
    for clause in extra_filters:
        if "=" not in clause:
            raise ValueError(f"Invalid filter expression '{clause}'. Expected format key=value.")
        key, value = clause.split("=", 1)
        resolved[key.strip()] = _coerce_value(value.strip())
    return resolved


def load_filtered_dataframe(
    path_to_csv: Path,
    solver_name: Optional[str],
    filter_dict: Dict[str, object],
    extra_filters: Sequence[str],
) -> Tuple[pd.DataFrame, str]:
    if not path_to_csv.exists():
        raise FileNotFoundError(f"CSV file '{path_to_csv}' was not found.")

    resolved_filters = _resolve_filters(filter_dict, extra_filters)
    df = pd.read_csv(path_to_csv)
    subset = filter_dataframe(df, solver=solver_name, filters=resolved_filters)

    solver_context = solver_name if solver_name is not None else "ALL"
    filter_text = _format_filter_text(resolved_filters)
    context = f"solver={solver_context}; {filter_text}"
    return subset, context


def plot_metric_violin(ax: plt.Axes, df: pd.DataFrame, metric: MetricSpec) -> int:
    snr_values = np.sort(df[SNR_COLUMN].unique())
    if metric.has_pre_post:
        counts_df = df[[metric.pre_column, metric.post_column]].dropna(how="all")
    else:
        counts_df = df[[metric.value_column]].dropna()
    total_samples = int(len(counts_df))
    if snr_values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return total_samples

    rng = np.random.default_rng(42)
    width = 0.35

    for idx, snr_value in enumerate(snr_values):
        snr_mask = df[SNR_COLUMN] == snr_value
        if metric.has_pre_post:
            pre_vals = df.loc[snr_mask, metric.pre_column].dropna().to_numpy()
            post_vals = df.loc[snr_mask, metric.post_column].dropna().to_numpy()
            x_center = idx + 1
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
        else:
            values = df.loc[snr_mask, metric.value_column].dropna().to_numpy()
            x_center = idx + 1
            if values.size:
                viol = ax.violinplot(values, positions=[x_center], widths=width * 0.9, showextrema=False)
                for body in viol["bodies"]:
                    body.set_facecolor(PRE_COLOR)
                    body.set_edgecolor(PRE_COLOR)
                    body.set_alpha(0.35)
                jitter = rng.normal(scale=0.02, size=values.size)
                ax.scatter(np.full(values.size, x_center) + jitter, values, color=PRE_COLOR, s=12, alpha=0.7)

    ax.set_xticks(np.arange(1, len(snr_values) + 1))
    ax.set_xticklabels([f"{snr:g}" for snr in snr_values])
    return total_samples


def plot_violin_metrics(
    df: pd.DataFrame,
    metrics: Iterable[MetricSpec],
    output_path: Path,
    show: bool = False,
    title_context: Optional[str] = None,
) -> None:
    # Remove inf and -inf values globally before plotting
    df = _sanitize_dataframe(df)
    # Filter out metrics that are all-NaN for this solver
    filtered_metrics = []
    for metric in metrics:
        if metric.has_pre_post:
            col_pre = metric.pre_column
            col_post = metric.post_column
            all_nan = df[col_pre].isna().all() and df[col_post].isna().all()
        else:
            col_val = metric.value_column
            all_nan = df[col_val].isna().all()
        if not all_nan:
            filtered_metrics.append(metric)
    metrics = filtered_metrics
    n_metrics = len(metrics)
    nrows = 2
    ncols = int(np.ceil(n_metrics / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    n_subjects, seeds_per_subject, subject_seed_sample_count = _compute_subject_seed_stats(df)
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        fallback_count = plot_metric_violin(ax, df, metric)
        sample_count = subject_seed_sample_count if subject_seed_sample_count is not None else fallback_count
        label_suffix = f" (n={sample_count})"
        ax.set_xlabel("SNR")
        ax.set_ylabel(f"{metric.label}{label_suffix}")
        ax.grid(True, axis="y", alpha=0.2)

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    if any(metric.has_pre_post for metric in metrics):
        legend_handles = [
            Patch(facecolor=PRE_COLOR, edgecolor=PRE_COLOR, alpha=0.35, label="Pre-calibration"),
            Patch(facecolor=POST_COLOR, edgecolor=POST_COLOR, alpha=0.35, label="Post-calibration"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.89), fontsize=16)

    subtitle = title_context or ""
    extra_info = (
        f"Subjects: {n_subjects if n_subjects is not None else 'N/A'}, Seeds/Subject: {seeds_per_subject:.1f}"
        if n_subjects and seeds_per_subject is not None
        else ""
    )
    if extra_info:
        subtitle = f"{subtitle}\n{extra_info}" if subtitle else extra_info
    # Improved title
    main_title = "Distribution of Calibration and Evaluation Metrics by SNR"
    fig.suptitle(main_title, fontsize=18, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(0.5, 0.94, subtitle.strip(), ha="center", va="top", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.85))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def summarize_metric(df: pd.DataFrame, metric: MetricSpec) -> pd.DataFrame:
    grouped = df.groupby(SNR_COLUMN)
    rows: List[Dict[str, float]] = []
    for snr_value, group in grouped:
        if metric.has_pre_post:
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
        else:
            rows.append(
                {
                    "snr": snr_value,
                    "value_mean": group[metric.value_column].mean(),
                    "value_std": group[metric.value_column].std(ddof=0),
                    "count": len(group),
                }
            )
    stats = pd.DataFrame(rows).sort_values("snr")
    if not stats.empty:
        stats = stats.fillna(0.0)
    return stats


def plot_summary_metrics(
    df: pd.DataFrame,
    metrics: Iterable[MetricSpec],
    output_path: Path,
    show: bool = False,
    title_context: Optional[str] = None,
) -> None:
    # Remove inf and -inf values globally before plotting
    df = _sanitize_dataframe(df)
    # Filter out metrics that are all-NaN for this solver
    filtered_metrics = []
    for metric in metrics:
        if metric.has_pre_post:
            col_pre = metric.pre_column
            col_post = metric.post_column
            all_nan = df[col_pre].isna().all() and df[col_post].isna().all()
        else:
            col_val = metric.value_column
            all_nan = df[col_val].isna().all()
        if not all_nan:
            filtered_metrics.append(metric)
    metrics = filtered_metrics
    n_metrics = len(metrics)
    nrows = 2
    ncols = int(np.ceil(n_metrics / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()


    n_subjects, seeds_per_subject, subject_seed_sample_count = _compute_subject_seed_stats(df)
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        fallback_count = _count_metric_samples(df, metric)
        sample_count = subject_seed_sample_count if subject_seed_sample_count is not None else fallback_count
        summary = summarize_metric(df, metric)
        if summary.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("SNR")
            ax.set_ylabel(f"{metric.label} (n={sample_count})")
            continue

        x_values = summary["snr"].to_numpy()

        if metric.has_pre_post:
            ax.errorbar(
                x_values,
                summary["pre_mean"],
                yerr=summary["pre_std"],
                label="Pre-calibration",
                marker="o",
                linestyle="-",
                capsize=4,
                color=PRE_COLOR,
            )
            _fill_confidence_band(ax, x_values, summary["pre_mean"], summary["pre_std"], PRE_COLOR)
            ax.errorbar(
                x_values,
                summary["post_mean"],
                yerr=summary["post_std"],
                label="Post-calibration",
                marker="s",
                linestyle="-",
                capsize=4,
                color=POST_COLOR,
            )
            _fill_confidence_band(ax, x_values, summary["post_mean"], summary["post_std"], POST_COLOR)
        else:
            ax.errorbar(
                x_values,
                summary["value_mean"],
                yerr=summary["value_std"],
                label=metric.label,
                marker="o",
                linestyle="-",
                capsize=4,
                color=PRE_COLOR,
            )
            _fill_confidence_band(ax, x_values, summary["value_mean"], summary["value_std"], PRE_COLOR)

        ax.set_xlabel("SNR")
        ax.set_ylabel(f"{metric.label} (n={sample_count})")
        ax.grid(True, alpha=0.25)

    # Shared legend outside all subplots if any metric has pre/post
    if any(metric.has_pre_post for metric in metrics):
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=PRE_COLOR, edgecolor=PRE_COLOR, alpha=0.35, label="Pre-calibration"),
            Patch(facecolor=POST_COLOR, edgecolor=POST_COLOR, alpha=0.35, label="Post-calibration"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.87), ncol=2, fontsize=16)

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    subtitle = title_context or ""
    extra_info = (
        f"Subjects: {n_subjects if n_subjects is not None else 'N/A'}, Seeds/Subject: {seeds_per_subject:.1f}"
        if n_subjects and seeds_per_subject is not None
        else ""
    )
    if extra_info:
        subtitle = f"{subtitle}\n{extra_info}" if subtitle else extra_info
    # Improved title
    main_title = "Calibration and Evaluation Metrics by SNR (mean ± std)"
    fig.suptitle(main_title, fontsize=18, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.93, subtitle.strip(), ha="center", va="top", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.85))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def _fill_confidence_band(ax: plt.Axes, x_values: np.ndarray, mean: pd.Series, std: pd.Series, color: str) -> None:
    mean_vals = mean.to_numpy()
    std_vals = std.to_numpy()
    if len(x_values) < 2:
        ax.fill_between(x_values, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.12)
        return
    x_dense = np.linspace(x_values.min(), x_values.max(), len(x_values) * 50)
    mean_dense = np.interp(x_dense, x_values, mean_vals)
    std_dense = np.interp(x_dense, x_values, std_vals)
    ax.fill_between(
        x_dense,
        mean_dense - std_dense,
        mean_dense + std_dense,
        color=color,
        alpha=0.12,
    )


def run_violin_plot(
    path_to_csv: Path,
    solver_name: Optional[str],
    output_path: Path,
    filter_dict: Dict[str, object],
    extra_filters: Sequence[str],
    show_figure: bool = False,
    metrics: Iterable[MetricSpec] = METRICS,
) -> None:
    subset, context = load_filtered_dataframe(path_to_csv, solver_name, filter_dict, extra_filters)
    plot_violin_metrics(subset, metrics, output_path=output_path, show=show_figure, title_context=context)
    print(f"Saved violin figure to {output_path}")


def run_summary_plot(
    path_to_csv: Path,
    solver_name: Optional[str],
    output_path: Path,
    filter_dict: Dict[str, object],
    extra_filters: Sequence[str],
    show_figure: bool = False,
    metrics: Iterable[MetricSpec] = METRICS,
) -> None:
    subset, context = load_filtered_dataframe(path_to_csv, solver_name, filter_dict, extra_filters)
    plot_summary_metrics(subset, metrics, output_path=output_path, show=show_figure, title_context=context)
    print(f"Saved summary figure to {output_path}")


def _derive_output_paths(csv_path: Path, figures_root: Path) -> Tuple[Path, Path]:
    context_str = csv_path.stem.replace("benchmark_results_", "")
    violin_output_path = figures_root / f"violin_metrics_{context_str}.png"
    boxplot_output_path = figures_root / f"boxplot_metrics_{context_str}.png"
    summary_output_path = figures_root / f"summary_metrics_{context_str}.png"
    violin_box_output_path = figures_root / f"violinbox_metrics_{context_str}.png"
    return violin_output_path, boxplot_output_path, violin_box_output_path, summary_output_path


def main() -> None:
    # --- User-configurable section -------------------------------------------------
    filename = 'benchmark_results_20260106_190753'
    path_to_csv = Path("results/benchmark_results/") / f"{filename}.csv"
    solver_name: Optional[str] = "BMN"
    figures_root = Path("results/benchmark_results/") / filename
    show_violin = False
    show_summary = False

    filter_dict: Dict[str, object] = {
        "nnz": 10,
        "orientation_type": "fixed",
        "noise_type": "baseline",
        "subject": None,
    }
    extra_filters: List[str] = []
    # -------------------------------------------------------------------------------

    subset, context = load_filtered_dataframe(path_to_csv, solver_name, filter_dict, extra_filters)
    violin_output_path, boxplot_output_path, violin_box_output_path, summary_output_path = _derive_output_paths(path_to_csv, figures_root)

    if violin_output_path is not None:
        plot_violin_metrics(subset, METRICS, output_path=violin_output_path, show=show_violin, title_context=context)
        print(f"Saved violin figure to {violin_output_path}")

    if boxplot_output_path is not None:
        plot_boxplot_metrics(subset, METRICS, output_path=boxplot_output_path, show=show_violin, title_context=context)
        print(f"Saved boxplot figure to {boxplot_output_path}")

    if violin_box_output_path is not None:
        plot_violin_box_metrics(subset, METRICS, output_path=violin_box_output_path, show=show_violin, title_context=context)
        print(f"Saved violin+boxplot figure to {violin_box_output_path}")

    if summary_output_path is not None:
        plot_summary_metrics(subset, METRICS, output_path=summary_output_path, show=show_summary, title_context=context)
        print(f"Saved summary figure to {summary_output_path}")


if __name__ == "__main__":
    main()
