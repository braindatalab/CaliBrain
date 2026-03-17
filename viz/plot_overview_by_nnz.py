"""
Raincloud grid comparing metrics across solver × noise-type combinations,
but with nnz (number of active sources) on the x-axis instead of alpha_SNR.
"""

from __future__ import annotations
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "axes.titlesize": 14,
        "axes.labelsize": 16,
    }
)
from scipy.stats import gaussian_kde

DEFAULT_OUT_DIRNAME = "figures/calibration"
DEFAULT_OUT_BASENAME = "solver_noise_raincloud_by_nnz.png"
MEAN_STD_BASENAME = "solver_noise_meanSD_by_nnz.png"

C_PRE = "#1f77b4"
C_POST = "#d62728"
C_SINGLE = "#4A4A4A"
FACE_COLOR = "#F7F7F7"
GRID_COLOR = "#D8D8D8"
RNG = np.random.default_rng(2026)

VIOLIN_WIDTH = 0.32
VIOLIN_OFFSET = 0.18
BOX_WIDTH = 0.08
BOX_MARGIN_FACTOR = 0.5

# Match the subplot sizing used in plot_overview_by_snr.py
RAIN_FIG_WIDTH_SCALE = 3.2
RAIN_FIG_HEIGHT_SCALE = 2.4
RAIN_FIG_WIDTH_PAD = 2.0
RAIN_FIG_HEIGHT_PAD = 0.8
MEANSTD_FIG_WIDTH_SCALE = RAIN_FIG_WIDTH_SCALE
MEANSTD_FIG_HEIGHT_SCALE = RAIN_FIG_HEIGHT_SCALE
MEANSTD_FIG_WIDTH_PAD = RAIN_FIG_WIDTH_PAD
MEANSTD_FIG_HEIGHT_PAD = RAIN_FIG_HEIGHT_PAD

METRIC_ROWS = [
    {"key": "emd", "label": "EMD", "columns": ["emd"], "paired": False},
    {"key": "mean_posterior_std", "label": "MPS", "columns": ["mean_posterior_std"], "paired": False},
    {
        "key": "mean_signed_deviation",
        "label": "MSD",
        "columns": ["pre_cal_mean_signed_deviation", "post_cal_mean_signed_deviation"],
        "paired": True,
    },
    {
        "key": "mean_absolute_deviation",
        "label": "MAD",
        "columns": ["pre_cal_mean_absolute_deviation", "post_cal_mean_absolute_deviation"],
        "paired": True,
    },
    {
        "key": "max_underconfidence",
        "label": "Max Underconf.",
        "columns": ["pre_cal_max_underconfidence_deviation", "post_cal_max_underconfidence_deviation"],
        "paired": True,
    },
    {
        "key": "max_overconfidence",
        "label": "Max Overconf.",
        "columns": ["pre_cal_max_overconfidence_deviation", "post_cal_max_overconfidence_deviation"],
        "paired": True,
    },
]


def _clean(label: str) -> str:
    return label.replace("_", " ")


def _ensure_values(df: pd.DataFrame, column: str, requested: Optional[Sequence]) -> List:
    unique_vals = list(dict.fromkeys(df[column]))
    if requested:
        req = list(dict.fromkeys(requested))
        missing = sorted(set(req) - set(unique_vals))
        if missing:
            raise ValueError(f"{column}: requested values not found {missing}")
        return req
    return sorted(unique_vals)


def _apply_filter(df: pd.DataFrame, column: str, values: Optional[Sequence]) -> pd.DataFrame:
    if values:
        return df[df[column].isin(values)]
    return df


def _default_output_path(csv_path: str) -> Path:
    csv_path = Path(csv_path)
    out_dir = DEFAULT_OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / DEFAULT_OUT_BASENAME


def _format_nnzs(values: Sequence[int]) -> List[str]:
    return [str(int(v)) for v in values]


def _draw_half_violin(ax, data: np.ndarray, x_pos: float, side: str, color: str):
    clean = data[np.isfinite(data)]
    if clean.size == 0:
        return
    if np.allclose(clean, clean[0]):
        width = VIOLIN_WIDTH if side == "both" else VIOLIN_WIDTH / 2
        left = x_pos - width if side in {"left", "both"} else x_pos
        right = x_pos + width if side in {"right", "both"} else x_pos
        ax.plot([left, right], [clean[0], clean[0]], color=color, linewidth=2)
        return
    if clean.size == 1:
        ax.scatter(
            [x_pos],
            [clean[0]],
            color=color,
            s=STRIP_SIZE * 1.5,
            alpha=0.8,
            zorder=5,
        )
        return
    kde = gaussian_kde(clean)
    y = np.linspace(clean.min(), clean.max(), 400)
    density = kde(y)
    scale = density / density.max() * VIOLIN_WIDTH
    if side == "left":
        ax.fill_betweenx(y, x_pos - scale, x_pos, color=color, alpha=0.7, linewidth=0)
        ax.plot(x_pos - scale, y, color=color, linewidth=1.1)
    elif side == "right":
        ax.fill_betweenx(y, x_pos, x_pos + scale, color=color, alpha=0.7, linewidth=0)
        ax.plot(x_pos + scale, y, color=color, linewidth=1.1)
    else:
        ax.fill_betweenx(y, x_pos - scale, x_pos + scale, color=color, alpha=0.55, linewidth=0)
        ax.plot(x_pos - scale, y, color=color, linewidth=1.0)
        ax.plot(x_pos + scale, y, color=color, linewidth=1.0)


def _plot_metric_panel(ax, df: pd.DataFrame, config: dict, nnz_levels: Sequence[int]):
    ax.set_facecolor(FACE_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for x_idx, nnz in enumerate(nnz_levels):
        slice_df = df[df["nnz"] == nnz]
        if slice_df.empty:
            continue
        if config["paired"]:
            pre = slice_df[config["columns"][0]].to_numpy(dtype=float)
            post = slice_df[config["columns"][1]].to_numpy(dtype=float)
            pre_center = x_idx - VIOLIN_OFFSET
            post_center = x_idx + VIOLIN_OFFSET
            _draw_half_violin(ax, pre, pre_center, "left", C_PRE)
            _draw_half_violin(ax, post, post_center, "right", C_POST)
            for values, color, box_pos in (
                (pre, C_PRE, pre_center + VIOLIN_OFFSET * BOX_MARGIN_FACTOR),
                (post, C_POST, post_center - VIOLIN_OFFSET * BOX_MARGIN_FACTOR),
            ):
                clean = values[np.isfinite(values)]
                if clean.size:
                    bp = ax.boxplot(
                        clean,
                        positions=[box_pos],
                        widths=BOX_WIDTH * 0.9,
                        vert=True,
                        patch_artist=True,
                        boxprops=dict(facecolor=to_rgba(color, 0.35), linewidth=0.0),
                        medianprops=dict(color="#222222", linewidth=1.2),
                        whiskerprops=dict(color=color, linewidth=0.9),
                        capprops=dict(color=color, linewidth=0.9),
                        flierprops=dict(
                            marker="o",
                            markersize=3,
                            markerfacecolor=color,
                            alpha=0.5,
                            markeredgewidth=0,
                        ),
                    )
                    for artist in bp["boxes"]:
                        artist.set_edgecolor(color)
                        artist.set_linewidth(1.1)
                        artist.set_alpha(1.0)
        else:
            values = slice_df[config["columns"][0]].to_numpy(dtype=float)
            _draw_half_violin(ax, values, x_idx, "both", C_SINGLE)
            clean = values[np.isfinite(values)]
            if clean.size:
                bp = ax.boxplot(
                    clean,
                    positions=[x_idx],
                    widths=BOX_WIDTH,
                    vert=True,
                    patch_artist=True,
                    boxprops=dict(facecolor=to_rgba(C_SINGLE, 0.3), linewidth=0.0),
                    medianprops=dict(color="#ffffff", linewidth=1.1),
                    whiskerprops=dict(color=C_SINGLE, linewidth=0.9),
                    capprops=dict(color=C_SINGLE, linewidth=0.9),
                    flierprops=dict(
                        marker="o",
                        markersize=3,
                        markerfacecolor=C_SINGLE,
                        alpha=0.5,
                        markeredgewidth=0,
                    ),
                )
                for artist in bp["boxes"]:
                    artist.set_edgecolor(C_SINGLE)
                    artist.set_linewidth(1.1)
                    artist.set_alpha(1.0)
    ax.set_xticks(range(len(nnz_levels)))
    ax.set_xticklabels(_format_nnzs(nnz_levels))
    ax.set_xlim(-0.6, len(nnz_levels) - 0.4)


def build_figure(
    df: pd.DataFrame,
    combos: Iterable[Tuple[str, str]],
    nnz_levels: Sequence[int],
    display_names: dict[str, str] | None = None,
) -> matplotlib.figure.Figure:
    combos = list(combos)
    n_rows = len(METRIC_ROWS)
    n_cols = len(combos)
    if n_cols == 0:
        raise ValueError("No solver/noise combinations available after filtering.")
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            RAIN_FIG_WIDTH_SCALE * n_cols + RAIN_FIG_WIDTH_PAD,
            RAIN_FIG_HEIGHT_SCALE * n_rows + RAIN_FIG_HEIGHT_PAD,
        ),
        sharex=False,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    noise_labels = [noise for _, noise in combos]
    solver_spans: List[Tuple[str, int, int]] = []
    last_solver = None
    span_start = 0

    for col_idx, (solver, noise) in enumerate(combos):
        combo_df = df[(df["solver_canonical"] == solver) & (df["noise_type"] == noise)]
        for row_idx, config in enumerate(METRIC_ROWS):
            ax = axes[row_idx][col_idx]
            if combo_df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#777777", fontsize=10)
                ax.set_xticks(range(len(nnz_levels)))
                ax.set_xticklabels(_format_nnzs(nnz_levels))
                ax.set_facecolor(FACE_COLOR)
                continue
            _plot_metric_panel(ax, combo_df, config, nnz_levels)
            if col_idx == 0:
                ax.set_ylabel(config["label"])
            else:
                ax.set_ylabel("")
            if row_idx == n_rows - 1:
                ax.set_xlabel("nnz")
            else:
                ax.set_xlabel("")
        if last_solver is None:
            last_solver = solver
            span_start = col_idx
        elif solver != last_solver:
            solver_spans.append((last_solver, span_start, col_idx - 1))
            last_solver = solver
            span_start = col_idx
    if last_solver is not None:
        solver_spans.append((last_solver, span_start, n_cols - 1))

    for col_idx, noise in enumerate(noise_labels):
        axes[0][col_idx].set_title(_clean(noise), fontsize=17, fontweight="semibold", pad=1)

    fig.tight_layout()
    for solver, start_col, end_col in solver_spans:
        label = display_names.get(solver, solver) if display_names else solver
        left = axes[0][start_col].get_position()
        right = axes[0][end_col].get_position()
        center_x = (left.x0 + right.x1) / 2
        fig.text(
            center_x,
            left.y1 - 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=25,
            fontweight="bold",
        )

    handles = [
        Patch(facecolor=C_PRE, label="Pre calibration"),
        Patch(facecolor=C_POST, label="Post calibration"),
        Patch(facecolor=C_SINGLE, label="Single metric"),
    ]
    fig.legend(handles=handles, loc="center left", ncol=1, frameon=False, fontsize=15, bbox_to_anchor=(0.955, 0.5))
    fig.subplots_adjust(top=0.88, right=0.955, wspace=0.18, hspace=0.35)
    return fig


def build_mean_std_figure(
    df: pd.DataFrame,
    combos: Iterable[Tuple[str, str]],
    nnz_levels: Sequence[int],
    display_names: dict[str, str] | None = None,
) -> matplotlib.figure.Figure:
    combos = list(combos)
    n_rows = len(METRIC_ROWS)
    n_cols = len(combos)
    if n_cols == 0:
        raise ValueError("No solver/noise combinations available after filtering.")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            MEANSTD_FIG_WIDTH_SCALE * n_cols + MEANSTD_FIG_WIDTH_PAD,
            MEANSTD_FIG_HEIGHT_SCALE * n_rows + MEANSTD_FIG_HEIGHT_PAD,
        ),
        sharex=False,
    )
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    noise_labels = [noise for _, noise in combos]
    solver_spans: List[Tuple[str, int, int]] = []
    last_solver = None
    span_start = 0

    for col_idx, (solver, noise) in enumerate(combos):
        combo_df = df[(df["solver_canonical"] == solver) & (df["noise_type"] == noise)]
        for row_idx, config in enumerate(METRIC_ROWS):
            ax = axes[row_idx][col_idx]
            if combo_df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#777777", fontsize=10)
                ax.set_xticks(range(len(nnz_levels)))
                ax.set_xticklabels(_format_nnzs(nnz_levels))
                ax.set_facecolor(FACE_COLOR)
                continue
            _plot_mean_std_panel(ax, combo_df, config, nnz_levels)
            if col_idx == 0:
                ax.set_ylabel(config["label"])
            else:
                ax.set_ylabel("")
            if row_idx == n_rows - 1:
                ax.set_xlabel("nnz")
            else:
                ax.set_xlabel("")
        if last_solver is None:
            last_solver = solver
            span_start = col_idx
        elif solver != last_solver:
            solver_spans.append((last_solver, span_start, col_idx - 1))
            last_solver = solver
            span_start = col_idx
    if last_solver is not None:
        solver_spans.append((last_solver, span_start, n_cols - 1))

    for col_idx, noise in enumerate(noise_labels):
        axes[0][col_idx].set_title(_clean(noise), fontsize=17, fontweight="semibold", pad=3)

    fig.tight_layout()
    for solver, start_col, end_col in solver_spans:
        label = display_names.get(solver, solver) if display_names else solver
        left = axes[0][start_col].get_position()
        right = axes[0][end_col].get_position()
        center_x = (left.x0 + right.x1) / 2
        fig.text(
            center_x,
            left.y1 - 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=25,
            fontweight="bold",
        )

    handles = [
        Patch(facecolor=C_PRE, label="Pre calibration"),
        Patch(facecolor=C_POST, label="Post calibration"),
        Patch(facecolor=C_SINGLE, label="Single metric"),
    ]
    fig.legend(handles=handles, loc="center left", ncol=1, frameon=False, fontsize=15, bbox_to_anchor=(0.955, 0.5))
    fig.subplots_adjust(top=0.88, right=0.955, wspace=0.18, hspace=0.35)
    return fig


def _plot_mean_std_panel(ax, df: pd.DataFrame, config: dict, nnz_levels: Sequence[int]):
    ax.set_facecolor(FACE_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    series = {}

    def _add_point(key: str, values: np.ndarray, x: float, color: str):
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        mean = float(np.mean(values))
        std = float(np.std(values))
        bucket = series.setdefault(key, {"color": color, "x": [], "mean": [], "std": []})
        bucket["x"].append(x)
        bucket["mean"].append(mean)
        bucket["std"].append(std)

    for idx, nnz in enumerate(nnz_levels):
        subset = df[df["nnz"] == nnz]
        if subset.empty:
            continue
        if config["paired"]:
            pre = subset[config["columns"][0]].to_numpy(dtype=float)
            post = subset[config["columns"][1]].to_numpy(dtype=float)
            _add_point("pre", pre, idx, C_PRE)
            _add_point("post", post, idx, C_POST)
        else:
            vals = subset[config["columns"][0]].to_numpy(dtype=float)
            _add_point("single", vals, idx, C_SINGLE)

    for data in series.values():
        xs = np.array(data["x"])
        means = np.array(data["mean"])
        stds = np.array(data["std"])
        order = np.argsort(xs)
        xs, means, stds = xs[order], means[order], stds[order]
        color = data["color"]
        ax.plot(xs, means, color=color, linewidth=2.0, marker="o", markersize=4.8, alpha=0.95)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.18, linewidth=0)
        ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=color, elinewidth=1.4, capsize=4, alpha=0.9)

    ax.set_xticks(range(len(nnz_levels)))
    ax.set_xticklabels(_format_nnzs(nnz_levels))
    ax.set_xlim(-0.6, len(nnz_levels) - 0.4)


def main():
    csv_path = Path("results/benchmark_results/benchmark_results_20260317_030208.csv")
    output_path = "results/figures/calibration"
    raincloud_filename = DEFAULT_OUT_BASENAME
    meanstd_filename = MEAN_STD_BASENAME
    
    show = False
    dpi = 600
    solvers = ["BMN", "BMN_joint", "sflex_gamma_map", "sflex_gamma_lambda_map"]
    noise_types = ["baseline", "oracle", "adaptive_joint_learning"]
    subjects = ["CC120166", "CC120264", "CC120309", "CC120313"]
    orientation_types = ["fixed"]
    snr_values = [0.5]
    nnz_values = [1, 3, 5, 10, 100]

    
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    solvers = _ensure_values(df, "solver", solvers)
    noise_types = _ensure_values(df, "noise_type", noise_types)
    df = df[df["solver"].isin(solvers) & df["noise_type"].isin(noise_types)]
    if df.empty:
        raise SystemExit("No data left after solver/noise selection.")

    solver_aliases = {"BMN_joint": "BMN", "sflex_gamma_lambda_map": "sflex_gamma_map"}
    solver_display = {"BMN": "BMN", "sflex_gamma_map": "sFLEX-Gamma MAP"}
    df = df.copy()
    df["solver_canonical"] = df["solver"].map(solver_aliases).fillna(df["solver"])

    df = _apply_filter(df, "subject", subjects)
    df = _apply_filter(df, "orientation_type", orientation_types)
    df = _apply_filter(df, "alpha_SNR", snr_values)
    if nnz_values:
        df = df[df["nnz"].isin(nnz_values)]
    if df.empty:
        raise SystemExit("No data left after subject/orientation/SNR/nnz filtering.")

    nnz_levels = sorted(df["nnz"].unique())
    if not nnz_levels:
        raise SystemExit("No nnz values available.")

    canonical_order = []
    for solver in solvers:
        canonical = solver_aliases.get(solver, solver)
        if canonical not in canonical_order:
            canonical_order.append(canonical)
    if not canonical_order:
        canonical_order = sorted(df["solver_canonical"].unique())
    combos = list(product(canonical_order, noise_types))

    fig = build_figure(df, combos, nnz_levels, solver_display)
    provided = Path(output_path)
    if provided.is_dir():
        raincloud_path = provided / raincloud_filename
    else:
        raincloud_path = provided
    raincloud_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(raincloud_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved raincloud figure to {raincloud_path}")

    meanstd_fig = build_mean_std_figure(df, combos, nnz_levels, solver_display)
    meanstd_path = raincloud_path.parent / meanstd_filename
    meanstd_fig.savefig(meanstd_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved mean±SD figure to {meanstd_path}")
    plt.close(meanstd_fig)

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
