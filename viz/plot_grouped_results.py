"""
Grouped figures for CaliBrain benchmark results.

Figures are organized into meaningful groups:
  1.  Solver comparison          – which estimator performs best?
  2.  Noise-type comparison      – oracle vs baseline (vs CV)
  3.  SNR effect                 – how does SNR affect performance?
  4.  NNZ effect                 – how does nnz affect performance?
  5.  Calibration improvement    – pre / post / improvement side-by-side
  6.  Subject variability        – consistency across subjects
  7.  Interaction: SNR × Solver  – heat-maps
  8.  Interaction: nnz × Solver  – heat-maps
  9.  Pre vs Post scatter        – per calibration metric

All grouping values are discovered dynamically from the CSV — nothing is
hard-coded. Only the requested metrics are used.

Every figure subtitle states explicitly what the distribution is computed over
(e.g. "across nruns=2, n_subjects=5, nnz=[10, 20], noise_type=[baseline, oracle]").
Individual data points are overlaid wherever the plot type allows it.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit only this block)
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH   = "results/benchmark_results/benchmark_results_20260306_172109.csv"
OUTPUT_DIR = "results/benchmark_results/figures/grouped_results"        # None → auto-detected next to CSV
SAVE_DPI   = 200
SHOW       = False       # set True to display figures interactively
STRIP_SIZE  = 10         # size of individual data-point markers
STRIP_ALPHA = 0.45       # transparency of individual data points

# ── Unified colour system ──────────────────────────────────────────────────
# Background elements (boxes, violins, bars, lines)
C_PRE    = "#E07B54"   # Pre-calibration  – warm sienna
C_POST   = "#5B9BD5"   # Post-calibration – steel blue
C_SINGLE = "#888888"   # Single-value metrics – neutral gray
C_DELTA  = "#70AD47"   # Improvement/Δ metrics – sage green
# Solver-differentiated background elements: Set2 muted/pastel palette
SOLVER_PALETTE = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
                  "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]
# Individual data-point colours (nnz): Dark2 – saturated, clearly distinct
# from the muted Set2 backgrounds and the warm/cool Pre/Post colours
NNZ_PALETTE  = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a",
                "#66a61e", "#e6ab02", "#a6761d", "#666666"]
# Markers per subject / secondary variable – 12 distinct filled markers,
# enough for up to 10 SNR levels or subjects with room to spare
SUBJ_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "<", ">"]
# Colors for per-group confidence ellipses (tab10 – distinct from all other palettes)
_ELLIPSE_PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
# ─────────────────────────────────────────────────────────────────────────────

# Calibration base names (pre/post/improvement columns are built from these)
CAL_BASE_METRICS = [
    "mean_calibration_error",
    "max_underconfidence_deviation",
    "max_overconfidence_deviation",
    "mean_absolute_deviation",
    "mean_signed_deviation",
]

# Single-value metrics (no pre/post variants)
SINGLE_METRICS = ["emd", "mean_posterior_std"]

# All metrics to consider (detected dynamically against CSV columns)
def _build_metric_list():
    cols = list(SINGLE_METRICS)
    for base in CAL_BASE_METRICS:
        cols.append(f"pre_cal_{base}")
        cols.append(f"post_cal_{base}")
        cols.append(f"improvement_{base}")
    return cols

ALL_METRICS = _build_metric_list()

# Nicer display labels
METRIC_LABELS = {
    "emd": "EMD",
    "mean_posterior_std": "Mean Posterior Std",
    **{f"pre_cal_{b}":   f"Pre  {b.replace('_', ' ').title()}"   for b in CAL_BASE_METRICS},
    **{f"post_cal_{b}":  f"Post {b.replace('_', ' ').title()}"   for b in CAL_BASE_METRICS},
    **{f"improvement_{b}": f"Δ {b.replace('_', ' ').title()} (%)" for b in CAL_BASE_METRICS},
}

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def available_metrics(df: pd.DataFrame) -> list:
    return [m for m in ALL_METRICS if m in df.columns]


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"  ✓  {path.relative_to(path.parents[3]) if path.parents[3].exists() else path}")
    if SHOW:
        plt.show()
    plt.close(fig)


def label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def dist_info(df: pd.DataFrame, on_axes: list) -> str:
    """
    Build a subtitle string describing what dimensions the distribution is
    computed over (all experimental variables NOT shown on the axes / hue).

    Parameters
    ----------
    df       : full (or filtered) dataframe for this figure
    on_axes  : list of column names already shown on the plot axes / hue
    """
    def _fmt(vals):
        """Return a clean list-like string, stripping numpy type wrappers."""
        cleaned = []
        for v in vals:
            try:
                f = float(v)
                cleaned.append(int(f) if f == int(f) else round(f, 4))
            except (TypeError, ValueError):
                cleaned.append(str(v))
        if len(cleaned) == 1:
            
            return str(cleaned[0])
        return "[" + ", ".join(str(c) for c in cleaned) + "]"

    dim_order = ["run_id", "subject", "nnz", "alpha_SNR", "noise_type", "solver"]
    parts = []
    for dim in dim_order:
        if dim in on_axes or dim not in df.columns:
            continue
        vals = sorted(df[dim].unique())
        n    = len(vals)
        if dim == "run_id":
            parts.append(f"n runs = {n}")
        elif dim == "subject":
            parts.append(f"n subjects = {n}")
        elif dim == "nnz":
            parts.append(f"nnz = {_fmt(vals)}")
        elif dim == "alpha_SNR":
            parts.append(f"SNR = {_fmt(vals)}")
        elif dim == "noise_type":
            parts.append(f"noise type = {_fmt(vals)}")
        elif dim == "solver":
            parts.append(f"solver = {_fmt(vals)}")
    return "Distribution across:\n" + "\n".join(parts)


def _add_dist_box(fig, text: str):
    """
    Render *text* inside a styled rounded box on the right side of the figure,
    below the legend.  Each parameter is listed on its own line.
    """
    fig.text(
        1.02, 0.32,
        text,
        transform=fig.transFigure,
        ha="left", va="top",
        fontsize=8.5, style="italic", color="#333333",
        multialignment="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f5f5f5",
            edgecolor="#aaaaaa",
            alpha=0.92,
            linewidth=0.9,
        ),
    )


def _build_paired_panels(metrics: list) -> list:
    """
    Return an ordered list of panel specifications for line-plot figures:
      [(title_str, [(col, linestyle, marker), ...]), ...]

    Pre- and post-calibration columns for the *same* base metric are collapsed
    into a single panel (solid line = pre, dashed line = post) so they can be
    compared directly.  EMD and improvement metrics appear as standalone panels.
    """
    panels    = []
    processed = set()

    if "emd" in metrics:
        panels.append(("EMD", [("emd", "-", "o")]))
        processed.add("emd")

    for base in CAL_BASE_METRICS:
        pre = f"pre_cal_{base}"
        post = f"post_cal_{base}"
        imp  = f"improvement_{base}"

        pre_in  = pre  in metrics and pre  not in processed
        post_in = post in metrics and post not in processed

        if pre_in and post_in:
            panels.append((
                base.replace("_", " ").title(),
                [(pre, "-", "o"), (post, "--", "s")]
            ))
            processed.update({pre, post})
        elif pre_in:
            panels.append((label(pre),  [(pre,  "-", "o")]))
            processed.add(pre)
        elif post_in:
            panels.append((label(post), [(post, "-", "o")]))
            processed.add(post)

        if imp in metrics and imp not in processed:
            panels.append((f"Δ {base.replace('_', ' ').title()}", [(imp, "-", "o")]))
            processed.add(imp)

    for m in metrics:
        if m not in processed:
            panels.append((label(m), [(m, "-", "o")]))

    return panels


def _make_nnz_subj_maps(df: pd.DataFrame):
    """
    Return (nnz_color_map, subj_marker_map) dicts for the given dataframe.
    Color is keyed by nnz value; marker is keyed by subject.
    """
    nnz_vals  = sorted(df["nnz"].unique())
    subj_vals = sorted(df["subject"].unique())
    nnz_color   = {v: NNZ_PALETTE[i % len(NNZ_PALETTE)]    for i, v in enumerate(nnz_vals)}
    subj_marker = {s: SUBJ_MARKERS[i % len(SUBJ_MARKERS)]  for i, s in enumerate(subj_vals)}
    return nnz_color, subj_marker


def _scatter_pts(ax, data: pd.DataFrame, x_pos, y_col: str,
                 nnz_color: dict, subj_marker: dict,
                 jitter_scale: float = 0.05,
                 dodge_offset: float = 0.0,
                 size: float = None, alpha: float = None):
    """
    Draw individual data points for one box/bar group.

    Each point is colored by its ``nnz`` value and shaped by its ``subject``
    value.  ``x_pos`` is the numeric x-centre of the box/bar; ``jitter_scale``
    controls horizontal spread; ``dodge_offset`` shifts the cloud left/right
    when box-dodge is active.
    """
    size  = size  or STRIP_SIZE * 4
    alpha = alpha or STRIP_ALPHA
    rng   = np.random.default_rng(abs(hash(y_col)) % (2**31))
    sub   = data[[y_col, "nnz", "subject"]].dropna(subset=[y_col])
    jitter = rng.uniform(-jitter_scale, jitter_scale, len(sub))
    for (nnz_val, subj), grp in sub.groupby(["nnz", "subject"]):
        idx  = grp.index
        ax.scatter(
            x_pos + dodge_offset + jitter[sub.index.get_indexer(idx)],
            grp[y_col].values,
            color=nnz_color.get(nnz_val, "gray"),
            marker=subj_marker.get(subj, "o"),
            s=size, alpha=alpha, zorder=5,
            edgecolors="white", linewidths=0.3,
        )


def _nnz_subj_legend_handles(nnz_color: dict, subj_marker: dict) -> list:
    """Build legend handle list: nnz color patches + subject marker lines."""
    handles = []
    for nnz_val, color in nnz_color.items():
        handles.append(mpatches.Patch(facecolor=color, label=f"nnz={int(nnz_val)}"))
    for subj, marker in subj_marker.items():
        handles.append(mlines.Line2D([], [], color="gray", marker=marker,
                                     linestyle="None", markersize=6,
                                     label=subj))
    return handles


def _confidence_ellipse(ax, x, y, color, n_std=1.5, alpha=0.12, lw=1.2):
    """
    Draw a covariance-based confidence ellipse at *n_std* standard deviations.
    Skips silently when fewer than 3 points are available.
    """
    from matplotlib.patches import Ellipse
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # sort descending
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle  = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(max(eigenvalues[0], 0))
    height = 2 * n_std * np.sqrt(max(eigenvalues[1], 0))
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=angle,
        facecolor=color, alpha=alpha,
        edgecolor=color, linewidth=lw, linestyle="--",
        zorder=2,
    )
    ax.add_patch(ell)


def _half_violin(ax, data, x_pos: float, side: str, color,
                 width: float = 0.35, alpha: float = 0.75,
                 linewidth: float = 0.9):
    """
    Draw a KDE-based half (or full) violin at *x_pos*.

    Parameters
    ----------
    side : 'left' | 'right' | 'both'
        'left'  → filled left of x_pos  (Pre-cal)
        'right' → filled right of x_pos (Post-cal)
        'both'  → symmetric full violin  (single-valued metrics)
    """
    from scipy.stats import gaussian_kde

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        if len(data) == 1:
            ax.scatter([x_pos], data, color=color, s=25, alpha=alpha, zorder=5)
        return
    if np.unique(data).size < 2:            # all identical – draw a tick line
        med = data[0]
        if side == "left":
            ax.plot([x_pos - width, x_pos], [med, med], color=color, linewidth=2)
        elif side == "right":
            ax.plot([x_pos, x_pos + width], [med, med], color=color, linewidth=2)
        else:
            ax.plot([x_pos - width, x_pos + width], [med, med], color=color, linewidth=2)
        return

    try:
        kde    = gaussian_kde(data, bw_method="scott")
    except Exception:
        return

    y_vals = np.linspace(data.min(), data.max(), 300)
    dens   = kde(y_vals)
    dens_s = dens / dens.max() * width        # normalise to requested half-width

    if side == "left":
        ax.fill_betweenx(y_vals, x_pos - dens_s, x_pos,
                         alpha=alpha, color=color, linewidth=0)
        ax.plot(x_pos - dens_s, y_vals, color=color, linewidth=linewidth)
    elif side == "right":
        ax.fill_betweenx(y_vals, x_pos, x_pos + dens_s,
                         alpha=alpha, color=color, linewidth=0)
        ax.plot(x_pos + dens_s, y_vals, color=color, linewidth=linewidth)
    else:                                     # symmetric
        ax.fill_betweenx(y_vals, x_pos - dens_s, x_pos + dens_s,
                         alpha=alpha, color=color, linewidth=0)
        ax.plot(x_pos - dens_s, y_vals, color=color, linewidth=linewidth)
        ax.plot(x_pos + dens_s, y_vals, color=color, linewidth=linewidth)

    # White median tick
    med   = np.median(data)
    med_d = float(kde(np.array([med]))[0]) / dens.max() * width
    if side == "left":
        ax.plot([x_pos - med_d, x_pos], [med, med],
                color="white", linewidth=1.8, zorder=7, solid_capstyle="round")
    elif side == "right":
        ax.plot([x_pos, x_pos + med_d], [med, med],
                color="white", linewidth=1.8, zorder=7, solid_capstyle="round")
    else:
        ax.plot([x_pos - med_d, x_pos + med_d], [med, med],
                color="white", linewidth=1.8, zorder=7, solid_capstyle="round")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Solver comparison
#    One panel per base metric: Pre and Post side-by-side (hue=Phase, x=solver)
#    Standalone panels for EMD and improvement metrics
# ─────────────────────────────────────────────────────────────────────────────
def fig_solver_comparison(df: pd.DataFrame, metrics: list, out: Path):
    print("\n[1] Solver comparison")
    solvers     = sorted(df["solver"].unique())
    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]

    # Build panel list: (title, "paired"|("single"), col_or_tuple)
    panels = []
    if "emd" in df.columns:
        panels.append(("EMD", "single", "emd"))
    if "mean_posterior_std" in df.columns:
        panels.append(("Mean Post. Std", "single", "mean_posterior_std"))
    for base in valid_bases:
        panels.append((
            base.replace("_", " ").title(), "paired",
            (f"pre_cal_{base}", f"post_cal_{base}")
        ))
    for base in valid_bases:
        imp = f"improvement_{base}"
        if imp in df.columns:
            panels.append((f"\u0394 {base.replace('_', ' ').title()}", "single", imp))

    n             = len(panels)
    ncols         = 3
    nrows         = (n + ncols - 1) // ncols
    phase_palette  = {"Pre": C_PRE, "Post": C_POST}
    solver_palette = SOLVER_PALETTE[:len(solvers)]
    nnz_color, subj_marker = _make_nnz_subj_maps(df)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, (title, kind, cols) in enumerate(panels):
        ax = axes[idx]
        if kind == "paired":
            pre_col, post_col = cols
            melted = pd.melt(
                df[["solver", "nnz", "subject", pre_col, post_col]],
                id_vars=["solver", "nnz", "subject"], value_vars=[pre_col, post_col],
                var_name="_c", value_name="value",
            )
            melted["Phase"] = melted["_c"].map({pre_col: "Pre", post_col: "Post"})
            sns.boxplot(
                data=melted, x="solver", y="value",
                hue="Phase", hue_order=["Pre", "Post"],
                order=solvers, palette=phase_palette,
                width=0.55, dodge=True, legend=False,
                flierprops=dict(marker="", alpha=0), ax=ax,
            )
            # Scatter: pre left (-0.14), post right (+0.14) within each solver tick
            for s_idx, solver in enumerate(solvers):
                for phase, pre_c, offset in (("Pre", pre_col, -0.14), ("Post", post_col, +0.14)):
                    sub = df[df["solver"] == solver][[pre_c, "nnz", "subject"]].rename(columns={pre_c: "value"})
                    _scatter_pts(ax, sub, s_idx, "value", nnz_color, subj_marker,
                                 jitter_scale=0.08, dodge_offset=offset)
        else:
            sns.boxplot(
                data=df, x="solver", y=cols,
                order=solvers, hue="solver", hue_order=solvers,
                palette=solver_palette, legend=False,
                flierprops=dict(marker="", alpha=0), ax=ax,
            )
            for s_idx, solver in enumerate(solvers):
                sub = df[df["solver"] == solver][[cols, "nnz", "subject"]].rename(columns={cols: "value"})
                _scatter_pts(ax, sub, s_idx, "value", nnz_color, subj_marker)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)

    for ax in axes[n:]:
        ax.set_visible(False)

    leg_handles = [
        mpatches.Patch(facecolor=phase_palette["Pre"],  label="Pre-cal"),
        mpatches.Patch(facecolor=phase_palette["Post"], label="Post-cal"),
        mlines.Line2D([], [], color="none", label=""),
    ] + _nnz_subj_legend_handles(nnz_color, subj_marker)
    fig.legend(handles=leg_handles, loc="center left", ncol=1,
               bbox_to_anchor=(1.01, 0.5))

    subtitle = dist_info(df, on_axes=["solver"])
    fig.suptitle("Distribution by Solver  (Pre vs Post within each metric)",
                 fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "1_solver_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Noise-type comparison
#    One panel per base metric: x = noise_type × Phase (interleaved),
#    hue = solver.  Dotted separator lines visually group Pre/Post pairs.
# ─────────────────────────────────────────────────────────────────────────────
def fig_noise_type_comparison(df: pd.DataFrame, metrics: list, out: Path):
    print("[2] Noise-type comparison")
    noise_types = sorted(df["noise_type"].unique())
    solvers     = sorted(df["solver"].unique())
    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]

    panels = []
    if "emd" in df.columns:
        panels.append(("EMD", "single", "emd"))
    if "mean_posterior_std" in df.columns:
        panels.append(("Mean Post. Std", "single", "mean_posterior_std"))
    for base in valid_bases:
        panels.append((
            base.replace("_", " ").title(), "paired",
            (f"pre_cal_{base}", f"post_cal_{base}")
        ))
    for base in valid_bases:
        imp = f"improvement_{base}"
        if imp in df.columns:
            panels.append((f"\u0394 {base.replace('_', ' ').title()}", "single", imp))

    # Combined x-order: for each noise_type, Pre then Post
    x_order_paired = [f"{nt}\n{ph}" for nt in noise_types for ph in ("Pre", "Post")]
    n     = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    palette = SOLVER_PALETTE[:len(solvers)]
    nnz_color, subj_marker = _make_nnz_subj_maps(df)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, (title, kind, cols) in enumerate(panels):
        ax = axes[idx]
        if kind == "paired":
            pre_col, post_col = cols
            melted = pd.melt(
                df[["solver", "noise_type", "nnz", "subject", pre_col, post_col]],
                id_vars=["solver", "noise_type", "nnz", "subject"],
                value_vars=[pre_col, post_col],
                var_name="_c", value_name="value",
            )
            melted["Phase"]   = melted["_c"].map({pre_col: "Pre", post_col: "Post"})
            melted["x_label"] = melted["noise_type"] + "\n" + melted["Phase"]
            sns.boxplot(
                data=melted, x="x_label", y="value",
                hue="solver", hue_order=solvers,
                order=x_order_paired, palette=palette,
                width=0.55, ax=ax, flierprops=dict(marker="", alpha=0),
            )
            for x_idx, x_lbl in enumerate(x_order_paired):
                nt, ph = x_lbl.split("\n")
                val_col = pre_col if ph == "Pre" else post_col
                for s_idx, solver in enumerate(solvers):
                    sub = df[(df["noise_type"] == nt) & (df["solver"] == solver)][[val_col, "nnz", "subject"]].rename(columns={val_col: "value"})
                    n_s = len(solvers)
                    dodge_off = (s_idx - (n_s - 1) / 2) * (0.55 / n_s)
                    _scatter_pts(ax, sub, x_idx, "value", nnz_color, subj_marker,
                                 jitter_scale=0.04, dodge_offset=dodge_off)
            # Dotted separators between noise_type groups
            for i in range(1, len(noise_types)):
                ax.axvline(i * 2 - 0.5, color="gray", linewidth=0.8,
                           linestyle=":", alpha=0.6)
        else:
            sns.boxplot(
                data=df, x="noise_type", y=cols,
                hue="solver", order=noise_types, hue_order=solvers,
                palette=palette, width=0.55, ax=ax,
                flierprops=dict(marker="", alpha=0),
            )
            for x_idx, nt in enumerate(noise_types):
                for s_idx, solver in enumerate(solvers):
                    sub = df[(df["noise_type"] == nt) & (df["solver"] == solver)][[cols, "nnz", "subject"]].rename(columns={cols: "value"})
                    n_s = len(solvers)
                    dodge_off = (s_idx - (n_s - 1) / 2) * (0.55 / n_s)
                    _scatter_pts(ax, sub, x_idx, "value", nnz_color, subj_marker,
                                 jitter_scale=0.04, dodge_offset=dodge_off)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        if ax.get_legend():
            ax.legend_.remove()

    handles, labels_ = axes[0].get_legend_handles_labels()
    solver_handles = [mpatches.Patch(facecolor=palette[i], label=s)
                      for i, s in enumerate(solvers)]
    fig.legend(handles=solver_handles + [mlines.Line2D([], [], color="none", label="")] +
               _nnz_subj_legend_handles(nnz_color, subj_marker),
               title="", loc="center left", ncol=1, bbox_to_anchor=(1.01, 0.5))

    for ax in axes[n:]:
        ax.set_visible(False)

    subtitle = dist_info(df, on_axes=["noise_type", "solver"])
    fig.suptitle("Distribution by Noise Type  (Pre vs Post, hue = Solver)",
                 fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "2_noise_type_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SNR effect  – line + error-band per solver
#    Pre & post of the same metric share one panel (solid=pre, dashed=post)
# ─────────────────────────────────────────────────────────────────────────────
def fig_snr_effect(df: pd.DataFrame, metrics: list, out: Path):
    print("[3] SNR effect")
    solvers   = sorted(df["solver"].unique())
    snr_vals  = sorted(df["alpha_SNR"].unique())
    panels    = _build_paired_panels(metrics)
    n         = len(panels)
    has_pairs = any(len(specs) > 1 for _, specs in panels)
    ncols     = 3
    nrows     = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    palette = SOLVER_PALETTE[:len(solvers)]
    nnz_color, subj_marker = _make_nnz_subj_maps(df)

    for idx, (title, line_specs) in enumerate(panels):
        ax = axes[idx]
        for s_idx, solver in enumerate(solvers):
            sub   = df[df["solver"] == solver]
            color = palette[s_idx]
            for col, lstyle, mrkr in line_specs:
                grp     = sub.groupby("alpha_SNR")[col].agg(["mean", "std"]).reindex(snr_vals)
                alpha_v = 0.75 if "post_cal" in col else 1.0
                # Individual data points colored by nnz, shaped by subject
                for snr in snr_vals:
                    pts_df = sub[sub["alpha_SNR"] == snr][[col, "nnz", "subject"]].dropna(subset=[col]).rename(columns={col: "value"})
                    _scatter_pts(ax, pts_df, snr, "value", nnz_color, subj_marker,
                                 jitter_scale=0.012, alpha=STRIP_ALPHA * alpha_v, size=STRIP_SIZE * 3)
                # Mean line + std band
                ax.plot(snr_vals, grp["mean"], marker=mrkr, linestyle=lstyle,
                        color=color, linewidth=2, alpha=alpha_v, zorder=4)
                ax.fill_between(snr_vals,
                                grp["mean"] - grp["std"],
                                grp["mean"] + grp["std"],
                                alpha=0.10, color=color)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("α SNR")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    for ax in axes[n:]:
        ax.set_visible(False)

    # Legend: colour encodes solver line, nnz color, subject marker
    leg = [mlines.Line2D([], [], color=palette[i], marker="o", linestyle="-",
                         linewidth=2, label=s)
           for i, s in enumerate(solvers)]
    if has_pairs:
        leg += [
            mlines.Line2D([], [], color="gray", linestyle="-",  marker="o",
                          linewidth=2, label="Pre-cal"),
            mlines.Line2D([], [], color="gray", linestyle="--", marker="s",
                          linewidth=2, label="Post-cal"),
        ]
    leg += [mlines.Line2D([], [], color="none", label="")] + _nnz_subj_legend_handles(nnz_color, subj_marker)
    fig.legend(handles=leg, loc="center left", ncol=1,
               bbox_to_anchor=(1.01, 0.5))

    subtitle = dist_info(df, on_axes=["alpha_SNR", "solver"])
    fig.suptitle("Effect of SNR  (mean ± std + individual points)",
                 fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "3_snr_effect.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. NNZ effect  – line + error-band per solver
#    Pre & post of the same metric share one panel (solid=pre, dashed=post)
# ─────────────────────────────────────────────────────────────────────────────
def fig_nnz_effect(df: pd.DataFrame, metrics: list, out: Path):
    print("[4] NNZ effect")
    solvers   = sorted(df["solver"].unique())
    nnz_vals  = sorted(df["nnz"].unique())
    panels    = _build_paired_panels(metrics)
    n         = len(panels)
    has_pairs = any(len(specs) > 1 for _, specs in panels)
    ncols     = 3
    nrows     = (n + ncols - 1) // ncols
    x_span    = (max(nnz_vals) - min(nnz_vals)) * 0.02 if len(nnz_vals) > 1 else 0.3

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    palette = SOLVER_PALETTE[:len(solvers)]
    nnz_color, subj_marker = _make_nnz_subj_maps(df)

    for idx, (title, line_specs) in enumerate(panels):
        ax = axes[idx]
        for s_idx, solver in enumerate(solvers):
            sub   = df[df["solver"] == solver]
            color = palette[s_idx]
            for col, lstyle, mrkr in line_specs:
                grp     = sub.groupby("nnz")[col].agg(["mean", "std"]).reindex(nnz_vals)
                alpha_v = 0.75 if "post_cal" in col else 1.0
                # Individual data points colored by nnz, shaped by subject
                for nnz in nnz_vals:
                    pts_df = sub[sub["nnz"] == nnz][[col, "nnz", "subject"]].dropna(subset=[col]).rename(columns={col: "value"})
                    _scatter_pts(ax, pts_df, nnz, "value", nnz_color, subj_marker,
                                 jitter_scale=x_span, alpha=STRIP_ALPHA * alpha_v, size=STRIP_SIZE * 3)
                # Mean line + std band
                ax.plot(nnz_vals, grp["mean"], marker=mrkr, linestyle=lstyle,
                        color=color, linewidth=2, alpha=alpha_v, zorder=4)
                ax.fill_between(nnz_vals,
                                grp["mean"] - grp["std"],
                                grp["mean"] + grp["std"],
                                alpha=0.10, color=color)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("nnz")
        ax.set_ylabel("")

    for ax in axes[n:]:
        ax.set_visible(False)

    # Legend: solver lines, pre/post style, nnz color, subject marker
    leg = [mlines.Line2D([], [], color=palette[i], marker="o", linestyle="-",
                         linewidth=2, label=s)
           for i, s in enumerate(solvers)]
    if has_pairs:
        leg += [
            mlines.Line2D([], [], color="gray", linestyle="-",  marker="o",
                          linewidth=2, label="Pre-cal"),
            mlines.Line2D([], [], color="gray", linestyle="--", marker="s",
                          linewidth=2, label="Post-cal"),
        ]
    leg += [mlines.Line2D([], [], color="none", label="")] + _nnz_subj_legend_handles(nnz_color, subj_marker)
    fig.legend(handles=leg, loc="center left", ncol=1,
               bbox_to_anchor=(1.01, 0.5))

    subtitle = dist_info(df, on_axes=["nnz", "solver"])
    fig.suptitle("Effect of NNZ  (mean ± std + individual points)",
                 fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "4_nnz_effect.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Calibration improvement
#    Panel A: grouped bars pre + post side-by-side per solver (same y-axis)
#    Panel B: Improvement (Δ) bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_calibration_improvement(df: pd.DataFrame, out: Path):
    print("[5] Calibration improvement (pre / post / Δ)")
    solvers    = sorted(df["solver"].unique())
    pre_color  = C_PRE
    post_color = C_POST
    imp_color  = C_DELTA
    width = 0.35
    x     = np.arange(len(solvers))
    nnz_color, subj_marker = _make_nnz_subj_maps(df)

    valid_bases = [b for b in CAL_BASE_METRICS
                   if (f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns)
                   or f"improvement_{b}" in df.columns]

    nrows = len(valid_bases)
    # Each metric gets 2 panels: Pre/Post bars + Δ bar
    ncols = 2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for r_idx, base in enumerate(valid_bases):
        pre_col  = f"pre_cal_{base}"
        post_col = f"post_cal_{base}"
        imp_col  = f"improvement_{base}"
        has_pair = pre_col in df.columns and post_col in df.columns
        has_imp  = imp_col in df.columns

        # ── Col 0: grouped pre / post bars ───────────────────────────────────
        ax = axes[r_idx][0]
        if has_pair:
            agg     = df.groupby("solver")[[pre_col, post_col]].mean().reindex(solvers)
            agg_std = df.groupby("solver")[[pre_col, post_col]].std().reindex(solvers)

            bars_pre  = ax.bar(x - width / 2, agg[pre_col].values,  width,
                               label="Pre-cal",  color=pre_color,  alpha=0.78,
                               edgecolor="black", linewidth=0.5, zorder=2)
            bars_post = ax.bar(x + width / 2, agg[post_col].values, width,
                               label="Post-cal", color=post_color, alpha=0.78,
                               edgecolor="black", linewidth=0.5, zorder=2)
            ax.errorbar(x - width / 2, agg[pre_col].values,  yerr=agg_std[pre_col].values,
                        fmt="none", color="black", linewidth=1.2, capsize=4, capthick=1.2, zorder=5)
            ax.errorbar(x + width / 2, agg[post_col].values, yerr=agg_std[post_col].values,
                        fmt="none", color="black", linewidth=1.2, capsize=4, capthick=1.2, zorder=5)

            for i, solver in enumerate(solvers):
                pre_df  = df[df["solver"] == solver][[pre_col,  "nnz", "subject"]].rename(columns={pre_col:  "value"})
                post_df = df[df["solver"] == solver][[post_col, "nnz", "subject"]].rename(columns={post_col: "value"})
                _scatter_pts(ax, pre_df,  i - width / 2, "value", nnz_color, subj_marker,
                             jitter_scale=width * 0.38)
                _scatter_pts(ax, post_df, i + width / 2, "value", nnz_color, subj_marker,
                             jitter_scale=width * 0.38)

            for bars, col in [(bars_pre, pre_col), (bars_post, post_col)]:
                for bar, val in zip(bars, df.groupby("solver")[col].mean().reindex(solvers)):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val * 1.02 if val >= 0 else val * 0.98,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=8, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(solvers, rotation=30, ha="right", fontsize=8)
            ax.set_xlabel("Solver", fontsize=8)
            ax.set_ylabel(base.replace("_", " ").title(), fontsize=9, fontweight="bold")
            ax.legend(loc="best", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            if r_idx == 0:
                ax.set_title("Pre vs Post", fontsize=10, fontweight="bold")
        else:
            ax.set_visible(False)

        # ── Col 1: improvement (Δ) ────────────────────────────────────────────
        ax = axes[r_idx][1]
        if has_imp:
            agg_imp     = df.groupby("solver")[imp_col].mean().reindex(solvers)
            agg_imp_std = df.groupby("solver")[imp_col].std().reindex(solvers)

            ax.bar(range(len(solvers)), agg_imp.values,
                   color=imp_color, alpha=0.75, edgecolor="black",
                   linewidth=0.5, zorder=2)
            ax.errorbar(range(len(solvers)), agg_imp.values, yerr=agg_imp_std.values,
                        fmt="none", color="black", linewidth=1.2, capsize=4, capthick=1.2, zorder=5)

            for i, solver in enumerate(solvers):
                imp_df = df[df["solver"] == solver][[imp_col, "nnz", "subject"]].rename(columns={imp_col: "value"})
                _scatter_pts(ax, imp_df, i, "value", nnz_color, subj_marker, jitter_scale=0.18)

            for i, val in enumerate(agg_imp.values):
                ax.text(i, val * 1.02 if val >= 0 else val * 0.98,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

            ax.set_xticks(range(len(solvers)))
            ax.set_xticklabels(solvers, rotation=30, ha="right", fontsize=8)
            ax.set_xlabel("Solver", fontsize=8)
            ax.set_ylabel(label(imp_col), fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if r_idx == 0:
                ax.set_title("Improvement (Δ)", fontsize=10, fontweight="bold")
        else:
            ax.set_visible(False)

    subtitle = dist_info(df, on_axes=["solver"])
    fig.suptitle("Calibration Improvement: Pre / Post / Δ",
                 fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "5_calibration_improvement.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Subject variability
#    One panel per base metric: x = subject × Phase (interleaved), hue = solver
# ─────────────────────────────────────────────────────────────────────────────
def fig_subject_variability(df: pd.DataFrame, metrics: list, out: Path):
    if df["subject"].nunique() < 2:
        print("[6] Skipping subject variability – fewer than 2 subjects")
        return
    print("[6] Subject variability")
    subjects    = sorted(df["subject"].unique())
    solvers     = sorted(df["solver"].unique())
    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]

    panels = []
    if "emd" in df.columns:
        panels.append(("EMD", "single", "emd"))
    if "mean_posterior_std" in df.columns:
        panels.append(("Mean Post. Std", "single", "mean_posterior_std"))
    for base in valid_bases:
        panels.append((
            base.replace("_", " ").title(), "paired",
            (f"pre_cal_{base}", f"post_cal_{base}")
        ))
    for base in valid_bases:
        imp = f"improvement_{base}"
        if imp in df.columns:
            panels.append((f"\u0394 {base.replace('_', ' ').title()}", "single", imp))

    x_order_paired = [f"{s}\n{ph}" for s in subjects for ph in ("Pre", "Post")]
    n     = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    palette = SOLVER_PALETTE[:len(solvers)]
    nnz_color, subj_marker = _make_nnz_subj_maps(df)

    # Wider figure to handle more x-tick labels
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, (title, kind, cols) in enumerate(panels):
        ax = axes[idx]
        if kind == "paired":
            pre_col, post_col = cols
            melted = pd.melt(
                df[["solver", "subject", "nnz", pre_col, post_col]],
                id_vars=["solver", "subject", "nnz"],
                value_vars=[pre_col, post_col],
                var_name="_c", value_name="value",
            )
            melted["Phase"]   = melted["_c"].map({pre_col: "Pre", post_col: "Post"})
            melted["x_label"] = melted["subject"] + "\n" + melted["Phase"]
            sns.boxplot(
                data=melted, x="x_label", y="value",
                hue="solver", hue_order=solvers,
                order=x_order_paired, palette=palette,
                width=0.55, ax=ax, flierprops=dict(marker="", alpha=0),
            )
            for x_idx, x_lbl in enumerate(x_order_paired):
                subj, ph = x_lbl.split("\n")
                val_col = pre_col if ph == "Pre" else post_col
                for s_idx, solver in enumerate(solvers):
                    sub = df[(df["subject"] == subj) & (df["solver"] == solver)][[val_col, "nnz", "subject"]].rename(columns={val_col: "value"})
                    n_s = len(solvers)
                    dodge_off = (s_idx - (n_s - 1) / 2) * (0.55 / n_s)
                    _scatter_pts(ax, sub, x_idx, "value", nnz_color, subj_marker,
                                 jitter_scale=0.04, dodge_offset=dodge_off)
            # Dotted separators between subject groups
            for i in range(1, len(subjects)):
                ax.axvline(i * 2 - 0.5, color="gray", linewidth=0.8,
                           linestyle=":", alpha=0.6)
        else:
            sns.boxplot(
                data=df, x="subject", y=cols,
                hue="solver", hue_order=solvers,
                order=subjects, palette=palette,
                width=0.55, ax=ax, flierprops=dict(marker="", alpha=0),
            )
            for x_idx, subj in enumerate(subjects):
                for s_idx, solver in enumerate(solvers):
                    sub = df[(df["subject"] == subj) & (df["solver"] == solver)][[cols, "nnz", "subject"]].rename(columns={cols: "value"})
                    n_s = len(solvers)
                    dodge_off = (s_idx - (n_s - 1) / 2) * (0.55 / n_s)
                    _scatter_pts(ax, sub, x_idx, "value", nnz_color, subj_marker,
                                 jitter_scale=0.04, dodge_offset=dodge_off)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        if ax.get_legend():
            ax.legend_.remove()

    solver_handles = [mpatches.Patch(facecolor=palette[i], label=s)
                      for i, s in enumerate(solvers)]
    fig.legend(handles=solver_handles + [mlines.Line2D([], [], color="none", label="")] +
               _nnz_subj_legend_handles(nnz_color, subj_marker),
               title="", loc="center left", ncol=1, bbox_to_anchor=(1.01, 0.5))

    for ax in axes[n:]:
        ax.set_visible(False)

    subtitle = dist_info(df, on_axes=["subject", "solver"])
    fig.suptitle("Subject Variability  (Pre vs Post, hue = Solver)",
                 fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "6_subject_variability.png")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 8. Pre vs Post by grouping variable
#     For each base metric: one figure, one column per grouping parameter.
#     Pre and Post share the same panel (hue = phase) on a shared y-axis,
#     enabling direct magnitude comparison across all experimental conditions.
# ─────────────────────────────────────────────────────────────────────────────
def fig_pre_post_by_group(df: pd.DataFrame, out: Path):
    print("[8] Pre vs Post grouped by parameter")

    # (column_name,  x-axis label,  optional value-formatter for display)
    group_specs = [
        ("solver",     "Solver",     None),
        ("noise_type", "Noise Type", None),
        ("alpha_SNR",  "α SNR",      lambda v: f"{v:.2f}"),
        ("nnz",        "nnz",        lambda v: str(int(v))),
    ]
    group_specs = [(g, lbl, fmt) for g, lbl, fmt in group_specs if g in df.columns]

    phase_palette = {"Pre": C_PRE, "Post": C_POST}
    phase_order   = ["Pre", "Post"]
    nnz_color, subj_marker = _make_nnz_subj_maps(df)

    valid_bases = [
        b for b in CAL_BASE_METRICS
        if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns
    ]

    nrows = len(valid_bases)
    ncols = len(group_specs)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 4.5 * nrows),
        sharey="row",
        squeeze=False,
    )

    for r_idx, base in enumerate(valid_bases):
        pre_col  = f"pre_cal_{base}"
        post_col = f"post_cal_{base}"

        id_vars    = [g for g, _, _ in group_specs]
        extra_cols = [c for c in ["nnz", "subject"] if c not in id_vars]
        melted = pd.melt(
            df[id_vars + extra_cols + [pre_col, post_col]],
            id_vars=id_vars + extra_cols,
            value_vars=[pre_col, post_col],
            var_name="phase",
            value_name="value",
        )
        melted["phase"] = melted["phase"].map({pre_col: "Pre", post_col: "Post"})

        for c_idx, (gvar, glabel, fmt) in enumerate(group_specs):
            ax      = axes[r_idx][c_idx]
            plot_df = melted.copy()

            if fmt is not None:
                orig_vals = sorted(df[gvar].unique())
                order     = [fmt(v) for v in orig_vals]
                plot_df[gvar] = plot_df[gvar].apply(fmt)
            else:
                order = sorted(plot_df[gvar].unique())

            sns.boxplot(
                data=plot_df, x=gvar, y="value",
                hue="phase", order=order, hue_order=phase_order,
                palette=phase_palette, width=0.55, ax=ax,
                flierprops=dict(marker="", alpha=0),
            )
            for x_idx, x_val in enumerate(order):
                for ph_idx, ph in enumerate(phase_order):
                    mask = plot_df[gvar] == x_val
                    sub  = plot_df[mask & (plot_df["phase"] == ph)][["value", "nnz", "subject"]]
                    dodge_off = (ph_idx - 0.5) * (0.55 / 2)
                    _scatter_pts(ax, sub, x_idx, "value", nnz_color, subj_marker,
                                 jitter_scale=0.05, dodge_offset=dodge_off)

            # Column header on first row only
            if r_idx == 0:
                ax.set_title(glabel, fontsize=10, fontweight="bold")
            # Row (metric) label on leftmost column only
            ax.set_ylabel(
                base.replace("_", " ").title() if c_idx == 0 else "",
                fontsize=9, fontweight="bold",
            )
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.25)
            if ax.get_legend():
                ax.legend_.remove()

    # Single shared legend
    phase_handles = [mpatches.Patch(facecolor=phase_palette[ph], label=ph)
                     for ph in phase_order]
    fig.legend(
        handles=phase_handles + [mlines.Line2D([], [], color="none", label="")] +
                _nnz_subj_legend_handles(nnz_color, subj_marker),
        title="", loc="center left", ncol=1,
        bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9,
    )

    subtitle = dist_info(df, on_axes=[g for g, _, _ in group_specs])
    fig.suptitle("Pre vs Post by Group Parameter", fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "8_pre_post_by_group.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
# 9. Per-solver SNR violin
#     One figure per solver.  Rows = noise_type; cols = metrics.
#     At each SNR x-tick: left half-violin = Pre, right = Post.
#     Metrics without pre/post (EMD, improvement) get a single full violin.
#     Individual points: colour = subject, shape = nnz.
# ─────────────────────────────────────────────────────────────────────────────
def fig_snr_violin_per_solver(df: pd.DataFrame, out: Path):
    print("[9] Per-solver SNR violin")
    solvers     = sorted(df["solver"].unique())
    noise_types = sorted(df["noise_type"].unique())
    snr_vals    = sorted(df["alpha_SNR"].unique())
    subjects    = sorted(df["subject"].unique())
    nnz_vals    = sorted(df["nnz"].unique())

    # Point encodings: subject → colour, nnz → marker shape
    nnz_color, subj_marker = _make_nnz_subj_maps(df)

    pre_color    = C_PRE
    post_color   = C_POST
    single_color = C_SINGLE

    # Col specs: (label, 'paired'|'single', col_or_(pre,post))
    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]
    col_specs = []
    if "emd" in df.columns:
        col_specs.append(("EMD", "single", "emd"))
    if "mean_posterior_std" in df.columns:
        col_specs.append(("Mean Post. Std", "single", "mean_posterior_std"))
    for base in valid_bases:
        col_specs.append((
            base.replace("_", " ").title(), "paired",
            (f"pre_cal_{base}", f"post_cal_{base}")
        ))

    nrows = len(noise_types)
    ncols = len(col_specs)

    for solver in solvers:
        print(f"    solver={solver}")
        df_s = df[df["solver"] == solver]

        fig, axes = plt.subplots(  
            nrows, ncols,
            figsize=(5.5 * ncols, 2.8 * nrows),
            squeeze=False,
        )
        rng = np.random.default_rng(7)

        for r_idx, nt in enumerate(noise_types):
            df_nt = df_s[df_s["noise_type"] == nt]
            for c_idx, (col_label, kind, cols) in enumerate(col_specs):
                ax      = axes[r_idx][c_idx]
                df_cell = df_nt

                if kind == "paired":
                    pre_col, post_col = cols
                    for s_idx, snr in enumerate(snr_vals):
                        sub = df_cell[df_cell["alpha_SNR"] == snr]
                        _half_violin(ax, sub[pre_col].values,  s_idx, "left",  pre_color)
                        _half_violin(ax, sub[post_col].values, s_idx, "right", post_color)
                        # Individual points
                        for (subj, nnz_v), grp in sub.groupby(["subject", "nnz"]):
                            pre_pts  = grp[pre_col].dropna().values
                            post_pts = grp[post_col].dropna().values
                            kw = dict(
                                color=nnz_color[nnz_v],
                                marker=subj_marker[subj],
                                s=STRIP_SIZE * 5,
                                alpha=STRIP_ALPHA + 0.15,
                                zorder=6,
                                edgecolors="white",
                                linewidths=0.3,
                            )
                            if len(pre_pts):
                                jit = rng.uniform(-0.06, 0.06, len(pre_pts))
                                ax.scatter(s_idx - 0.13 + jit, pre_pts,  **kw)
                            if len(post_pts):
                                jit = rng.uniform(-0.06, 0.06, len(post_pts))
                                ax.scatter(s_idx + 0.13 + jit, post_pts, **kw)

                else:  # single metric
                    val_col = cols
                    for s_idx, snr in enumerate(snr_vals):
                        sub = df_cell[df_cell["alpha_SNR"] == snr]
                        _half_violin(ax, sub[val_col].values, s_idx, "both", single_color)
                        for (subj, nnz_v), grp in sub.groupby(["subject", "nnz"]):
                            pts = grp[val_col].dropna().values
                            if len(pts):
                                jit = rng.uniform(-0.12, 0.12, len(pts))
                                ax.scatter(
                                    s_idx + jit, pts,
                                    color=nnz_color[nnz_v],
                                    marker=subj_marker[subj],
                                    s=STRIP_SIZE * 5, alpha=STRIP_ALPHA + 0.15,
                                    zorder=6, edgecolors="white", linewidths=0.3,
                                )

                # Axes decoration
                ax.set_xticks(range(len(snr_vals)))
                ax.set_xticklabels([f"{v:.2f}" for v in snr_vals],
                                   fontsize=8, rotation=30)
                ax.set_xlim(-0.55, len(snr_vals) - 0.45)
                ax.set_ylabel(nt if c_idx == 0 else "", fontsize=9, fontweight="bold")
                ax.set_xlabel("α SNR" if r_idx == nrows - 1 else "", fontsize=8)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="y", alpha=0.25)
                if r_idx == 0:
                    ax.set_title(col_label, fontsize=10, fontweight="bold")

        # ── Legend ──────────────────────────────────────────────────────────
        sep = [mlines.Line2D([], [], color="none", label="")]
        phase_h = [
            mpatches.Patch(facecolor=pre_color,    label="Pre-cal"),
            mpatches.Patch(facecolor=post_color,   label="Post-cal"),
            mpatches.Patch(facecolor=single_color, label="Single metric"),
        ]
        subj_h = [mlines.Line2D([], [], color="gray", marker=m, linestyle="None",
                                 markersize=7, label=s)
                  for s, m in subj_marker.items()]
        nnz_h  = [mpatches.Patch(facecolor=c,
                                  label=f"nnz={int(v)}")
                  for v, c in nnz_color.items()]
        fig.legend(
            handles=phase_h + sep + subj_h + sep + nnz_h,
            loc="center left", ncol=1,
            bbox_to_anchor=(1.01, 0.5), fontsize=8,
            title="", framealpha=0.9,
        )

        solver_safe = solver.replace(" ", "_")
        subtitle = dist_info(df_s, on_axes=["alpha_SNR", "noise_type", "solver"])
        fig.suptitle(
            f"Solver: {solver}  \u2013  Pre vs Post Calibration by SNR & Noise Type",
            fontsize=13, fontweight="bold", y=1.01,
        )
        _add_dist_box(fig, subtitle)
        fig.tight_layout()
        save_fig(fig, out / f"9_snr_violin_{solver_safe}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Per-solver SNR mean ± std
#     Same layout as fig 12 (rows=noise_type, cols=metric) but each cell shows
#     mean ± 1 std as an error-bar line connected with a shaded ±std band.
#     Paired metrics: two lines (Pre=solid, Post=dashed) in C_PRE / C_POST.
#     Single metrics (EMD): one line in C_SINGLE.
#     Individual points overlaid: colour = nnz, shape = subject.
# ─────────────────────────────────────────────────────────────────────────────
def fig_snr_meanstd_per_solver(df: pd.DataFrame, out: Path):
    print("[10] Per-solver SNR mean \u00b1 std")
    solvers     = sorted(df["solver"].unique())
    noise_types = sorted(df["noise_type"].unique())
    snr_vals    = sorted(df["alpha_SNR"].unique())
    x           = np.arange(len(snr_vals))

    pre_color    = C_PRE
    post_color   = C_POST
    single_color = C_SINGLE

    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]
    col_specs = []
    if "emd" in df.columns:
        col_specs.append(("EMD", "single", "emd"))
    if "mean_posterior_std" in df.columns:
        col_specs.append(("Mean Post. Std", "single", "mean_posterior_std"))
    for base in valid_bases:
        col_specs.append((
            base.replace("_", " ").title(), "paired",
            (f"pre_cal_{base}", f"post_cal_{base}")
        ))

    nrows = len(noise_types)
    ncols = len(col_specs)

    for solver in solvers:
        print(f"    solver={solver}")
        df_s = df[df["solver"] == solver]

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 2.8 * nrows),
            squeeze=False,
        )

        for r_idx, nt in enumerate(noise_types):
            df_nt = df_s[df_s["noise_type"] == nt]
            for c_idx, (col_label, kind, cols) in enumerate(col_specs):
                ax      = axes[r_idx][c_idx]
                df_cell = df_nt

                if kind == "paired":
                    pre_col, post_col = cols
                    for line_col, color, ls, jit_off in [
                        (pre_col,  pre_color,  "-",  -0.06),
                        (post_col, post_color, "--", +0.06),
                    ]:
                        vals = [df_cell[df_cell["alpha_SNR"] == snr][line_col].dropna().values
                                for snr in snr_vals]
                        means = np.array([v.mean() if len(v) else np.nan for v in vals])
                        stds  = np.array([v.std()  if len(v) else np.nan for v in vals])
                        ax.plot(x, means, color=color, linestyle=ls,
                                linewidth=2, marker="o", markersize=5, zorder=4)
                        ax.errorbar(x, means, yerr=stds, fmt="none",
                                    color=color, linewidth=1.2,
                                    capsize=4, capthick=1.2, zorder=4)
                        ax.fill_between(x, means - stds, means + stds,
                                        color=color, alpha=0.15, linewidth=0)

                else:  # single metric
                    val_col = cols
                    vals  = [df_cell[df_cell["alpha_SNR"] == snr][val_col].dropna().values
                             for snr in snr_vals]
                    means = np.array([v.mean() if len(v) else np.nan for v in vals])
                    stds  = np.array([v.std()  if len(v) else np.nan for v in vals])
                    ax.plot(x, means, color=single_color, linestyle="-",
                            linewidth=2, marker="o", markersize=5, zorder=4)
                    ax.errorbar(x, means, yerr=stds, fmt="none",
                                color=single_color, linewidth=1.2,
                                capsize=4, capthick=1.2, zorder=4)
                    ax.fill_between(x, means - stds, means + stds,
                                    color=single_color, alpha=0.15, linewidth=0)

                # Axes decoration
                ax.set_xticks(x)
                ax.set_xticklabels([f"{v:.2f}" for v in snr_vals],
                                   fontsize=8, rotation=30)
                ax.set_xlim(-0.55, len(snr_vals) - 0.45)
                ax.set_ylabel(nt if c_idx == 0 else "", fontsize=9, fontweight="bold")
                ax.set_xlabel("α SNR" if r_idx == nrows - 1 else "", fontsize=8)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="y", alpha=0.25)
                if r_idx == 0:
                    ax.set_title(col_label, fontsize=10, fontweight="bold")

        # ── Legend ──────────────────────────────────────────────────────────────
        phase_h = [
            mlines.Line2D([], [], color=pre_color,    linestyle="-",  linewidth=2,
                          marker="o", markersize=5, label="Pre-cal"),
            mlines.Line2D([], [], color=post_color,   linestyle="--", linewidth=2,
                          marker="o", markersize=5, label="Post-cal"),
            mlines.Line2D([], [], color=single_color, linestyle="-",  linewidth=2,
                          marker="o", markersize=5, label="Single metric"),
        ]
        fig.legend(
            handles=phase_h,
            loc="center left", ncol=1,
            bbox_to_anchor=(1.01, 0.5), fontsize=8,
            title="", framealpha=0.9,
        )

        solver_safe = solver.replace(" ", "_")
        subtitle = dist_info(df_s, on_axes=["alpha_SNR", "noise_type", "solver"])
        fig.suptitle(
            f"Solver: {solver}  \u2013  Mean \u00b1 Std by SNR & Noise Type",
            fontsize=13, fontweight="bold", y=1.01,
        )
        _add_dist_box(fig, subtitle)
        fig.tight_layout()
        save_fig(fig, out / f"10_snr_meanstd_{solver_safe}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Per-solver SNR mean ± std – all noise types overlaid (single row)
# ─────────────────────────────────────────────────────────────────────────────
def fig_snr_meanstd_overlaid_noise(df: pd.DataFrame, out: Path):
    print("[11] Per-solver SNR mean \u00b1 std – noise types overlaid")
    solvers     = sorted(df["solver"].unique())
    noise_types = sorted(df["noise_type"].unique())
    snr_vals    = sorted(df["alpha_SNR"].unique())
    x           = np.arange(len(snr_vals))

    # One color per noise_type – reuse SOLVER_PALETTE for consistency
    nt_color = {nt: SOLVER_PALETTE[i % len(SOLVER_PALETTE)]
                for i, nt in enumerate(noise_types)}

    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]
    col_specs = []
    if "emd" in df.columns:
        col_specs.append(("EMD", "single", "emd"))
    if "mean_posterior_std" in df.columns:
        col_specs.append(("Mean Post. Std", "single", "mean_posterior_std"))
    for base in valid_bases:
        col_specs.append((
            base.replace("_", " ").title(), "paired",
            (f"pre_cal_{base}", f"post_cal_{base}")
        ))

    ncols = len(col_specs)

    for solver in solvers:
        print(f"    solver={solver}")
        df_s = df[df["solver"] == solver]

        fig, axes = plt.subplots(
            1, ncols,
            figsize=(5.5 * ncols, 3.5),
            squeeze=False,
        )

        for c_idx, (col_label, kind, cols) in enumerate(col_specs):
            ax = axes[0][c_idx]

            for nt in noise_types:
                df_nt = df_s[df_s["noise_type"] == nt]
                color = nt_color[nt]

                if kind == "paired":
                    pre_col, post_col = cols
                    for line_col, ls in [(pre_col, "-"), (post_col, "--")]:
                        vals  = [df_nt[df_nt["alpha_SNR"] == snr][line_col].dropna().values
                                 for snr in snr_vals]
                        means = np.array([v.mean() if len(v) else np.nan for v in vals])
                        stds  = np.array([v.std()  if len(v) else np.nan for v in vals])
                        ax.plot(x, means, color=color, linestyle=ls,
                                linewidth=2, marker="o", markersize=5, zorder=4)
                        ax.errorbar(x, means, yerr=stds, fmt="none",
                                    color=color, linewidth=1.2,
                                    capsize=4, capthick=1.2, zorder=4)
                        ax.fill_between(x, means - stds, means + stds,
                                        color=color, alpha=0.12, linewidth=0)

                else:  # single metric
                    val_col = cols
                    vals  = [df_nt[df_nt["alpha_SNR"] == snr][val_col].dropna().values
                             for snr in snr_vals]
                    means = np.array([v.mean() if len(v) else np.nan for v in vals])
                    stds  = np.array([v.std()  if len(v) else np.nan for v in vals])
                    ax.plot(x, means, color=color, linestyle="-",
                            linewidth=2, marker="o", markersize=5, zorder=4)
                    ax.errorbar(x, means, yerr=stds, fmt="none",
                                color=color, linewidth=1.2,
                                capsize=4, capthick=1.2, zorder=4)
                    ax.fill_between(x, means - stds, means + stds,
                                    color=color, alpha=0.12, linewidth=0)

            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:.2f}" for v in snr_vals], fontsize=8, rotation=30)
            ax.set_xlim(-0.55, len(snr_vals) - 0.45)
            ax.set_ylabel("" if c_idx else "Value", fontsize=9)
            ax.set_xlabel("\u03b1 SNR", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(col_label, fontsize=10, fontweight="bold")

        # ── Legend ──────────────────────────────────────────────────────────────
        nt_h = [mpatches.Patch(facecolor=nt_color[nt], label=nt)
                for nt in noise_types]
        phase_h = [
            mlines.Line2D([], [], color="black", linestyle="-",  linewidth=2,
                          markersize=0, label="Pre-cal"),
            mlines.Line2D([], [], color="black", linestyle="--", linewidth=2,
                          markersize=0, label="Post-cal"),
        ]
        sep = [mlines.Line2D([], [], color="none", label="")]
        fig.legend(
            handles=nt_h + sep + phase_h,
            loc="center left", ncol=1,
            bbox_to_anchor=(1.01, 0.5), fontsize=8,
            title="", framealpha=0.9, handlelength=3,
        )

        solver_safe = solver.replace(" ", "_")
        subtitle = dist_info(df_s, on_axes=["alpha_SNR", "noise_type", "solver"])
        fig.suptitle(
            f"Solver: {solver}  \u2013  Mean \u00b1 Std by SNR (noise types overlaid)",
            fontsize=13, fontweight="bold", y=1.01,
        )
        _add_dist_box(fig, subtitle)
        fig.tight_layout()
        save_fig(fig, out / f"11_snr_meanstd_overlaid_{solver_safe}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Pre vs Post calibration scatter – encoded by nnz
#     Each nnz level gets a distinct fill colour (NNZ_PALETTE).
# ─────────────────────────────────────────────────────────────────────────────
def fig_pre_vs_post_scatter_nnz(df: pd.DataFrame, out: Path):
    print("[12] Pre vs Post scatter – color=solver, shape=nnz")

    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]
    if not valid_bases:
        return

    solvers  = sorted(df["solver"].unique())
    nnz_vals = sorted(df["nnz"].unique())

    solver_color = {s: SOLVER_PALETTE[i % len(SOLVER_PALETTE)] for i, s in enumerate(solvers)}
    _NNZ_MARKERS = SUBJ_MARKERS
    nnz_marker   = {v: _NNZ_MARKERS[i % len(_NNZ_MARKERS)] for i, v in enumerate(nnz_vals)}

    ncols = min(3, len(valid_bases))
    nrows = (len(valid_bases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, base in enumerate(valid_bases):
        ax   = axes_flat[idx]
        pre  = f"pre_cal_{base}"
        post = f"post_cal_{base}"

        for solver in solvers:
            for nnz_val in nnz_vals:
                sub = df[(df["solver"] == solver) & (df["nnz"] == nnz_val)][
                    [pre, post]].dropna()
                if sub.empty:
                    continue
                ax.scatter(sub[pre], sub[post],
                           color=solver_color[solver],
                           marker=nnz_marker[nnz_val],
                           alpha=0.60, s=45,
                           edgecolors="white", linewidths=0.35)

        all_vals = pd.concat([df[pre], df[post]]).dropna()
        lo, hi = all_vals.min(), all_vals.max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.45)
        ax.set_xlabel(f"Pre  {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_ylabel(f"Post {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_title(base.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(valid_bases):]:
        ax.set_visible(False)

    sep = [mlines.Line2D([], [], color="none", label="")]
    solver_h  = [mpatches.Patch(facecolor=solver_color[s], label=s) for s in solvers]
    nnz_h     = [mlines.Line2D([], [], color="gray", marker=nnz_marker[v],
                                linestyle="None", markersize=7, label=f"nnz={int(v)}")
                 for v in nnz_vals]
    identity_h = [mlines.Line2D([], [], color="k", linestyle="--", label="y = x")]
    fig.legend(handles=solver_h + sep + nnz_h + sep + identity_h,
               loc="center left", ncol=1, bbox_to_anchor=(1.01, 0.5), fontsize=9)

    subtitle = dist_info(df, on_axes=["solver", "nnz"])
    fig.suptitle("Pre vs Post Calibration  –  color = solver  |  shape = nnz\n(below diagonal = improved)",
                 fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "12_pre_vs_post_scatter_nnz.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Pre vs Post calibration scatter – encoded by subject
#     Each subject gets a distinct colour + marker combination.
# ─────────────────────────────────────────────────────────────────────────────
def fig_pre_vs_post_scatter_subject(df: pd.DataFrame, out: Path):
    print("[13] Pre vs Post scatter – color=solver, shape=subject")

    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]
    if not valid_bases:
        return

    solvers  = sorted(df["solver"].unique())
    subjects = sorted(df["subject"].unique())

    solver_color = {s: SOLVER_PALETTE[i % len(SOLVER_PALETTE)] for i, s in enumerate(solvers)}
    subj_marker  = {s: SUBJ_MARKERS[i % len(SUBJ_MARKERS)]     for i, s in enumerate(subjects)}

    ncols = min(3, len(valid_bases))
    nrows = (len(valid_bases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, base in enumerate(valid_bases):
        ax   = axes_flat[idx]
        pre  = f"pre_cal_{base}"
        post = f"post_cal_{base}"

        for solver in solvers:
            for subj in subjects:
                sub = df[(df["solver"] == solver) & (df["subject"] == subj)][
                    [pre, post]].dropna()
                if sub.empty:
                    continue
                ax.scatter(sub[pre], sub[post],
                           color=solver_color[solver],
                           marker=subj_marker[subj],
                           alpha=0.65, s=48,
                           edgecolors="white", linewidths=0.35)

        all_vals = pd.concat([df[pre], df[post]]).dropna()
        lo, hi = all_vals.min(), all_vals.max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.45)
        ax.set_xlabel(f"Pre  {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_ylabel(f"Post {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_title(base.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(valid_bases):]:
        ax.set_visible(False)

    sep = [mlines.Line2D([], [], color="none", label="")]
    solver_h  = [mpatches.Patch(facecolor=solver_color[s], label=s) for s in solvers]
    subj_h    = [mlines.Line2D([], [], color="gray", marker=subj_marker[s],
                                linestyle="None", markersize=7, label=s)
                 for s in subjects]
    identity_h = [mlines.Line2D([], [], color="k", linestyle="--", label="y = x")]
    fig.legend(handles=solver_h + sep + subj_h + sep + identity_h,
               loc="center left", ncol=1, bbox_to_anchor=(1.01, 0.5), fontsize=9)

    subtitle = dist_info(df, on_axes=["solver", "subject"])
    fig.suptitle("Pre vs Post Calibration  –  color = solver  |  shape = subject\n(below diagonal = improved)",
                 fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "13_pre_vs_post_scatter_subject.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Pre vs Post calibration scatter – encoded by noise variance method
#     Each noise_type gets a distinct fill colour (SOLVER_PALETTE).
# ─────────────────────────────────────────────────────────────────────────────
def fig_pre_vs_post_scatter_noise(df: pd.DataFrame, out: Path):
    print("[14] Pre vs Post scatter – color=solver, shape=noise type")

    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]
    if not valid_bases:
        return

    solvers     = sorted(df["solver"].unique())
    noise_types = sorted(df["noise_type"].unique())

    solver_color = {s: SOLVER_PALETTE[i % len(SOLVER_PALETTE)] for i, s in enumerate(solvers)}
    _NT_MARKERS  = SUBJ_MARKERS
    nt_marker    = {nt: _NT_MARKERS[i % len(_NT_MARKERS)] for i, nt in enumerate(noise_types)}

    ncols = min(3, len(valid_bases))
    nrows = (len(valid_bases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, base in enumerate(valid_bases):
        ax   = axes_flat[idx]
        pre  = f"pre_cal_{base}"
        post = f"post_cal_{base}"

        for solver in solvers:
            for nt in noise_types:
                sub = df[(df["solver"] == solver) & (df["noise_type"] == nt)][
                    [pre, post]].dropna()
                if sub.empty:
                    continue
                ax.scatter(sub[pre], sub[post],
                           color=solver_color[solver],
                           marker=nt_marker[nt],
                           alpha=0.60, s=45,
                           edgecolors="white", linewidths=0.35)

        all_vals = pd.concat([df[pre], df[post]]).dropna()
        lo, hi = all_vals.min(), all_vals.max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.45)
        ax.set_xlabel(f"Pre  {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_ylabel(f"Post {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_title(base.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(valid_bases):]:
        ax.set_visible(False)

    sep = [mlines.Line2D([], [], color="none", label="")]
    solver_h  = [mpatches.Patch(facecolor=solver_color[s], label=s) for s in solvers]
    nt_h      = [mlines.Line2D([], [], color="gray", marker=nt_marker[nt],
                                linestyle="None", markersize=7, label=nt)
                 for nt in noise_types]
    identity_h = [mlines.Line2D([], [], color="k", linestyle="--", label="y = x")]
    fig.legend(handles=solver_h + sep + nt_h + sep + identity_h,
               loc="center left", ncol=1, bbox_to_anchor=(1.01, 0.5), fontsize=9)

    subtitle = dist_info(df, on_axes=["solver", "noise_type"])
    fig.suptitle("Pre vs Post Calibration  –  color = solver  |  shape = noise variance method\n(below diagonal = improved)",
                 fontsize=13, fontweight="bold", y=1.01)
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "14_pre_vs_post_scatter_noise.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Pre vs Post calibration scatter – encoded by SNR level
#     color = solver (SOLVER_PALETTE)  |  shape = alpha_SNR level
# ─────────────────────────────────────────────────────────────────────────────
def fig_pre_vs_post_scatter_snr(df: pd.DataFrame, out: Path):
    print("[15] Pre vs Post scatter – color=solver, shape=SNR")

    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]
    if not valid_bases:
        return

    solvers   = sorted(df["solver"].unique())
    snr_vals  = sorted(df["alpha_SNR"].unique())

    solver_color = {s: SOLVER_PALETTE[i % len(SOLVER_PALETTE)] for i, s in enumerate(solvers)}
    _SNR_MARKERS = SUBJ_MARKERS
    snr_marker   = {v: _SNR_MARKERS[i % len(_SNR_MARKERS)] for i, v in enumerate(snr_vals)}

    ncols = min(3, len(valid_bases))
    nrows = (len(valid_bases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, base in enumerate(valid_bases):
        ax   = axes_flat[idx]
        pre  = f"pre_cal_{base}"
        post = f"post_cal_{base}"

        for solver in solvers:
            for snr in snr_vals:
                sub = df[(df["solver"] == solver) & (df["alpha_SNR"] == snr)][
                    [pre, post]].dropna()
                if sub.empty:
                    continue
                ax.scatter(sub[pre], sub[post],
                           color=solver_color[solver],
                           marker=snr_marker[snr],
                           alpha=0.60, s=45,
                           edgecolors="white", linewidths=0.35)

        all_vals = pd.concat([df[pre], df[post]]).dropna()
        lo, hi = all_vals.min(), all_vals.max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.45)
        ax.set_xlabel(f"Pre  {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_ylabel(f"Post {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_title(base.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(valid_bases):]:
        ax.set_visible(False)

    sep = [mlines.Line2D([], [], color="none", label="")]
    solver_h  = [mpatches.Patch(facecolor=solver_color[s], label=s) for s in solvers]
    snr_h     = [mlines.Line2D([], [], color="gray", marker=snr_marker[v],
                                linestyle="None", markersize=7,
                                label=f"SNR={round(float(v), 4)}")
                 for v in snr_vals]
    identity_h = [mlines.Line2D([], [], color="k", linestyle="--", label="y = x")]
    fig.legend(handles=solver_h + sep + snr_h + sep + identity_h,
               loc="center left", ncol=1, bbox_to_anchor=(1.01, 0.5), fontsize=9)

    subtitle = dist_info(df, on_axes=["solver", "alpha_SNR"])
    fig.suptitle(
        "Pre vs Post Calibration  –  color = solver  |  shape = SNR\n"
        "(below diagonal = improved)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "15_pre_vs_post_scatter_snr.png")


# ─────────────────────────────────────────────────────────────────────────────
# 16. Pre vs Post calibration scatter – solver only
#     Single encoding: color = solver.  No secondary variable.
#     Clean baseline view of pre/post relationship per estimator.
# ─────────────────────────────────────────────────────────────────────────────
def fig_pre_vs_post_scatter_solver(df: pd.DataFrame, out: Path):
    print("[16] Pre vs Post scatter – color = solver only")

    valid_bases = [b for b in CAL_BASE_METRICS
                   if f"pre_cal_{b}" in df.columns and f"post_cal_{b}" in df.columns]
    if not valid_bases:
        return

    solvers      = sorted(df["solver"].unique())
    solver_color = {s: SOLVER_PALETTE[i % len(SOLVER_PALETTE)] for i, s in enumerate(solvers)}

    ncols = min(3, len(valid_bases))
    nrows = (len(valid_bases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, base in enumerate(valid_bases):
        ax   = axes_flat[idx]
        pre  = f"pre_cal_{base}"
        post = f"post_cal_{base}"

        for solver in solvers:
            sub = df[df["solver"] == solver][[pre, post]].dropna()
            if sub.empty:
                continue
            ax.scatter(sub[pre], sub[post],
                       color=solver_color[solver],
                       marker="o", alpha=0.60, s=45,
                       edgecolors="white", linewidths=0.35)

        all_vals = pd.concat([df[pre], df[post]]).dropna()
        lo, hi = all_vals.min(), all_vals.max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.45)
        ax.set_xlabel(f"Pre  {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_ylabel(f"Post {base.replace('_', ' ').title()}", fontsize=9)
        ax.set_title(base.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(valid_bases):]:
        ax.set_visible(False)

    sep        = [mlines.Line2D([], [], color="none", label="")]
    solver_h   = [mpatches.Patch(facecolor=solver_color[s], label=s) for s in solvers]
    identity_h = [mlines.Line2D([], [], color="k", linestyle="--", label="y = x")]
    fig.legend(handles=solver_h + sep + identity_h,
               loc="center left", ncol=1, bbox_to_anchor=(1.01, 0.5), fontsize=9)

    subtitle = dist_info(df, on_axes=["solver"])
    fig.suptitle(
        "Pre vs Post Calibration  (below diagonal = improved)\n"
        "color = solver",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _add_dist_box(fig, subtitle)
    fig.tight_layout()
    save_fig(fig, out / "16_pre_vs_post_scatter_solver.png")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("  CaliBrain – grouped benchmark result figures")
    print("=" * 65)

    df = load_data(CSV_PATH)
    metrics = available_metrics(df)

    out = Path(OUTPUT_DIR) if OUTPUT_DIR else (
        Path(CSV_PATH).parent / "figures" / "grouped"
    )
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory : {out}")
    print(f"Rows in CSV      : {len(df)}")
    print(f"Solvers          : {sorted(df['solver'].unique())}")
    print(f"Noise types      : {sorted(df['noise_type'].unique())}")
    print(f"SNR levels       : {sorted(df['alpha_SNR'].unique())}")
    print(f"nnz levels       : {sorted(df['nnz'].unique())}")
    print(f"Subjects         : {sorted(df['subject'].unique())}")
    print(f"Runs (run_id)    : {sorted(df['run_id'].unique())}")
    print(f"Metrics found    : {len(metrics)}")
    print()

    fig_solver_comparison(df, metrics, out)
    fig_noise_type_comparison(df, metrics, out)
    fig_snr_effect(df, metrics, out)
    fig_nnz_effect(df, metrics, out)
    fig_calibration_improvement(df, out)
    fig_subject_variability(df, metrics, out)
    fig_pre_vs_post_scatter_nnz(df, out)
    fig_pre_vs_post_scatter_subject(df, out)
    fig_pre_vs_post_scatter_noise(df, out)
    fig_pre_vs_post_scatter_snr(df, out)
    fig_pre_vs_post_scatter_solver(df, out)
    fig_pre_post_by_group(df, out)
    fig_snr_violin_per_solver(df, out)
    fig_snr_meanstd_per_solver(df, out)
    fig_snr_meanstd_overlaid_noise(df, out)

    print("\n" + "=" * 65)
    print(f"  All figures saved to: {out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
