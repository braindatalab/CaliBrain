#!/usr/bin/env python
"""
Paper-style multi-panel calibration curve figures.

This script is tailored to the updated figure requirements:

Figure 2 (main): fixed orientation at default setting (SNR=0.5, NNZ=5)
  - Multi-panel grid: rows = solvers, cols = noise variance methods.
  - In each panel, plot 4 strategies:
      * pre-calibration
      * post-oracle
      * post-pooled
      * post-pooled-mismatch

Figure S1: fixed vs free pre-calibration (default setting)
  - Same grid (solver x noise); in each panel overlay:
      * fixed precal
      * free precal
  - Uses paired evaluation runs when possible (intersection by subject/run_id/seed).

Figure S2: fixed vs free post-calibration (default setting)
  - Same grid; in each panel overlay post-calibrated curves for:
      * post-oracle (fixed vs free)
      * post-pooled (fixed vs free)
      * post-pooled-mismatch (fixed vs free)
  - Recalibration is learned separately for fixed and free (handled by using the
    corresponding calibration JSON outputs).

Inputs
------
Calibration JSON outputs produced by `calibrain/workflows/calibration.py`.
Each JSON contains `eval_source` pointing to the aggregated per-run NPZ. This
script reads the NPZ to recover solver/noise_type/orientation/alpha_SNR/nnz/etc.

IMPORTANT: For robust grouping, aggregated NPZs should include scalar metadata
fields like `noise_type`, `alpha_SNR`, `nnz`, `seed`, `run_id`. The aggregation
workflow writes these when available.

Usage
-----
Edit CALIBRATION_JSON_ROOT_CANDIDATES below to point at one or more directories
that contain calibration JSON outputs (recursively).

Then run:
  python calibrain/workflows/plot_paper_calibration_figures.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# User configuration (edit)
# ---------------------------------------------------------------------------

# Folders containing calibration JSON files (recursive search).
#
# This repo's `configs/calibration_default.py` writes to:
#   /data/orabem/calibrain/results/calibration_eval/<orientation>/<solver>/<noise>/<experiment>/...
#
# If you run everything inside the repo with relative paths, you may also have:
#   results/calibration_eval/<orientation>/<solver>/<noise>/<experiment>/...
#
# By default we auto-detect whichever of the following roots exists.
CALIBRATION_JSON_ROOT_CANDIDATES = (
    Path("results/calibration_eval"),
    Path("/data/orabem/calibrain/results/calibration_eval"),
)

# Default setting for the paper figures.
DEFAULT_ALPHA_SNR = 0.5
DEFAULT_NNZ = 5

# Expected heads (used only for reporting/run-count sanity checks).
HEADS = ("CC120166", "CC120264", "CC120309", "CC120313")

# Strategy names as used in your folder naming / config selectors.
STRATEGIES = ("precal", "post_oracle", "post_pooled", "post_pooled_mismatch")

# Solver/noise ordering on the grid (edit to match your paper).
SOLVER_ORDER = ("BMN", "BMN_joint", "gamma_map_sflex", "gamma_lambda_map_sflex")
NOISE_ORDER = ("oracle", "baseline", "adaptive_joint_learning")

# Use consistent colors across figures for orientation comparisons.
ORIENTATION_COLORS = {
    "fixed": "#1f77b4",
    "free": "#ff7f0e",
}

# Optional aliasing for display/grouping.
SOLVER_ALIASES = {
    "BMN_joint": "BMN",
    "gamma_lambda_map_sflex": "gamma_map_sflex",
}
SOLVER_DISPLAY = {
    "BMN": "BMN",
    "gamma_map_sflex": "sFLEX-Gamma MAP",
}
NOISE_DISPLAY = {
    "oracle": "Oracle",
    "baseline": "Baseline",
    "adaptive_joint_learning": "Adaptive noise learning",
}

OUTPUT_DIR = Path("results/figures/paper_calibration")

# Plot uncertainty as standard deviation across runs (not SEM).
SHOW_STD_BAND = True
SHOW_STD_ERRORBARS = True
ERRORBAR_MARKERSIZE = 2.5
ERRORBAR_CAPSIZE = 2

# Output formats
SAVE_PNG = True
SAVE_PDF = False
SAVE_SVG = True  # can be large
DPI = 600

# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        # Typography (safe defaults, good PDF embedding)
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Lines
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.0,
        # Savefig
        "savefig.dpi": DPI,
        "figure.dpi": 110,
    }
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz_scalar(npz: Mapping[str, object], key: str) -> Optional[object]:
    if key not in npz:
        return None
    value = npz[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


@dataclass(frozen=True)
class RunKey:
    """Key used to pair evaluation runs across orientations."""

    subject: str
    run_id: int
    seed: int


@dataclass
class CurveRecord:
    strategy: str
    orientation: str
    solver: str
    noise_type: str
    alpha_snr: float
    nnz: int
    run_key: RunKey
    nominal: np.ndarray
    pre: np.ndarray
    post: np.ndarray


def _infer_strategy_from_path(path: Path) -> Optional[str]:
    parts = {p for p in path.parts}
    for strategy in STRATEGIES:
        if strategy in parts:
            return strategy
    return None


def _load_curve_record(path: Path) -> Optional[CurveRecord]:
    payload = _read_json(path)
    eval_source = payload.get("eval_source")
    if not eval_source:
        return None
    eval_path = Path(eval_source)
    if not eval_path.exists():
        return None

    strategy = _infer_strategy_from_path(path)
    if not strategy:
        # Fall back to run_name prefix if it contains strategy tokens.
        run_name = str(payload.get("run_name") or "")
        for token in STRATEGIES:
            if token in run_name:
                strategy = token
                break
    if not strategy:
        return None

    with np.load(eval_path) as npz:
        subject = str(_npz_scalar(npz, "subject") or "")
        solver = str(_npz_scalar(npz, "solver") or "")
        noise_type = str(_npz_scalar(npz, "noise_type") or "")
        orientation = str(_npz_scalar(npz, "orientation_type") or "fixed")
        alpha_snr = float(_npz_scalar(npz, "alpha_SNR") or np.nan)
        nnz = int(_npz_scalar(npz, "nnz") or -1)
        run_id = int(_npz_scalar(npz, "run_id") or -1)
        seed = int(_npz_scalar(npz, "seed") or -1)

    if not subject or run_id < 0 or seed < 0 or np.isnan(alpha_snr) or nnz < 0:
        # Missing metadata needed for filtering/pairing; ignore quietly.
        return None

    solver = SOLVER_ALIASES.get(solver, solver)
    noise_type = NOISE_TYPE_NORMALIZE.get(noise_type, noise_type)

    pre_block = payload.get("pre_calibration") or {}
    post_block = payload.get("post_calibration") or {}
    nominal = np.asarray(pre_block.get("nominal_coverages") or post_block.get("nominal_coverages"), dtype=float)
    pre = np.asarray(pre_block.get("empirical_coverages"), dtype=float)
    post = np.asarray(post_block.get("empirical_coverages"), dtype=float)
    if nominal.size == 0 or pre.shape != nominal.shape or post.shape != nominal.shape:
        return None

    return CurveRecord(
        strategy=strategy,
        orientation=orientation,
        solver=solver,
        noise_type=noise_type,
        alpha_snr=alpha_snr,
        nnz=nnz,
        run_key=RunKey(subject=subject, run_id=run_id, seed=seed),
        nominal=nominal,
        pre=pre,
        post=post,
    )


NOISE_TYPE_NORMALIZE = {
    # keep as-is for now, but allow future synonyms here
}


def _iter_calibration_jsons(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        yield from root.rglob("calibration_*.json")


def _mean_and_std(curves: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.vstack(curves)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    return mean, std


def _run_count_warning(name: str, count: int) -> str:
    if count == 100:
        return ""
    return f" [WARN expected 100, got {count}]"


def _filter_default_setting(records: Sequence[CurveRecord]) -> List[CurveRecord]:
    return [
        r
        for r in records
        if r.alpha_snr == DEFAULT_ALPHA_SNR and r.nnz == DEFAULT_NNZ
    ]


def _group_records(records: Sequence[CurveRecord]):
    grouped: MutableMapping[Tuple[str, str, str, str], List[CurveRecord]] = {}
    for r in records:
        key = (r.orientation, r.solver, r.noise_type, r.strategy)
        grouped.setdefault(key, []).append(r)
    return grouped


def _paired_intersection(
    fixed: Sequence[CurveRecord],
    free: Sequence[CurveRecord],
) -> Tuple[List[CurveRecord], List[CurveRecord]]:
    fixed_map = {r.run_key: r for r in fixed}
    free_map = {r.run_key: r for r in free}
    common = sorted(set(fixed_map) & set(free_map), key=lambda k: (k.subject, k.run_id, k.seed))
    return [fixed_map[k] for k in common], [free_map[k] for k in common]


def _plot_curve(
    ax: plt.Axes,
    nominal: np.ndarray,
    curves: Sequence[np.ndarray],
    label: str,
    color: str,
    linestyle: str = "-",
):
    mean, std = _mean_and_std(list(curves))
    ax.plot(nominal, mean, color=color, linestyle=linestyle, label=label, linewidth=2)
    if SHOW_STD_BAND:
        ax.fill_between(
            nominal,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    if SHOW_STD_ERRORBARS:
        ax.errorbar(
            nominal,
            mean,
            yerr=std,
            fmt="o" if linestyle == "-" else "s",
            color=color,
            ecolor=color,
            elinewidth=1,
            capsize=ERRORBAR_CAPSIZE,
            markersize=ERRORBAR_MARKERSIZE,
            alpha=0.9,
        )


def _setup_grid(n_rows: int, n_cols: int, title: str):
    # Use an extra column for a dedicated legend area to avoid tight_layout warnings.
    fig = plt.figure(figsize=(3.6 * n_cols + 2.0, 3.0 * n_rows + 0.9))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.55],
        wspace=0.18,
        hspace=0.15,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    return fig, axes, legend_ax


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", linestyle="--", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    if SAVE_PNG:
        fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def _clean_noise(noise: str) -> str:
    return NOISE_DISPLAY.get(noise, noise).replace("_", " ")


def _clean_solver(solver: str) -> str:
    return SOLVER_DISPLAY.get(solver, solver).replace("_", " ")


def _dedup_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def plot_figure_2(records: Sequence[CurveRecord], output_dir: Path) -> Path:
    # fixed orientation only
    records = [r for r in records if r.orientation == "fixed"]
    grouped = _group_records(records)

    # Use canonical keys for lookup (and deduplicate after aliasing).
    canonical_solvers = _dedup_preserve_order([SOLVER_ALIASES.get(s, s) for s in SOLVER_ORDER])
    noise_order = list(NOISE_ORDER)

    fig, axes, legend_ax = _setup_grid(
        len(canonical_solvers),
        len(noise_order),
        "Figure 2: Fixed-orientation calibration curves (default setting)",
    )

    colors = {
        "precal": "#1f77b4",
        "post_oracle": "#ff7f0e",
        "post_pooled": "#2ca02c",
        "post_pooled_mismatch": "#d62728",
    }

    for c, noise in enumerate(noise_order):
        axes[0][c].set_title(_clean_noise(noise), fontsize=12, fontweight="bold")

    for r, solver in enumerate(canonical_solvers):
        axes[r][0].annotate(
            _clean_solver(solver),
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-0.22, 0.5),
            textcoords="axes fraction",
            ha="right",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
        )
        for c, noise in enumerate(noise_order):
            ax = axes[r][c]
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.35, label="perfect")
            any_plotted = False
            for strategy in STRATEGIES:
                key = ("fixed", solver, noise, strategy)
                runs = grouped.get(key, [])
                if not runs:
                    continue
                nominal = runs[0].nominal
                # For the main figure we plot PRE for precal and POST for post-* strategies.
                curves = [rr.pre for rr in runs] if strategy == "precal" else [rr.post for rr in runs]
                _plot_curve(
                    ax,
                    nominal,
                    curves,
                    label=f"{strategy}{_run_count_warning(strategy, len(curves))}",
                    color=colors[strategy],
                )
                any_plotted = True
            if not any_plotted:
                ax.axis("off")
                continue
            _style_axis(ax)
            if r == len(canonical_solvers) - 1:
                ax.set_xlabel("Nominal coverage")
            if c == 0:
                ax.set_ylabel("Empirical coverage")

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center left", frameon=False)
    out_path = output_dir / "Figure2_fixed_default_calibration_curves.png"
    _save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_figure_s1(records: Sequence[CurveRecord], output_dir: Path) -> Path:
    # Pre-calibration only, overlay fixed vs free
    records = [r for r in records if r.strategy == "precal"]
    grouped = _group_records(records)
    canonical_solvers = _dedup_preserve_order([SOLVER_ALIASES.get(s, s) for s in SOLVER_ORDER])
    noise_order = list(NOISE_ORDER)

    fig, axes, legend_ax = _setup_grid(
        len(canonical_solvers),
        len(noise_order),
        "Figure S1: Pre-calibration (fixed vs free, default setting)",
    )

    for c, noise in enumerate(noise_order):
        axes[0][c].set_title(_clean_noise(noise), fontsize=12, fontweight="bold")

    for r, solver in enumerate(canonical_solvers):
        axes[r][0].annotate(
            _clean_solver(solver),
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-0.22, 0.5),
            textcoords="axes fraction",
            ha="right",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
        )
        for c, noise in enumerate(noise_order):
            ax = axes[r][c]
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.35, label="perfect")
            fixed_runs = grouped.get(("fixed", solver, noise, "precal"), [])
            free_runs = grouped.get(("free", solver, noise, "precal"), [])
            if not fixed_runs and not free_runs:
                ax.axis("off")
                continue

            if fixed_runs and free_runs:
                fixed_runs, free_runs = _paired_intersection(fixed_runs, free_runs)

            if fixed_runs:
                nominal = fixed_runs[0].nominal
                _plot_curve(
                    ax,
                    nominal,
                    [rr.pre for rr in fixed_runs],
                    label=f"fixed{_run_count_warning('fixed', len(fixed_runs))}",
                    color=ORIENTATION_COLORS["fixed"],
                    linestyle="-",
                )
            if free_runs:
                nominal = free_runs[0].nominal
                _plot_curve(
                    ax,
                    nominal,
                    [rr.pre for rr in free_runs],
                    label=f"free{_run_count_warning('free', len(free_runs))}",
                    color=ORIENTATION_COLORS["free"],
                    linestyle="-",
                )

            _style_axis(ax)
            if r == len(canonical_solvers) - 1:
                ax.set_xlabel("Nominal coverage")
            if c == 0:
                ax.set_ylabel("Empirical coverage")

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center left", frameon=False)
    out_path = output_dir / "FigureS1_precal_fixed_vs_free_default.png"
    _save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_figure_s2(records: Sequence[CurveRecord], output_dir: Path) -> Path:
    # Post-calibration curves only, overlay fixed vs free per strategy.
    records = [r for r in records if r.strategy in {"post_oracle", "post_pooled", "post_pooled_mismatch"}]
    grouped = _group_records(records)
    canonical_solvers = _dedup_preserve_order([SOLVER_ALIASES.get(s, s) for s in SOLVER_ORDER])
    noise_order = list(NOISE_ORDER)

    fig, axes, legend_ax = _setup_grid(
        len(canonical_solvers),
        len(noise_order),
        "Figure S2: Post-calibration (fixed vs free, default setting)",
    )

    # Match Figure S1 colors: fixed=blue, free=orange. Use line style to
    # differentiate post-calibration strategy.
    strategy_linestyles = {
        "post_oracle": "-",
        "post_pooled": "--",
        "post_pooled_mismatch": ":",
    }

    for c, noise in enumerate(noise_order):
        axes[0][c].set_title(_clean_noise(noise), fontsize=12, fontweight="bold")

    for r, solver in enumerate(canonical_solvers):
        axes[r][0].annotate(
            _clean_solver(solver),
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-0.22, 0.5),
            textcoords="axes fraction",
            ha="right",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
        )
        for c, noise in enumerate(noise_order):
            ax = axes[r][c]
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.35, label="perfect")
            any_plotted = False
            for strategy in ("post_oracle", "post_pooled", "post_pooled_mismatch"):
                fixed_runs = grouped.get(("fixed", solver, noise, strategy), [])
                free_runs = grouped.get(("free", solver, noise, strategy), [])
                if not fixed_runs and not free_runs:
                    continue
                if fixed_runs and free_runs:
                    fixed_runs, free_runs = _paired_intersection(fixed_runs, free_runs)
                linestyle = strategy_linestyles[strategy]
                if fixed_runs:
                    nominal = fixed_runs[0].nominal
                    _plot_curve(
                        ax,
                        nominal,
                        [rr.post for rr in fixed_runs],
                        label=f"fixed {strategy}{_run_count_warning('fixed', len(fixed_runs))}",
                        color=ORIENTATION_COLORS["fixed"],
                        linestyle=linestyle,
                    )
                    any_plotted = True
                if free_runs:
                    nominal = free_runs[0].nominal
                    _plot_curve(
                        ax,
                        nominal,
                        [rr.post for rr in free_runs],
                        label=f"free {strategy}{_run_count_warning('free', len(free_runs))}",
                        color=ORIENTATION_COLORS["free"],
                        linestyle=linestyle,
                    )
                    any_plotted = True

            if not any_plotted:
                ax.axis("off")
                continue
            _style_axis(ax)
            if r == len(canonical_solvers) - 1:
                ax.set_xlabel("Nominal coverage")
            if c == 0:
                ax.set_ylabel("Empirical coverage")

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center left", frameon=False)
    out_path = output_dir / "FigureS2_postcal_fixed_vs_free_default.png"
    _save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    roots = [p for p in CALIBRATION_JSON_ROOT_CANDIDATES if p.exists()]
    if not roots:
        raise SystemExit(
            "No calibration JSON roots found. Edit CALIBRATION_JSON_ROOT_CANDIDATES in "
            "calibrain/workflows/plot_paper_calibration_figures.py to point at your calibration output directory."
        )
    json_paths = list(_iter_calibration_jsons(roots))
    if not json_paths:
        raise SystemExit(
            f"No calibration JSON files found under: {[str(p) for p in roots]}. "
            "Make sure calibration outputs (calibration_*.json) exist and the root is correct."
        )
    records: List[CurveRecord] = []
    for path in json_paths:
        rec = _load_curve_record(path)
        if rec is None:
            continue
        records.append(rec)

    records = _filter_default_setting(records)
    if not records:
        raise SystemExit(
            "No calibration JSON records matched DEFAULT_ALPHA_SNR/DEFAULT_NNZ. "
            "Either you haven't run calibration for the default setting yet, or the NPZ metadata "
            "(alpha_SNR/nnz) is missing in the aggregated eval datasets."
        )

    fig2 = plot_figure_2(records, OUTPUT_DIR)
    s1 = plot_figure_s1(records, OUTPUT_DIR)
    s2 = plot_figure_s2(records, OUTPUT_DIR)
    print("Saved:")
    print(f"  {fig2}")
    print(f"  {s1}")
    print(f"  {s2}")


if __name__ == "__main__":
    main()
