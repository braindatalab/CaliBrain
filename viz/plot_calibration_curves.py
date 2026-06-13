#!/usr/bin/env python
"""Aggregate calibration curves across runs and plot solver/noise grids."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from calibrain.calibration_storage import iter_calibration_records
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "axes.titlesize": 14,
        "axes.labelsize": 14,
    }
)

AXIS_LABEL_FONTSIZE = 12

GroupKey = Tuple[str, str]
CurveStore = Dict[GroupKey, Dict[str, List[np.ndarray]]]


def _matches(meta: Dict[str, object], values: Sequence | None, key: str) -> bool:
    if not values:
        return True
    return meta.get(key) in values


def _collect_records(
    records_dir: Path,
    nnz: Sequence[int] | None,
    subjects: Sequence[str] | None,
    alpha_snr: Sequence[float] | None,
    solvers: Sequence[str] | None,
    noise_types: Sequence[str] | None,
) -> CurveStore:
    if not records_dir.exists():
        raise FileNotFoundError(f"No calibration records found at {records_dir}")

    groups: CurveStore = defaultdict(lambda: {"nominal": None, "pre": [], "post": []})
    for record in iter_calibration_records(records_dir):
        meta = record.get("metadata", {})
        if not (
            _matches(meta, nnz, "nnz")
            and _matches(meta, subjects, "subject")
            and _matches(meta, alpha_snr, "alpha_SNR")
            and _matches(meta, solvers, "solver")
            and _matches(meta, noise_types, "noise_type")
        ):
            continue
        solver = str(meta.get("solver"))
        noise = str(meta.get("noise_type"))
        key = (solver, noise)

        for stage in ("pre_calibration", "post_calibration"):
            if stage not in record:
                raise KeyError(f"Record missing '{stage}' payload: {record.get('path')}")

        store = groups[key]
        pre_payload = record["pre_calibration"]
        post_payload = record["post_calibration"]
        nominal = np.asarray(pre_payload.get("nominal_coverages"), dtype=float)
        if store["nominal"] is None:
            store["nominal"] = nominal
        else:
            if not np.allclose(store["nominal"], nominal):
                raise ValueError("Nominal coverage grid mismatch across records")

        pre_curve = np.asarray(pre_payload.get("empirical_coverages"), dtype=float)
        post_curve = np.asarray(post_payload.get("empirical_coverages"), dtype=float)
        if pre_curve.shape != nominal.shape:
            raise ValueError("Pre-calibration curve shape mismatch with nominal grid")
        if post_curve.shape != nominal.shape:
            raise ValueError("Post-calibration curve shape mismatch with nominal grid")
        store["pre"].append(pre_curve)
        store["post"].append(post_curve)

    return groups


def _merge_solver_groups(
    groups: CurveStore,
    alias_map: Dict[str, str],
) -> CurveStore:
    merged: CurveStore = defaultdict(lambda: {"nominal": None, "pre": [], "post": []})
    for (solver, noise), payload in groups.items():
        canonical = alias_map.get(solver, solver)
        key = (canonical, noise)
        target = merged[key]
        if target["nominal"] is None:
            target["nominal"] = payload["nominal"]
        else:
            if not np.allclose(target["nominal"], payload["nominal"]):
                raise ValueError("Nominal coverage mismatch when merging solver aliases")
        target["pre"].extend(payload["pre"])
        target["post"].extend(payload["post"])
    return merged


def _mean_and_sem(curves: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.vstack(curves)
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=0) / np.sqrt(arr.shape[0])
    return mean, sem


def _plot_group(ax: plt.Axes, nominal: np.ndarray, curves: Dict[str, List[np.ndarray]]):
    pre_mean, pre_sem = _mean_and_sem(curves["pre"])
    post_mean, post_sem = _mean_and_sem(curves["post"])

    ax.plot(nominal, pre_mean, color="#1f77b4", label="Pre")
    ax.fill_between(
        nominal,
        np.clip(pre_mean - pre_sem, 0.0, 1.0),
        np.clip(pre_mean + pre_sem, 0.0, 1.0),
        color="#1f77b4",
        alpha=0.2,
    )
    ax.plot(nominal, post_mean, color="#d62728", label="Post")
    ax.fill_between(
        nominal,
        np.clip(post_mean - post_sem, 0.0, 1.0),
        np.clip(post_mean + post_sem, 0.0, 1.0),
        color="#d62728",
        alpha=0.2,
    )
    ax.errorbar(
        nominal,
        pre_mean,
        yerr=pre_sem,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=3,
        markersize=4,
        alpha=0.9,
    )
    ax.errorbar(
        nominal,
        post_mean,
        yerr=post_sem,
        fmt="s",
        color="#d62728",
        ecolor="#d62728",
        capsize=3,
        markersize=4,
        alpha=0.9,
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)


def main():
    # ---- Configuration ----
    dpi = 600
    nnz = [5]
    subjects = ["CC120166", "CC120264", "CC120309", "CC120313"]
    snr = [0.5]
    solvers = ["BMN", "BMN_joint", "gamma_map_sflex", "gamma_lambda_map_sflex"]
    noise_types = ["baseline", "oracle", "adaptive_joint_learning"]

    solver_aliases = {
        "BMN_joint": "BMN",
        "gamma_lambda_map_sflex": "gamma_map_sflex",
    }
    solver_display = {
        "BMN": "BMN",
        "gamma_map_sflex": "sFLEX-Gamma MAP",
    }
    
    records_dir = Path("results") / f"calibration_records_nnz={nnz[0]}"
    output_path = Path("results") / "figures" / "calibration" / f"calibration_curves_NNZ={nnz[0]}_SNR={snr[0]}.png"

    groups = _collect_records(
        records_dir=records_dir,
        nnz=nnz or [],
        subjects=subjects or [],
        alpha_snr=snr or [],
        solvers=solvers or [],
        noise_types=noise_types or [],
    )
    if not groups:
        raise RuntimeError("No calibration records matched the provided filters")
    groups = _merge_solver_groups(groups, solver_aliases)

    def _clean(label: str) -> str:
        return label.replace("_", " ")

    available_solvers = sorted({solver for solver, _ in groups.keys()})
    available_noise = sorted({noise for _, noise in groups.keys()})
    if solvers:
        solver_order = []
        for solver in solvers:
            canonical = solver_aliases.get(solver, solver)
            if canonical not in solver_order:
                solver_order.append(canonical)
    else:
        solver_order = available_solvers
    noise_order = list(noise_types) if noise_types else available_noise
    n_rows = len(solver_order)
    n_cols = len(noise_order)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.05, "wspace": -0.3},
    )

    for c, noise in enumerate(noise_order):
        axes[0][c].set_title(_clean(noise), fontsize=12, fontweight="bold", pad=12)

    for r, solver in enumerate(solver_order):
        axes[r][0].annotate(
            solver_display.get(solver, _clean(solver)),
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-0.35, 0.5),
            textcoords="axes fraction",
            ha="right",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
        )
        for c, noise in enumerate(noise_order):
            ax = axes[r][c]
            key = (solver, noise)
            if key in groups and groups[key]["pre"]:
                _plot_group(ax, groups[key]["nominal"], groups[key])
            else:
                ax.axis("off")
                continue
            if r == n_rows - 1:
                ax.set_xlabel("Nominal confidence level", fontsize=AXIS_LABEL_FONTSIZE)
            if c == 0:
                ax.set_ylabel("Empirical coverage", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_aspect('equal', adjustable='box')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}"))

    handles = [
        Patch(facecolor="#1f77b4", edgecolor="#1f77b4", label="Pre-calibration"),
        Patch(facecolor="#d62728", edgecolor="#d62728", label="Post-calibration"),
        plt.Line2D([0], [0], linestyle="--", color="k", label="Perfect calibration"),
    ]
    labels = [h.get_label() for h in handles]
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        borderaxespad=0,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
