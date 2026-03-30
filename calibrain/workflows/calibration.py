from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import block_diag

from calibrain import MetricEvaluator, UncertaintyCalibrator, UncertaintyEstimator
from calibrain.workflows.common import load_python_config

DEFAULT_CONFIG_PATH = Path("configs/calibration_default.py")


def build_uncertainty_components(
    *, nominal_coverages: Iterable[float] | None, logger
) -> Tuple[UncertaintyEstimator, MetricEvaluator]:
    coverages = (
        np.asarray(list(nominal_coverages), dtype=float)
        if nominal_coverages is not None
        else np.linspace(0.0, 1.0, num=11)
    )
    uncertainty_estimator = UncertaintyEstimator(
        nominal_coverages=coverages,
        logger=logger,
    )
    metrics_config = {
        "evaluation": (
            "mean_posterior_std",
            "emd",
        ),
        "calibration": (
            "mean_signed_deviation",
            "mean_absolute_deviation",
            "max_underconfidence_deviation",
            "max_overconfidence_deviation",
        ),
    }
    metric_evaluator = MetricEvaluator(
        uncertainty_estimator,
        evaluation_metrics=metrics_config["evaluation"],
        calibration_metrics=metrics_config["calibration"],
        logger=logger,
    )
    return uncertainty_estimator, metric_evaluator


def _slugify(value: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip()]
    slug = "".join(cleaned).strip("-_")
    return slug or "run"


def _expand_path_spec(value: Any, pattern: str) -> List[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        paths: List[Path] = []
        for item in value:
            paths.extend(_expand_path_spec(item, pattern))
        return paths
    path = Path(value)
    if path.is_dir():
        matched = sorted(path.glob(pattern))
        return [p for p in matched if p.is_file()]
    return [path]


def _collect_dataset_paths(
    *,
    dataset: Any = None,
    directory: Any = None,
    pattern: str = "*.npz",
    label: str,
) -> List[Path]:
    paths: List[Path] = []
    if dataset is not None:
        paths.extend(_expand_path_spec(dataset, pattern))
    if directory is not None:
        paths.extend(_expand_path_spec(directory, pattern))
    deduped: List[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(f"{label} dataset not found: {resolved}")
        if resolved.is_dir():
            raise ValueError(f"{label} path must reference .npz files, not directories: {resolved}")
        deduped.append(resolved)
        seen.add(resolved)
    return deduped


def _combine_datasets(datasets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not datasets:
        raise ValueError("At least one dataset is required to build the training pool.")
    if len(datasets) == 1:
        combined = dict(datasets[0])
        combined.pop("metadata", None)
        return combined

    first = datasets[0]
    orientation = first.get("orientation_type")
    coil_type = first.get("coil_type")
    n_times = first.get("n_times")

    x_true_list: List[np.ndarray] = []
    x_hat_list: List[np.ndarray] = []
    cov_list: List[np.ndarray] = []
    q_basis_list: List[np.ndarray] = []

    for ds in datasets:
        if ds.get("orientation_type") != orientation:
            raise ValueError("All training datasets must share orientation_type.")
        if ds.get("coil_type") != coil_type:
            raise ValueError("All training datasets must share coil_type.")
        if n_times and ds.get("n_times") not in (None, n_times):
            raise ValueError("All training datasets must share n_times.")

        x_true_list.append(np.asarray(ds["x_true"], dtype=float))
        x_hat_list.append(np.asarray(ds["x_hat"], dtype=float))
        cov_list.append(np.asarray(ds["posterior_cov"], dtype=float))
        q_basis = ds.get("Q_basis")
        if q_basis is not None:
            q_basis_arr = np.asarray(q_basis, dtype=float)
            if q_basis_arr.ndim >= 2:
                q_basis_list.append(q_basis_arr)

    combined_true = np.concatenate(x_true_list, axis=0)
    combined_hat = np.concatenate(x_hat_list, axis=0)
    combined_cov = block_diag(*cov_list)

    combined: Dict[str, Any] = {
        "orientation_type": orientation,
        "coil_type": coil_type,
        "x_true": combined_true,
        "x_hat": combined_hat,
        "posterior_cov": combined_cov,
        "n_sources": combined_true.shape[0],
        "n_times": int(n_times or combined_true.shape[-1]),
    }
    if q_basis_list:
        combined["Q_basis"] = np.concatenate(q_basis_list, axis=0)
    return combined


def _read_scalar_field(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _read_str_field(data, key, default=None):
    value = _read_scalar_field(data, key, default)
    if value is None:
        return default
    return str(value)


def _load_npz_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        orientation_type = _read_str_field(data, "orientation_type", "fixed")
        payload: Dict[str, Any] = {
            "orientation_type": orientation_type,
            "coil_type": _read_scalar_field(data, "coil_type"),
            "n_sources": int(_read_scalar_field(data, "n_sources", 0) or 0),
            "n_times": int(_read_scalar_field(data, "n_times", 0) or 0),
        }

        if "posterior_cov" not in data:
            raise ValueError(
                f"Dataset {path} is missing 'posterior_cov'. Regenerate the aggregation dataset with the current workflow."
            )

        payload["x_true"] = data["x_true"]
        payload["x_hat"] = data["x_hat"]
        payload["posterior_cov"] = data["posterior_cov"]
        if "Q_basis" in data:
            payload["Q_basis"] = data["Q_basis"]

        if payload["n_sources"] == 0:
            if "x_true" in payload:
                payload["n_sources"] = payload["x_true"].shape[0]
        if payload["n_times"] == 0:
            if "x_true" in payload:
                payload["n_times"] = payload["x_true"].shape[-1]
        return payload


def _serialize_array(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_serialize_array(v) for v in value]
    return value


def _serialize_block(block: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if block is None:
        return None
    return {key: _serialize_array(val) for key, val in block.items()}


def _plot_curves(pre_block: Dict[str, Any], post_block: Dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    nominal = np.array(pre_block.get("nominal_coverages", []), dtype=float)
    pre_emp = np.array(pre_block.get("empirical_coverages", []), dtype=float)
    post_emp = np.array(post_block.get("empirical_coverages", []), dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", label="perfect calibration")
    ax.plot(nominal, pre_emp, "o-", color="tab:blue", label="pre-calibration")
    ax.plot(nominal, post_emp, "s-", color="tab:red", label="post-calibration")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_aggregated_curves(
    nominal: np.ndarray,
    pre_runs: List[np.ndarray],
    post_runs: List[np.ndarray],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    pre_arr = np.vstack(pre_runs)
    post_arr = np.vstack(post_runs)
    pre_mean = pre_arr.mean(axis=0)
    pre_std = pre_arr.std(axis=0)
    post_mean = post_arr.mean(axis=0)
    post_std = post_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", label="perfect calibration")
    ax.errorbar(
        nominal,
        pre_mean,
        yerr=pre_std,
        fmt="o-",
        color="tab:blue",
        label="pre-calibration (mean ± std)",
    )
    ax.errorbar(
        nominal,
        post_mean,
        yerr=post_std,
        fmt="o-",
        color="tab:red",
        label="post-calibration (mean ± std)",
    )
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_calibration(config: Union[str, Path, Dict[str, Any]]) -> Path:
    if isinstance(config, (str, Path)):
        config = load_python_config(config)

    logger = logging.getLogger(__name__)
    train_paths = _collect_dataset_paths(
        dataset=config.get("train_dataset"),
        directory=config.get("train_dir"),
        pattern=config.get("train_pattern", "*.npz"),
        label="Train",
    )
    if not train_paths:
        raise ValueError("Provide at least one train dataset via 'train_dataset' or 'train_dir'.")
    train_sources = [str(p) for p in train_paths]

    eval_paths = _collect_dataset_paths(
        dataset=config.get("test_dataset"),
        directory=config.get("eval_dir"),
        pattern=config.get("eval_pattern", "*.npz"),
        label="Eval",
    )
    if not eval_paths:
        eval_paths = [train_paths[0]]
    eval_limit = config.get("eval_limit")
    if isinstance(eval_limit, int) and eval_limit > 0:
        eval_paths = eval_paths[:eval_limit]

    output_dir = Path(config.get("output_dir", "results/calibration_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_curve = bool(config.get("plot_curve", False))
    run_name = config.get("run_name")
    run_slug = _slugify(run_name) if isinstance(run_name, str) and run_name.strip() else None

    train_datasets = [_load_npz_dataset(path) for path in train_paths]
    train_data = _combine_datasets(train_datasets)

    logger.info("Loaded %d training datasets.", len(train_paths))
    for path in train_paths:
        logger.debug("  train -> %s", path)

    uncertainty_estimator, metric_evaluator = build_uncertainty_components(
        nominal_coverages=config.get("nominal_coverages"),
        logger=logger,
    )
    outputs: List[Path] = []
    aggregated_nominal: np.ndarray | None = None
    aggregated_pre_runs: List[np.ndarray] = []
    aggregated_post_runs: List[np.ndarray] = []

    for idx, eval_path in enumerate(eval_paths, start=1):
        print(f"[calibration] Evaluating {eval_path.name} ({idx}/{len(eval_paths)})")
        logger.info("Loaded eval dataset: %s", eval_path)
        eval_data = _load_npz_dataset(eval_path)
        calibrator = UncertaintyCalibrator(
            uncertainty_estimator=uncertainty_estimator,
            metric_evaluator=metric_evaluator,
        )
        calibration_results = calibrator.calibrate(
            train_data=train_data,
            test_data=eval_data,
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if len(eval_paths) == 1 and run_slug:
            slug = run_slug
        else:
            base = run_name or "calibration"
            slug = _slugify(f"{base}_{eval_path.stem}")
        stem = f"calibration_{slug}_{timestamp}"
        output_path = output_dir / f"{stem}.json"

        pre_block = calibration_results.get("pre_calibration", {})
        post_block = calibration_results.get("post_calibration", {})
        sanitized_pre = {k: v for k, v in pre_block.items() if k not in {"ci_lowers", "ci_uppers"}}
        sanitized_post = {k: v for k, v in post_block.items() if k not in {"ci_lowers", "ci_uppers"}}

        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "train_sources": train_sources,
            "eval_source": str(eval_path),
            "run_name": run_name,
            "pre_calibration": _serialize_block(sanitized_pre),
            "post_calibration": _serialize_block(sanitized_post),
            "train_empirical_coverages": _serialize_array(
                calibration_results.get("train_empirical_coverages")
            ),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved calibration summary -> %s", output_path)
        print(f"[calibration] Saved summary -> {output_path}")
        outputs.append(output_path)

        if plot_curve:
            curve_path = output_dir / f"calibration_curve_{slug}_{timestamp}.png"
            _plot_curves(pre_block, post_block, curve_path)
            logger.info("Saved calibration curve -> %s", curve_path)
            print(f"[calibration] Saved calibration curve -> {curve_path}")

        # Collect data for aggregated plot.
        nominal_values = pre_block.get("nominal_coverages")
        if nominal_values is None:
            nominal_values = post_block.get("nominal_coverages")
        if nominal_values is None:
            nominal_values = uncertainty_estimator.nominal_coverages
        nominal_values = np.asarray(nominal_values, dtype=float)
        pre_empirical = np.asarray(pre_block.get("empirical_coverages"), dtype=float)
        post_empirical = np.asarray(post_block.get("empirical_coverages"), dtype=float)
        if aggregated_nominal is None:
            aggregated_nominal = nominal_values
        elif not np.allclose(aggregated_nominal, nominal_values):
            raise ValueError("Nominal coverage grids differ across evaluation runs.")
        aggregated_pre_runs.append(pre_empirical)
        aggregated_post_runs.append(post_empirical)

    logger.info("Calibration artifacts written to %s", output_dir)
    print(f"[calibration] All artifacts stored in -> {output_dir}")

    if plot_curve and aggregated_nominal is not None and aggregated_pre_runs:
        agg_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        agg_slug = run_slug or _slugify(run_name or "calibration")
        agg_path = output_dir / f"calibration_curve_{agg_slug}_aggregate_{agg_timestamp}.png"
        _plot_aggregated_curves(
            aggregated_nominal,
            aggregated_pre_runs,
            aggregated_post_runs,
            agg_path,
        )
        logger.info("Saved aggregated calibration curve -> %s", agg_path)
        print(f"[calibration] Saved aggregated calibration curve -> {agg_path}")

    return outputs[-1] if outputs else output_dir


if __name__ == "__main__":
    run_calibration(DEFAULT_CONFIG_PATH)
