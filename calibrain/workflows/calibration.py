from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from calibrain import UncertaintyCalibrator
from calibrain.workflows.common import build_uncertainty_components, load_python_config

DEFAULT_CONFIG_PATH = Path("configs/calibration_default.py")


def _slugify(value: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip()]
    slug = "".join(cleaned).strip("-_")
    return slug or "run"


def _load_npz_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "x_true": data["x_true"],
            "x_hat": data["x_hat"],
            "posterior_std": data["posterior_std"],
        }


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


def run_calibration(config: Union[str, Path, Dict[str, Any]]) -> Path:
    if isinstance(config, (str, Path)):
        config = load_python_config(config)

    logger = logging.getLogger(__name__)
    train_path = Path(config["train_dataset"])
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")

    test_dataset = config.get("test_dataset")
    test_path = Path(test_dataset) if test_dataset else train_path
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    output_dir = Path(config.get("output_dir", "results/calibration_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_curve = bool(config.get("plot_curve", False))
    run_name = config.get("run_name")
    run_slug = _slugify(run_name) if isinstance(run_name, str) and run_name.strip() else None

    train_data = _load_npz_dataset(train_path)
    test_data = _load_npz_dataset(test_path)

    logger.info("Loaded train dataset: %s", train_path)
    logger.info("Loaded eval dataset: %s", test_path)

    uncertainty_estimator, metric_evaluator = build_uncertainty_components(
        nominal_coverages=config.get("nominal_coverages"),
        logger=logger,
    )
    calibrator = UncertaintyCalibrator(
        uncertainty_estimator=uncertainty_estimator,
        metric_evaluator=metric_evaluator,
    )
    calibration_results = calibrator.calibrate(
        train_data=train_data,
        test_data=test_data,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"calibration_{run_slug}_{timestamp}" if run_slug else f"calibration_{timestamp}"
    output_path = output_dir / f"{stem}.json"
    pre_block = calibration_results.get("pre_calibration", {})
    post_block = calibration_results.get("post_calibration", {})
    sanitized_pre = {k: v for k, v in pre_block.items() if k not in {"ci_lowers", "ci_uppers"}}
    sanitized_post = {k: v for k, v in post_block.items() if k not in {"ci_lowers", "ci_uppers"}}

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_dataset": str(train_path),
        "test_dataset": str(test_path),
        "run_name": run_name,
        "pre_calibration": _serialize_block(sanitized_pre),
        "post_calibration": _serialize_block(sanitized_post),
        "train_empirical_coverages": _serialize_array(
            calibration_results.get("train_empirical_coverages")
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved calibration summary -> %s", output_path)

    if plot_curve:
        curve_stem = (
            f"calibration_curve_{run_slug}_{timestamp}"
            if run_slug
            else f"calibration_curve_{timestamp}"
        )
        plot_path = output_dir / f"{curve_stem}.png"
        _plot_curves(pre_block, post_block, plot_path)
        logger.info("Saved calibration curve -> %s", plot_path)

    logger.info("Calibration artifacts written to %s", output_dir)
    return output_path


if __name__ == "__main__":
    run_calibration(DEFAULT_CONFIG_PATH)
