from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from calibrain.calibration_dataset import (
    concatenate_summaries,
    discover_posterior_summaries,
    filter_summaries_by_metadata,
)
from calibrain.workflows.common import load_python_config

DEFAULT_CONFIG_PATH = Path("configs/aggregate_default.py")


def _serialize_filter(criteria: Dict[str, Any] | None) -> Dict[str, Any]:
    if not criteria:
        return {}

    def _convert(value: Any) -> Any:
        if callable(value):
            doc = getattr(value, "__doc__", None)
            return doc.strip() if isinstance(doc, str) else repr(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_convert(v) for v in value]
        return value

    return {key: _convert(val) for key, val in criteria.items()}

def _build_output_path(base_dir: Path, metadata: Dict[str, Any]) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = metadata.get("run_id") or metadata.get("global_run_id")
    subject = metadata.get("subject")
    solver = metadata.get("solver")
    seed = metadata.get("seed")
    parts = []
    if subject:
        parts.append(str(subject))
    if solver:
        parts.append(str(solver))
    if run_id is not None:
        parts.append(f"run{int(run_id):08d}")
    if seed is not None:
        parts.append(f"seed{seed}")
    if not parts:
        parts.append("summary")
    stem = "_".join(parts)
    return base_dir / f"{stem}.npz"


def _write_dataset(
    *,
    dataset: Dict[str, Any],
    output_path: Path,
    summaries,
    summaries_root: Path,
    split_name: str,
    criteria: Dict[str, Any] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    scalars: Dict[str, Any] = {}
    for key, value in dataset.items():
        if key == "metadata":
            continue
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif value is None:
            continue
        else:
            scalars[key] = value

    scalar_arrays = {key: np.array(val) for key, val in scalars.items()}
    np.savez_compressed(output_path, **arrays, **scalar_arrays)

    dataset_meta = {
        "orientation_type": dataset.get("orientation_type"),
        "coil_type": dataset.get("coil_type"),
        "sensor_kind": dataset.get("sensor_kind"),
        "n_sources": int(dataset.get("n_sources", 0)),
        "n_times": int(dataset.get("n_times", 0)),
    }
    meta_path = output_path.with_suffix(".json")
    meta_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_root": str(summaries_root),
        "summary_count": len(summaries),
        "n_sources": dataset_meta["n_sources"],
        "n_times": dataset_meta["n_times"],
        "split": split_name,
        "criteria": _serialize_filter(criteria),
        "dataset_info": dataset_meta,
        "summaries": [str(summary.path) for summary in summaries],
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")


def _aggregate_single(config: Dict[str, Any], tag: str | None = None) -> None:
    if isinstance(config, (str, Path)):
        config = load_python_config(config)

    summaries_root = Path(config["summaries_root"])
    if not summaries_root.exists():
        raise FileNotFoundError(f"Summary root does not exist: {summaries_root}")

    output_dir = Path(config.get("output_dir", "results/calibration_datasets"))
    criteria = config.get("filter")

    summaries = list(discover_posterior_summaries(summaries_root))
    if not summaries:
        raise FileNotFoundError(
            f"No posterior summary files found under {summaries_root}"
        )

    if criteria:
        summaries = filter_summaries_by_metadata(summaries, criteria)
        if not summaries:
            raise ValueError("No summaries matched the provided metadata filter.")

    for summary in summaries:
        dataset = concatenate_summaries([summary])
        meta = (dataset.get("metadata") or [{}])[0]
        output_path = _build_output_path(output_dir, meta or {})
        _write_dataset(
            dataset=dataset,
            output_path=output_path,
            summaries=[summary],
            summaries_root=summaries_root,
            split_name="single",
            criteria=criteria,
        )
    if tag:
        print(f"[aggregation] Completed split '{tag}' with {len(summaries)} summaries.")


def aggregate_posteriors(config: Union[str, Path, Dict[str, Any]]) -> None:
    if isinstance(config, (str, Path)):
        config = load_python_config(config)

    if "splits" in config:
        splits = config["splits"]
        if isinstance(splits, dict):
            items = splits.items()
        else:
            items = enumerate(splits)
        for name, sub_config in items:
            print(f"[aggregation] Running split '{name}'")
            _aggregate_single(sub_config, tag=str(name))
        return

    _aggregate_single(config)


if __name__ == "__main__":
    aggregate_posteriors(DEFAULT_CONFIG_PATH)
