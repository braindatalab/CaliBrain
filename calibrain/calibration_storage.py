from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SCHEMA_VERSION = 1
_RECORD_GLOB = "calibration_run-*.json"

Serializable = Union[str, int, float, None, bool, Sequence["Serializable"], Mapping[str, "Serializable"]]


def _to_serializable(value: Any) -> Serializable:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value  # type: ignore[arg-type]


def _infer_n_observations(stage_data: Mapping[str, Any]) -> Optional[int]:
    if "n_observations" in stage_data and stage_data["n_observations"] is not None:
        return int(stage_data["n_observations"])
    ci_lowers = stage_data.get("ci_lowers")
    if ci_lowers is not None:
        return int(np.asarray(ci_lowers).shape[-1])
    ci_counts = stage_data.get("ci_counts")
    empirical = stage_data.get("empirical_coverages")
    if ci_counts is not None and empirical is not None:
        counts = np.asarray(ci_counts, dtype=float)
        empirical = np.asarray(empirical, dtype=float)
        valid = empirical > 0
        if np.any(valid):
            denom = counts[valid] / empirical[valid]
            return int(round(float(np.median(denom))))
    return None


def _compact_stage(stage_data: Optional[Mapping[str, Any]], stage: str) -> Dict[str, Any]:
    if not stage_data:
        return {}
    payload: Dict[str, Any] = {}
    for key in ("nominal_coverages", "empirical_coverages"):
        value = stage_data.get(key)
        if value is not None:
            payload[key] = value
    metrics = stage_data.get("calibration_metrics")
    if metrics:
        payload["calibration_metrics"] = metrics
    if stage == "pre":
        ci_counts = stage_data.get("ci_counts")
        if ci_counts is not None:
            payload["ci_counts"] = ci_counts
        n_obs = _infer_n_observations(stage_data)
        if n_obs is not None:
            payload["n_observations"] = n_obs
    if stage == "post":
        recal = stage_data.get("recalibrated_nominal_coverages")
        if recal is not None:
            payload["recalibrated_nominal_coverages"] = recal
    return payload


def save_calibration_record(
    output_dir: Union[str, Path],
    record_name: str,
    metadata: Mapping[str, Any],
    pre_calibration: Mapping[str, Any],
    post_calibration: Mapping[str, Any],
) -> Path:
    """Persist calibration payload alongside metadata in JSON format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": _to_serializable(dict(metadata)),
        "pre_calibration": _to_serializable(_compact_stage(pre_calibration, "pre")),
        "post_calibration": _to_serializable(_compact_stage(post_calibration, "post")),
    }
    record_path = output_dir / f"{record_name}.json"
    with record_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return record_path


def load_calibration_record(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["path"] = path.as_posix()
    return data


def iter_calibration_records(
    root: Union[str, Path],
    predicate: Optional[Callable[[Mapping[str, Any]], bool]] = None,
) -> Iterator[Dict[str, Any]]:
    root_path = Path(root)
    if root_path.is_file():
        paths = [root_path]
    else:
        paths = sorted(root_path.rglob(_RECORD_GLOB))
    for path in paths:
        record = load_calibration_record(path)
        meta = record.get("metadata", {})
        if predicate and not predicate(meta):
            continue
        yield record


def metadata_matcher(**expected: Any) -> Callable[[Mapping[str, Any]], bool]:
    def _match(meta: Mapping[str, Any]) -> bool:
        for key, value in expected.items():
            if meta.get(key) != value:
                return False
        return True

    return _match


def stack_empirical_curves(
    records: Iterable[Mapping[str, Any]],
    stage: str = "pre",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    curves = []
    weights = []
    nominal = None
    for record in records:
        stage_payload = record.get(f"{stage}_calibration") or {}
        empirical = stage_payload.get("empirical_coverages")
        if empirical is None:
            continue
        empirical_arr = np.asarray(empirical, dtype=float)
        curves.append(empirical_arr)
        if nominal is None and stage_payload.get("nominal_coverages") is not None:
            nominal = np.asarray(stage_payload["nominal_coverages"], dtype=float)
        if stage == "pre":
            weights.append(stage_payload.get("n_observations", 1.0))
        else:
            weights.append(stage_payload.get("weight", 1.0))
    if not curves:
        raise ValueError(f"No calibration curves found for stage '{stage}'")
    if nominal is None:
        nominal = np.arange(curves[0].shape[-1])
    curves_arr = np.vstack(curves)
    weights_arr = np.asarray(weights, dtype=float)
    if not np.any(weights_arr):
        weights_arr = None
    return nominal, curves_arr, weights_arr
