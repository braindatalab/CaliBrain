from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from calibrain.calibration_dataset import (
    concatenate_summaries,
    filter_summaries_by_metadata,
)
from calibrain.run_manifest import summaries_from_manifest
from calibrain.utils import load_python_config

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
    metadata_entries = dataset.get("metadata") or []
    subject_candidates = {
        entry.get("subject")
        for entry in metadata_entries
        if isinstance(entry, dict) and entry.get("subject")
    }
    primary_subject = subject_candidates.pop() if len(subject_candidates) == 1 else None

    solver_candidates = {
        entry.get("solver")
        for entry in metadata_entries
        if isinstance(entry, dict) and entry.get("solver")
    }
    primary_solver = solver_candidates.pop() if len(solver_candidates) == 1 else None

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

    if primary_subject is not None:
        scalars["subject"] = primary_subject
    if primary_solver is not None:
        scalars["solver"] = primary_solver

    # Persist additional per-run metadata that is useful for downstream grouping
    # (plots, pooled filtering, pairing fixed/free runs). When aggregating a
    # single posterior summary (the default), these are unambiguous.
    def _unique_or_none(values):
        uniq = {v for v in values if v is not None}
        return uniq.pop() if len(uniq) == 1 else None

    def _collect_unique(key: str):
        return _unique_or_none(
            entry.get(key)
            for entry in metadata_entries
            if isinstance(entry, dict)
        )

    for key in ("noise_type", "alpha_SNR", "nnz", "seed", "run_id", "global_run_id"):
        value = _collect_unique(key)
        if value is not None:
            scalars[key] = value

    scalar_arrays = {key: np.array(val) for key, val in scalars.items()}
    # Write NPZ atomically to avoid corrupted zip archives if the job is interrupted.
    # NOTE: numpy appends ".npz" when the provided filename does not end with it.
    # Use a tmp path that ends with ".npz" so we can safely os.replace it.
    tmp_npz = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays, **scalar_arrays)
    os.replace(tmp_npz, output_path)

    dataset_meta = {
        "orientation_type": dataset.get("orientation_type"),
        "coil_type": dataset.get("coil_type"),
        "sensor_kind": dataset.get("sensor_kind"),
        "n_sources": int(dataset.get("n_sources", 0)),
        "n_times": int(dataset.get("n_times", 0)),
        "subject": primary_subject,
        "solver": primary_solver,
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
        "solver": primary_solver,
        "summaries": [str(summary.path) for summary in summaries],
    }
    tmp_meta = meta_path.with_name(f"{meta_path.name}.tmp")
    tmp_meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    os.replace(tmp_meta, meta_path)


def _aggregate_single(config: Dict[str, Any], tag: str | None = None) -> None:
    if isinstance(config, (str, Path)):
        config = load_python_config(config)

    if "manifest_path" not in config:
        raise KeyError(
            "Aggregation config must specify 'manifest_path' (CSV) so discovery does not "
            "scan the filesystem."
        )
    manifest_path = Path(config["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {manifest_path}")

    summaries_root = Path(config.get("summaries_root") or manifest_path.parent)

    output_dir = Path(config.get("output_dir", "results/calibration_datasets"))
    criteria = config.get("filter")

    summaries = summaries_from_manifest(manifest_path)
    if not summaries:
        raise FileNotFoundError(
            f"No posterior summary files found in manifest {manifest_path}"
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
            if isinstance(sub_config, dict):
                sub_config = dict(sub_config)
                if "manifest_path" not in sub_config and "manifest_path" in config:
                    sub_config["manifest_path"] = config["manifest_path"]
                if "summaries_root" not in sub_config and "summaries_root" in config:
                    sub_config["summaries_root"] = config["summaries_root"]
            print(f"[aggregation] Running split '{name}'")
            _aggregate_single(sub_config, tag=str(name))
        return

    _aggregate_single(config)


if __name__ == "__main__":
    aggregate_posteriors(DEFAULT_CONFIG_PATH)
