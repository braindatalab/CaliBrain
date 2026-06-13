from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from calibrain.calibration_dataset import PosteriorSummary


@dataclass(frozen=True)
class ManifestRow:
    """Typed view of a single manifest CSV row."""

    summary_path: Path
    metadata: Dict[str, Any]


def _as_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _as_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_str(value: str | None) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _coerce_manifest_metadata(row: Mapping[str, str]) -> Dict[str, Any]:
    # Keep only fields that the calibration/aggregation pipeline expects to
    # filter/group on. Everything else is preserved as strings.
    meta: Dict[str, Any] = {}

    for key in (
        "global_run_id",
        "run_id",
        "seed",
        "nnz",
        "alpha_SNR",
        "subject",
        "orientation_type",
        "coil_type",
        "sensor_kind",
        "n_sources",
        "n_times",
        "solver",
        "noise_type",
        "posterior_summary",
    ):
        if key not in row:
            continue
        meta[key] = row.get(key)

    # Coerce types where practical.
    for key in ("global_run_id", "run_id", "seed", "nnz", "coil_type", "n_sources", "n_times"):
        if key in meta:
            meta[key] = _as_int(meta.get(key))
    if "alpha_SNR" in meta:
        meta["alpha_SNR"] = _as_float(meta.get("alpha_SNR"))
    for key in ("subject", "orientation_type", "solver", "noise_type"):
        if key in meta:
            meta[key] = _as_str(meta.get(key))

    # Preserve any remaining fields (as strings).
    for key, value in row.items():
        if key in meta:
            continue
        meta[key] = value

    return meta


def load_manifest_csv(path: str | Path) -> List[ManifestRow]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")

    rows: List[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV has no header row: {path}")
        for raw in reader:
            meta = _coerce_manifest_metadata(raw)
            summary_raw = raw.get("posterior_summary") or raw.get("summary_path") or raw.get("posterior_path")
            summary_str = _as_str(summary_raw)
            if not summary_str:
                # Skip rows that do not point to a posterior summary file.
                continue
            summary_path = Path(summary_str)
            rows.append(ManifestRow(summary_path=summary_path, metadata=meta))
    return rows


def summaries_from_manifest(path: str | Path) -> List[PosteriorSummary]:
    rows = load_manifest_csv(path)
    summaries: List[PosteriorSummary] = []
    for row in rows:
        summaries.append(PosteriorSummary(path=row.summary_path, metadata=row.metadata))
    return summaries
