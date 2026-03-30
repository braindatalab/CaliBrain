from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
from mne.io.constants import FIFF
from scipy.linalg import block_diag

EEG_COIL_TYPES = {FIFF.FIFFV_COIL_EEG}
MEG_COIL_TYPES = {
    FIFF.FIFFV_COIL_VV_MAG_T1,
    FIFF.FIFFV_COIL_VV_PLANAR_T1,
}
ALLOWED_COIL_TYPES = EEG_COIL_TYPES | MEG_COIL_TYPES


def _maybe_q_basis_array(array: Any) -> np.ndarray | None:
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        # Scalar placeholder (e.g., NaN for fixed orientation)
        return None
    return arr.astype(float, copy=False)




@dataclass
class PosteriorSummary:
    """Container describing a saved posterior summary file and metadata."""

    path: Path
    metadata: dict | None = None


def _normalize_orientation(orientation_type: str | None) -> str:
    if orientation_type is None:
        return "fixed"
    normalized = orientation_type.lower()
    if normalized not in {"fixed", "free"}:
        raise ValueError(f"Unsupported orientation_type '{orientation_type}'")
    return normalized


def _validate_coil_type(coil_type: int | None) -> None:
    if coil_type is None:
        return
    if coil_type not in ALLOWED_COIL_TYPES:
        raise ValueError(
            f"Unsupported coil_type '{coil_type}'. Expected one of "
            f"{sorted(ALLOWED_COIL_TYPES)} or None."
        )


def is_fixed_orientation(orientation_type: str | None) -> bool:
    return _normalize_orientation(orientation_type) == "fixed"


def is_free_eeg_orientation(
    orientation_type: str | None,
    coil_type: int | None,
    n_components: int | None = None,
) -> bool:
    if _normalize_orientation(orientation_type) != "free":
        return False
    if coil_type is not None:
        _validate_coil_type(coil_type)
        return coil_type in EEG_COIL_TYPES
    return n_components == 3


def is_free_meg_orientation(
    orientation_type: str | None,
    coil_type: int | None,
    n_components: int | None = None,
) -> bool:
    if _normalize_orientation(orientation_type) != "free":
        return False
    if coil_type is not None:
        _validate_coil_type(coil_type)
        return coil_type in MEG_COIL_TYPES
    return n_components == 2


def _load_h5_metadata(path: Path) -> dict | None:
    try:
        with h5py.File(path, "r") as handle:
            meta_json = handle.attrs.get("metadata_json")
            if meta_json is None:
                return None
            if isinstance(meta_json, bytes):
                meta_json = meta_json.decode("utf-8")
            return json.loads(meta_json)
    except Exception:
        return None


def _load_legacy_metadata(path: Path) -> dict | None:
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None

def load_posterior_summary(summary: PosteriorSummary) -> dict:
    """
    Load a single posterior summary file produced by benchmarking.
    Supports the new HDF5 format as well as legacy ``.npz`` files.
    """
    suffix = summary.path.suffix.lower()
    metadata = summary.metadata or {}
    orientation_type = metadata.get("orientation_type")
    coil_type = metadata.get("coil_type")
    _validate_coil_type(coil_type)

    if suffix in {".h5", ".hdf5"}:
        with h5py.File(summary.path, "r") as handle:
            def _require(name: str) -> np.ndarray:
                if name not in handle:
                    raise KeyError(f"Dataset '{name}' missing in {summary.path}")
                return handle[name][:]

            payload: Dict[str, Any] = {"metadata": metadata}
            payload["x_true"] = _require("x_true")
            payload["x_hat"] = _require("x_hat")
            payload["posterior_cov"] = _require("posterior_cov")
            if "Q_basis" in handle:
                q_dataset = handle["Q_basis"]
                q_data = q_dataset[()] if q_dataset.shape == () else q_dataset[:]
                q_basis = _maybe_q_basis_array(q_data)
                if q_basis is not None:
                    payload["Q_basis"] = q_basis
            return payload

    if suffix == ".npz":
        with np.load(summary.path) as data:
            payload = {"metadata": metadata}
            payload["x_true"] = data["x_true"]
            payload["x_hat"] = data["x_hat"]
            payload["posterior_cov"] = data["posterior_cov"]
            if "Q_basis" in data:
                q_basis = _maybe_q_basis_array(data["Q_basis"])
                if q_basis is not None:
                    payload["Q_basis"] = q_basis
        return payload

    raise ValueError(f"Unsupported posterior summary format: {summary.path}")


def discover_posterior_summaries(root: Path) -> Iterable[PosteriorSummary]:
    for path in root.rglob("posterior_summary*.h5"):
        metadata = _load_h5_metadata(path)
        yield PosteriorSummary(path=path, metadata=metadata)
    for path in root.rglob("posterior_summary*.npz"):
        metadata = _load_legacy_metadata(path)
        yield PosteriorSummary(path=path, metadata=metadata)


def metadata_matches(metadata: dict | None, criteria: dict | None) -> bool:
    if not criteria:
        return True
    metadata = metadata or {}
    for key, allowed in criteria.items():
        current_value = metadata.get(key)
        if callable(allowed):
            if not allowed(current_value):
                return False
            continue
        if isinstance(allowed, (list, tuple, set, frozenset)):
            if current_value not in allowed:
                return False
        else:
            if current_value != allowed:
                return False
    return True


def filter_summaries_by_metadata(
    summaries: Sequence[PosteriorSummary],
    criteria: dict | None,
) -> List[PosteriorSummary]:
    if not criteria:
        return list(summaries)
    return [s for s in summaries if metadata_matches(s.metadata, criteria)]


def split_summaries_by_metadata(
    summaries: Sequence[PosteriorSummary],
    train_criteria: dict | None,
    test_criteria: dict | None = None,
) -> Tuple[List[PosteriorSummary], List[PosteriorSummary]]:
    train = filter_summaries_by_metadata(summaries, train_criteria)
    train_paths = {s.path for s in train}
    remainder = [s for s in summaries if s.path not in train_paths]

    if test_criteria is None:
        test = remainder
    else:
        test = filter_summaries_by_metadata(remainder, test_criteria)

    return train, test


def concatenate_summaries(
    summaries: Sequence[PosteriorSummary],
) -> dict:
    if not summaries:
        raise ValueError("At least one posterior summary is required.")

    reference_orientation: str | None = None
    reference_coil: int | None = None
    reference_metadata: dict | None = None
    collected_metadata: List[dict | None] = []

    fixed_true: List[np.ndarray] = []
    fixed_hat: List[np.ndarray] = []
    fixed_covs: List[np.ndarray] = []

    eeg_true: List[np.ndarray] = []
    eeg_hat: List[np.ndarray] = []
    eeg_covs: List[np.ndarray] = []

    meg_true: List[np.ndarray] = []
    meg_hat: List[np.ndarray] = []
    meg_covs: List[np.ndarray] = []
    meg_qbasis: List[np.ndarray] = []

    for summary in summaries:
        payload = load_posterior_summary(summary)
        metadata = payload.get("metadata") or {}
        orientation = metadata.get("orientation_type")
        orientation_norm = _normalize_orientation(orientation)
        coil = metadata.get("coil_type")
        _validate_coil_type(coil)
        if orientation_norm == "free" and coil is None:
            raise ValueError(
                "Free-orientation summaries must record a supported coil_type."
            )

        if reference_orientation is None:
            reference_metadata = metadata
            reference_orientation = orientation_norm
            reference_coil = coil
        else:
            if orientation_norm != reference_orientation:
                raise ValueError(
                    "Cannot mix orientation_type values while concatenating summaries."
                )
            if reference_coil != coil:
                raise ValueError(
                    "Cannot mix coil_type values while concatenating summaries."
                )
        collected_metadata.append(metadata)

        x_true = np.asarray(payload["x_true"], dtype=float)
        x_hat = np.asarray(payload["x_hat"], dtype=float)
        posterior_cov = np.asarray(payload["posterior_cov"], dtype=float)

        if orientation_norm == "fixed":
            fixed_true.append(x_true)
            fixed_hat.append(x_hat)
            fixed_covs.append(posterior_cov)
            continue

        if orientation_norm == "free" and coil in EEG_COIL_TYPES:
            eeg_true.append(x_true)
            eeg_hat.append(x_hat)
            eeg_covs.append(posterior_cov)
            continue

        if orientation_norm == "free" and coil in MEG_COIL_TYPES:
            q_basis = _maybe_q_basis_array(payload.get("Q_basis"))
            if q_basis is None:
                raise ValueError(
                    "Free-orientation MEG summaries must include Q_basis."
                )
            meg_true.append(x_true)
            meg_hat.append(x_hat)
            meg_covs.append(posterior_cov)
            meg_qbasis.append(q_basis)
            continue

        raise ValueError("Unhandled orientation case while concatenating summaries.")

    if reference_orientation == "fixed":
        x_true = np.concatenate(fixed_true, axis=0)
        x_hat = np.concatenate(fixed_hat, axis=0)
        posterior_cov = (
            block_diag(*fixed_covs) if len(fixed_covs) > 1 else fixed_covs[0]
        )
        return {
            "orientation_type": reference_orientation,
            "coil_type": reference_coil,
            "sensor_kind": (reference_metadata or {}).get("sensor_kind"),
            "x_true": x_true,
            "x_hat": x_hat,
            "posterior_cov": posterior_cov,
            "metadata": collected_metadata,
            "n_sources": x_true.shape[0],
            "n_times": x_true.shape[-1],
        }

    if reference_orientation == "free" and reference_coil in EEG_COIL_TYPES:
        x_true = np.concatenate(eeg_true, axis=0)
        x_hat = np.concatenate(eeg_hat, axis=0)
        posterior_cov = (
            block_diag(*eeg_covs) if len(eeg_covs) > 1 else eeg_covs[0]
        )
        return {
            "orientation_type": reference_orientation,
            "coil_type": reference_coil,
            "sensor_kind": (reference_metadata or {}).get("sensor_kind"),
            "x_true": x_true,
            "x_hat": x_hat,
            "posterior_cov": posterior_cov,
            "metadata": collected_metadata,
            "n_sources": x_true.shape[0],
            "n_times": x_true.shape[-1],
        }

    if reference_orientation == "free" and reference_coil in MEG_COIL_TYPES:
        x_true_cat = np.concatenate(meg_true, axis=0)
        x_hat_cat = np.concatenate(meg_hat, axis=0)
        posterior_cov = (
            block_diag(*meg_covs)
            if len(meg_covs) > 1
            else meg_covs[0]
        )
        q_basis_cat = np.concatenate(meg_qbasis, axis=0)
        return {
            "orientation_type": reference_orientation,
            "coil_type": reference_coil,
            "sensor_kind": (reference_metadata or {}).get("sensor_kind"),
            "x_true": x_true_cat,
            "x_hat": x_hat_cat,
            "posterior_cov": posterior_cov,
            "Q_basis": q_basis_cat,
            "metadata": collected_metadata,
            "n_sources": x_true_cat.shape[0],
            "n_times": x_true_cat.shape[-1],
        }

    raise RuntimeError("Unhandled orientation case while concatenating summaries.")
