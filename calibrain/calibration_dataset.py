from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import numpy as np


@dataclass
class PosteriorSummary:
    """Container describing a saved posterior summary file and metadata."""

    path: Path
    metadata: dict | None = None


def load_posterior_summary(summary: PosteriorSummary) -> dict:
    """
    Load a single posterior summary file that was produced by the benchmark loop.
    Supports the new HDF5 format as well as legacy ``.npz`` files.
    """
    suffix = summary.path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(summary.path, "r") as handle:
            payload = {
                "x_true": handle["x_true"][:],
                "x_hat": handle["x_hat"][:],
                "posterior_std": handle["posterior_std"][:],
            }
        return payload

    if suffix == ".npz":
        with np.load(summary.path) as data:
            payload = {
                "x_true": data["x_true"],
                "x_hat": data["x_hat"],
                "posterior_std": data["posterior_std"],
            }
        return payload

    raise ValueError(f"Unsupported posterior summary format: {summary.path}")


def concatenate_summaries(
    summaries: Sequence[PosteriorSummary],
) -> dict:
    """
    Concatenate multiple posterior summaries along the source dimension so they
    can be fed into the refactored ``UncertaintyCalibrator`` as aggregated
    train/test datasets.
    """
    if not summaries:
        raise ValueError("At least one posterior summary is required.")

    x_true_blocks: List[np.ndarray] = []
    x_hat_blocks: List[np.ndarray] = []
    posterior_std_blocks: List[np.ndarray] = []
    collected_metadata: List[dict | None] = []

    for summary in summaries:
        payload = load_posterior_summary(summary)
        x_true_blocks.append(payload["x_true"])
        x_hat_blocks.append(payload["x_hat"])
        posterior_std_blocks.append(payload["posterior_std"])
        collected_metadata.append(summary.metadata)

    x_true = np.concatenate(x_true_blocks, axis=0)
    x_hat = np.concatenate(x_hat_blocks, axis=0)
    posterior_std = np.concatenate(posterior_std_blocks, axis=0)

    return {
        "x_true": x_true,
        "x_hat": x_hat,
        "posterior_std": posterior_std,
        "metadata": collected_metadata,
        "n_sources": x_true.shape[0],
        "n_times": x_true.shape[-1],
    }


def discover_posterior_summaries(root: Path) -> Iterable[PosteriorSummary]:
    """
    Yield ``PosteriorSummary`` objects for every stored summary file (HDF5 or legacy NPZ)
    found under ``root``.
    """
    # Prefer the new HDF5 outputs
    for path in root.rglob("posterior_summary*.h5"):
        metadata = _load_h5_metadata(path)
        yield PosteriorSummary(path=path, metadata=metadata)

    # Fall back to legacy NPZ/JSON pairs if they exist
    for path in root.rglob("posterior_summary*.npz"):
        metadata = _load_legacy_metadata(path)
        yield PosteriorSummary(path=path, metadata=metadata)


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


def metadata_matches(metadata: dict | None, criteria: dict | None) -> bool:
    """
    Return True if ``metadata`` satisfies the ``criteria`` mapping.

    ``criteria`` is a dict mapping metadata keys to allowed values. Each value
    can be a scalar, a list/tuple/set of allowed scalars, or a callable that
    receives the metadata value and returns a boolean.
    """
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
    """Return summaries matching the provided metadata criteria."""
    if not criteria:
        return list(summaries)
    return [s for s in summaries if metadata_matches(s.metadata, criteria)]


def split_summaries_by_metadata(
    summaries: Sequence[PosteriorSummary],
    train_criteria: dict | None,
    test_criteria: dict | None = None,
) -> Tuple[List[PosteriorSummary], List[PosteriorSummary]]:
    """
    Split summaries into train/test lists based on metadata criteria.

    - ``train_criteria`` selects summaries for the training split.
    - ``test_criteria`` (optional) selects a subset of the remaining summaries.
      When omitted, every summary not in the training list goes to the test list.
    """
    train = filter_summaries_by_metadata(summaries, train_criteria)
    train_paths = {s.path for s in train}
    remainder = [s for s in summaries if s.path not in train_paths]

    if test_criteria is None:
        test = remainder
    else:
        test = filter_summaries_by_metadata(remainder, test_criteria)

    return train, test
