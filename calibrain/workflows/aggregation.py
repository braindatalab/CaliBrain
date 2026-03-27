from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from calibrain.calibration_dataset import (
    concatenate_summaries,
    discover_posterior_summaries,
    split_summaries_by_metadata,
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

def _derive_split_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")


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
    np.savez_compressed(
        output_path,
        x_true=dataset["x_true"],
        x_hat=dataset["x_hat"],
        posterior_cov=dataset["posterior_cov"],
    )
    meta_path = output_path.with_suffix(".json")
    meta_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_root": str(summaries_root),
        "summary_count": len(summaries),
        "n_sources": int(dataset["n_sources"]),
        "n_times": int(dataset["n_times"]),
        "split": split_name,
        "criteria": _serialize_filter(criteria),
        "summaries": [str(summary.path) for summary in summaries],
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    print(f"[aggregation] Wrote {split_name} split with {len(summaries)} summaries -> {output_path}")


def aggregate_posteriors(config: Union[str, Path, Dict[str, Any]]) -> None:
    if isinstance(config, (str, Path)):
        config = load_python_config(config)

    summaries_root = Path(config["summaries_root"])
    if not summaries_root.exists():
        raise FileNotFoundError(f"Summary root does not exist: {summaries_root}")

    output_path = Path(config["output_path"])
    train_filter = config.get("train_filter")
    test_filter = config.get("test_filter")
    train_output = (
        Path(config["train_output"])
        if config.get("train_output")
        else _derive_split_path(output_path, "_train")
    )
    test_output = (
        Path(config["test_output"])
        if config.get("test_output")
        else _derive_split_path(output_path, "_test")
    )

    summaries = list(discover_posterior_summaries(summaries_root))
    if not summaries:
        raise FileNotFoundError(
            f"No posterior summary files found under {summaries_root}"
        )

    if train_filter or test_filter:
        train_summaries, test_summaries = split_summaries_by_metadata(
            summaries,
            train_filter,
            test_filter,
        )
        if not train_summaries:
            raise ValueError("No summaries matched the training metadata filters.")
        if not test_summaries:
            raise ValueError("No summaries available for the test split.")

        train_dataset = concatenate_summaries(train_summaries)
        _write_dataset(
            dataset=train_dataset,
            output_path=train_output,
            summaries=train_summaries,
            summaries_root=summaries_root,
            split_name="train",
            criteria=train_filter,
        )

        test_dataset = concatenate_summaries(test_summaries)
        _write_dataset(
            dataset=test_dataset,
            output_path=test_output,
            summaries=test_summaries,
            summaries_root=summaries_root,
            split_name="test",
            criteria=test_filter or {"rule": "remainder"},
        )
        return

    dataset = concatenate_summaries(summaries)
    _write_dataset(
        dataset=dataset,
        output_path=output_path,
        summaries=summaries,
        summaries_root=summaries_root,
        split_name="all",
        criteria=None,
    )


if __name__ == "__main__":
    aggregate_posteriors(DEFAULT_CONFIG_PATH)
