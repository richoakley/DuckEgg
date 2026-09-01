"""Checkpoint and structured-log helpers for EGGROLL training."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

CHECKPOINT_FORMAT = "mjlab-microduck-eggroll-posttrain"
CHECKPOINT_VERSION = 1


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save a trusted local training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        **payload,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        pickle.dump(envelope, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint created by this trainer.

    Pickle is used because Optax optimizer states are nested named tuples.  As
    with any pickle file, callers must only load checkpoints they trust.
    """
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid checkpoint payload in {path}")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a Microduck EGGROLL checkpoint")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload.get('version')!r} in {path}"
        )
    return payload


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True))
        stream.write("\n")
