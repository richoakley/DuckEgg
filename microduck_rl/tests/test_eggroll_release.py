"""Release-manifest and deterministic hero-selection regression tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mjlab_microduck.eggroll.hero import (
    select_nominal_pair,
    select_shifted_pairs,
)
from mjlab_microduck.eggroll.release import sha256_file, verify_release_manifest

_ROOT = Path(__file__).parents[1]
_RELEASE = (
    _ROOT / "policies/eggroll_posttraining/alpha_stand_lag16_v1/manifest.json"
)


def test_canonical_release_is_self_verifying() -> None:
    result = verify_release_manifest(_RELEASE)
    assert result["release_passed"] is True
    assert result["output_layer_only"] is True
    assert result["episode_banks_pairwise_disjoint"] is True
    assert result["maximum_runtime_error"] < 1.0e-5


def test_release_refuses_modified_evidence(tmp_path: Path) -> None:
    copied = tmp_path / "release"
    shutil.copytree(_RELEASE.parent, copied)
    manifest = json.loads((copied / "manifest.json").read_text())
    evidence = copied / manifest["evaluation"]["source_shifted"]["path"]
    evidence.write_text(evidence.read_text().replace("0.53125", "0.53126", 1))
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        verify_release_manifest(copied / "manifest.json")


def test_hero_selection_is_predeclared_and_paired() -> None:
    manifest = json.loads(_RELEASE.read_text())
    summaries = {
        role: json.loads((_RELEASE.parent / record["path"]).read_text())
        for role, record in manifest["evaluation"].items()
    }
    shifted = select_shifted_pairs(
        summaries["source_shifted"], summaries["adapted_shifted"]
    )
    assert [selection.pose for selection in shifted] == [
        "standing",
        "sitting",
        "face-down",
        "face-up",
    ]
    for selection in shifted:
        index = selection.episode_index
        assert summaries["source_shifted"]["episodes"]["terminal_success"][index] is False
        assert summaries["adapted_shifted"]["episodes"]["terminal_success"][index] is True
    nominal = select_nominal_pair(
        summaries["source_nominal"], summaries["adapted_nominal"]
    )
    assert nominal.pose == "face-down"


def test_product_artifact_index_matches_committed_bytes() -> None:
    index = json.loads(
        (_ROOT / "docs/experiments/eggroll_posttraining_2026-08.artifacts.json").read_text()
    )
    release = _RELEASE.parent
    product = index["product_release"]
    expected = {
        release / "manifest.json": product["manifest_sha256"],
        release / "runtime_verification.json": product[
            "adapted_runtime_verification_sha256"
        ],
        release / "microduck_updater/model-stand-1.0.0.tar.zst": product[
            "adapted_model_bundle_sha256"
        ],
        release / "rollback/source_runtime_verification.json": product[
            "source_runtime_verification_sha256"
        ],
        release / "rollback/microduck_updater/model-stand-0.9.0.tar.zst": product[
            "source_model_bundle_sha256"
        ],
        _ROOT / "docs/assets/eggroll_posttraining/eggroll_posttraining_hero_v1.mp4": product[
            "hero_video_sha256"
        ],
        _ROOT
        / "docs/assets/eggroll_posttraining/eggroll_posttraining_hero_v1.mp4.json": product[
            "hero_sidecar_sha256"
        ],
    }
    for path, digest in expected.items():
        assert sha256_file(path) == digest, path
