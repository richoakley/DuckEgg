from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "hf" / "resume_eggroll_autopatch_hf.py"
SPEC = importlib.util.spec_from_file_location("resume_eggroll_autopatch_hf", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _bootstrap() -> str:
    return """mkdir -p /work/run /work/output
(
  while true; do
    sleep 300
  done
) &
uv run --no-sync eggroll-autopatch train-walking-campaign \\
  --campaign .artifacts/input/campaign.json \\
  --output-dir /work/run \\
  --device cuda:0
"""


def test_resume_bootstrap_hydrates_before_uploader_and_uses_strict_resume() -> None:
    resumed = MODULE._resume_bootstrap(_bootstrap())

    assert resumed.index("hf download") < resumed.index("while true")
    assert "test -f /work/restore/run/last.pkl" in resumed
    assert "cp -a /work/restore/run/. /work/run/" in resumed
    assert "--resume /work/run/last.pkl" in resumed


def test_resume_bootstrap_rejects_drift_or_double_resume() -> None:
    with pytest.raises(ValueError, match="unexpected run-directory"):
        MODULE._resume_bootstrap(_bootstrap().replace("/work/output", "/work/out"))
    with pytest.raises(ValueError, match="already a resume"):
        MODULE._resume_bootstrap(
            _bootstrap().replace(
                "--device cuda:0", "--resume /work/run/last.pkl --device cuda:0"
            )
        )
