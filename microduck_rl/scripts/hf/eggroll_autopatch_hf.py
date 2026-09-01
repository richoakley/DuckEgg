"""Submit an EGGROLL Autopatch smoke or frozen campaign to HF Jobs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, Volume, get_token

from mjlab_microduck.autopatch.contracts import PatchCampaign
from mjlab_microduck.hf_jobs import (
    DEFAULT_IMAGE,
    _await_scheduling,
    _pick_namespace,
    _repo_root,
)

BOOTSTRAP = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -qq -y --no-install-recommends \
  git curl ca-certificates build-essential ffmpeg libegl1 libgl1 libgles2 >/dev/null
curl -LsSf https://astral.sh/uv/0.11.30/install.sh | sh >/dev/null
export PATH="/root/.local/bin:$PATH"
export UV_LINK_MODE=copy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

mkdir -p /work
cd /work
tar -xzf "/src/$SRC_TARBALL"
cd /work/microduck_rl
uv sync --frozen --extra eggroll --no-editable --no-progress

mkdir -p /work/run /work/output
(
  while true; do
    sleep 300
    uv run --no-sync hf upload "$OUTPUT_REPO" /work/run run \
      --repo-type model --private --token "$HF_TOKEN" || true
  done
) &
UPLOADER_PID=$!

set +e
uv run --no-sync eggroll-autopatch train-walking-campaign \
  --campaign .artifacts/input/campaign.json \
  --runtime-repo /work/microduck \
  --output-dir /work/run \
  --device cuda:0
RUN_RC=$?
set -e

kill "$UPLOADER_PID" 2>/dev/null || true
cp .artifacts/input/source_manifest.json /work/output/source_manifest.json
cp .artifacts/input/campaign.json /work/output/campaign.json

if [ "$RUN_RC" -eq 0 ]; then
  CHECKPOINT_ARGS=()
  while IFS= read -r checkpoint; do
    CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
  done < <(find /work/run/candidates -name '*.npz' -type f 2>/dev/null | sort)
  if [ "${#CHECKPOINT_ARGS[@]}" -eq 0 ]; then
    echo "no nominal-retained candidate checkpoint was produced" >&2
    RUN_RC=1
  else
    uv run --no-sync eggroll-autopatch select-export \
      --campaign .artifacts/input/campaign.json \
      --runtime-repo /work/microduck \
      "${CHECKPOINT_ARGS[@]}" \
      --output-policy /work/output/adapted_policy.onnx \
      --output-record /work/output/selection.json || RUN_RC=$?
  fi
fi

uv run --no-sync hf upload "$OUTPUT_REPO" /work/run run \
  --repo-type model --private --token "$HF_TOKEN" || true
uv run --no-sync hf upload "$OUTPUT_REPO" /work/output . \
  --repo-type model --private --token "$HF_TOKEN" || true
exit "$RUN_RC"
"""


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repo).decode().strip()


def _require_clean_repo(repo: Path) -> None:
    dirty = _git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "Refusing to upload an uncommitted source snapshot. Use a clean worktree:\n"
            f"{dirty}"
        )


def _tracked_files(repo: Path) -> list[str]:
    tracked = _git_output(repo, "ls-files")
    return tracked.splitlines() if tracked else []


def _campaign_document(path: Path, *, mode: str) -> tuple[dict[str, Any], str]:
    base = PatchCampaign.from_json(path.read_text())
    document = base.canonical_dict()
    if mode == "smoke":
        document["campaign_id"] = f"{base.campaign_id}-cuda-smoke"
        document["optimizer"] = {
            **document["optimizer"],
            "population": 16,
            "generations": 1,
        }
    execution = PatchCampaign.from_dict(document)
    return execution.canonical_dict(), execution.sha256


def _build_bundle(
    *,
    repo: Path,
    output: Path,
    policy: Path,
    campaign_path: Path,
    mode: str,
) -> dict[str, Any]:
    _require_clean_repo(repo)
    campaign, campaign_sha256 = _campaign_document(campaign_path, mode=mode)
    base_campaign = PatchCampaign.from_json(campaign_path.read_text())
    manifest = {
        "schema": "eggroll-autopatch-hf-source-v1",
        "mode": mode,
        "evidence_role": "non-evidence CUDA smoke" if mode == "smoke" else "frozen campaign",
        "source_commit": _git_output(repo, "rev-parse", "HEAD"),
        "source_branch": _git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "source_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "base_campaign_sha256": base_campaign.sha256,
        "execution_campaign_sha256": campaign_sha256,
    }
    with tarfile.open(output, "w:gz") as archive:
        for relative in _tracked_files(repo):
            path = repo / relative
            if path.is_file() and ".DS_Store" not in path.parts:
                archive.add(path, arcname=str(Path("microduck_rl") / relative))
        archive.add(
            policy,
            arcname="microduck/example_policies/alpha_walking.onnx",
        )
        for name, payload in (
            ("campaign.json", campaign),
            ("source_manifest.json", manifest),
        ):
            raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo(
                f"microduck_rl/.artifacts/input/{name}"
            )
            info.size = len(raw)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(raw))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--output-repo")
    parser.add_argument("--flavor", default="a10g-large")
    parser.add_argument("--timeout", default="8h")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--detach", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo = _repo_root()
    policy = args.policy.resolve()
    campaign = args.campaign.resolve()
    for path in (policy, campaign):
        if not path.is_file():
            raise FileNotFoundError(path)
    token = get_token()
    if not token:
        print("error: no cached Hugging Face token", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    namespace = _pick_namespace(api, args.namespace)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    source_repo = f"{namespace}/eggroll-autopatch-source-{stamp}"
    output_repo = args.output_repo or f"eggroll-walking-wedge-{args.mode}-{stamp}"
    if "/" not in output_repo:
        output_repo = f"{namespace}/{output_repo}"

    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary) / f"source-{stamp}.tar.gz"
        manifest = _build_bundle(
            repo=repo,
            output=bundle,
            policy=policy,
            campaign_path=campaign,
            mode=args.mode,
        )
        api.create_repo(source_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(bundle),
            path_in_repo=bundle.name,
            repo_id=source_repo,
            repo_type="dataset",
        )
        api.create_repo(output_repo, repo_type="model", private=True, exist_ok=True)
        job = api.run_job(
            image=args.image,
            command=["bash", "-c", BOOTSTRAP],
            env={
                "SRC_TARBALL": bundle.name,
                "OUTPUT_REPO": output_repo,
            },
            secrets={"HF_TOKEN": token},
            flavor=args.flavor,
            timeout=args.timeout,
            volumes=[
                Volume(
                    type="dataset",
                    source=source_repo,
                    mount_path="/src",
                    read_only=True,
                )
            ],
            namespace=namespace,
        )
    print(f"[job] {job.id}")
    print(f"[artifacts] https://huggingface.co/{output_repo}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.detach:
        return 0
    stage, message = _await_scheduling(api, job.id, namespace)
    if stage == "ERROR":
        print(f"error: {message}", file=sys.stderr)
        return 1
    while True:
        try:
            for line in api.fetch_job_logs(
                job_id=job.id, namespace=namespace, follow=True
            ):
                print(line)
        except Exception as error:  # noqa: BLE001
            print(f"[job] log stream dropped ({error}); re-attaching")
        status = api.inspect_job(job_id=job.id, namespace=namespace).status
        if status.stage == "COMPLETED":
            return 0
        if status.stage in {"ERROR", "DELETED", "CANCELED"}:
            print(f"error: job {status.stage}: {status.message}", file=sys.stderr)
            return 1
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
