"""Submit a self-contained EGGROLL post-training stage to Hugging Face Jobs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token

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

mkdir -p /work/output
cd /work
tar -xzf "/src/$SRC_TARBALL"
cd /work/microduck_rl
uv sync --frozen --extra eggroll --no-editable --no-progress

UPLOADER_PID=""
if [ "$MODE" = "train" ]; then
  (
    while true; do
      sleep 300
      if [ -d runs/eggroll-posttrain ]; then
        uv run --no-sync hf upload "$OUTPUT_REPO" runs/eggroll-posttrain runs/eggroll-posttrain \
          --repo-type model --private --token "$HF_TOKEN" || true
      fi
    done
  ) &
  UPLOADER_PID=$!
fi

set +e
eval "$POSTTRAIN_COMMAND"
RUN_RC=$?
set -e

if [ -n "$UPLOADER_PID" ]; then
  kill "$UPLOADER_PID" 2>/dev/null || true
fi
COLLECT_RC=0
eval "$COLLECT_COMMAND" || COLLECT_RC=$?
uv run --no-sync hf upload "$OUTPUT_REPO" /work/output . \
  --repo-type model --private --token "$HF_TOKEN" || true
if [ "$RUN_RC" -ne 0 ]; then
  exit "$RUN_RC"
fi
exit "$COLLECT_RC"
"""


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repo).decode().strip()


def _require_clean_repo(repo: Path) -> None:
    dirty = _git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "Refusing to upload an uncommitted source snapshot. Commit or stash "
            f"these paths first:\n{dirty}"
        )


def _tracked_files(repo: Path) -> list[str]:
    tracked = _git_output(repo, "ls-files")
    return tracked.splitlines() if tracked else []


def _build_bundle(
    *,
    repo: Path,
    output: Path,
    policy: Path,
    calibration: Path | None,
) -> None:
    _require_clean_repo(repo)
    manifest = {
        "schema_version": 1,
        "source_commit": _git_output(repo, "rev-parse", "HEAD"),
        "source_branch": _git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "source_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "calibration_sha256": (
            hashlib.sha256(calibration.read_bytes()).hexdigest()
            if calibration is not None
            else None
        ),
    }
    with tarfile.open(output, "w:gz") as archive:
        for relative in _tracked_files(repo):
            path = repo / relative
            if path.is_file() and ".DS_Store" not in path.parts:
                archive.add(path, arcname=str(Path("microduck_rl") / relative))
        archive.add(
            policy,
            arcname="microduck_rl/.artifacts/input/source_policy.onnx",
        )
        if calibration is not None:
            archive.add(
                calibration,
                arcname="microduck_rl/.artifacts/input/calibration.json",
            )
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo(
            "microduck_rl/.artifacts/input/source_manifest.json"
        )
        info.size = len(manifest_bytes)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "smoke", "train", "evaluate"))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--config",
        default="configs/eggroll_posttrain/alpha_stand_output_layer_v1.toml",
    )
    parser.add_argument("--profile")
    parser.add_argument("--episodes-per-pose", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--namespace")
    parser.add_argument("--output-repo")
    parser.add_argument("--flavor", default="a10g-large")
    parser.add_argument("--timeout", default="6h")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--detach", action="store_true")
    return parser


def _commands(args: argparse.Namespace) -> tuple[str, str]:
    script = "uv run --no-sync python scripts/eggroll_posttrain.py"
    policy = ".artifacts/input/source_policy.onnx"
    if args.mode == "calibrate":
        command = [
            *shlex.split(script),
            "calibrate-shift",
            "--policy",
            policy,
            "--device",
            "cuda:0",
            "--seed",
            str(args.seed),
            "--episodes-per-pose",
            str(args.episodes_per_pose),
            "--output-dir",
            "/work/output/calibration",
        ]
        return shlex.join(command), "true"
    if args.mode in {"smoke", "train"}:
        if args.calibration is None:
            raise ValueError(f"{args.mode} requires --calibration")
        config = (
            "configs/eggroll_posttrain/smoke.toml"
            if args.mode == "smoke"
            else args.config
        )
        command = [
            *shlex.split(script),
            "train",
            "--policy",
            policy,
            "--calibration",
            ".artifacts/input/calibration.json",
            "--config",
            config,
        ]
        checkpoint_fallback = (
            'if [ -z "$CHECKPOINT" ]; then '
            "CHECKPOINT=$(find runs/eggroll-posttrain -name last.pkl -print -quit); "
            "fi; "
            if args.mode == "smoke"
            else ""
        )
        collect = (
            "mkdir -p /work/output/run && "
            "cp -a runs/eggroll-posttrain/. /work/output/run/ && "
            "CHECKPOINT=$(find runs/eggroll-posttrain -name best.pkl -print -quit); "
            + checkpoint_fallback
            + 'if [ -n "$CHECKPOINT" ]; then '
            "uv run --no-sync python scripts/eggroll_posttrain.py export "
            '--checkpoint "$CHECKPOINT" '
            "--output /work/output/adapted_policy.onnx; "
            "fi"
        )
        return shlex.join(command), collect
    if args.profile is None:
        raise ValueError("evaluate requires --profile")
    command = [
        *shlex.split(script),
        "evaluate",
        "--policy",
        policy,
        "--profile",
        args.profile,
        "--device",
        "cuda:0",
        "--seed",
        str(args.seed),
        "--episodes-per-pose",
        str(args.episodes_per_pose),
        "--output-dir",
        "/work/output/evaluation",
    ]
    if args.video:
        command.append("--video")
    return shlex.join(command), "true"


def main() -> int:
    args = _parser().parse_args()
    policy = args.policy.resolve()
    calibration = args.calibration.resolve() if args.calibration else None
    if not policy.is_file():
        raise FileNotFoundError(policy)
    if calibration is not None and not calibration.is_file():
        raise FileNotFoundError(calibration)
    command, collect = _commands(args)

    token = get_token()
    if not token:
        print("error: no cached Hugging Face token", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    namespace = _pick_namespace(api, args.namespace)
    repo = _repo_root()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    source_repo = f"{namespace}/eggroll-posttrain-source-{stamp}"
    output_repo = (
        args.output_repo or f"{namespace}/eggroll-posttrain-{args.mode}-{stamp}"
    )
    if "/" not in output_repo:
        output_repo = f"{namespace}/{output_repo}"

    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary) / f"source-{stamp}.tar.gz"
        _build_bundle(
            repo=repo,
            output=bundle,
            policy=policy,
            calibration=calibration,
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
                "MODE": "train" if args.mode in {"smoke", "train"} else args.mode,
                "POSTTRAIN_COMMAND": command,
                "COLLECT_COMMAND": collect,
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
