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

from mjlab_microduck.autopatch.contracts import PatchCampaign, ReleaseScope
from mjlab_microduck.autopatch.qualification import QualificationPlan
from mjlab_microduck.autopatch.qualification_command import CommandQualificationSpec
from mjlab_microduck.hf_jobs import (
    DEFAULT_IMAGE,
    _await_scheduling,
    _pick_namespace,
    _repo_root,
)

PRODUCTION_BUCKET = "richoakley/jobs-artifacts"
PRODUCTION_RUNTIME_VOLUME_PATH = "microduck-runtime-evidence-20260901-v1-837c0f59"
REFERENCE_POLICIES_VOLUME_PATH = "eggroll-wedge-confirmation-inputs-65b33979"
PRODUCTION_RUNTIME_IDENTITIES = {
    "source_commit": "590b986bd8c0d50ae02cb3ea2f59c463b6828168",
    "working_tree_diff_sha256": (
        "8ab65a1acda82f20d6ffefc857363ccd53686ba00cc8704ccc97362ff29170ab"
    ),
    "untracked_path_list_sha256": (
        "4f1a1d2ee651b23b50757d4132847afd0f323eb6b797cf5ee09cfff4296b6c8c"
    ),
    "robotd_main_sha256": (
        "e3e225002f189bda06ca9d38c16d09e57b1d2fe754d42d0cef1026533c06a33d"
    ),
    "sim_transport_sha256": (
        "3bbdb52f03cf103123415428a0478229be49f8e8493c49124c3c413193365619"
    ),
    "cargo_lock_sha256": (
        "308998851339511fc1e8a5ab1fa11bcc02741b8604372c7d95b24fb2888237b1"
    ),
}

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

if [ -d /inputs/microduck ]; then
  cp -a /inputs/microduck/. /work/microduck/
  mkdir -p /work/microduck/example_policies
  cp -a /inputs/reference_policies/example_policies/. /work/microduck/example_policies/
  test "$(sha256sum /work/microduck/robotd/src/main.rs | cut -d' ' -f1)" = \
    "e3e225002f189bda06ca9d38c16d09e57b1d2fe754d42d0cef1026533c06a33d"
  test "$(sha256sum /work/microduck/duck-control/src/sim.rs | cut -d' ' -f1)" = \
    "3bbdb52f03cf103123415428a0478229be49f8e8493c49124c3c413193365619"
  test "$(sha256sum /work/microduck/Cargo.lock | cut -d' ' -f1)" = \
    "308998851339511fc1e8a5ab1fa11bcc02741b8604372c7d95b24fb2888237b1"
  test "$(sha256sum /work/microduck/example_policies/alpha_walking.onnx | cut -d' ' -f1)" = \
    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"

  cp scripts/runtime_support/profile_scoped_model_activation.rs \
    /work/microduck/updater/src/profile_scoped_model_activation.rs
  cp scripts/runtime_support/profile_scoped_policy_lab.rs \
    /work/microduck/updater/examples/profile_scoped_policy_lab.rs
  grep -q '^pub mod profile_scoped_model_activation;' /work/microduck/updater/src/lib.rs || \
    sed -i '/^pub mod preflight;/a pub mod profile_scoped_model_activation;' \
      /work/microduck/updater/src/lib.rs

  apt-get install -qq -y --no-install-recommends pkg-config >/dev/null
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
  export PATH="/root/.cargo/bin:$PATH"
  cd /work/microduck
  cargo build --release --locked -p robotd
  cargo build --release --locked -p updater \
    --example policy_patch_lab --example profile_scoped_policy_lab
  cd /work/microduck_rl
  ORT_DYLIB_PATH="$(uv run --no-sync python -c \
    'from pathlib import Path; import onnxruntime as ort; root=Path(ort.__file__).parent; print(sorted(root.rglob("libonnxruntime.so*"))[0])')"
  test -f "$ORT_DYLIB_PATH"
  export EGGROLL_PRODUCTION_ROBOTD=/work/microduck/target/release/robotd
  export EGGROLL_PRODUCTION_ORT_DYLIB="$ORT_DYLIB_PATH"
  export EGGROLL_PROFILE_ROUTING_HARNESS=\
/work/microduck/target/release/examples/profile_scoped_policy_lab
  export EGGROLL_SIGNED_UPDATER_HARNESS=\
/work/microduck/target/release/examples/policy_patch_lab
fi

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
RELEASE_SCOPE_ARGS=()
if [ -f .artifacts/input/release_scope.json ]; then
  RELEASE_SCOPE_ARGS+=(--release-scope .artifacts/input/release_scope.json)
fi
QUALIFICATION_ARGS=()
if [ -f .artifacts/input/qualification_plan.json ]; then
  QUALIFICATION_ARGS+=(--qualification-plan .artifacts/input/qualification_plan.json)
  QUALIFICATION_ARGS+=(--qualification-command-spec .artifacts/input/qualification_command_spec.json)
fi
PROFILE_ARGS=()
if [ "$RUN_MODE" = "smoke" ]; then
  PROFILE_ARGS+=(--profile-generation 0 --profile-generation 1)
fi
uv run --no-sync eggroll-autopatch train-walking-campaign \
  --campaign .artifacts/input/campaign.json \
  "${RELEASE_SCOPE_ARGS[@]}" \
  "${QUALIFICATION_ARGS[@]}" \
  "${PROFILE_ARGS[@]}" \
  --runtime-repo /work/microduck \
  --output-dir /work/run \
  --device cuda:0
RUN_RC=$?
set -e

kill "$UPLOADER_PID" 2>/dev/null || true
cp .artifacts/input/source_manifest.json /work/output/source_manifest.json
cp .artifacts/input/campaign.json /work/output/campaign.json
if [ -f .artifacts/input/release_scope.json ]; then
  cp .artifacts/input/release_scope.json /work/output/release_scope.json
fi
if [ -f .artifacts/input/qualification_plan.json ]; then
  cp .artifacts/input/qualification_plan.json /work/output/qualification_plan.json
  cp .artifacts/input/qualification_command_spec.json /work/output/qualification_command_spec.json
fi

if [ "$RUN_RC" -eq 0 ] && [ "$RUN_MODE" = "smoke" ]; then
  uv run --no-sync eggroll-autopatch verify-cuda-smoke-run \
    --run-dir /work/run \
    --campaign .artifacts/input/campaign.json \
    "${RELEASE_SCOPE_ARGS[@]}" \
    --source-manifest .artifacts/input/source_manifest.json \
    --expected-candidate-evaluations 16 \
    --expected-optimization-world-rollouts 68 \
    --expected-optimization-requested-steps 17000 \
    --expected-total-world-rollouts 196 \
    --expected-total-requested-steps 49000 \
    --output /work/output/smoke_validation.json || RUN_RC=$?
elif [ "$RUN_RC" -eq 0 ]; then
  CHECKPOINT_ARGS=()
  if [ -f .artifacts/input/qualification_plan.json ]; then
    ELIGIBLE_CHECKPOINT="$(uv run --no-sync python -c \
      'import json; from pathlib import Path; q=json.loads(Path("/work/run/qualification.json").read_text()); g=q.get("stop_generation"); assert isinstance(g, int) and g > 0, "campaign ended without complete release qualification"; p=Path(f"/work/run/qualification_candidates/generation-{g:06d}.npz"); assert p.is_file(), f"eligible checkpoint is absent: {p}"; print(p)')" || RUN_RC=$?
    if [ "$RUN_RC" -eq 0 ]; then
      CHECKPOINT_ARGS+=(--checkpoint "$ELIGIBLE_CHECKPOINT")
    fi
  else
    while IFS= read -r checkpoint; do
      CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
    done < <(find /work/run/candidates -name '*.npz' -type f 2>/dev/null | sort)
  fi
  if [ "${#CHECKPOINT_ARGS[@]}" -eq 0 ]; then
    echo "no release-eligible candidate checkpoint was produced" >&2
    RUN_RC=1
  else
    uv run --no-sync eggroll-autopatch select-export \
      --campaign .artifacts/input/campaign.json \
      --runtime-repo /work/microduck \
      "${CHECKPOINT_ARGS[@]}" \
      --output-policy /work/output/adapted_policy.onnx \
      --output-record /work/output/selection.json || RUN_RC=$?
    if [ "$RUN_RC" -eq 0 ] && [ -f .artifacts/input/qualification_plan.json ]; then
      uv run --no-sync eggroll-autopatch verify-integrated-early-stop-run \
        --run-dir /work/run \
        --campaign .artifacts/input/campaign.json \
        --release-scope .artifacts/input/release_scope.json \
        --qualification-plan .artifacts/input/qualification_plan.json \
        --qualification-command-spec .artifacts/input/qualification_command_spec.json \
        --source-manifest .artifacts/input/source_manifest.json \
        --selection-record /work/output/selection.json \
        --output-policy /work/output/adapted_policy.onnx \
        --max-requested-optimization-steps "$MAX_REQUESTED_OPTIMIZATION_STEPS" \
        --output /work/output/integrated_validation.json || RUN_RC=$?
    fi
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
    release_scope_path: Path | None,
    mode: str,
    qualification_plan_path: Path | None = None,
    qualification_command_spec_path: Path | None = None,
    max_requested_optimization_steps: int = 5_120_000,
    hf_hardware_flavor: str = "unbound-local-test",
) -> dict[str, Any]:
    _require_clean_repo(repo)
    if not hf_hardware_flavor:
        raise ValueError("HF hardware flavor cannot be empty")
    if max_requested_optimization_steps <= 0:
        raise ValueError(
            "requested optimization simulator-step ceiling must be positive"
        )
    campaign, campaign_sha256 = _campaign_document(campaign_path, mode=mode)
    base_campaign = PatchCampaign.from_json(campaign_path.read_text())
    needs_release_scope = (
        base_campaign.objective.objective_id
        == "locomotion-release-scope-lexicographic-v2"
    )
    if needs_release_scope != (release_scope_path is not None):
        raise ValueError(
            "the v2 release-scope objective requires a release scope, while "
            "historical campaign objectives forbid one"
        )
    release_scope = (
        None
        if release_scope_path is None
        else ReleaseScope.from_json(release_scope_path.read_text())
    )
    if (qualification_plan_path is None) != (qualification_command_spec_path is None):
        raise ValueError("qualification plan and command spec must be bundled together")
    qualification_plan = (
        None
        if qualification_plan_path is None
        else QualificationPlan.from_json(qualification_plan_path.read_text())
    )
    qualification_command_spec = (
        None
        if qualification_command_spec_path is None
        else CommandQualificationSpec.from_json(
            qualification_command_spec_path.read_text()
        )
    )
    if qualification_plan is not None and qualification_command_spec is not None:
        qualification_command_spec.validate_plan(qualification_plan)
    source_policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
    if source_policy_sha256 != base_campaign.artifact_sha256:
        raise ValueError("source policy bytes do not match campaign artifact")
    if (
        release_scope is not None
        and release_scope.source_fallback_sha256 != base_campaign.artifact_sha256
    ):
        raise ValueError("release scope fallback does not match campaign source")
    manifest = {
        "schema": "eggroll-autopatch-hf-source-v1",
        "mode": mode,
        "evidence_role": (
            "non-evidence CUDA smoke"
            if mode == "smoke"
            else (
                "production qualification environment preflight"
                if mode == "qualification-preflight"
                else "frozen campaign"
            )
        ),
        "source_commit": _git_output(repo, "rev-parse", "HEAD"),
        "source_branch": _git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "hf_hardware_flavor": hf_hardware_flavor,
        "source_policy_sha256": source_policy_sha256,
        "base_campaign_sha256": base_campaign.sha256,
        "execution_campaign_sha256": campaign_sha256,
        "release_scope_sha256": (
            None if release_scope is None else release_scope.sha256
        ),
        "qualification_plan_sha256": (
            None if qualification_plan is None else qualification_plan.sha256
        ),
        "qualification_command_spec_sha256": (
            None
            if qualification_command_spec is None
            else qualification_command_spec.sha256
        ),
        "requested_optimization_simulator_step_ceiling": (
            max_requested_optimization_steps
        ),
        "production_runtime_inputs": (
            None
            if qualification_command_spec is None
            else {
                "bucket": PRODUCTION_BUCKET,
                "runtime_volume_path": PRODUCTION_RUNTIME_VOLUME_PATH,
                "reference_policies_volume_path": REFERENCE_POLICIES_VOLUME_PATH,
                "runtime_identities": PRODUCTION_RUNTIME_IDENTITIES,
                "profile_router_overlay_sha256": hashlib.sha256(
                    (
                        repo
                        / "scripts/runtime_support/profile_scoped_model_activation.rs"
                    ).read_bytes()
                ).hexdigest(),
                "profile_harness_overlay_sha256": hashlib.sha256(
                    (
                        repo / "scripts/runtime_support/profile_scoped_policy_lab.rs"
                    ).read_bytes()
                ).hexdigest(),
            }
        ),
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
        documents = [
            ("campaign.json", campaign),
            ("source_manifest.json", manifest),
        ]
        if release_scope is not None:
            documents.append(("release_scope.json", release_scope.canonical_dict()))
        if qualification_plan is not None and qualification_command_spec is not None:
            documents.extend(
                (
                    ("qualification_plan.json", qualification_plan.canonical_dict),
                    (
                        "qualification_command_spec.json",
                        qualification_command_spec.canonical_dict,
                    ),
                )
            )
        for name, payload in documents:
            raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo(f"microduck_rl/.artifacts/input/{name}")
            info.size = len(raw)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(raw))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--release-scope",
        type=Path,
        help="required for v2; omit for immutable historical campaign objectives",
    )
    parser.add_argument("--qualification-plan", type=Path)
    parser.add_argument("--qualification-command-spec", type=Path)
    parser.add_argument("--namespace")
    parser.add_argument("--output-repo")
    parser.add_argument("--flavor", default="a10g-large")
    parser.add_argument("--timeout", default="8h")
    parser.add_argument(
        "--max-requested-optimization-steps", type=int, default=5_120_000
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--detach", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    repo = _repo_root()
    policy = args.policy.resolve()
    campaign = args.campaign.resolve()
    release_scope = args.release_scope.resolve() if args.release_scope else None
    qualification_plan = (
        args.qualification_plan.resolve() if args.qualification_plan else None
    )
    qualification_command_spec = (
        args.qualification_command_spec.resolve()
        if args.qualification_command_spec
        else None
    )
    for path in (
        policy,
        campaign,
        release_scope,
        qualification_plan,
        qualification_command_spec,
    ):
        if path is None:
            continue
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
            release_scope_path=release_scope,
            mode=args.mode,
            qualification_plan_path=qualification_plan,
            qualification_command_spec_path=qualification_command_spec,
            max_requested_optimization_steps=args.max_requested_optimization_steps,
            hf_hardware_flavor=args.flavor,
        )
        api.create_repo(source_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(bundle),
            path_in_repo=bundle.name,
            repo_id=source_repo,
            repo_type="dataset",
        )
        api.create_repo(output_repo, repo_type="model", private=True, exist_ok=True)
        volumes = [
            Volume(
                type="dataset",
                source=source_repo,
                mount_path="/src",
                read_only=True,
            )
        ]
        if qualification_command_spec is not None:
            volumes.extend(
                [
                    Volume(
                        type="bucket",
                        source=PRODUCTION_BUCKET,
                        path=PRODUCTION_RUNTIME_VOLUME_PATH,
                        mount_path="/inputs/microduck",
                        read_only=True,
                    ),
                    Volume(
                        type="bucket",
                        source=PRODUCTION_BUCKET,
                        path=REFERENCE_POLICIES_VOLUME_PATH,
                        mount_path="/inputs/reference_policies",
                        read_only=True,
                    ),
                ]
            )
        job = api.run_job(
            image=args.image,
            command=["bash", "-c", BOOTSTRAP],
            env={
                "SRC_TARBALL": bundle.name,
                "OUTPUT_REPO": output_repo,
                "RUN_MODE": args.mode,
                "MAX_REQUESTED_OPTIMIZATION_STEPS": str(
                    args.max_requested_optimization_steps
                ),
                "EGGROLL_HF_HARDWARE_FLAVOR": args.flavor,
            },
            secrets={"HF_TOKEN": token},
            flavor=args.flavor,
            timeout=args.timeout,
            volumes=volumes,
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
