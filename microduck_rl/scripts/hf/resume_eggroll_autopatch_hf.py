"""Resume timed-out EGGROLL Autopatch HF jobs from their uploaded exact state."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from huggingface_hub import HfApi, get_token

_HYDRATE_ANCHOR = "mkdir -p /work/run /work/output\n"
_TRAIN_ANCHOR = "  --output-dir /work/run \\" + "\n  --device cuda:0"
_REQUIRED_RUN_FILES = {
    "run/accounting.json",
    "run/config.json",
    "run/last.pkl",
    "run/metrics.jsonl",
    "run/qualification.json",
    "run/source_baseline.json",
}


def _resume_bootstrap(bootstrap: str) -> str:
    """Inject fail-closed state hydration and the trainer's strict resume flag."""
    if "--resume /work/run/last.pkl" in bootstrap:
        raise ValueError("job bootstrap is already a resume bootstrap")
    if bootstrap.count(_HYDRATE_ANCHOR) != 1:
        raise ValueError("job bootstrap has an unexpected run-directory boundary")
    if bootstrap.count(_TRAIN_ANCHOR) != 1:
        raise ValueError("job bootstrap has an unexpected trainer invocation")
    hydration = _HYDRATE_ANCHOR + (
        "mkdir -p /work/restore\n"
        'uv run --no-sync hf download "$OUTPUT_REPO" --repo-type model \\'
        "\n"
        '  --local-dir /work/restore --token "$HF_TOKEN"\n'
        "test -f /work/restore/run/last.pkl\n"
        "cp -a /work/restore/run/. /work/run/\n"
    )
    bootstrap = bootstrap.replace(_HYDRATE_ANCHOR, hydration, 1)
    return bootstrap.replace(
        _TRAIN_ANCHOR,
        "  --output-dir /work/run \\"
        "\n  --resume /work/run/last.pkl \\"
        "\n  --device cuda:0",
        1,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume timed-out EGGROLL Autopatch jobs from exact periodically uploaded "
            "state while reusing the original immutable source and runtime volumes."
        )
    )
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--timeout", default="2h")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    token = get_token()
    if not token:
        print("error: no cached Hugging Face token", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    submissions: list[dict[str, str]] = []
    for origin_id in args.job_id:
        origin = api.inspect_job(job_id=origin_id)
        stage = str(origin.status.stage)
        message = str(origin.status.message)
        if stage != "ERROR" or message != "Job timeout":
            raise ValueError(
                f"origin {origin_id} is not an exact timeout: {stage=} {message=}"
            )
        if list(origin.command[:2]) != ["bash", "-c"] or len(origin.command) != 3:
            raise ValueError(f"origin {origin_id} has an unexpected command")
        environment = dict(origin.environment)
        output_repo = environment.get("OUTPUT_REPO")
        if not isinstance(output_repo, str) or not output_repo:
            raise ValueError(f"origin {origin_id} has no output repository")
        if environment.get("RUN_MODE") != "train":
            raise ValueError(f"origin {origin_id} is not a training job")
        remote_files = set(api.list_repo_files(repo_id=output_repo, repo_type="model"))
        missing = sorted(_REQUIRED_RUN_FILES - remote_files)
        if missing:
            raise ValueError(
                f"origin {origin_id} has no complete resumable upload: {missing}"
            )
        owner = origin.owner.name
        resumed = api.run_job(
            image=origin.docker_image,
            command=["bash", "-c", _resume_bootstrap(origin.command[2])],
            env=environment,
            secrets={"HF_TOKEN": token},
            flavor=origin.flavor,
            timeout=args.timeout,
            labels={"eggroll_resume_of": origin_id},
            volumes=list(origin.volumes),
            namespace=owner,
        )
        submissions.append(
            {
                "origin_job_id": origin_id,
                "resume_job_id": resumed.id,
                "output_repository": output_repo,
                "timeout": args.timeout,
            }
        )
    print(json.dumps(submissions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
