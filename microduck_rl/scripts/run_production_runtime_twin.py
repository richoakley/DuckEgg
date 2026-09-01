#!/usr/bin/env python3
"""Run one StandUp episode through the real Rust MicroDuck policy runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from mjlab_microduck.eggroll.deployment import (
    PROFILES,
    Scenario,
    asymmetric_left_leg_profile,
)
from mjlab_microduck.sim.standup_runtime_twin import StandupRuntimeBody, start_server

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_ROBOTD = WORKSPACE / "microduck" / "target" / "debug" / "robotd"
DEFAULT_SOURCE = WORKSPACE / "microduck" / "example_policies" / "alpha_stand.onnx"
DEFAULT_ADAPTED = (
    ROOT
    / "policies"
    / "eggroll_posttraining"
    / "alpha_stand_lag16_v1"
    / "alpha_stand_eggroll_lag16_v1.onnx"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_config(policy: Path) -> str:
    rendered = str(policy.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    return f"""
[policy]
enabled = true
mode = "walk"
walk = "{rendered}"
stand = "{rendered}"
sitstand = "none"
ground_pick = "none"
kick_left = "none"
kick_right = "none"
roulade = "none"
action_scale = 1.0
standing_action_scale = 1.0
standing_gain_ratio = 1.0
gain = 200
head_lowpass = 1.0
legs_lowpass = 1.0
voltage_adapt = false

[safety]
limp_fall = false
battery_empty_shutdown = false

[audio]
enabled = false
greet = false
""".lstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--robotd", type=Path, default=DEFAULT_ROBOTD)
    parser.add_argument(
        "--pose",
        choices=("standing", "sitting", "face-down", "face-up"),
        default="face-down",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="nominal-fixed-median-v1"
    )
    parser.add_argument(
        "--asymmetric-effectiveness",
        type=float,
        help="evaluation-only left knee/ankle torque effectiveness; uses nominal base profile",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ort-dylib",
        type=Path,
        default=(
            ROOT
            / ".venv"
            / "lib"
            / "python3.12"
            / "site-packages"
            / "onnxruntime"
            / "capi"
            / "libonnxruntime.1.24.4.dylib"
        ),
    )
    args = parser.parse_args()

    for name, path in (
        ("policy", args.policy),
        ("robotd", args.robotd),
        ("ONNX Runtime", args.ort_dylib),
    ):
        if not path.is_file():
            raise SystemExit(f"{name} not found: {path}")
    profile = (
        asymmetric_left_leg_profile(args.asymmetric_effectiveness)
        if args.asymmetric_effectiveness is not None
        else PROFILES[args.profile]
    )
    scenario = Scenario(
        scenario_id=f"runtime-{args.pose}-{args.seed}",
        pose=args.pose,
        seed=args.seed,
        profile_name=profile.name,
        profile_sha256=profile.sha256,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    body = StandupRuntimeBody(
        scenario=scenario,
        profile=profile,
        device=args.device,
        record_video=args.video,
    )
    server, thread = start_server(body)
    host, port = server.server_address
    process: subprocess.Popen[str] | None = None
    with tempfile.TemporaryDirectory(prefix="microduck-runtime-twin-") as temporary:
        scratch = Path(temporary)
        params = scratch / "robotd.toml"
        socket_path = scratch / "robotd.sock"
        params.write_text(runtime_config(args.policy))
        environment = os.environ.copy()
        environment["ORT_DYLIB_PATH"] = str(args.ort_dylib.resolve())
        environment.setdefault("RUST_LOG", "info")
        command = [
            str(args.robotd.resolve()),
            "--params",
            str(params),
            "--socket",
            str(socket_path),
            "--sim",
            f"{host}:{port}",
            "--sim-eval",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            if not body.done.wait(args.timeout):
                raise TimeoutError(
                    f"runtime twin did not finish {body.horizon} steps in {args.timeout}s"
                )
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            output = (
                ""
                if process is None or process.stdout is None
                else process.stdout.read()
            )
            (args.output_dir / "robotd.log").write_text(output)
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    try:
        result = body.save(args.output_dir)
        video_replay = (
            body.render_trace_video(args.output_dir / "episode.mp4")
            if args.video
            else None
        )
        manifest = {
            "schema": "microduck-production-runtime-twin-run-v1",
            "claim_scope": "production-runtime digital twin; no physical robot",
            "task": "Mjlab-StandUp-Flat-MicroDuck",
            "policy": {
                "path": str(args.policy.resolve()),
                "sha256": sha256(args.policy),
            },
            "runtime": {
                "path": str(args.robotd.resolve()),
                "sha256": sha256(args.robotd),
                "arguments": command[1:],
            },
            "scenario": scenario.to_dict(),
            "profile": profile.canonical_dict(),
            "profile_sha256": profile.sha256,
            "artifacts": {
                "summary": "summary.json",
                "trace": "trace.jsonl",
                "render_states": "render_states.npz"
                if video_replay is not None
                else None,
                "runtime_log": "robotd.log",
                "video": "episode.mp4" if video_replay is not None else None,
            },
            "video_replay": video_replay,
            "result": result.summary,
            "timing": result.timing,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        body.close()


if __name__ == "__main__":
    main()
