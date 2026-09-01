#!/usr/bin/env python3
"""Build the source/adapted production-runtime digital-twin evidence pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from mjlab_microduck.eggroll.deployment import (
    CALIBRATION_LAG16,
    NOMINAL_PROFILE,
    bank_sha256,
    make_balanced_bank,
)

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MICRODUCK = WORKSPACE / "microduck"
SOURCE = WORKSPACE / "microduck" / "example_policies" / "alpha_stand.onnx"
ADAPTED = (
    ROOT
    / "policies"
    / "eggroll_posttraining"
    / "alpha_stand_lag16_v1"
    / "alpha_stand_eggroll_lag16_v1.onnx"
)
RUNNER = ROOT / "scripts" / "run_production_runtime_twin.py"
VERIFIER = ROOT / "scripts" / "verify_runtime_trace.py"
PROBE = WORKSPACE / "microduck" / "target" / "debug" / "examples" / "policy_probe"
UPDATER_HARNESS = MICRODUCK / "target" / "debug" / "examples" / "policy_patch_lab"
ORT_DYLIB = (
    ROOT
    / ".venv"
    / "lib"
    / "python3.12"
    / "site-packages"
    / "onnxruntime"
    / "capi"
    / "libonnxruntime.1.24.4.dylib"
)


def run(command: list[str], *, environment: dict[str, str], cwd: Path = ROOT) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_episode(
    *,
    policy_path: Path,
    pose: str,
    seed: int,
    profile_name: str,
    episode_dir: Path,
    video: bool,
    environment: dict[str, str],
) -> tuple[dict, dict]:
    command = [
        sys.executable,
        str(RUNNER),
        "--policy",
        str(policy_path),
        "--pose",
        pose,
        "--seed",
        str(seed),
        "--profile",
        profile_name,
        "--output-dir",
        str(episode_dir),
    ]
    if video:
        command.append("--video")
    run(command, environment=environment)
    parity_path = episode_dir / "parity.json"
    run(
        [
            sys.executable,
            str(VERIFIER),
            "--trace",
            str(episode_dir / "trace.jsonl"),
            "--policy",
            str(policy_path),
            "--probe",
            str(PROBE),
            "--output",
            str(parity_path),
        ],
        environment=environment,
    )
    row = json.loads((episode_dir / "manifest.json").read_text())
    parity = json.loads(parity_path.read_text())
    if parity["first_divergence"] is not None:
        raise RuntimeError(f"trace divergence in {episode_dir}")
    return row, parity


def aggregate(rows: list[dict]) -> dict:
    by_pose: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pose[row["scenario"]["pose"]].append(row)
    metric_names = (
        "stable_hold_s",
        "max_trunk_height_m",
        "final_trunk_height_m",
        "final_upright_cosine",
        "time_upright_s",
        "task_return",
    )

    def summarize(selected: list[dict]) -> dict:
        return {
            "episodes": len(selected),
            "terminal_successes": sum(
                bool(row["result"]["terminal_success"]) for row in selected
            ),
            **{
                f"mean_{name}": mean(float(row["result"][name]) for row in selected)
                for name in metric_names
            },
        }

    return {
        "overall": summarize(rows),
        "per_pose": {pose: summarize(selected) for pose, selected in by_pose.items()},
    }


def updater_proof(
    *,
    output: Path,
    source: Path,
    adapted: Path,
    harness: Path,
    environment: dict[str, str],
) -> dict:
    if not harness.is_file():
        run(
            ["cargo", "build", "-p", "updater", "--example", "policy_patch_lab"],
            environment=environment,
            cwd=MICRODUCK,
        )
    if not harness.is_file():
        raise RuntimeError(f"updater proof harness was not built: {harness}")
    root = output / "updater"
    run(
        [
            str(harness),
            "activate",
            "--root",
            str(root),
            "--source",
            str(source.resolve()),
            "--adapted",
            str(adapted.resolve()),
        ],
        environment=environment,
        cwd=MICRODUCK,
    )
    activation = json.loads((root / "activation.json").read_text())
    current = root / "install" / "current" / "policy.onnx"
    if sha256(current) != sha256(adapted):
        raise RuntimeError("updater activation did not expose exact adapted bytes")
    scenario = {
        "pose": "sitting",
        "seed": 20260901,
        "profile": CALIBRATION_LAG16.name,
    }
    adapted_row, adapted_parity = run_episode(
        policy_path=current,
        pose=scenario["pose"],
        seed=scenario["seed"],
        profile_name=scenario["profile"],
        episode_dir=root / "behavior-adapted",
        video=False,
        environment=environment,
    )
    run(
        [str(harness), "rollback", "--root", str(root)],
        environment=environment,
        cwd=MICRODUCK,
    )
    rollback = json.loads((root / "rollback.json").read_text())
    if sha256(current) != sha256(source):
        raise RuntimeError("updater rollback did not restore exact source bytes")
    source_row, source_parity = run_episode(
        policy_path=current,
        pose=scenario["pose"],
        seed=scenario["seed"],
        profile_name=scenario["profile"],
        episode_dir=root / "behavior-rollback-source",
        video=False,
        environment=environment,
    )
    if adapted_row["scenario"] != source_row["scenario"]:
        raise RuntimeError("updater behavior replay did not use an identical scenario")
    return {
        "schema": "eggroll-policy-patch-updater-proof-v1",
        "activation": activation,
        "rollback": rollback,
        "behavior_scenario": scenario,
        "adapted_behavior": adapted_row["result"],
        "rollback_source_behavior": source_row["result"],
        "adapted_trace_first_divergence": adapted_parity["first_divergence"],
        "rollback_trace_first_divergence": source_parity["first_divergence"],
        "exact_source_restored": rollback["exact_source_restored"],
        "claim_scope": "real updater state machine plus production-runtime digital twin",
    }


def render_runtime_hero(
    *, output: Path, artifacts: list[dict], comparisons: dict[str, dict]
) -> dict:
    import mediapy
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    shifted = [row for row in artifacts if row["profile"] == CALIBRATION_LAG16.name]
    by_scenario: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in shifted:
        by_scenario[row["scenario_id"]][row["policy"]] = row
    complete = [
        (scenario_id, pair)
        for scenario_id, pair in sorted(by_scenario.items())
        if set(pair) == {"source", "adapted"}
    ]
    improved = [
        item
        for item in complete
        if not item[1]["source"]["terminal_success"]
        and item[1]["adapted"]["terminal_success"]
    ]
    if improved:
        scenario_id, pair = improved[0]
        selection_rule = "lowest scenario id with source failure and adapted success"
    elif complete:
        scenario_id, pair = max(
            complete,
            key=lambda item: (
                float(item[1]["adapted"]["result"]["terminal_progress"])
                - float(item[1]["source"]["result"]["terminal_progress"])
            ),
        )
        selection_rule = "largest adapted-minus-source terminal progress"
    else:
        raise RuntimeError("hero renderer found no complete shifted policy pair")

    source_path = output / pair["source"]["path"] / "episode.mp4"
    adapted_path = output / pair["adapted"]["path"] / "episode.mp4"
    if not source_path.is_file() or not adapted_path.is_file():
        raise RuntimeError("hero renderer requires --video episode artifacts")
    source_video = np.asarray(mediapy.read_video(source_path))
    adapted_video = np.asarray(mediapy.read_video(adapted_path))
    width, height, fps = 1280, 720, 50
    background = (10, 14, 22)
    white = (245, 247, 252)
    muted = (164, 174, 192)
    source_color = (255, 177, 66)
    adapted_color = (78, 218, 150)

    def font(size: int, bold: bool = False):
        candidates = (
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        for candidate in candidates:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size, index=0)
        return ImageFont.load_default()

    def card(lines: list[tuple[str, tuple[int, int, int], int]], count: int):
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        y = 210
        for text_value, color, size in lines:
            bounds = draw.textbbox((0, 0), text_value, font=font(size, True))
            draw.text(
                ((width - (bounds[2] - bounds[0])) / 2, y),
                text_value,
                font=font(size, True),
                fill=color,
            )
            y += size + 24
        return [np.asarray(image)] * count

    shifted_overall = comparisons[CALIBRATION_LAG16.name]
    source_total = shifted_overall["source"]["aggregate"]["overall"]
    adapted_total = shifted_overall["adapted"]["aggregate"]["overall"]
    frames = card(
        [
            ("A deployed policy meets a new failure mode.", white, 44),
            ("EGGROLL patches it without gradients.", adapted_color, 52),
            ("Production-runtime digital twin", muted, 26),
        ],
        fps * 2,
    )
    source_result = pair["source"]["result"]
    adapted_result = pair["adapted"]["result"]
    count = min(len(source_video), len(adapted_video))
    for index in range(count):
        canvas = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (28, 16),
            "Hidden deployment shift: 6.5 V + sag 0.2 + lag 16",
            font=font(31, True),
            fill=white,
        )
        draw.text(
            (28, 56),
            (
                f"Paired bank: source {source_total['terminal_successes']}/"
                f"{source_total['episodes']} → EGGROLL "
                f"{adapted_total['terminal_successes']}/{adapted_total['episodes']}"
            ),
            font=font(19),
            fill=muted,
        )
        left = Image.fromarray(source_video[index]).resize((600, 450))
        right = Image.fromarray(adapted_video[index]).resize((600, 450))
        canvas.paste(left, (24, 105))
        canvas.paste(right, (656, 105))
        draw.rectangle((24, 105, 624, 145), fill=(24, 30, 40))
        draw.rectangle((656, 105, 1256, 145), fill=(24, 30, 40))
        draw.text((38, 113), "Source PPO", font=font(21, True), fill=source_color)
        draw.text((670, 113), "EGGROLL patch", font=font(21, True), fill=adapted_color)
        for x, result, color in (
            (30, source_result, source_color),
            (662, adapted_result, adapted_color),
        ):
            status = "SUCCESS" if result["terminal_success"] else "FAILURE"
            draw.text((x, 570), f"terminal: {status}", font=font(20, True), fill=color)
            draw.text(
                (x, 600),
                f"final height {result['final_trunk_height_m']:.3f} m  |  "
                f"upright {result['final_upright_cosine']:.3f}",
                font=font(17),
                fill=white,
            )
            draw.text(
                (x, 628),
                f"stable hold {result['stable_hold_s']:.2f} s  |  "
                f"return {result['task_return']:.2f} (diagnostic)",
                font=font(17),
                fill=white,
            )
        draw.text(
            (28, 684),
            f"scenario {scenario_id}  |  t={index / fps:.2f}s",
            font=font(14),
            fill=muted,
        )
        frames.append(np.asarray(canvas))
    frames.extend(
        card(
            [
                ("Same 61D → 14D deployment contract.", white, 40),
                ("1,806 parameters patched.", adapted_color, 54),
                ("Signed activation. Health gate. Exact rollback.", white, 30),
                ("Simulation evidence; no physical robot claim.", muted, 22),
            ],
            fps * 2,
        )
    )
    path = output / "eggroll_policy_patch_lab_hero.mp4"
    mediapy.write_video(path, frames, fps=fps, codec="h264", crf=20)
    result = {
        "schema": "eggroll-policy-patch-lab-hero-v1",
        "path": path.name,
        "sha256": sha256(path),
        "frames": len(frames),
        "fps": fps,
        "duration_s": len(frames) / fps,
        "scenario_id": scenario_id,
        "selection_rule": selection_rule,
        "source_video_sha256": sha256(source_path),
        "adapted_video_sha256": sha256(adapted_path),
    }
    (path.with_suffix(".mp4.json")).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def write_record(output: Path, manifest: dict) -> None:
    comparisons = manifest["comparisons"]
    lines = [
        "# EGGROLL Policy Patch Lab runtime evidence",
        "",
        "This pack is evidence from the **production-runtime digital twin**. It uses the ",
        "actual registered `Mjlab-StandUp-Flat-MicroDuck` environment behind MicroDuck's ",
        "real Rust `RobotIo` seam. It is not evidence from a physical robot.",
        "",
        "| Profile | Policy | Terminal success | Mean terminal hold | Mean task return |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for profile_name, policies in comparisons.items():
        for policy_name, payload in policies.items():
            overall = payload["aggregate"]["overall"]
            lines.append(
                f"| {profile_name} | {policy_name} | "
                f"{overall['terminal_successes']}/{overall['episodes']} | "
                f"{overall['mean_stable_hold_s']:.3f} s | "
                f"{overall['mean_task_return']:.2f} |"
            )
    lines.extend(
        [
            "",
            "Every episode includes raw sensor fixtures, actor observations, absolute targets, ",
            "task diagnostics, the `robotd` log, policy/runtime SHA-256 identities, and a strict ",
            "step-by-step Rust/ONNX trace-parity report. Task return is diagnostic only.",
            "",
            (
                "The updater proof used the real signed `model-stand` state machine: signatures "
                "and artifact hashes were verified, the adapted bytes passed the health gate, "
                "and a fresh process restored the exact source SHA before replaying its behavior."
                if manifest["updater_proof"] is not None
                else "Updater proof was explicitly skipped for this run."
            ),
            "",
            (
                f"Hero video: `{manifest['hero']['path']}` (deterministically selected pair)."
                if manifest["hero"] is not None
                else "Run with `--video` to produce per-episode evidence and the hero video."
            ),
            "",
            "Reproduce:",
            "",
            "```bash",
            manifest["reproduction_command"],
            "```",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-pose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--adapted", type=Path, default=ADAPTED)
    parser.add_argument("--updater-harness", type=Path, default=UPDATER_HARNESS)
    parser.add_argument(
        "--skip-updater-proof",
        action="store_true",
        help="skip only the signed activation/health-gate/rollback proof",
    )
    args = parser.parse_args()
    if args.episodes_per_pose < 1:
        raise SystemExit("--episodes-per-pose must be positive")
    for name, path in (
        ("source policy", args.source),
        ("adapted policy", args.adapted),
        ("Rust policy probe", PROBE),
        ("ONNX Runtime dylib", ORT_DYLIB),
    ):
        if not path.is_file():
            raise SystemExit(f"{name} not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/eggroll-mpl")
    environment.setdefault("ORT_DYLIB_PATH", str(ORT_DYLIB))
    profiles = (NOMINAL_PROFILE, CALIBRATION_LAG16)
    policies = (("source", args.source), ("adapted", args.adapted))
    comparisons: dict[str, dict] = {}
    episode_artifacts: list[dict] = []

    for profile in profiles:
        bank = make_balanced_bank(
            profile=profile,
            base_seed=args.seed,
            episodes_per_pose=args.episodes_per_pose,
            prefix=f"runtime-{profile.name}",
        )
        profile_rows: dict[str, dict] = {}
        for policy_name, policy_path in policies:
            rows: list[dict] = []
            for scenario in bank:
                episode_dir = (
                    args.output_dir
                    / "episodes"
                    / profile.name
                    / policy_name
                    / scenario.scenario_id
                )
                row, parity = run_episode(
                    policy_path=policy_path,
                    pose=scenario.pose,
                    seed=scenario.seed,
                    profile_name=profile.name,
                    episode_dir=episode_dir,
                    video=args.video,
                    environment=environment,
                )
                rows.append(row)
                episode_artifacts.append(
                    {
                        "profile": profile.name,
                        "policy": policy_name,
                        "scenario_id": scenario.scenario_id,
                        "path": str(episode_dir.relative_to(args.output_dir)),
                        "terminal_success": row["result"]["terminal_success"],
                        "result": row["result"],
                        "policy_sha256": row["policy"]["sha256"],
                        "runtime_sha256": row["runtime"]["sha256"],
                        "runtime_path": row["runtime"]["path"],
                        "timing": row["timing"],
                        "profile_sha256": row["profile_sha256"],
                        "runtime_safety_clamp_steps": parity[
                            "runtime_safety_clamp_steps"
                        ],
                        "task_vs_runtime_previous_action_error": parity[
                            "max_task_vs_runtime_previous_action_error"
                        ],
                        "first_divergence": parity["first_divergence"],
                        "parity_max_error": max(
                            float(parity["max_sensor_derived_observation_error"]),
                            float(parity["max_runtime_previous_action_chain_error"]),
                            float(parity["max_rust_vs_onnx_action_error"]),
                            float(parity["max_rust_proposed_target_formula_error"]),
                            float(parity["max_rust_applied_target_formula_error"]),
                            float(parity["max_runtime_applied_action_vs_rust_error"]),
                            float(parity["max_runtime_applied_target_vs_rust_error"]),
                        ),
                    }
                )
            profile_rows[policy_name] = {
                "policy_sha256": rows[0]["policy"]["sha256"],
                "bank_sha256": bank_sha256(bank),
                "bank": [scenario.to_dict() for scenario in bank],
                "aggregate": aggregate(rows),
            }
        if profile_rows["source"]["bank"] != profile_rows["adapted"]["bank"]:
            raise RuntimeError(f"unpaired source/adapted bank for {profile.name}")
        comparisons[profile.name] = profile_rows

    updater = (
        None
        if args.skip_updater_proof
        else updater_proof(
            output=args.output_dir,
            source=args.source,
            adapted=args.adapted,
            harness=args.updater_harness,
            environment=environment,
        )
    )
    hero = (
        render_runtime_hero(
            output=args.output_dir,
            artifacts=episode_artifacts,
            comparisons=comparisons,
        )
        if args.video
        else None
    )

    reproduction = (
        f"PYTHONPATH=src {sys.executable} scripts/eggroll_patch_lab.py "
        f"--output-dir {args.output_dir} --episodes-per-pose {args.episodes_per_pose} "
        f"--seed {args.seed}{' --video' if args.video else ''}"
    )
    runtime_sha256_values = {row["runtime_sha256"] for row in episode_artifacts}
    if len(runtime_sha256_values) != 1:
        raise RuntimeError("episode set used more than one robotd runtime binary")
    manifest = {
        "schema": "eggroll-policy-patch-lab-evidence-v1",
        "claim_scope": "production-runtime digital twin; no physical robot",
        "task": "Mjlab-StandUp-Flat-MicroDuck",
        "seed": args.seed,
        "episodes_per_pose": args.episodes_per_pose,
        "paired_banks": True,
        "trace_parity_required": True,
        "all_trace_parity_passed": all(
            row["parity_max_error"] == 0.0 for row in episode_artifacts
        ),
        "task_return_role": "diagnostic_only",
        "runtime": {
            "upstream": "https://github.com/pollen-robotics/microduck",
            "binary": episode_artifacts[0]["runtime_path"],
            "robotd_sha256": next(iter(runtime_sha256_values)),
            "control_rate_hz": 50.0,
            "boundary": (
                "actual StandUp simulator supplies raw RobotIo sensors and accepts "
                "absolute targets; Rust owns observation, ONNX inference, safety "
                "clamping, previous raw action, and loop timing"
            ),
        },
        "actor_contract": {
            "observation_shape": [1, 61],
            "action_shape": [1, 14],
            "observation_blocks": {
                "gyro": [0, 3],
                "projected_gravity": [3, 6],
                "joint_position_relative_to_home": [6, 20],
                "joint_velocity": [20, 34],
                "previous_raw_policy_action": [34, 48],
                "twist_command": [48, 51],
                "head_command": [51, 55],
                "body_xy_command": [55, 57],
                "body_z_roll_pitch_command": [57, 60],
                "body_yaw_command": [60, 61],
            },
            "command_values": [0.0] * 13,
            "action_scale": 1.0,
            "absolute_target": "home_pose + action",
            "safety_boundary": "absolute targets clamped to actuator range [-pi, pi]",
            "task_internal_history_note": (
                "physics records the prior safety-applied action; Rust actor history "
                "correctly remains the prior raw policy output"
            ),
        },
        "policy_sha256": {
            "source": sha256(args.source),
            "adapted": sha256(args.adapted),
        },
        "comparisons": comparisons,
        "episode_artifacts": episode_artifacts,
        "updater_proof": updater,
        "hero": hero,
        "reproduction_command": reproduction,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_record(args.output_dir, manifest)
    print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
