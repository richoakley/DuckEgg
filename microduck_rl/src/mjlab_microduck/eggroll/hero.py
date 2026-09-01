"""Deterministic source-versus-adapted EGGROLL demonstration renderer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import write_json
from .release import POSES, sha256_file, verify_release_manifest

WIDTH = 1280
HEIGHT = 720
FPS = 50
BACKGROUND = (10, 14, 22)
SOURCE_COLOR = (255, 177, 66)
ADAPTED_COLOR = (78, 218, 150)
MUTED = (164, 174, 192)
WHITE = (245, 247, 252)


@dataclass(frozen=True)
class PairSelection:
    pose: str
    scenario_id: str
    episode_index: int
    selection_rule: str


def load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def select_shifted_pairs(
    source: Mapping[str, Any], adapted: Mapping[str, Any]
) -> tuple[PairSelection, ...]:
    """Select the first paired source failure per pose, never a hand-picked clip."""

    if source["bank_sha256"] != adapted["bank_sha256"]:
        raise ValueError("Shifted playback summaries do not use the same bank")
    source_terminal = source["episodes"]["terminal_success"]
    adapted_terminal = adapted["episodes"]["terminal_success"]
    selections: list[PairSelection] = []
    for pose in POSES:
        candidates = [
            index
            for index, scenario in enumerate(source["bank"])
            if scenario["pose"] == pose
        ]
        failures = [
            index
            for index in candidates
            if not bool(source_terminal[index]) and bool(adapted_terminal[index])
        ]
        if not failures:
            raise ValueError(f"No source-failure/adapted-success pair for {pose}")
        index = failures[0]
        if source["bank"][index] != adapted["bank"][index]:
            raise ValueError("Paired playback scenario payloads differ")
        selections.append(
            PairSelection(
                pose=pose,
                scenario_id=str(source["bank"][index]["scenario_id"]),
                episode_index=index,
                selection_rule="lowest-index source failure with adapted terminal success",
            )
        )
    return tuple(selections)


def select_nominal_pair(
    source: Mapping[str, Any], adapted: Mapping[str, Any]
) -> PairSelection:
    if source["bank_sha256"] != adapted["bank_sha256"]:
        raise ValueError("Nominal playback summaries do not use the same bank")
    for index, scenario in enumerate(source["bank"]):
        if (
            scenario["pose"] == "face-down"
            and bool(source["episodes"]["terminal_success"][index])
            and bool(adapted["episodes"]["terminal_success"][index])
        ):
            return PairSelection(
                pose="face-down",
                scenario_id=str(scenario["scenario_id"]),
                episode_index=index,
                selection_rule="lowest-index paired nominal face-down success",
            )
    raise ValueError("Nominal evidence has no paired face-down success")


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size, index=0)
    return ImageFont.load_default(size=size)


def _draw_centered(draw: Any, text: str, y: int, *, font: Any, fill: tuple[int, ...]) -> None:
    left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((WIDTH - (right - left)) / 2, y), text, font=font, fill=fill)


def _title_card(lines: Sequence[tuple[str, tuple[int, ...], int]], frames: int) -> list[np.ndarray]:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    total = sum(size + 20 for _text, _fill, size in lines)
    y = (HEIGHT - total) // 2
    for text, fill, size in lines:
        _draw_centered(draw, text, y, font=_font(size, bold=size >= 42), fill=fill)
        y += size + 20
    frame = np.asarray(image)
    return [frame] * frames


def _resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS))


def _episode_metrics(summary: Mapping[str, Any], index: int) -> list[str]:
    episodes = summary["episodes"]
    success = "SUCCESS" if bool(episodes["terminal_success"][index]) else "FAILURE"
    return [
        f"terminal: {success}",
        (
            f"max / final z: {episodes['max_trunk_height_m'][index]:.3f} / "
            f"{episodes['final_trunk_height_m'][index]:.3f} m"
        ),
        f"final upright: {episodes['final_upright_cosine'][index]:.3f}",
        (
            f"upright / stable hold: {episodes['time_upright_s'][index]:.2f} / "
            f"{episodes['stable_hold_s'][index]:.2f} s"
        ),
        f"task return (diagnostic): {episodes['task_return'][index]:.2f}",
    ]


def _paired_frames(
    *,
    source_video: np.ndarray,
    adapted_video: np.ndarray,
    source_summary: Mapping[str, Any],
    adapted_summary: Mapping[str, Any],
    selection: PairSelection,
    heading: str,
    aggregate: str,
    source_hash: str,
    adapted_hash: str,
    profile_hash: str,
) -> list[np.ndarray]:
    from PIL import Image, ImageDraw

    count = min(len(source_video), len(adapted_video))
    source_metrics = _episode_metrics(source_summary, selection.episode_index)
    adapted_metrics = _episode_metrics(adapted_summary, selection.episode_index)
    output: list[np.ndarray] = []
    for frame_index in range(count):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        draw.text((32, 18), heading, font=_font(34, bold=True), fill=WHITE)
        draw.text((32, 57), aggregate, font=_font(19), fill=MUTED)
        draw.text(
            (32, 84),
            f"reset: {selection.pose}  |  scenario: {selection.scenario_id}",
            font=_font(17),
            fill=MUTED,
        )
        left = _resize_frame(source_video[frame_index], (600, 450))
        right = _resize_frame(adapted_video[frame_index], (600, 450))
        canvas.paste(Image.fromarray(left), (24, 118))
        canvas.paste(Image.fromarray(right), (656, 118))
        draw.rectangle((24, 118, 624, 156), fill=(24, 30, 40))
        draw.rectangle((656, 118, 1256, 156), fill=(24, 30, 40))
        draw.text((40, 126), "Production PPO", font=_font(20, bold=True), fill=SOURCE_COLOR)
        draw.text((672, 126), "EGGROLL post-trained", font=_font(20, bold=True), fill=ADAPTED_COLOR)
        for row, text in enumerate(source_metrics):
            draw.text((32, 580 + row * 24), text, font=_font(16), fill=WHITE)
        for row, text in enumerate(adapted_metrics):
            draw.text((664, 580 + row * 24), text, font=_font(16), fill=WHITE)
        draw.text(
            (925, 95),
            f"t={frame_index / FPS:4.2f}s",
            font=_font(17),
            fill=MUTED,
        )
        draw.text(
            (1060, 95),
            f"profile {profile_hash[:10]}",
            font=_font(14),
            fill=MUTED,
        )
        draw.text((32, 695), f"source {source_hash[:12]}", font=_font(12), fill=MUTED)
        draw.text((664, 695), f"adapted {adapted_hash[:12]}", font=_font(12), fill=MUTED)
        output.append(np.asarray(canvas))
    return output


def render_hero(
    *,
    manifest_path: Path,
    source_shifted_dir: Path,
    adapted_shifted_dir: Path,
    source_nominal_dir: Path,
    adapted_nominal_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Render the complete, predeclared five-scene product demonstration."""

    import mediapy

    verify_release_manifest(manifest_path)
    manifest = load_summary(manifest_path)
    root = manifest_path.parent
    summaries = {
        role: load_summary(root / record["path"])
        for role, record in manifest["evaluation"].items()
    }
    shifted = select_shifted_pairs(
        summaries["source_shifted"], summaries["adapted_shifted"]
    )
    nominal = select_nominal_pair(
        summaries["source_nominal"], summaries["adapted_nominal"]
    )
    source_hash = manifest["base_policy"]["sha256"]
    adapted_hash = manifest["adapted_policy"]["sha256"]
    shifted_profile = manifest["adaptation"]["deployment_profile_sha256"]
    nominal_profile = summaries["source_nominal"]["profile_sha256"]

    frames: list[np.ndarray] = _title_card(
        [
            ("One deployed PPO policy.", WHITE, 48),
            ("One evaluation-only objective.", WHITE, 48),
            ("A new robot policy—without gradients.", ADAPTED_COLOR, 48),
            ("Actual Mjlab-StandUp-Flat-MicroDuck environment", MUTED, 22),
        ],
        FPS * 2,
    )
    chosen: list[dict[str, Any]] = []
    shifted_source = summaries["source_shifted"]
    shifted_adapted = summaries["adapted_shifted"]
    for selection in shifted:
        name = f"{selection.scenario_id}.mp4"
        source_path = source_shifted_dir / name
        adapted_path = adapted_shifted_dir / name
        source_pose = manifest["release_decision"]["shifted"]["source_per_pose"][selection.pose]
        adapted_pose = manifest["release_decision"]["shifted"]["adapted_per_pose"][selection.pose]
        frames.extend(
            _paired_frames(
                source_video=np.asarray(mediapy.read_video(source_path)),
                adapted_video=np.asarray(mediapy.read_video(adapted_path)),
                source_summary=shifted_source,
                adapted_summary=shifted_adapted,
                selection=selection,
                heading="Hidden deployment shift: 6.5 V + sag 0.2 + lag 16",
                aggregate=(
                    f"Full held-out bank: PPO 17/32 → EGGROLL 32/32  |  "
                    f"{selection.pose}: {source_pose}/8 → {adapted_pose}/8"
                ),
                source_hash=source_hash,
                adapted_hash=adapted_hash,
                profile_hash=shifted_profile,
            )
        )
        chosen.append(
            {
                **selection.__dict__,
                "profile": "shifted",
                "source_video_sha256": sha256_file(source_path),
                "adapted_video_sha256": sha256_file(adapted_path),
            }
        )

    nominal_source = summaries["source_nominal"]
    nominal_adapted = summaries["adapted_nominal"]
    nominal_name = f"{nominal.scenario_id}.mp4"
    nominal_source_path = source_nominal_dir / nominal_name
    nominal_adapted_path = adapted_nominal_dir / nominal_name
    frames.extend(
        _paired_frames(
            source_video=np.asarray(mediapy.read_video(nominal_source_path)),
            adapted_video=np.asarray(mediapy.read_video(nominal_adapted_path)),
            source_summary=nominal_source,
            adapted_summary=nominal_adapted,
            selection=nominal,
            heading="Nominal-retention gate",
            aggregate="Full held-out bank: PPO 32/32  |  EGGROLL 32/32",
            source_hash=source_hash,
            adapted_hash=adapted_hash,
            profile_hash=nominal_profile,
        )
    )
    chosen.append(
        {
            **nominal.__dict__,
            "profile": "nominal",
            "source_video_sha256": sha256_file(nominal_source_path),
            "adapted_video_sha256": sha256_file(nominal_adapted_path),
        }
    )
    frames.extend(
        _title_card(
            [
                ("17/32 → 32/32", ADAPTED_COLOR, 72),
                ("32/32 nominal capability retained", WHITE, 38),
                ("1,806 parameters changed. Same 61D → 14D ONNX contract.", WHITE, 28),
                ("Task return fell: the black-box terminal objective drove selection.", MUTED, 22),
            ],
            FPS * 3,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(output, frames, fps=FPS, codec="h264", crf=20)
    sidecar = {
        "format": "eggroll-posttraining-hero-v1",
        "manifest_sha256": sha256_file(manifest_path),
        "output_sha256": sha256_file(output),
        "frames": len(frames),
        "fps": FPS,
        "duration_s": len(frames) / FPS,
        "selection_policy": (
            "first paired source-failure/adapted-success per shifted pose; "
            "first paired nominal face-down success"
        ),
        "selections": chosen,
        "full_bank_results": manifest["release_decision"],
    }
    write_json(output.with_suffix(output.suffix + ".json"), sidecar)
    return sidecar
