"""Local safety and provenance tests for the Autopatch HF launcher."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from mjlab_microduck.autopatch.contracts import PatchCampaign
from mjlab_microduck.autopatch.foot_proof import (
    make_walking_proof_bank,
    walking_bank_sha256,
)
from mjlab_microduck.autopatch.walking_protocol import (
    walking_campaign_family_sha256,
)
from mjlab_microduck.eggroll.deployment import TRUNK_COM_CALIBRATION_PROFILES

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts/hf/eggroll_autopatch_hf.py"
GLOBALS = runpy.run_path(str(LAUNCHER))
build_bundle = GLOBALS["_build_bundle"]
BOOTSTRAP = GLOBALS["BOOTSTRAP"]
sys.path.insert(0, str(LAUNCHER.parent))
try:
    PREFLIGHT_GLOBALS = runpy.run_path(
        str(ROOT / "scripts/hf/eggroll_autopatch_qualification_preflight_hf.py")
    )
    COM_CALIBRATION_GLOBALS = runpy.run_path(
        str(ROOT / "scripts/hf/eggroll_autopatch_com_calibration_hf.py")
    )
    SOURCE_BEHAVIOR_GLOBALS = runpy.run_path(
        str(ROOT / "scripts/hf/eggroll_autopatch_source_behavior_hf.py")
    )
finally:
    sys.path.pop(0)
qualification_driver_source = PREFLIGHT_GLOBALS["_driver_source"]
qualification_bootstrap = PREFLIGHT_GLOBALS["_qualification_bootstrap"]
com_calibration_bootstrap = COM_CALIBRATION_GLOBALS["_calibration_bootstrap"]
build_com_calibration_bundle = COM_CALIBRATION_GLOBALS["_build_calibration_bundle"]
source_behavior_bootstrap = SOURCE_BEHAVIOR_GLOBALS["_source_behavior_bootstrap"]
build_source_behavior_bundle = SOURCE_BEHAVIOR_GLOBALS["_build_source_behavior_bundle"]


def _git(repo: Path, *arguments: str) -> None:
    subprocess.check_call(
        [
            "git",
            "-c",
            "user.name=Eggroll Test",
            "-c",
            "user.email=eggroll-test@example.invalid",
            *arguments,
        ],
        cwd=repo,
    )


def _documents(tmp_path: Path, policy_sha256: str) -> tuple[Path, Path]:
    campaign = json.loads(
        (
            ROOT
            / "docs/experiments/campaigns/walking_wedge_autopatch_release_scope_v2.json"
        ).read_text()
    )
    campaign["artifact_sha256"] = policy_sha256
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign))

    scope = json.loads(
        (
            ROOT
            / "docs/experiments/release_scopes/walking_wedge_gen85_profile_specific_v1.json"
        ).read_text()
    )
    scope["source_fallback_sha256"] = policy_sha256
    scope_path = tmp_path / "release_scope.json"
    scope_path.write_text(json.dumps(scope))
    return campaign_path, scope_path


def _qualification_documents(tmp_path: Path) -> tuple[Path, Path]:
    plan_path = (
        ROOT / "docs/experiments/qualification_plans/walking_wedge_release_v1.json"
    )
    plan = json.loads(plan_path.read_text())
    spec = {
        "spec_id": "launcher-fixture-v1",
        "commands": [
            {
                "stage": stage,
                "argv": (
                    "qualification-tool",
                    "{candidate_checkpoint}",
                    "{result_path}",
                ),
                "result_path": f"generation-{{generation}}/{stage}.json",
                "timeout_seconds": 30.0,
            }
            for stage in plan["required_stages"]
        ],
    }
    copied_plan = tmp_path / "qualification_plan.json"
    copied_plan.write_text(json.dumps(plan))
    spec_path = tmp_path / "qualification_command_spec.json"
    spec_path.write_text(json.dumps(spec))
    return copied_plan, spec_path


def test_bundle_binds_release_scope_and_requires_clean_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("committed\n")
    support = repo / "scripts/runtime_support"
    support.mkdir(parents=True)
    (support / "profile_scoped_model_activation.rs").write_text("router overlay\n")
    (support / "profile_scoped_policy_lab.rs").write_text("harness overlay\n")
    _git(repo, "add", "tracked.txt", "scripts/runtime_support")
    _git(repo, "commit", "-q", "-m", "fixture")

    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"exact-policy-bytes")
    policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
    campaign, scope = _documents(tmp_path, policy_sha256)
    qualification_plan, qualification_spec = _qualification_documents(tmp_path)
    output = tmp_path / "source.tar.gz"

    leak = repo / "untracked-secret.txt"
    leak.write_text("must not upload\n")
    with pytest.raises(RuntimeError, match="uncommitted source snapshot"):
        build_bundle(
            repo=repo,
            output=output,
            policy=policy,
            campaign_path=campaign,
            release_scope_path=scope,
            mode="train",
        )

    leak.unlink()
    manifest = build_bundle(
        repo=repo,
        output=output,
        policy=policy,
        campaign_path=campaign,
        release_scope_path=scope,
        mode="train",
        qualification_plan_path=qualification_plan,
        qualification_command_spec_path=qualification_spec,
    )
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert "microduck_rl/.artifacts/input/release_scope.json" in names
        assert "microduck_rl/.artifacts/input/qualification_plan.json" in names
        assert "microduck_rl/.artifacts/input/qualification_command_spec.json" in names
        scope_file = archive.extractfile(
            "microduck_rl/.artifacts/input/release_scope.json"
        )
        assert scope_file is not None
        bundled_scope = json.load(scope_file)

    assert manifest["source_policy_sha256"] == policy_sha256
    assert (
        manifest["release_scope_sha256"]
        == GLOBALS["ReleaseScope"].from_dict(bundled_scope).sha256
    )
    assert "--release-scope .artifacts/input/release_scope.json" in BOOTSTRAP
    assert manifest["qualification_plan_sha256"] is not None
    assert manifest["qualification_command_spec_sha256"] is not None
    assert manifest["production_runtime_inputs"]["runtime_volume_path"] == (
        "microduck-runtime-evidence-20260901-v1-837c0f59"
    )
    assert (
        manifest["production_runtime_inputs"]["profile_router_overlay_sha256"]
        == hashlib.sha256(b"router overlay\n").hexdigest()
    )
    assert "--qualification-command-spec" in BOOTSTRAP
    assert "profile_scoped_policy_lab" in BOOTSTRAP
    assert 'q.get("stop_generation")' in BOOTSTRAP
    assert "qualification_candidates/generation-{g:06d}.npz" in BOOTSTRAP
    assert "eggroll-autopatch verify-integrated-early-stop-run" in BOOTSTRAP
    assert "--max-requested-optimization-steps" in BOOTSTRAP
    assert manifest["requested_optimization_simulator_step_ceiling"] == 5_120_000


def test_smoke_verifies_optimizer_and_accounting_without_requiring_export() -> None:
    subprocess.run(["bash", "-n"], input=BOOTSTRAP, text=True, check=True)
    assert 'if [ "$RUN_MODE" = "smoke" ]; then' in BOOTSTRAP
    assert "--profile-generation 0 --profile-generation 1" in BOOTSTRAP
    assert 'if [ "$RUN_RC" -eq 0 ] && [ "$RUN_MODE" = "smoke" ]; then' in BOOTSTRAP
    assert "eggroll-autopatch verify-cuda-smoke-run" in BOOTSTRAP
    assert "--expected-candidate-evaluations 16" in BOOTSTRAP
    assert "--expected-optimization-world-rollouts 68" in BOOTSTRAP
    assert "--expected-optimization-requested-steps 17000" in BOOTSTRAP
    assert "--expected-total-world-rollouts 196" in BOOTSTRAP
    assert "--expected-total-requested-steps 49000" in BOOTSTRAP
    assert BOOTSTRAP.count('"${RELEASE_SCOPE_ARGS[@]}"') == 2
    assert 'elif [ "$RUN_RC" -eq 0 ]; then' in BOOTSTRAP
    assert '"RUN_MODE": args.mode' in LAUNCHER.read_text()


def test_qualification_preflight_scripts_are_syntactically_valid(
    tmp_path: Path,
) -> None:
    subprocess.run(
        ["bash", "-n"], input=qualification_bootstrap(), text=True, check=True
    )
    driver = tmp_path / "preflight_driver.py"
    driver.write_text(qualification_driver_source(85))
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(driver)],
        check=True,
    )
    assert "CommandQualificationBackend" in driver.read_text()
    assert "not a new training or efficiency result" in driver.read_text()

    candidate_driver = tmp_path / "candidate_driver.py"
    candidate_driver.write_text(
        qualification_driver_source(4, "candidate-qualification")
    )
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(candidate_driver)],
        check=True,
    )
    candidate_source = candidate_driver.read_text()
    assert "eggroll-autopatch-candidate-qualification-v1" in candidate_source
    assert "not physical-robot evidence" in candidate_source
    assert "nominal adapted-policy evaluation remains diagnostic" in candidate_source


def test_trunk_com_calibration_bundle_and_bootstrap_are_frozen(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("committed\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    policy = tmp_path / "alpha_walking.onnx"
    policy.write_bytes(b"exact-calibration-source")
    protocol = json.loads(
        (
            ROOT / "docs/experiments/walking_trunk_com_cross_failure_protocol_v2.json"
        ).read_text()
    )
    protocol["source"]["policy_sha256"] = hashlib.sha256(
        policy.read_bytes()
    ).hexdigest()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    bundle = tmp_path / "source.tar.gz"

    manifest = build_com_calibration_bundle(
        repo=repo,
        output=bundle,
        policy=policy,
        protocol_path=protocol_path,
        hf_hardware_flavor="a10g-large",
    )
    with tarfile.open(bundle, "r:gz") as archive:
        names = set(archive.getnames())
        protocol_file = archive.extractfile(
            "microduck_rl/.artifacts/input/calibration_protocol.json"
        )
        assert protocol_file is not None
        bundled_protocol = protocol_file.read()
    assert "microduck_rl/tracked.txt" in names
    assert "microduck_rl/.artifacts/input/source_manifest.json" in names
    assert "microduck_rl/.artifacts/input/calibration_protocol.json" in names
    assert "microduck/example_policies/alpha_walking.onnx" in names
    assert manifest["policy_episode_evaluation_ceiling"] == 160
    assert manifest["requested_simulator_step_ceiling"] == 40_000
    assert manifest["candidate_optimization_evaluation_ceiling"] == 0
    assert manifest["hf_hardware_flavor"] == "a10g-large"
    assert bundled_protocol == protocol_path.read_bytes()
    assert manifest["protocol_sha256"] == hashlib.sha256(bundled_protocol).hexdigest()

    subprocess.run(
        ["bash", "-n"], input=com_calibration_bootstrap(), text=True, check=True
    )
    bootstrap = com_calibration_bootstrap()
    assert "eggroll-autopatch calibrate-trunk-com" in bootstrap
    assert "eggroll-autopatch verify-trunk-com-calibration" in bootstrap
    assert "--source-manifest .artifacts/input/source_manifest.json" in bootstrap
    assert "--base-seed 20293001" in bootstrap
    assert "--device cuda:0" in bootstrap
    assert "--attempts 1" in bootstrap
    assert "/inputs/reference_policies" not in bootstrap
    assert "Calibration bundles their exact source policy" in bootstrap


def test_payload_calibration_bundle_preserves_exact_protocol_bytes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("committed\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    policy = tmp_path / "alpha_walking.onnx"
    policy.write_bytes(b"exact-payload-calibration-source")
    protocol = json.loads(
        (
            ROOT
            / "docs/experiments/walking_trunk_payload_cross_failure_protocol_v1.json"
        ).read_text()
    )
    protocol["source"]["policy_sha256"] = hashlib.sha256(
        policy.read_bytes()
    ).hexdigest()
    protocol_path = tmp_path / "payload-protocol.json"
    protocol_path.write_text(json.dumps(protocol, separators=(",", ":")))
    bundle = tmp_path / "payload-source.tar.gz"

    manifest = build_com_calibration_bundle(
        repo=repo,
        output=bundle,
        policy=policy,
        protocol_path=protocol_path,
        hf_hardware_flavor="a10g-large",
    )
    with tarfile.open(bundle, "r:gz") as archive:
        bundled = archive.extractfile(
            "microduck_rl/.artifacts/input/calibration_protocol.json"
        )
        assert bundled is not None
        bundled_raw = bundled.read()
    assert bundled_raw == protocol_path.read_bytes()
    assert manifest["schema"] == (
        "eggroll-autopatch-trunk-payload-calibration-source-v1"
    )
    bootstrap = com_calibration_bootstrap(
        calibrate_command="calibrate-trunk-payload",
        verify_command="verify-trunk-payload-calibration",
        base_seed=20794001,
    )
    subprocess.run(["bash", "-n"], input=bootstrap, text=True, check=True)
    assert "eggroll-autopatch calibrate-trunk-payload" in bootstrap
    assert "eggroll-autopatch verify-trunk-payload-calibration" in bootstrap
    assert "--base-seed 20794001" in bootstrap


def test_source_behavior_bootstrap_is_source_only_and_syntactically_valid() -> None:
    bootstrap = source_behavior_bootstrap()
    subprocess.run(["bash", "-n"], input=bootstrap, text=True, check=True)
    assert "eggroll-autopatch capture-walking-source-behavior" in bootstrap
    assert "--calibration-validation .artifacts/input/calibration_validation.json" in (
        bootstrap
    )
    assert "train-walking-campaign" not in bootstrap
    assert "qualify-release-stage" not in bootstrap
    assert "/inputs/reference_policies" not in bootstrap
    assert "Source-behavior capture bundles its exact source policy" in bootstrap


def test_source_behavior_bundle_binds_campaign_calibration_and_exact_cost(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tracked.txt").write_text("committed\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    policy = tmp_path / "alpha_walking.onnx"
    policy.write_bytes(b"source-behavior-policy")
    source_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
    campaign_document = PatchCampaign.from_json(
        (
            ROOT
            / "docs/experiments/campaigns/walking_wedge_autopatch_release_scope_v2.json"
        ).read_text()
    ).canonical_dict()
    profile = TRUNK_COM_CALIBRATION_PROFILES[2]
    campaign_document["campaign_id"] = "alpha-walking-trunk-com-bundle-test-v1"
    campaign_document["optimizer"]["seed"] = 21_000_001
    campaign_document["optimizer"]["generations"] = 9
    campaign_document["artifact_sha256"] = source_sha256
    campaign_document["condition"] = {
        "condition_id": profile.name,
        "adapter": "mjlab-trunk-com-shift-profile-v1",
        "parameters": [
            ["profile_name", profile.name],
            ["profile_sha256", profile.sha256],
            ["body", "trunk_base"],
            ["offset_x_m", profile.offset_m[0]],
        ],
        "hidden_from_actor": True,
        "description": "test fixture",
    }
    campaign_document["calibration_bank_sha256"] = walking_bank_sha256(
        make_walking_proof_bank(
            base_seed=20293001,
            episodes_per_command=8,
            prefix="calibration-trunk-com",
        )
    )
    campaign_document["held_out_bank_sha256"] = walking_bank_sha256(
        make_walking_proof_bank(
            base_seed=20393004,
            episodes_per_command=8,
            prefix="selection-trunk-com",
        )
    )
    campaign = PatchCampaign.from_dict(campaign_document)
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign.canonical_dict()))
    calibration_path = tmp_path / "calibration_validation.json"
    calibration_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "selected_profile_sha256": profile.sha256,
                "source_policy_sha256": source_sha256,
                "hf_hardware_flavor": "a10g-large",
            }
        )
    )
    bundle = tmp_path / "source.tar.gz"

    manifest = build_source_behavior_bundle(
        repo=repo,
        output=bundle,
        policy=policy,
        campaign_path=campaign_path,
        calibration_validation_path=calibration_path,
        hf_hardware_flavor="a10g-large",
    )
    assert manifest["capture_campaign_sha256"] == campaign.sha256
    assert manifest["campaign_family_sha256"] == walking_campaign_family_sha256(
        campaign
    )
    assert manifest["activation_profile_sha256"] == profile.sha256
    assert manifest["policy_episode_evaluation_ceiling"] == 64
    assert manifest["requested_simulator_step_ceiling"] == 16_000
    assert manifest["candidate_optimization_evaluation_ceiling"] == 0
    assert manifest["hf_hardware_flavor"] == "a10g-large"
    with tarfile.open(bundle, "r:gz") as archive:
        names = set(archive.getnames())
    assert "microduck_rl/.artifacts/input/source_manifest.json" in names
    assert "microduck_rl/.artifacts/input/campaign.json" in names
    assert "microduck_rl/.artifacts/input/calibration_validation.json" in names


def test_bundle_rejects_release_scope_for_another_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("committed\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"exact-policy-bytes")
    campaign, scope = _documents(
        tmp_path, hashlib.sha256(policy.read_bytes()).hexdigest()
    )
    scope_document = json.loads(scope.read_text())
    scope_document["source_fallback_sha256"] = "0" * 64
    scope.write_text(json.dumps(scope_document))

    with pytest.raises(ValueError, match="fallback does not match"):
        build_bundle(
            repo=repo,
            output=tmp_path / "source.tar.gz",
            policy=policy,
            campaign_path=campaign,
            release_scope_path=scope,
            mode="train",
        )


def test_v2_requires_scope_but_historical_v1_forbids_and_omits_it(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("committed\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"exact-policy-bytes")
    policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()

    v2_campaign, scope = _documents(tmp_path, policy_sha256)
    with pytest.raises(ValueError, match="v2 release-scope objective requires"):
        build_bundle(
            repo=repo,
            output=tmp_path / "missing-scope.tar.gz",
            policy=policy,
            campaign_path=v2_campaign,
            release_scope_path=None,
            mode="train",
        )

    historical = json.loads(
        (
            ROOT / "docs/experiments/campaigns/walking_wedge_autopatch_v1.json"
        ).read_text()
    )
    historical["artifact_sha256"] = policy_sha256
    historical_path = tmp_path / "historical.json"
    historical_path.write_text(json.dumps(historical))
    with pytest.raises(ValueError, match="historical campaign objectives forbid"):
        build_bundle(
            repo=repo,
            output=tmp_path / "unexpected-scope.tar.gz",
            policy=policy,
            campaign_path=historical_path,
            release_scope_path=scope,
            mode="train",
        )

    output = tmp_path / "historical.tar.gz"
    manifest = build_bundle(
        repo=repo,
        output=output,
        policy=policy,
        campaign_path=historical_path,
        release_scope_path=None,
        mode="train",
    )
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "microduck_rl/.artifacts/input/release_scope.json" not in names
    assert manifest["release_scope_sha256"] is None
    assert "if [ -f .artifacts/input/release_scope.json ]; then" in BOOTSTRAP
