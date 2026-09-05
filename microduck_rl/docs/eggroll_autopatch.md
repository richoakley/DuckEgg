# EGGROLL Policy Autopatch for MicroDuck

Status: implemented and validated across successful StandUp and walking-policy repairs,
including release-aware early stopping, in a production-runtime digital twin on
2026-09-05. No physical robot has been used.

## Product claim

Autopatch is a policy-maintenance system for a robot fleet. Given an immutable deployed
ONNX policy, a capability contract, a reversible deployment change, an evaluation-only
objective, and hard release gates, it can:

1. reproduce the policy in its actual registered simulation task;
2. drive the real Rust `robotd` scheduler and policy loop;
3. detect a capability regression using behavior rather than task return;
4. prepare forward-only EGGROLL post-training over a narrow parameter scope;
5. account exactly for candidate evaluations, simulator rollouts and compute;
6. stop at the first candidate that passes the complete release gate;
7. validate a candidate on capability nodes and scheduler-transition edges; and
8. package it for the existing signed, health-gated, reversible model updater.

StandUp is the first successful patch, not a special architecture. The same evaluator and
runtime contract now cover every production policy shipped in Pollen Robotics'
`microduck` repository.

Walking is the second successful patch and the first continuous-locomotion proof. Across
two independent sealed banks, a 15 degree hidden replacement-foot geometry reduced the
deployed source to 47/64 terminal successes in production-runtime playback; the selected
DuckEgg derivative achieved 64/64, retaining every source success and repairing all 17
observed failures.

The defensible scope is a **production-runtime digital twin**. This proves execution
through production software against actual Mjlab tasks. It does not prove physical
transfer, hardware safety, unattended fleet operation, or adaptation to arbitrary future
changes.

## Four immutable contracts

Every campaign is content-addressed and separates:

- `PolicyArtifactSpec`: source ONNX hash, 61D/14D API, production slot, mode and task;
- `CapabilitySpec`: command protocol, initial-state protocol, lifecycle and behavioral
  success semantics;
- `DeploymentCondition`: reversible model, actuator, sensor, runtime, terrain or object
  mutation; and
- `PatchCampaign`: artifact plus condition, lexicographic objective, optimizer, disjoint
  banks and hard release gates.

The campaign hash binds metrics, checkpoints, exports and releases to those semantics.
Task return is diagnostic unless explicitly named in the objective. Checkpoint selection
uses the objective's ordered behavioral metrics.

## Production policy inventory

| Artifact | Capability | Runtime slot | Actual task used by source acceptance |
| --- | --- | --- | --- |
| `alpha_walking.onnx` | legged locomotion | `walk` | `Mjlab-Velocity-Flat-MicroDuck` |
| `alpha_stand.onnx` | standing/recovery/body pose | `stand` | `Mjlab-StandUp-Flat-MicroDuck` |
| `alpha_sitstand.onnx` | sit/rise transition | `sitstand` | `Mjlab-SitStand-Flat-MicroDuck` |
| `alpha_ground_pick.onnx` | ground pick | `ground_pick` | `Mjlab-GroundPick-Flat-MicroDuck` |
| `ball_kick_left.onnx` | left kick | `kick_left` | mirrored left-foot state in registered BallKick |
| `ball_kick_right.onnx` | right kick | `kick_right` | `Mjlab-BallKick-Flat-MicroDuck` |
| `roller.onnx` | roller locomotion | `walk` in roller mode | `Mjlab-Velocity-Flat-MicroDuck-Rollers` |
| `roller_crouch.onnx` | roller crouch | `ground_pick` in roller mode | `Mjlab-RollerCrouch-Flat-MicroDuck` |
| `roulade.onnx` | forward roll | `roulade` | `Mjlab-Roulade-Flat-MicroDuck` |

Training tasks with no corresponding deployed artifact are recorded as gaps rather than
invented policies: Velocity Swizzle, Roller Slope, Roller StandUp and Spin.

## Runtime authority and trace gate

Python supplies raw simulated `RobotIo` sensors, scenarios and independent task
diagnostics. Rust owns the deployed semantics:

```text
raw joints and IMU
        ↓
robotd 61D observation
        ↓
selected ONNX and exact SHA
        ↓
14D raw action
        ↓
home offset and mode-specific action scale
        ↓
head/leg low-pass filters
        ↓
safety clamp
        ↓
absolute targets actually written through RobotIo
```

In evaluation-only `--sim-eval` mode, `robotd` publishes this hidden-stage trace. The
Python auditor independently checks dimensions and finiteness, ONNX Runtime parity,
previous raw action, action scales, filters, state targets, safety targets and the exact
write observed by the simulation body. A missing or timed-out write rejects the attempt;
it is never treated as a policy outcome. Physical and fake-device startup behavior is
unchanged.

## Current evidence

### Entire production fleet

The deterministic source suite passed all nine artifacts and all eight registered graph
edges. It directly observed stand→walk→stand, sit→rise→stand, skill→stand and
roller-crouch→roller handoffs. Each accepted episode passed the full runtime trace audit.

- evidence: `/private/tmp/autopatch-source-fleet-20260831-v7/manifest.json`
- manifest SHA-256:
  `dc6d0505646ad5f2d9474b14e506022f0675d5d292938e2a67258ba9717f2e25`
- result: 9/9 capability nodes, 8/8 transition edges, zero reported graph gaps
- transport audit: the first walking attempt had one missing tick and was retained as
  rejected evidence; its clean second attempt passed

This is one seeded source-acceptance case per artifact, not a robustness estimate and not
evidence that EGGROLL has adapted all nine policies.

### StandUp reference patch through the generic engine

The existing output-layer EGGROLL derivative was replayed through the same generic paired
evaluator used by the fleet. Source and adapted policies used identical standing,
sitting, face-down and face-up cases.

| Profile | Source PPO | EGGROLL derivative |
| --- | ---: | ---: |
| nominal | 4/4 | 4/4 |
| 6.5 V, sag 0.2, lag 16 | 1/4 | **4/4** |

- source SHA-256:
  `1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b`
- derivative SHA-256:
  `bd2bfbb22d7a0942a2f83e3164c035fa887580e29e66eb28c9a809e9ed8be8db`
- evidence: `/private/tmp/autopatch-standup-ab-generic-20260831-v1/manifest.json`
- manifest SHA-256:
  `a4c6951e65fc2a3bbb749cef2a897666fe8d599bf0d282aa43de4df204f13ff8`

All 16 source/adapted runtime episodes passed without a retry. This small replay validates
the generalized workflow. The stronger prior held-out result remains lag-16 17/32→32/32
with nominal 32/32 retained, documented in `eggroll_policy_patch_lab.md`.

The generalized updater harness then activated exact source 0.9.0 and derivative 1.0.0 in
`model-stand`, recording two signature/artifact verifications, two model-API checks and two
health gates. A separate rollback process restored the exact source hash and passed another
health gate.

- activation evidence: `/private/tmp/autopatch-updater-standup-generic-v1/activation.json`
  (`9a37b38c467df215c58275e2f6fbdeaf4a993af1098f720a10c5817d3044b035`)
- rollback evidence: `/private/tmp/autopatch-updater-standup-generic-v1/rollback.json`
  (`c68f26bfc70c6949c3ba452becf0030c4f70bd99803b4b0fddf0757b7af9af7c`)

### Walking replacement-foot patch

The frozen walking campaign completed 100 generations and selected generation 85. It
modified only the final 1,806 affine parameters of the production walking actor. The
independently exported ONNX bytes exactly matched the remote export and passed numerical
parity at `4.77e-7`.

| Actual production-runtime playback | Source PPO | EGGROLL patch |
| --- | ---: | ---: |
| replacement foot, first sealed bank | 24/32 | **32/32** |
| replacement foot, independent confirmation bank | 23/32 | **32/32** |
| **replacement foot, combined** | **47/64** | **64/64** |
| nominal | **32/32** | 31/32 |

Every command speed reached 8/8 shifted successes after adaptation. The source achieved
4/8, 7/8, 8/8 and 5/8 at 0.28, 0.32, 0.36 and 0.40 m/s respectively. All 128 accepted
source/adapted episodes passed the Rust trace audit and produced videos. This validates
the exact ONNX selection, 61D observation, raw previous action, command smoothing, action
scale, low-pass filters, safety target and RobotIo write—not merely Python-side inference.

- training artifacts: `richoakley/eggroll-walking-wedge-train-20260831-v1` (private)
- local playback evidence:
  `/private/tmp/eggroll-walking-wedge-paired-evidence-v1/manifest.json`
- playback manifest SHA-256:
  `018a05616e1eb3581d5248852a17d9bc86018b4ca38381698de09a3a199fd714`
- confirmation job: `6a9671d921c5aa7c8364ab6b`
- confirmation artifacts: `richoakley/eggroll-walking-wedge-confirmation-20260901-v1`
  (private)
- confirmation manifest SHA-256:
  `3a694e6b845c203789e5979882ec3132b98f6e6baac9f46728a72bc9e44f9d4a`
- two-bank non-regression record:
  `docs/experiments/walking_wedge_gen85_two_bank_non_regression_v2.json`
- two-bank record SHA-256:
  `ffe2011d2c59a186abbcfaa6e128ffdb53cc7781be53e609ef98864162f26879`
- selected policy SHA-256:
  `f6c2378b415cbf5449e21d1bd4f1c1df72ac7817e8d4f39caa943c8c22221b5c`
- machine-readable repository record:
  `docs/experiments/eggroll_walking_autopatch_2026-09.artifacts.json`
- visual comparison:
  `docs/assets/eggroll_autopatch/walking_wedge_gen85_hero.mp4`

Absolute search cost was 51,200 candidate evaluations, 204,800 optimization rollouts,
51.2 million requested simulator steps and 13,698.7 seconds of campaign wall time on one
A10G Large. No random-search or generic-ES baseline was run, so this supports an absolute
efficiency report but no comparative sample-efficiency claim.

The current release-aware workflow no longer assumes that 100 generations are required.
Three independent seeds each reached a fully qualified profile-specific derivative at
generation 6, with a median of 3,072 candidate evaluations and 3,078,000 requested
optimization simulator steps: 6.012% of the historical walking budget. Those jobs retained
all candidate prefixes but actually completed nine generations, so they are evidence of
interaction cost to first eligibility, not an early-stop wall-time measurement. A separate
integrated campaign then trained and qualified candidates in one job and stopped at the
first complete pass, generation 5, after 2,560 candidate evaluations and 2,565,000
requested optimization steps, plus 144,000 requested qualification steps. The selected
candidate passed both production-runtime banks, ONNX parity, profile routing, signed
activation, exact source fallback and rollback. Its recorded 4,840.9 seconds covers the
qualification commands only; no comparable end-to-end campaign wall time is claimed.
Complete accounting and execution boundaries are in
`docs/experiments/eggroll_autopatch_efficiency_v1.md`.

The confirmation bank contributed 32 new cases with no seed overlap with the first bank.
The source passed 8/8, 5/8, 6/8 and 4/8 at 0.28, 0.32, 0.36 and 0.40 m/s; DuckEgg passed
8/8 at every speed. Across both banks, the source passed 47/64 and DuckEgg passed 64/64.
DuckEgg lost zero of the source's 47 successes and repaired all 17 failures. All 64
accepted confirmation episodes passed the Rust trace audit at effectively machine
precision. Together with the first bank, this closes the two-bank profile-specific
behavioral gate with 128 accepted in-profile source/adapted traces.

## One workflow

Inspect the sealed registry:

```bash
eggroll-autopatch registry
```

Run the nine-policy source acceptance suite after building `robotd`:

```bash
eggroll-autopatch evaluate-source-fleet \
  --runtime-repo ../microduck \
  --robotd ../microduck/target/debug/robotd \
  --ort-dylib /path/to/libonnxruntime.dylib \
  --seed 20260831 \
  --output-dir /private/tmp/autopatch-source-fleet
```

Resolve a campaign without launching search:

```bash
eggroll-autopatch autopatch \
  --campaign campaign.json \
  --runtime-repo ../microduck \
  --policy-slot stand \
  --capability stationary-body-control \
  --deployment calibration-6p5V-lag16 \
  --acceptance terminal-standing-v1 \
  --output-dir /private/tmp/autopatch-plan
```

The plan verifies the source bytes, campaign selectors, disjoint bank hashes and release
node/edge coverage. The current walking proof executes EGGROLL only; naive ES and random
search are explicitly excluded. Planning never authorizes or launches training.

Candidate checkpoints use a non-pickle NPZ envelope bound to the campaign hash, source
policy hash, generation, output-layer-only scope and held-out metrics. Select and export
without allowing shaped task return to override the objective:

```bash
eggroll-autopatch select-export \
  --campaign campaign.json \
  --runtime-repo ../microduck \
  --checkpoint checkpoint-000050.npz \
  --checkpoint checkpoint-000100.npz \
  --output-policy /private/tmp/derivative.onnx \
  --output-record /private/tmp/selection.json
```

The exporter structurally discovers the source output layer, changes only its weight and
bias, and independently requires ONNX parity below `1e-5`.

Run a generic paired A/B by providing a JSON list of task cases:

```bash
eggroll-autopatch evaluate-ab \
  --runtime-repo ../microduck \
  --robotd ../microduck/target/debug/robotd \
  --ort-dylib /path/to/libonnxruntime.dylib \
  --artifact alpha-stand \
  --adapted-policy /path/to/derivative.onnx \
  --profile nominal=nominal-fixed-median-v1 \
  --profile shifted=calibration-6p5V-lag16 \
  --bank /path/to/bank.json \
  --output-dir /private/tmp/autopatch-ab
```

Bind at least two disjoint paired manifests into the strict non-regression decision:

```bash
eggroll-autopatch verify-non-regression \
  --artifact alpha-walking \
  --adapted-policy /path/to/derivative.onnx \
  --manifest /path/to/paired-bank-a/manifest.json \
  --manifest /path/to/paired-bank-b/manifest.json \
  --release-scope docs/experiments/release_scopes/walking_wedge_gen85_profile_specific_v1.json \
  --output /path/to/non-regression.json
```

The release scope is separate from the immutable training campaign. It declares which
profile hashes the derivative is allowed to run on, which roles require case-by-case
source retention, the exact rollback bytes, and the fail-closed behavior for an unknown
profile. Two modes are supported:

- `profile_specific`: one derivative for one attested hardware profile. Original or
  unknown robots retain the exact source bytes.
- `multi_profile`: one derivative intended to operate across every declared profile. It
  must retain every source success on every profile.

## Search and release boundaries

The generic search engine consumes only finite candidate fitness values and invokes the
real HyperscaleES EGGROLL update; it has no task-specific behavior and no backward path.
The current default patch scope is the structurally discovered final affine layer.
Expansion to additional affine layers must be explicitly declared and cannot change the
61D/14D actor API, normalizer or frozen architecture.

The generic code retains the ability to predeclare controlled optimizer comparisons, but
the walking campaign does not execute one. Its evidence supports a controlled comparison
against DuckEgg's own frozen 51.2-million-step EGGROLL reference, not a comparative claim
against retraining or an unrun optimizer baseline. The runner hard-requires the real
HyperscaleES EGGROLL implementation so a generic ES cannot silently substitute.

A release envelope requires exact source and derivative hashes, output-layer-only proof,
independent ONNX parity below `1e-5`, the production Rust loader probe, all capability-node
and relevant scheduler-edge evidence, every campaign release gate, and an exact source
rollback hash. Release envelope v3 additionally binds a content-addressed deployment scope,
at least two independent paired production-runtime banks, and a production updater routing
attestation. A derivative is rejected if it fails even one capability case that the sealed
source passes on any required retention profile. Aggregate retention such as 31/32 is not
sufficient within that scope. Task return and continuous diagnostics cannot override a
regression.

This is a finite, behavioral guarantee: **zero observed regressions under the capability's
terminal-success semantics on the sealed banks**. It is not a mathematical guarantee over
all possible states. Known hardware-specific patches can add an operational safeguard by
activating only on an attested hardware profile while nominal robots retain the exact
source bytes.

Each artifact declares a unique `model-<artifact>` updater component; this is not derived
from reused runtime slots such as `walk` or `ground_pick`. Eligible bytes are signed by
`cargo xtask`; the real updater verifies signature, artifact, model API and health before
activation.

## Honest limitations and next proof

What exists today is a general deployment/evaluation/release substrate plus successful
repairs in two policy classes: stationary recovery/body control and continuous walking.
This is evidence for the reusable policy-maintenance pattern, not evidence that every
hardware or environment change is repairable.

Strict walking re-selection showed that all four archived source-improving candidates lost
at least one original-foot success when the derivative was replayed outside its intended
wedge-foot hardware profile. That is a cross-profile robustness result, not automatically a
release regression for a profile-specific replacement-foot patch. Generation 85 preserved
all 47 source successes and repaired all 17 source failures across two seed-disjoint
wedge-foot banks. The two-bank profile-specific behavioral gate has therefore passed.
The historical generation-85 derivative still lacks its own production routing
attestation. The later integrated candidate passed signed profile routing, activation,
exact source fallback and fresh-process rollback, proving that loop for the current
release-aware workflow. Original-foot and unknown robots still retain the exact source
bytes. Physical transfer and safety validation remain separate gates before any robot
deployment claim.
