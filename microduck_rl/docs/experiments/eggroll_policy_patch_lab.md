# EGGROLL Policy Patch Lab

Status: **implemented and smoke-validated in a production-runtime digital twin**

Evidence date: 2026-08-31

## Product claim

The Policy Patch Lab demonstrates a closed deployment loop, not only a training
curve: EGGROLL can detect a specific failure of an already-deployed robotics
policy, optimize a narrow patch using forward evaluation of a non-differentiable
success criterion, and deliver that patch through the robot's real model-update
software with validation and exact rollback.

The defensible scope is a **production-runtime digital twin**. The simulation
uses the actual registered `Mjlab-StandUp-Flat-MicroDuck` task, while the policy
loop is the Rust runtime from Pollen Robotics' official
[`microduck`](https://github.com/pollen-robotics/microduck) repository. There has
been no physical-robot execution, transfer proof, safety certification,
on-device optimization, or fleet deployment.

## What runs where

```text
Mjlab StandUp physics
  raw joint/IMU sensors
          ↓
Rust RobotIo simulation transport
          ↓
real robotd loop at 50 Hz
  61D observation → ONNX → 14D action
  previous raw action + home offsets + safety clamp
          ↓
absolute joint targets
          ↓
Mjlab StandUp physics and task diagnostics
```

Rust—not the Python evaluator—owns the deployed observation construction,
ONNX inference, previous-action state, action processing, safety range clamp,
and control-loop timing. Python owns the digital twin, raw-sensor transport,
deterministic scenarios, task-semantic diagnostics, and independent audit.
The generic raw-XML body server remains a transport diagnostic only and is not
used as behavioral evidence.

The evaluation-only `--sim-eval` path bypasses homing, seated-boot, and IMU
warm-up operations that are inappropriate for deterministic episode replay. It
does not replace the production policy loop, and it does not change the normal
physical or fake-device startup paths.

## Frozen actor contract

| Observation block | Indices | Semantics |
| --- | ---: | --- |
| gyro | 0:3 | raw IMU angular velocity |
| projected gravity | 3:6 | gravity in body frame |
| joint position | 6:20 | 14 non-mouth joints relative to production home pose |
| joint velocity | 20:34 | 14 non-mouth joint velocities |
| previous action | 34:48 | previous raw ONNX output, zero on reset |
| twist command | 48:51 | all zero for StandUp |
| head command | 51:55 | all zero for StandUp |
| body XY command | 55:57 | all zero for StandUp |
| body Z/roll/pitch | 57:60 | all zero for StandUp |
| body yaw | 60:61 | zero for StandUp |

The output is 14D. Action scale is 1.0 and absolute targets are the production
home pose plus the raw action, followed by the runtime actuator-range clamp.
The mouth is outside the policy's 14 joints and retains the runtime's separate
closed-mouth overlay.

The task-internal action-history tensor can differ after a safety clamp because
the physics receives the applied action. The deployed actor correctly observes
the previous **raw** policy output. The parity audit treats these as two named
stages instead of forcing an incorrect equality.

## Step-by-step parity gate

Every episode records raw sensor fixtures, task actor observations, Rust actions,
absolute targets, task diagnostics, runtime logs, timing, and policy/runtime
SHA-256 identities. The verifier independently replays the entire trace through
the Rust observation types and ONNX loader. It gates:

- every sensor-derived observation feature;
- zero previous action on reset and the subsequent raw-action chain;
- all command slots;
- Rust versus ONNX raw action;
- proposed `home + action` targets;
- safety-applied target formulas;
- the 14D action and absolute targets that actually crossed `RobotIo`; and
- first divergence, with step, feature, block, and both values.

The complete smoke pack passed all gated comparisons at zero reported maximum
error. Safety clamping was not hypothetical: 14 of 18 replayed episodes reached
the clamp, with as many as 53 clamped control steps. The separately reported
task-history difference and five-degree mouth overlay are expected consequences
of the named contracts, not ignored parity failures.

## Paired behavioral proof

The one-command smoke used seed `20260901`, one episode from each of standing,
sitting, face-down, and face-up, and identical source/adapted banks:

| Deployment profile | Source PPO | EGGROLL patch |
| --- | ---: | ---: |
| nominal | 4/4 terminal successes | 4/4 terminal successes |
| 6.5 V, sag 0.2, lag 16 | 0/4 terminal successes | **4/4 terminal successes** |

This four-world run validates the software and evidence path. The statistical
behavioral result remains the earlier independent 32-world actual-environment
replay: lag-16 source 17/32 versus adapted 32/32, and nominal source/adapted both
32/32. Task return is diagnostic only; terminal stable-standing success is the
release criterion.

The deterministic hero is the lexicographically first paired shifted scenario
where source fails and adapted succeeds. In the validated smoke it is a
face-down recovery. The pack also contains all 16 source/adapted, profile, and
pose episode videos plus machine-readable diagnostics.

## Real update and rollback proof

The workflow packages the source as `model-stand` 0.9.0 and the patch as 1.0.0,
then calls the actual updater engine. The engine verifies the signed manifest,
artifact hash, model API, and health gate before committing. The live policy
slot contains exact adapted SHA-256
`bd2bfbb22d7a0942a2f83e3164c035fa887580e29e66eb28c9a809e9ed8be8db`.

It then evaluates the policy through that live slot, starts a fresh updater
process, rolls back, verifies exact source SHA-256
`1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b`,
and replays the identical scenario. The adapted version succeeds; the restored
source version fails. This proves both byte identity and the corresponding
behavioral reversal.

## Reproduce

Build the Rust runtime and proof helpers once:

```bash
cd ../microduck
cargo build -p robotd
cargo build -p duck-control --example policy_probe
cargo build -p updater --example policy_patch_lab
```

Then, from `microduck_rl`, create the complete local evidence pack:

```bash
PYTHONPATH=src .venv/bin/python scripts/eggroll_patch_lab.py \
  --output-dir /private/tmp/eggroll-policy-patch-lab \
  --episodes-per-pose 1 \
  --seed 20260901 \
  --video
```

The output directory contains `manifest.json`, a human-readable `README.md`,
`eggroll_policy_patch_lab_hero.mp4`, per-episode MP4s, traces, parity reports,
and updater evidence. Generated videos, traces, and release stores belong in an
external evidence directory such as `/private/tmp` or private artifact storage,
not Git.

The final local audit pack is
`/private/tmp/eggroll-policy-patch-lab-final`. Its manifest SHA-256 is
`d85681b9ee485054a379d9befdf814725f763979f85d5e478fba4b0f803891b3`.
The 501-frame, 10.02-second hero SHA-256 is
`1c7b823c767fe6803033835c7293d69187efb6c1a02eb809d67d4c67b1638f16`.
These paths identify the local evidence generated on the stated date; the
reproduction command and content hashes are the durable identifiers.

## Next calibrated objective

An evaluation-only asymmetric fault now reduces only the left knee and ankle
torque authority. The frozen 25% profile produces source success of 17/32:
face-down 3/8, face-up 8/8, sitting 2/8, and standing 4/8. Its exact profile and
bank hashes are frozen in code and in
`eggroll_next_objective_asymmetric_leg.md`. This is a useful second deployment
failure, but no optimization or expensive baseline has been approved or run.

The comparison against EGGROLL, full-vector naive ES, random search, and a
privileged differentiable upper bound is predeclared in
`eggroll_asymmetric_optimizer_comparison.md`.
