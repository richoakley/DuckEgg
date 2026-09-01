# EGGROLL Policy Autopatch deployment

This is the production deployment boundary for an EGGROLL-derived MicroDuck policy. The
training repository owns campaign semantics and actual-environment evidence; this runtime
repository owns policy loading, scheduling, signed model components, health gates and
rollback.

Autopatch is policy-agnostic. The first released derivative is the StandUp policy, but the
same mechanism supports every configured model slot. This document describes a
production-runtime digital twin and release mechanism, not a physical-robot result or
permission to operate hardware.

## Runtime contract

In evaluation-only `--sim-eval` mode, Python supplies raw `RobotIo` sensors from the
registered Mjlab task. The real Rust loop retains authority over:

- the exact 61D observation;
- selected network and policy SHA-256;
- previous raw action;
- ONNX inference;
- mode-specific action scaling and home offsets;
- head and leg low-pass filters;
- safety clamps; and
- absolute targets written through `RobotIo` at 50 Hz.

`RobotState.policy_trace` exposes these hidden stages only in `--sim-eval`. Normal physical
and fake-device paths leave it absent. The evaluator rejects a missing write, a policy hash
mismatch or a divergence above its declared tolerance.

The canonical registry and evaluator live in the adjacent `microduck_rl` repository under
`mjlab_microduck.autopatch`. Its nine-policy source suite currently covers walking,
standing/recovery, sit/rise, ground-pick, left/right kicks, roller locomotion,
roller-crouch and roulade, plus the scheduler edges connecting them.

## Package an eligible derivative

An Autopatch release envelope must identify the sealed source, exact derivative, allowed
patch scope, campaign hash, ONNX parity, production-loader probe, capability-node tests,
all relevant scheduler-edge tests, release gates and rollback target.

Package the model into the unique updater component declared by its artifact. Do not infer
this solely from the runtime slot: walking/roller and ground-pick/roller-crouch deliberately
reuse runtime slots but require distinct update components.

```bash
cargo xtask package-model \
  --version 1.0.0 \
  --channel model-<artifact> \
  --policy /secure/evidence/derivative.onnx \
  --evidence-manifest /secure/evidence/manifest.json \
  --runtime-verification /secure/evidence/runtime_verification.json \
  --revision <source-revision> \
  --model-api 1 \
  --out dist/model-<artifact>-1.0.0
```

The channel must start with `model-`. `package-model` rejects a zero model API, evidence
for different policy bytes, a policy whose hash differs from the manifest, or a runtime
probe that did not pass through `duck_control::policy::Policy::load`.

Sign with the normal release key outside either source tree:

```bash
cargo xtask sign --dir dist/model-<artifact>-1.0.0 --key /secure/path/release.key
```

Developer keys must remain separately trusted and confined to developer boards.

## Provision and activate a slot

Configure a separate updater component and point `robotd` at its current symlink. For
example, the standing slot uses:

```toml
[policy]
stand = "/opt/robot/model/stand/current/policy.onnx"
```

Install the independently packaged source version before the derivative so rollback has a
real previous release. The daemon currently reads policy paths at startup, so switching a
model component requires a `robotd` restart and socket health gate; live policy reload is
not implemented.

The generalized updater proof accepts any safe `model-<artifact>` component:

```bash
cargo run -p updater --example policy_patch_lab -- activate \
  --root /private/tmp/autopatch-updater \
  --source /path/to/source.onnx \
  --adapted /path/to/derivative.onnx \
  --component model-<artifact>

cargo run -p updater --example policy_patch_lab -- rollback \
  --root /private/tmp/autopatch-updater
```

The first command uses the real updater engine to verify and activate exact source then
derivative bytes. The second starts a separate process, reads persisted updater state and
verifies that rollback restored the exact source SHA-256.

On a board, use `robotctl update apply/select/rollback` and confirm both updater component
version and the `policy loaded` SHA in `robotd` logs. Never infer the active policy from a
filename.

## StandUp reference result

The first derivative changes only the final affine layer of `alpha_stand.onnx` (1,806
parameters). Prior 32-world actual-environment evaluation improved terminal success under
the selected lag-16/low-voltage profile from 17/32 to 32/32 while retaining nominal 32/32.
The generic evaluator independently replayed a four-pose bank as nominal 4/4→4/4 and
lag-16 1/4→4/4.

- source SHA-256:
  `1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b`
- derivative SHA-256:
  `bd2bfbb22d7a0942a2f83e3164c035fa887580e29e66eb28c9a809e9ed8be8db`

Task return was diagnostic. Terminal stable-standing success and nominal retention were the
release criteria.

## Physical-operation boundary

Simulation completion is not permission to operate hardware. A physical A/B requires
explicit authorization, a supported and current-limited robot, named operator and
observer, reachable power cut, predeclared stop criteria, source policy installed as the
rollback target, and confirmation of active component and SHA after every switch.
