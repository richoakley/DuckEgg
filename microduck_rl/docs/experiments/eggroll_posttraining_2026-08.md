# EGGROLL output-layer post-training experiment record

Status: **complete; scoped go**

Decision date: 2026-08-30

## Decision

The experiment succeeds on its predeclared claim. Starting from the exact
production PPO actor, EGGROLL learned a materially more robust closed-loop
StandUp policy using forward evaluations of a non-differentiable terminal
success objective. It changed only the actor's 1,806-parameter output layer,
retained terminal success under the nominal profile, exported as the same ONNX
graph contract, and reproduced on a fresh paired bank in the actual registered
environment.

The clearest result is the independent CUDA replay under the calibrated hidden
deployment shift:

- production source policy: **17/32 terminal successes**;
- EGGROLL-adapted policy: **32/32 terminal successes**;
- nominal source policy: **32/32 terminal successes**; and
- nominal adapted policy: **32/32 terminal successes**.

This is a go for demonstrating EGGROLL as a black-box policy post-training
method. It is not evidence that the tested formulation can discover the full
PPO-size StandUp policy from random initialization, replace PPO generally, or
transfer to hardware without the normal robot safety and sim-to-real checks.

## Question tested

Can EGGROLL adapt a competent deployed PPO policy to a hidden actuator condition
using scalar rollout evaluation only, while preserving nominal task success and
the production inference contract?

The hidden condition was fixed 6.5 V battery voltage, 0.2 voltage-sag gain, and
16 steps of actuator-command lag. None of those values was added to the actor's
61D observation. Success required a supported, height- and orientation-qualified
stand held through the terminal window; registered task return was diagnostic
only.

## Frozen experiment contract

| Item | Value |
| --- | --- |
| Source actor | `61 -> 512 -> 256 -> 128 -> 14` production ONNX |
| Source SHA-256 | `1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b` |
| Frozen parameters | normalizer and first three `Gemm + ELU` blocks |
| Trainable parameters | final `14 x 128` weight and 14D bias, 1,806 total |
| Task | `Mjlab-StandUp-Flat-MicroDuck` |
| Reset categories | equal standing, sitting, face-down, and face-up |
| Objective | terminal stable-standing lexicographic ordering |
| Retention gate | nominal terminal success within five points of source |
| EGGROLL | population 512, rank 4, sigma 0.015, Adam 0.003 |
| Run | 100 generations, fresh balanced CRN bank each generation |
| Selection | fixed 64-episode shifted bank and 32-episode nominal bank |

No reward weight, observation normalization, actor architecture, action
semantics, PPO code, task reset, or environment contract changed.

## Calibration and smoke gates

CUDA calibration selected `calibration-6p5V-lag16`, with profile SHA-256
`3b55ad4ff9c3d7a400ad9c94b2a102a7210be203c7db84a9a94ce3d15e3b5f81`.
The source policy achieved 12/32 terminal successes at this point: standing 5/8,
face-up 4/8, face-down 2/8, and sitting 1/8. The next ladder point, lag 20,
was catastrophic at 0/32, so lag 16 was the hardest admissible condition.

The corrected one-generation smoke produced 16 distinct candidate fitnesses, a
non-zero parameter delta of 0.12749, 4/4 nominal terminal successes, a
round-trippable checkpoint, and an independently reproduced ONNX export.

## Training result

The source policy began at 26/64 shifted terminal successes and 32/32 nominal
successes on the fixed selection banks. Progress was not inferred from return:

| Completed generation | Shifted terminal success | Nominal terminal success |
| ---: | ---: | ---: |
| source | 26/64 | 32/32 |
| 5 | 33/64 | 31/32 |
| 30 | 48/64 | 32/32 |
| 35 | 54/64 | 32/32 |
| 55 | 58/64 | 32/32 |
| 60 | 61/64 | 32/32 |
| 75 | 63/64 | 32/32 |
| 80 | 64/64 | 32/32 |
| 95 | 64/64 | 32/32 |
| 100, selected | 64/64 | 32/32 |

Generation 100 remained perfect on both banks and superseded generation 95 on
the lexicographic secondary metrics, including mean terminal hold. Its
checkpoint SHA-256 is
`ceaf5ef1cb5becff991aebcd298541ba34f6d76d0bef4f198f363cae81c8b4cc`.

The remote collection-time exporter measured maximum absolute runtime error
`1.34e-5` and correctly made the job wrapper report an error because that was
above the strict `1e-5` gate. The complete checkpoint had already been uploaded.
Independent export on the deployment host measured `9.5367431640625e-7`; the
resulting ONNX SHA-256 is
`bd2bfbb22d7a0942a2f83e3164c035fa887580e29e66eb28c9a809e9ed8be8db`.
That exact ONNX then completed both CUDA replay jobs, so the remote export
warning is recorded as a runtime-specific numerical-check discrepancy rather
than hidden training success.

## Fresh paired actual-environment replay

All four successful replay jobs used seed `20260901`, eight episodes per pose,
the actual registered task on CUDA, and 301-frame videos for every episode.
Within each profile, source and adapted bank payloads and hashes matched exactly.

| Profile and policy | Terminal | Stable hold | Final trunk z | Final upright cosine | Upright time | Task return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lag 16 source | 17/32 | 2.556 s | 0.0996 m | 0.8481 | 3.602 s | 35.94 |
| lag 16 adapted | **32/32** | **4.369 s** | **0.1082 m** | **0.9944** | **5.225 s** | 33.67 |
| nominal source | 32/32 | 5.724 s | 0.1151 m | 0.99994 | 5.796 s | 65.24 |
| nominal adapted | 32/32 | 4.985 s | 0.1098 m | 0.99591 | 5.492 s | 39.27 |

The adapted policy therefore retained the required nominal terminal capability
but did not preserve the source policy's nominal return or exact standing style.
That distinction limits the claim to the declared capability and retention gate.

### Shifted success by reset pose

| Reset pose | Source | Adapted |
| --- | ---: | ---: |
| standing | 3/8 | **8/8** |
| sitting | 2/8 | **8/8** |
| face-down | 5/8 | **8/8** |
| face-up | 7/8 | **8/8** |

Both nominal policies achieved 8/8 from every reset pose. Mean shifted maximum
trunk height was 0.1211 m for source and 0.1127 m for adapted; peak height alone
did not predict terminal success. Mean time to recovery improved from 2.821 s to
1.558 s under lag 16.

The video sets contain 128 valid 6.02-second MP4s in total. Every file has 301
frames. A paired spot check of a source failure from face-down showed both
policies rising transiently; the source subsequently lost the qualified stand,
while the adapted policy remained supported and upright through the terminal
window. This agrees with the episode diagnostics and is why transient recovery
was not treated as success.

## Why this is evidence for EGGROLL

The replay isolates the causal change: identical task code, hidden actuator
profile, reset seeds, commands, episode horizon, observations, actions, and
success semantics; the only behavioral difference is the EGGROLL-optimized
output weight and bias. The largest improvement occurred on the weak reset
categories rather than only standing starts.

Most importantly, shifted mean task return fell from 35.94 to 33.67 while
terminal success rose from 17/32 to 32/32. The capability improvement therefore
cannot be explained by selecting a higher original reward. It demonstrates the
specific advantage under test: forward-only policy learning against an explicit
non-differentiable deployment objective.

## Operational findings

- Training, checkpointing, and all 100 generations completed despite the final
  job wrapper's export-gate status.
- Nominal replay initially exposed an evaluator bug: it used the fixed four-step
  lag as the compiled buffer capacity, while the delay runtime requires capacity
  of at least six. Commit `4605d74` sizes the buffer to six but still applies the
  exact four-step nominal lag; focused regression tests pass.
- Actual-environment replay, not the local viewer and not registered return, is
  the behavioral authority.
- The output-layer subspace was sufficient for this adaptation. This does not
  reverse the archived no-go on discovering recovery from a random full-sized
  actor.

## Production-runtime digital-twin proof

The follow-on Policy Patch Lab closes the gap between actual-environment replay
and deployment software. The actual StandUp environment now supplies raw sensors
to the Rust runtime from Pollen Robotics' official `microduck` repository; Rust
owns the 61D observation, ONNX call, prior raw action, home-pose target conversion,
safety clamp, and 50 Hz timing. Absolute targets return through `RobotIo` to the
same registered task.

On a deterministic one-per-pose smoke bank, source/adapted terminal success was
0/4 versus 4/4 under lag 16 and 4/4 versus 4/4 under nominal conditions. Every
gated step-level parity error was zero. The real signed updater engine activated
the exact adapted bytes behind its health gate, and a fresh rollback process
restored the exact source SHA and its failing behavior on the same scenario.

The complete architecture, contract, reproduction command, and limits are in
[the Policy Patch Lab record](eggroll_policy_patch_lab.md). This adds deployment
software evidence but does not add a physical-robot claim.

## Recommendation

Proceed with the ambitious demo as a paired policy-adaptation story:

1. show the same lag-16 seeds side by side, especially sitting and standing
   source failures followed by adapted terminal success;
2. show the nominal pair to demonstrate retained terminal capability;
3. state the black-box terminal objective and the lower task-return result
   explicitly; and
4. treat deployment to physical Microduck as a separate safety-gated validation,
   beginning with supported, current-limited trials and the normal production
   rollback path.

Do not present this as from-scratch replacement of PPO. The defensible claim is
stronger and more specific: EGGROLL can perform RL-like closed-loop policy
post-training against an objective that need not be differentiable or aligned
with the original training reward.

## Authoritative artifacts

The private job IDs, repositories, hashes, banks, and summary digests are listed
in [the artifact index](eggroll_posttraining_2026-08.artifacts.json). The promoted
generation-100 ONNX, evidence manifest, runtime verifications, unsigned source and
adapted `model-stand` bundles, and deterministic hero are now retained in the active
repository under `policies/eggroll_posttraining/alpha_stand_lag16_v1` and
`docs/assets/eggroll_posttraining`. The complete 128 raw videos and checkpoints remain
in the private repositories rather than being committed to source control.
