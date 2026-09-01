# Predeclaration: asymmetric left-leg StandUp derivative

Status: **evaluator implemented; profile and bank frozen; optimization not approved or launched**

Predeclaration date: 2026-08-30

Calibration amendment: 2026-08-31, recorded before evaluating the fallback ladder

## Product question

Can the same sealed production PPO StandUp policy be post-trained into a second,
independently validated derivative that compensates for an asymmetric actuator
degradation it cannot observe, while retaining nominal standing and the exact 61D/14D
ONNX contract?

This is the best second demonstration because it is visibly different from the first
global voltage/latency shift. The unadapted robot should rise unevenly, rotate or collapse
toward the weakened side; a successful derivative should discover a compensating recovery
strategy. It tests the reusable derivative model rather than extending the first result by
another point on the same lag curve.

No optimization may start until the evaluator implementation, calibration result, hashes,
smoke test, expected compute cost, and bounded approval are reviewed.

## Frozen policy and search contract

| Item | Predeclared value |
| --- | --- |
| Base policy | production `alpha_stand.onnx`, SHA-256 `1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b` |
| Actor contract | `float32[1,61] -> float32[1,14]`; baked normalizer unchanged |
| Frozen scope | normalizer and first three `Gemm + ELU` blocks |
| Candidate scope | final `14 x 128` weight and 14D bias only, 1,806 parameters |
| Task | actual registered `Mjlab-StandUp-Flat-MicroDuck` |
| Commands | fixed zero 13D command, identical to the first proof |
| Reset categories | balanced standing, sitting, face-down, face-up |
| EGGROLL hypothesis | population 512, rank 4, sigma 0.015, Adam 0.003, 100 generations |
| Task return | diagnostic only; never a selector or release gate |

The first EGGROLL derivative is evidence that this search scope can adapt feedback
behavior, but it is not the starting policy for this test. Starting both derivatives from
the same sealed PPO bytes makes their provenance and rollback unambiguous.

## Evaluation-only deployment condition

Add a versioned `AsymmetricActuatorProfile`, separate from the existing global
`DeploymentProfile`. Under otherwise nominal hardware conditions (7.35 V, sag gain 0.10,
four-step lag), cap the simulated motor-torque authority of `left_knee` and `left_ankle`
to a fixed fraction of their normal BAM M6 limits. Do not expose the fraction or affected
joints in the actor observation.

The initial predeclared ladder was implemented and smoke-tested with one episode per pose:
`0.85`, `0.75`, `0.65`, and `0.55` all retained 4/4 terminal success. It therefore
contained no useful deployment gap and no level was selected. This was a calibration
no-go, not an optimization result.

Before evaluating any new fraction, this amendment predeclares one non-overlapping
fallback ladder: `0.45`, `0.35`, `0.25`, and `0.15`. Smoke each fraction with one episode
per pose. If at least one fraction is non-catastrophic, run the unchanged source-policy
calibration bank of eight episodes per pose across the fallback ladder. Select the hardest
fraction satisfying all of:

- aggregate source terminal success from 10% through 90%;
- at least one source terminal success in every reset category;
- at least a ten-point gap from the same bank under the nominal profile; and
- identical reset, friction, mass, sensor-noise, command, and delay realizations across
  fractions wherever the profile allows.

If no fallback fraction qualifies, stop. There is no third ladder and no interpolation.
Do not pick a visually dramatic catastrophic condition or retune EGGROLL to compensate.
Unit tests must prove that only the named left-side actuator limits change, that
applying/restoring profiles does not accumulate state, and that the profile payload and
hash include joint names and effectiveness.

## Calibration result and frozen benchmark

The fallback smoke produced 4/4 nominal success, 4/4 at 45%, 4/4 at 35%, 1/4 at
25%, and 0/4 at 15%. The predeclared 32-world calibration then selected the 25%
condition without interpolation:

| Profile | Overall | Face-down | Face-up | Sitting | Standing | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| nominal | 32/32 | 8/8 | 8/8 | 8/8 | 8/8 | reference |
| 45% authority | 32/32 | 8/8 | 8/8 | 8/8 | 8/8 | too easy |
| 35% authority | 32/32 | 8/8 | 8/8 | 8/8 | 8/8 | too easy |
| **25% authority** | **17/32** | **3/8** | **8/8** | **2/8** | **4/8** | **selected** |
| 15% authority | 2/32 | 1/8 | 1/8 | 0/8 | 0/8 | catastrophic |

The selected profile is `asymmetric-left-knee-ankle-25pct-v1`, SHA-256
`4ff1208b44ac154772939fb07c2902b4c902e47b0b51def94fb5d163fa8a925a`.
Its seed-`20261001`, eight-episodes-per-pose bank is frozen in code with SHA-256
`6df62d7ef310e06d7d44437c8e3ac0a9c423d3b12d5c4bd97cdf6e0274dbe4fb`.
Every intended start remains recoverable, while the 53.125% aggregate source
success leaves room to distinguish a real repair from noise or ceiling effects.

This result only calibrates the deployment objective. No candidate policy was
evaluated, no EGGROLL update occurred, and no new training job is authorized by
this record.

## Black-box objective

Candidate order is lexicographic:

1. terminal stable-standing successes;
2. worst-pose terminal success count;
3. total terminal qualified-hold time;
4. worst-scenario terminal progress;
5. mean terminal progress;
6. negative mean time to first stable recovery among successful episodes; and
7. post-success upright/support quality.

Terminal success retains the first proof's task semantics: target-height tolerance,
orientation threshold, leg-pose threshold, foot-support threshold, and a one-second stable
hold through the terminal window. Maximum height, transient upright time, original task
return, and a momentary recovery are diagnostics only.

## Episode-bank separation

Use trainer seed `20260920` and preserve the existing deterministic rules:

| Role | Size | Seed rule |
| --- | ---: | --- |
| optimization | 4 worlds per generation | `20260920 + generation_index * 10007` |
| checkpoint selection | 64 worlds | `20260920 + 1000003` |
| nominal retention | 32 worlds | `20260920 + 2000003` |
| final asymmetric held-out | 32 worlds | base seed `20261001` |
| final nominal held-out | 32 worlds | base seed `20261002` |

The manifest must record each bank payload and SHA-256 and reject any shared scenario
fingerprint across roles. Source and candidate must use identical banks within each paired
comparison.

## Selection, retention, and release gates

A checkpoint is eligible only when its shifted lexicographic key improves and its fixed
nominal terminal-success rate is at least 31/32. Release additionally requires, on the
fresh final banks:

- at least 28/32 asymmetric-profile terminal successes;
- at least eight more successes than the paired source policy;
- at least 6/8 successes in every reset category;
- at least 31/32 nominal terminal successes;
- no observation, action, normalizer, graph-topology, or action-scaling change;
- output-layer-only tensor differences and independent ONNX parity below `1e-5`; and
- successful loading and warm-up inference through MicroDuck's production Rust loader.

These thresholds are fixed before calibration or optimization. A result below them is a
no-go, even if its task return, maximum trunk height, or selected video looks better.

## Expected artifact and demo

If approved and successful, publish it as a second `model-stand` release derivative with a
distinct ID such as `alpha-stand-left-leg-v1`, its own manifest, profile hash, banks, paired
videos, and rollback target. The hero comparison should show the source and adapted policy
from identical face-down and sitting starts, with a visible left/right actuator-authority
indicator and per-pose aggregate results.

The physical analogue is a deliberately configured current/torque limit, not a damaged
robot. Hardware execution remains a separate, explicitly approved test with physical
support, current limits, an observer, power-cut access, stop criteria, and the source policy
preinstalled as the rollback target.
