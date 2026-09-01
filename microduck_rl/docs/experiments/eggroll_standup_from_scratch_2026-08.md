# EGGROLL StandUp from-scratch experiment record

Status: **complete and archived**
Decision date: 2026-08-30

## Decision

Rank-one HyperscaleES EGGROLL learned a genuine closed-loop Microduck balance
policy from random initialization using forward rollouts only. The result was
repeatable across two seeds and survived actual-environment replay. The tested
direct 197,774-parameter formulation did **not** learn stand-up recovery and is
closed as a no-go. It must not be revived by merely increasing population,
extending the run, changing the stage gate, or treating task return as success.

The live repository now pursues a different hypothesis: local post-training of
a competent deployed PPO policy in a compact parameter subspace against a
hidden deployment shift.

## Immutable source backup

- branch: `archive/eggroll-standup-pre-posttrain-2026-08-30`
- source commit: `5cafd566f88214dbbb9c2e5d6986ab784beed40c`
- annotated tag: `eggroll-standup-pre-posttrain-2026-08-30`
- local bundle: `../archives/eggroll-standup-pre-posttrain-2026-08-30.bundle`
- bundle SHA-256:
  `1f1c64eaeed30fe47333fe0e988407d9955a1ef2683a8c811cb83657c44dd096`

The verified bundle contains complete history for both the archival branch and
tag. It also contains the production ONNX policies and trusted local resume
checkpoints that existed immediately before the post-training reset. Generated
evaluation and run directories were moved intact to
`../archives/pre-posttrain-local-artifacts/`.

## Question tested

Could EGGROLL train the PPO-size deterministic actor
`61 -> 512 -> 256 -> 128 -> 14` from random initialization into a genuine
closed-loop StandUp and recovery policy using scalar rollout evaluations in
`Mjlab-StandUp-Flat-MicroDuck`?

Success was terminal behavior, not reward or a transient pose: the policy had
to finish an episode after a continuous supported stand. The diagnostic
standing envelope was trunk height `0.115 +/- 0.015 m`, tilt at most 20 degrees,
leg-pose RMS error at most 0.4 rad, supported feet, and a continuous hold of at
least one second.

## Environment and deployment findings

- The actor observation is 61D: angular velocity (3), projected gravity (3),
  joint position relative to HOME (14), joint velocity (14), previous raw
  action (14), twist command (3), head command (4), and body command (6).
- The action is 14 unfiltered joint-position offsets with the registered
  default-pose convention and scale.
- The local MuJoCo viewer is useful for interface rehearsal but is not a
  faithful behavioral evaluator for the registered StandUp task.
- The registered task has no native boolean success signal and deliberately
  allows fallen states. Dense return can therefore reach approximately 15--18
  without a stand-up recovery.
- The original base evaluator reset curriculum counters, so late logged
  evaluation returns did not describe the final training distribution.
- `alpha_stand.onnx` was the positive PPO control and finished stable in the
  initial ten registered-environment episodes: four standing and two each from
  sitting, face-down, and face-up.

## Estimator defects discovered and corrected

The first experiment matrix violated common random numbers in three hidden
paths:

1. battery voltage and voltage-sag gain differed across candidates;
2. BAM actuator delay sampled independently for every candidate and physics
   step; and
3. expanded MuJoCo fields were `TorchArray` proxies, so a literal
   `torch.Tensor` check skipped mass, inertia, CoM, armature, and friction.

The corrected evaluator broadcast all proxy-backed model fields, motor-plant
state, and delay draws. The strict identity audit measured zero spread in
exogenous model state, actuator state, reset observations, and lag span across
eight identical-policy worlds. Small later contact divergence remained and
caused two CPU/CUDA boundary cases, but did not change any gate decision.

## Corrected results

All primary arms used A10G Large, population 1,024, rank 1, sigma 0.01, Adam
0.0003 where applicable, ten common scenarios per candidate, and a fixed
64-world held-out bank.

| Arm | Horizon | Best terminal | Final terminal | Final return |
| --- | ---: | ---: | ---: | ---: |
| staged task return, seed 42 | 100 | 49/64 at generation 93 | 48/64 | 55.12 |
| staged task return, seed 7 | 100 | 46/64 at generation 68 | 46/64 | 53.31 |
| terminal capability v3 | 50 | 1/64 at generation 47 | 0/64 | 19.15 |
| PPO reset schedule with task return | 50 | 2/64 at generation 23 | 0/64 | 14.78 |

The successful standing episodes stayed upright for all six seconds and ended
near 0.115 m trunk height with upright cosine near 1.0. The seed-42 generation
93 checkpoint reproduced 47/64 terminal successes on CPU versus 49/64 in the
CUDA trainer. Neither seed reached the predeclared 52/64 gate twice, so both
remained on standing resets and never trained recovery.

Forced-pose replay of generation 93 gave:

| Pose | Stable success | Mean final trunk z | Mean upright time |
| --- | ---: | ---: | ---: |
| standing | 3/3 | 0.116 m | 6.000 s |
| sitting | 0/3 | 0.042 m | 0.153 s |
| face-down | 0/3 | 0.037 m | 0.000 s |
| face-up | 0/3 | 0.041 m | 0.000 s |

## Final objective-switch test

The same generation-93 checkpoint was resumed into two otherwise identical
30-generation continuations with stage 0 frozen:

| Continuation | Objective | Best terminal | Final terminal | Final return | Worst terminal progress |
| --- | --- | ---: | ---: | ---: | ---: |
| control | mean task return | 49/64 | 49/64 | 55.92 | 0.0000221 |
| treatment | standing robustness | 49/64 | 49/64 | 56.33 | 0.0000501 |

Neither continuation recorded a single 52/64 evaluation. The treatment
improved worst terminal progress by about 2.3 times and reached 49/64 earlier,
but converted no additional held-out world. This rejects the narrow hypothesis
that lexicographic lower-tail ordering alone would break the plateau at the
existing search scale.

The selected treatment checkpoint replayed at 47/64 on CPU. All 17 failures
fell from their standing reset, ended near 0.052--0.054 m with upright cosine
about 0.16--0.23, and had no terminal hold.

Authoritative final repositories:

- control job `6a93ced045686a1580c17532`, repository
  `richoakley/eggroll-v3j-g93-control-mean-g123`;
- treatment job `6a93cee2984507d9db4ec98b`, repository
  `richoakley/eggroll-v3j-g93-robustness-g123`.

## Retained conclusions

1. EGGROLL can learn a real feedback policy from rollout-only evaluation.
2. Sparse terminal capability is unsuitable for discovering this behavior from
   a random 197,774-parameter policy.
3. Shaped return discovers balance but did not make it robust enough to unlock
   recovery.
4. Population 4,096 cost about 1.85 times as much per generation as 1,024 in
   the measured setup without a one-step benefit.
5. Rank one is not a one-dimensional accumulated policy update; it is the rank
   of each sampled matrix perturbation.
6. Actual-environment terminal replay, not viewer appearance or task return,
   is the decision source.
7. Common-random-numbers matching must include model proxies, motor plant,
   actuator state, sensor draws, commands, pushes, and delays.

## Future hypotheses, deliberately not retained as live functionality

- output-layer post-training of a competent deployed policy;
- mergeable low-rank adapters across frozen PPO layers;
- exact quantized candidate evaluation;
- hardware-in-the-loop adaptation to battery, latency, friction, payload,
  backlash, energy, impact, or human preference;
- distributed evaluation across heterogeneous robots.

Each is a new experiment. None should be represented by an untested CLI flag
or a dormant training mode.
