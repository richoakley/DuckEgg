# Predeclaration: asymmetric policy-patch optimizer comparison

Status: **protocol frozen; no optimizer run approved or launched**

Predeclaration date: 2026-08-31

## Question

Given the same sealed PPO actor and the same non-differentiable terminal-standing
objective, does EGGROLL find a deployable policy patch more efficiently than
ordinary black-box baselines? A privileged gradient arm is included only as a
diagnostic upper bound; it is outside the product's black-box setting.

The deployment condition is the frozen
`asymmetric-left-knee-ankle-25pct-v1` profile. Its source result is 17/32 on the
frozen calibration bank, with at least one success in every reset category. The
profile SHA-256 is
`4ff1208b44ac154772939fb07c2902b4c902e47b0b51def94fb5d163fa8a925a` and
the bank SHA-256 is
`6df62d7ef310e06d7d44437c8e3ac0a9c423d3b12d5c4bd97cdf6e0274dbe4fb`.

## Shared contract

All arms start from source policy SHA-256
`1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b`.
Only the final `14 x 128` weight and 14D bias may change: 1,806 parameters.
The normalizer, feature extractor, 61D/14D contract, action scale, registered
environment, reset distribution, success semantics, objective order, selection
bank, held-out bank, and nominal-retention bank are identical.

The common black-box budget is at most 51,200 candidate evaluations: population
512 for 100 generations. Each candidate receives the same balanced four-world
common-random-number bank within a generation. Report both candidate evaluations
and simulator episodes, including any selection or retention evaluation. Wall
time and peak memory are diagnostics, not substitutes for evaluation count.

## Predeclared arms

| Arm | Update rule | Information allowed |
| --- | --- | --- |
| **EGGROLL** | HyperscaleES `EggRoll`, rank 4, antithetic population 512, sigma 0.015, rank-converted fitness, Adam 0.003 | scalar lexicographic fitness only |
| **Naive ES** | full 1,806D isotropic antithetic Gaussian perturbations, population 512, sigma 0.015, the same converted fitness and Adam 0.003 | scalar lexicographic fitness only; no matrix-aware low-rank perturbation |
| **Random search** | independent isotropic Gaussian samples around the unchanged source at sigma 0.015; retain best eligible candidate but perform no distribution update | scalar lexicographic fitness only |
| **Privileged gradient upper bound** | backpropagation through a differentiable simulator/surrogate and differentiable proxy for the terminal objective, restricted to the same output layer | simulator gradients and internal state; explicitly unavailable to the product method |

Random search draws exactly as many candidate parameter vectors as EGGROLL. The
naive ES arm uses the same antithetic pairing and optimizer settings so that the
structural perturbation method, rather than population or learning rate, is the
intended difference. If a faithful full-vector ES implementation cannot expose
the same fitness conversion, stop and amend this document before running it.

The privileged arm is not a head-to-head proof because the terminal lexicographic
objective is non-differentiable. Its proxy must be written down before seeing its
results, and both proxy performance and the real terminal gate must be reported.
Its simulator-step count and backward-pass cost must be measured separately.

## Seeds, selection, and release gates

- optimization seed: `20260920`, generation bank seed
  `20260920 + generation_index * 10007`;
- fixed 64-world selection seed: `20260920 + 1000003`;
- fixed 32-world nominal-retention seed: `20260920 + 2000003`;
- fresh final asymmetric seed: `20261001`, eight episodes per pose; and
- fresh final nominal seed: `20261002`, eight episodes per pose.

Every arm uses the terminal-success lexicographic objective in
`eggroll_next_objective_asymmetric_leg.md`. Task return and transient upright
time remain diagnostics. A release still requires at least 28/32 asymmetric
successes, an eight-success paired improvement over source, at least 6/8 in
every pose, at least 31/32 nominal successes, output-layer-only changes, ONNX
parity below `1e-5`, and successful production Rust loading.

## Decision rule

Primary comparison: candidate evaluations required to first produce a checkpoint
that passes every final gate. Secondary comparisons: best final lexicographic
key at equal budget, success across repeat seeds, simulator episodes, wall time,
and nominal degradation. A single lucky checkpoint is not enough: the eventual
approved experiment must predeclare at least three optimizer seeds or explicitly
remain a one-seed pilot.

No expensive baseline, EGGROLL optimization, or privileged-gradient run is
authorized by this predeclaration.
