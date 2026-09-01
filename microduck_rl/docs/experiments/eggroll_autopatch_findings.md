# EGGROLL robotics findings and supersession record

This record preserves the knowledge from the August 2026 MicroDuck experiments while
keeping the supported product surface small.

## Durable findings

1. Improving task return was not sufficient evidence of stable stand-up. Early
   from-scratch EGGROLL runs exploited shaped reward/local optima and required explicit
   terminal-stability diagnostics in the actual task.
2. The local standalone MuJoCo viewer was not an authoritative deployment test. Reset
   distribution, scheduler commands, previous-action state and runtime processing had to
   be reproduced through the real environment and Rust policy loop.
3. EGGROLL did not reliably learn the complete StandUp behavior from scratch under the
   tested budgets. That is a negative result about that formulation, not a proof that
   evaluation-only optimization cannot improve robotics policies.
4. Post-training a competent PPO source policy was successful. The released StandUp
   output-layer patch improved the selected lag/voltage condition from 17/32 to 32/32
   terminal successes while retaining 32/32 nominal success.
5. The successful product pattern is deployment repair: freeze the source representation,
   define a behavioral failure and nominal-retention gate, optimize a narrow patch with
   forward evaluations, and ship only after actual-runtime A/B and rollback proof.
6. One successful StandUp repair does not establish generality across policy classes,
   hardware changes or physical robots. The nine-policy source suite establishes that a
   common runtime/evidence contract is possible; future approved campaigns must establish
   adaptation breadth.
7. Walking source-only calibration rejected three misleading condition families before
   search: a geometry-only sole did not expose an active-command gap, equal-priority foot
   friction did not own MuJoCo pair friction, and a corrected high-priority low-grip sole
   still did not impair the production gait. The first wedge run then exposed an
   entity-local versus global geom-index bug. After fixing the mutation and recomputing
   derived simulator constants, a 15 degree symmetric wedge produced the intended partial
   failure (2/4), while 20 degrees was catastrophic (0/4) and was rejected.
8. The next proof is therefore continuous-locomotion repair against the frozen 15 degree
   wedge, not another StandUp variant. The campaign executes HyperscaleES EGGROLL only.
   Naive ES and random search are excluded; efficiency will be reported absolutely in
   candidate evaluations, simulator rollouts, simulator steps and wall/GPU time, with no
   comparative sample-efficiency claim.
9. That walking proof has now succeeded. EGGROLL patched the deployed walking policy for
   the hidden 15 degree replacement-foot geometry while preserving the frozen 61D/14D
   contract and meeting the nominal-retention floor. This establishes successful
   adaptation in a second policy class; it still does not establish arbitrary hardware
   adaptation or physical transfer.
10. The independent confirmation bank reproduced and strengthened the walking result.
    Across two seed-disjoint 32-case banks, the source passed 47/64 and DuckEgg passed
    64/64. The derivative preserved all 47 source successes and repaired all 17 observed
    failures, closing the profile-specific two-bank behavioral gate.

## Supported surface now

The supported implementation is `mjlab_microduck.autopatch` and the
`eggroll-autopatch` CLI. It owns production-artifact inventory, campaign contracts,
registered-task evaluation, Rust trace auditing, capability graph coverage, generic paired
A/B, equal-budget protocol planning and release eligibility.

The canonical product overview is `docs/eggroll_autopatch.md`.

## Historical reference, not extension points

The following StandUp-specific assets are retained only because they reproduce the first
patch, videos, signed updater activation and exact rollback:

- `scripts/eggroll_patch_lab.py`;
- `scripts/run_production_runtime_twin.py`;
- `scripts/verify_runtime_trace.py`;
- `scripts/verify_runtime_policy_parity.py`;
- `mjlab_microduck.sim.standup_runtime_twin`; and
- `mjlab_microduck.eggroll.release`.

Do not add another policy-specific lab by copying them. New policies and conditions go
through the generic registry, evaluator, campaign and release contracts. Once the generic
release workflow also emits the historical hero-video pack, these reference wrappers can
be removed without losing evidence.

## Experiment documents retained as evidence

- `eggroll_standup_from_scratch_2026-08.md`: negative from-scratch result;
- `eggroll_posttraining_2026-08.md`: successful PPO-source post-training;
- `eggroll_policy_patch_lab.md`: production-runtime and updater proof;
- `eggroll_next_objective_asymmetric_leg.md`: frozen but unlaunched next condition; and
- `eggroll_asymmetric_optimizer_comparison.md`: frozen but unlaunched comparison.

The unlaunched documents are predeclarations, not completed results or adopted roadmap.

## Frozen walking proof

The authoritative source calibration is
`/private/tmp/autopatch-walking-wedge-foot-calibration-20260831-v5/manifest.json`.
It binds production `alpha_walking.onnx` SHA-256
`e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`, active-command
bank SHA-256 `e1150ed24b619160d4cf490720bda3e186c6e154568903c50f71e44566c563c5`,
and the selected 15 degree profile SHA-256
`3410b59527e069c993212671ce463ac05183777968a1ed8e15872affb46912a2`.
Calibration used 20 source-policy episodes, 5,000 requested simulator steps, zero
candidate evaluations and zero accepted transport retries. Nominal passed 4/4, 5 and 10
degrees passed 4/4, 15 degrees passed 2/4, and the rejected 20 degree condition passed
0/4.

The immutable EGGROLL-only campaign is
`docs/experiments/campaigns/walking_wedge_autopatch_v1.json`, campaign SHA-256
`e2ddf3a989c14aad12301ef2e9713390377be5cb52030bfbce58fd30c9ffdd20`. It uses the
already validated StandUp post-training settings—population 512, rank 4, sigma 0.015,
Adam 0.003, 100 generations—and changes only the final 1,806 affine parameters. Each
generation uses one matched world per active speed (0.28, 0.32, 0.36 and 0.40 m/s).
The fixed shifted selection bank has 32 disjoint episodes and SHA-256
`9deed06a134abe4f7179cc1ad5bc1a328d057415574552bbb5a731f5f0173093`.

The release target was at least 28/32 shifted terminal successes, at least 6/8 at every
speed, and at least 31/32 nominal terminal successes, all twice consecutively. It also
required output-layer-only export, ONNX parity below `1e-5`, production Rust trace parity,
the walking capability node, stand-to-walk and walk-to-stand edges, signed activation and
exact rollback. Task return remained diagnostic only.

## Walking proof result

The frozen 100-generation campaign completed as Hugging Face job
`6a95de9721c5aa7c8364a0ec`. Its private authoritative artifact repository is
`richoakley/eggroll-walking-wedge-train-20260831-v1`. The source commit, campaign,
production policy, shifted profile and held-out bank hashes all matched the predeclared
contract.

The initial full-run evaluator measured the source at 19/32 shifted successes and 31/32
nominal successes. EGGROLL reached 32/32 shifted success by generation 5, initially at the
cost of nominal behavior. Nominal success then recovered. Generations 50 and 55 passed
the behavioral gates consecutively:

| Generation | Shifted | Worst command | Nominal | Retained |
| ---: | ---: | ---: | ---: | --- |
| 50 | 32/32 | 8/8 | 31/32 | yes |
| 55 | 32/32 | 8/8 | 32/32 | yes |

Generation 85 was selected lexicographically from nominal-retained candidates. It changes
only the final 1,806 affine parameters. Independent export reproduced the exact remote
ONNX SHA-256
`f6c2378b415cbf5449e21d1bd4f1c1df72ac7817e8d4f39caa943c8c22221b5c`
with maximum absolute parity error `4.76837158203125e-7`.

The production-runtime playback then ran source and adapted policies over the same 32
seeds under both the shifted and nominal profiles. Each episode used the actual
`Mjlab-Velocity-Flat-MicroDuck` task, the real Rust `robotd` policy loop and the simulator
`RobotIo` seam.

| Profile | Source PPO | EGGROLL patch | Per-command source | Per-command patch |
| --- | ---: | ---: | --- | --- |
| 15 degree replacement foot | 24/32 | **32/32** | 4, 7, 8, 5 | **8, 8, 8, 8** |
| nominal | **32/32** | 31/32 | 8, 8, 8, 8 | 8, 8, 7, 8 |

The shifted patch had 32/32 terminal-stability passes, upright fraction 1.0, mean signed
forward displacement 1.809 m and mean forward-velocity RMSE 0.110 m/s, versus 25/32,
0.963, 1.054 m and 0.143 m/s for the source. Nominal terminal stability remained 32/32;
one 0.36 m/s episode missed the compound terminal-success criterion, leaving the adapted
policy exactly at the 31/32 retention floor.

All 128 accepted episodes passed the Rust trace audit. Maximum error was zero for ONNX
actions, previous raw actions, scale/home offsets, state targets, safety targets and
RobotIo writes; the largest filter error was floating-point noise at `8.88e-16`. Three
transport attempts with an incomplete RobotIo write were rejected, retained as evidence
and rerun cleanly. There are 128 accepted videos plus those three rejected-attempt videos.

Absolute optimization cost was 51,200 candidate evaluations, 204,800 world rollouts,
51.2 million requested simulator steps and 13,698.7 seconds of campaign wall time on one
A10G Large. No comparative sample-efficiency claim is made because no baseline optimizer
was run. Complete machine-readable results are in
`eggroll_walking_autopatch_2026-09.artifacts.json`; the evidence-bound side-by-side video
is `docs/assets/eggroll_autopatch/walking_wedge_gen85_hero.mp4`.

### Causal conclusion

The original apparent walking success was an evaluator failure: it omitted `robotd`'s
command EMA, raw previous-action observation and head/leg target filters, and it reused a
single startup-randomized physical identity across case seeds. Mirroring the production
transport and freezing a fresh seeded startup identity for every world restored the real
failure distribution. Under that corrected contract, EGGROLL produced a genuine walking
repair rather than a task-return or viewer artifact.

This is a go for the product thesis that a frozen deployed neural policy can be repaired
against a non-differentiable behavioral deployment objective using forward evaluations
only. It is not yet a go for autonomous physical deployment, arbitrary component swaps,
or universal policy repair.

## Zero-regression audit

The original walking campaign deliberately allowed one nominal miss: its hard retention
floor was 31/32. That was sufficient for the repair proof but not for the stronger product
claim that Autopatch introduces no observed regression. The first strengthened contract
treated original-foot nominal retention as mandatory for every patch. That was the wrong
universal abstraction for a permanent hardware replacement: a wedge-foot derivative is
only eligible on a robot attested to have the wedge feet, while an original-foot robot
retains the exact source policy. Release envelope v3 therefore separates immutable training
campaigns from content-addressed deployment scope:

- `profile_specific` requires case-by-case source retention on the single attested
  activation profile and fails closed to the source bytes everywhere else;
- `multi_profile` requires case-by-case source retention on every declared profile and is
  appropriate for conditions, such as voltage or latency, that the same robot may traverse.

Both modes require two disjoint paired production-runtime banks, disjoint scenario seeds,
exact source and derivative hashes, complete bank-to-row binding, and a production updater
routing attestation. Aggregate success, task return and continuous diagnostics cannot
override a lost source success inside the declared deployment scope.

All four campaign-bound source-improving candidates were independently exported and
screened under the production Rust runtime. The original-foot results remain useful
cross-profile diagnostics:

| Generation | Campaign nominal | Production-runtime audit | First regression |
| ---: | ---: | --- | --- |
| 50 | 31/32 | rejected | `heldout-wedge-vx-0.36-001` |
| 55 | 32/32 | shifted 32/32, nominal **31/32** | `heldout-wedge-vx-0.36-004` |
| 75 | 31/32 | rejected | `heldout-wedge-vx-0.36-004` |
| 85 | 31/32 | shifted 32/32, nominal **31/32** | `heldout-wedge-vx-0.36-004` |

Generation 55 is the important counterexample. The campaign-side evaluator measured
32/32 nominal, but the production-runtime replay measured 31/32. The robot remained
upright for the full failed episode but made only 0.144 m of the required 0.450 m signed
progress. This demonstrates why strict paired playback must be a release invariant rather
than inferred from campaign metrics.

Generation 85 is eligible for continued profile-specific verification because, on the
first wedge-foot bank, it retained all 24 source successes and repaired all eight failures:
24/32 to 32/32 with zero observed in-profile regressions. Its 31/32 original-foot result
does not qualify it as a multi-profile patch, and it must not be routed to original-foot or
unknown robots. The exact source bytes remain the fallback for those robots.

No new training was launched during this audit. The predeclared second bank was executed
as job `6a9671d921c5aa7c8364ab6b` with no seed overlap with the first bank. The source passed
23/32 and DuckEgg passed 32/32, with zero lost source successes and nine repaired source
failures. Per-command results were source 8/8, 5/8, 6/8 and 4/8 versus DuckEgg 8/8 at all
four speeds. Its manifest SHA-256 is
`3a694e6b845c203789e5979882ec3132b98f6e6baac9f46728a72bc9e44f9d4a`.

Combined across both sealed banks, the source passed 47/64 and DuckEgg passed 64/64. The
derivative retained all 47 source successes and repaired all 17 failures. All 128 accepted
in-profile source/adapted episodes passed the independent Rust trace audit. This is the
finite, profile-scoped result required for **zero observed in-profile regression across
two independent banks**. The generated v2 record is
`walking_wedge_gen85_two_bank_non_regression_v2.json` with SHA-256
`ffe2011d2c59a186abbcfaa6e128ffdb53cc7781be53e609ef98864162f26879`.
Release-envelope v3 and signed-updater eligibility remain blocked until genuine production
routing evidence exists. The original-foot 31/32 result remains a cross-profile diagnostic,
and neither simulation result is a physical-robot guarantee.

Jobs `6a966c1b0718b0f6d890b66e` and `6a966eef0718b0f6d890b6dc` failed before
rollout because of harness/runtime-snapshot issues. They produced no behavioral cases and
are excluded from every success, regression and efficiency total above.
