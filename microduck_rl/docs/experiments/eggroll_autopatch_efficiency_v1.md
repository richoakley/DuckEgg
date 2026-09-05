# EGGROLL Autopatch evaluation efficiency v1

## Benchmark contract

This study freezes the successful walking replacement-foot campaign and asks how many
simulated robot interactions are required to reach the **first independently confirmed,
release-eligible EGGROLL patch**. The source is production `alpha_walking.onnx` SHA-256
`e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c` under the hidden
15 degree replacement-wedge-foot profile SHA-256
`3410b59527e069c993212671ce463ac05183777968a1ed8e15872affb46912a2`.

The actor remains exactly 61 observations to 14 actions. The only trainable parameters
are the final affine weight and bias: 1,806 parameters. Training uses the real
HyperscaleES EGGROLL implementation with antithetic pairs and common random numbers.
Registered task return is diagnostic only. Selection, independent confirmation and
release evidence remain disjoint from training. Release is profile-specific: only the
attested wedge-foot profile may activate the derivative; original and unknown profiles
retain the exact source bytes.

The frozen behavioral standard is paired, case-by-case retention on two independent
production-runtime wedge banks. In the historical mixed-platform record, the reference
source passed 47/64 cases and generation 85 passed 64/64, retaining all 47 source
successes and repairing all 17 observed source failures. A negative remote preflight later
showed that the same source passes 21 rather than 24 production-bank cases on Linux CPU;
the Linux CUDA confirmation count remains 23. The new remote standard therefore retains
all 44 source successes in the exact sealed Linux CPU plus Linux CUDA environments and
repairs all 20 observed failures. The cases, episode lengths and terminal-success
semantics are unchanged, and the exact source-failure case set is bound per
platform/device. ONNX parity must remain below `1e-5`; the Rust production-runtime trace,
profile routing, signed activation, source fallback and rollback gates are not weakened.

## Reference identities and original costs

The machine-readable reference is
[`eggroll_autopatch_efficiency_reference_v1.json`](eggroll_autopatch_efficiency_reference_v1.json).
It binds campaign canonical SHA-256
`e2ddf3a989c14aad12301ef2e9713390377be5cb52030bfbce58fd30c9ffdd20`, the fixed
selection bank, release scope, two-bank record, selected policy and original job.

| Cost kind | Frozen walking reference |
| --- | ---: |
| Candidate evaluations | 51,200 |
| Optimisation world rollouts | 204,800 |
| Requested optimisation simulator steps | 51,200,000 |
| Executed optimisation simulator steps | not historically recoverable |
| Source-baseline requested / executed steps | 16,000 / 13,982 |
| Periodic selection requested steps | 320,000 |
| Campaign wall time | 13,698.7 s |
| Accelerator | one A10G Large |
| Training seeds | one |

The 51.2 million figure is a **requested** budget: `512 candidates × 100 generations ×
4 worlds × 250 steps`. The rollout accumulator recorded each candidate's executed
`episode_steps`, but the trainer reduced this to mean survival and did not persist the
sum. Historical executed optimisation steps therefore cannot be reconstructed exactly.
The old `budget.json` also excluded source baselines, periodic selection, qualification,
confirmation and rejected transport attempts. New records must report these separately
and in aggregate.

The source baseline is partially reconstructable from its retained episode rows: the 32
shifted cases executed 5,982 of 8,000 requested steps and the 32 nominal cases executed
8,000 of 8,000. Periodic evaluations occurred every five generations (20 attempts, 64
worlds each), but their executed-step sums were not retained.

The successful StandUp post-training campaign used the same population, rank and 100
generations, with four six-second (300-control-step) pose worlds per candidate. Its exact
requested optimisation cost was therefore 51,200 candidate evaluations, 204,800 world
rollouts and 61,440,000 simulator steps (about 341 simulated robot-hours). Twenty fixed
selection attempts added 96 worlds or 28,800 requested steps each, 576,000 in total.
Historical summaries again omitted executed step totals. The job was reported as roughly
5.5 hours on one A10G Large; the sum of the 100 persisted generation timers is 5,492.625
seconds and excludes baseline, selection outside the generation timers, startup and job
wrapper overhead. These two wall figures are intentionally not conflated.

## Permitted changes

Permitted efficiency mechanisms are:

1. stop only after the complete predeclared qualification gate has passed;
2. rank training candidates with a new, versioned release-scope-aware lexicographic
   objective;
3. branch training-only candidate rollouts from a deterministically replayable state
   before the first discriminating failure, while keeping all selection and release
   evaluations full-episode and from reset; and
4. run a bounded, sequential population/rank study only after the preceding mechanisms
   are isolated.

The study may improve wall-clock overhead, but fewer generations or faster matrix
operations are not interaction-efficiency evidence unless candidate evaluations, world
rollouts, and requested and executed simulator steps also fall.

## Claim boundaries

No held-out, confirmation or release case may supply training data, frontier snapshots,
hyperparameter decisions or early-stopping signals beyond its predeclared gate role. A
campaign-side pass is not production-runtime evidence. A large shaping score cannot
compensate for behavioral failure. No gradient, autograd, PPO, CMA-ES, PGPE or generic ES
run may be presented as EGGROLL. Simulation does not establish physical-robot efficiency,
physical transfer, fleet deployment, autonomous repair or comparative superiority.

## Phase 1 findings

The archived metrics contain 100 complete generation rows and exactly tie to 51,200
candidate evaluations and 204,800 optimisation rollouts. Each generation used four
250-step worlds per candidate. Generation 5 was the first campaign-side 32/32 shifted
checkpoint, but it failed the historical nominal selection filter completely. Generations
50 and 55 were the first consecutive historical behavioral-gate passes. Generation 85
was selected lexicographically among the nominal-retained candidates and later passed the
two independent profile-specific production-runtime banks.

This history motivates qualification but does not prove that generation 5 was
release-eligible: it was never exported and replayed through the complete production
gate. A counterfactual stop at generation 5 would be only a hypothesis until the archived
checkpoint passes ONNX parity, paired source-retention, the independent confirmation bank,
Rust trace audit, routing, signed activation and rollback.

Profiling in the archived run is limited to generation wall time and construction counts.
Representative generation wall times were 66.3 s (generation 1), 52.7 s (5), 48.1 s
(50), 49.9 s (55), 45.3 s (85) and 50.7 s (100). The existing code did not separately
time MuJoCo physics, reset/world construction, policy inference, Torch/JAX/DLPack,
objective aggregation, evaluation, checkpointing, artifact generation or production
replay. The v1 implementation adds explicit phase timers and interaction ledgers; the
subsequent instrumented CUDA smoke below supplies that breakdown without inferring it from
wall time.

## Executed study

The new study achieves **strong success**. Three independent frozen seeds were trained on
A10G Large accelerators, and the first fully release-eligible checkpoint for every seed is
generation 6. Each reaches that checkpoint after 3,072 EGGROLL candidate evaluations,
12,312 optimisation world rollouts including 24 paid training-source reference rollouts,
and 3,078,000 requested optimisation simulator steps. The three executed optimisation
counts are 3,077,101, 3,077,239 and 3,077,345 steps. The median requested cost is therefore
3.078 million steps: 6.012% of the 51.2 million-step reference, a 16.634x improvement and a
93.988% reduction. This clears the 5.12 million-step strong-success threshold with all
three seeds rather than the required two.

Each generation-6 candidate independently passes the same six stages: release-scope
retention, final-affine-only ONNX parity, Linux CPU production-runtime replay, independent
Linux CUDA confirmation, signed profile routing, and signed activation plus fresh-process
rollback. Every candidate is 32/32 on both banks, retains every one of the 21 Linux CPU and
23 Linux CUDA source successes case-by-case, and passes every Rust trace. Maximum ONNX
absolute errors are `2.86102294921875e-6`, `3.337860107421875e-6`, and
`3.0994415283203125e-6`, all below the exclusive `1e-5` threshold. The adapted policy
SHA-256 values are, in seed order,
`111bbfcdc57c51aef61e52b01dcf7ad26eff75a0180a46234fc0cde37796dc68`,
`39bea1b92e2e5a07200f7c590bfb32231b927f006024a870d74ad3b5815239cf`, and
`3c65e5699a13caaccb488a90088a11f42f4d4204a13da340adbe950cb369dfc3`.

The immutable machine record is
[`eggroll_autopatch_efficiency_multiseed_20260903_v1.json`](eggroll_autopatch_efficiency_multiseed_20260903_v1.json).
It includes every accepted and rejected qualification attempt, requested and executed
interaction counts, training and output repository commits, candidate and exported-policy
hashes, cost boundaries and the leakage audit. Rejected candidates were not hidden:
seed 1 failed at generations 4 and 5 before passing 6; seed 2 failed 2, 4 and 5 before
passing 6; and seed 3 failed 5 before passing 6. Including all qualification attempts,
the requested all-in costs through first eligibility are 3.254, 3.286 and 3.254 million
steps. Their median is 3.254 million, still 15.734x below the reference optimisation
budget. Qualification itself executed 58,246, 87,646 and 58,800 simulator steps across
the sequential attempts. The exact executed optimisation prefixes are reported above;
the full nine-generation records also preserve exact all-prequalification execution, but
the trainer did not checkpoint a separate cumulative selection ledger at generation 6,
so the document does not invent an exact executed all-in prefix.

Summed generation timers through generation 6 are 471.423, 478.122 and 477.990 seconds;
they exclude startup, fixed evaluation, qualification and provider queue time. The summed
six-stage command time across every qualification attempt is 5,405.211, 7,736.337 and
4,737.365 seconds. At the provider's execution-time listing of `$0.025/min` for A10G Large,
creation-to-output timestamps give upper proxies of about `$7.445` for the initial plus
resume training jobs and `$8.585` for the nine candidate-qualification jobs, or `$16.03`
for the core study. These are not invoices: exact billed accelerator time was unavailable,
creation-to-output includes queue and setup, and smoke plus environment-preflight jobs are
excluded.

There is one important execution distinction. The nine-generation training jobs used the
correct profile-specific fitness but their cheap gate still hard-gated adapted-policy
nominal behavior. That is wrong for a release scope which activates the derivative only on
the attested profile and routes the exact source elsewhere. Commit `7e427c1` corrected the
gate interpretation without changing training, candidate bytes, banks, thresholds or any
production qualification stage. The generation-6 claim is therefore an exact persisted
training prefix plus independently executed, checkpoint-hash-bound qualification replay;
the original jobs actually ran all nine generations and are not represented as having
integrated early-stopped at six. At that stage an end-to-end early-stop rerun remained the
missing operational validation; the later integrated campaign reported below closes it.

For comparison, the reproducible historical replay record remains
[`eggroll_autopatch_efficiency_historical_replay_v1.json`](eggroll_autopatch_efficiency_historical_replay_v1.json).
It finds generation 55 as the first old campaign-side trigger and generation 85 as the only
historical checkpoint with later two-bank evidence. Neither establishes the new result.

## Integrated controller and orthogonal-condition follow-on

The missing operational proof has now passed. Hugging Face job
`6a9a9002259f8e97255ddc10`, from exact clean commit
`e59949bea76af683f54b91f3776f283fdd98beb6`, ran training and all qualification stages
inside one campaign and stopped at generation 5, the first eligible checkpoint. Earlier
generations were not stitched together: generation 1 failed the selection screen,
generation 2 lost a contemporaneous source success, and generations 3 and 4 failed
production-runtime casewise retention. Generation 5 then passed the complete six-stage
gate. It used 2,560 candidate evaluations, 2,565,000 requested and 2,564,376 executed
optimisation simulator steps, plus 144,000 requested and 138,263 executed qualification
steps. The selected ONNX SHA-256 is
`77970bcd949929e982b60c088c3055fe63504b748c43a2b5c496435b9be9c733`; its maximum
parity error is `2.86102294921875e-6`. The production and confirmation banks were each
32/32 with zero contemporaneous source-success regressions, and routing, exact source
fallback, signed activation and rollback all passed. The machine record is
[`eggroll_autopatch_efficiency_seed4_integrated_20260904_v8.json`](eggroll_autopatch_efficiency_seed4_integrated_20260904_v8.json).

The first orthogonal physical condition was a predeclared forward trunk-CoM ladder of
10, 15, 20 and 25 mm. The complete A10G Large source-only calibration ran 160 episodes,
40,000 requested simulator steps and zero optimizer evaluations. All five conditions,
including nominal, passed 32/32; increasing the forward offset reduced mean velocity RMSE
from `0.18982` to `0.13536`. Consequently no profile met the frozen 16--28/32 incident
band, the CoM branch stopped without training, and it does not establish cross-failure
generality. The result and all failed infrastructure attempts are recorded in
[`walking_trunk_com_calibration_result_v2.json`](walking_trunk_com_calibration_result_v2.json).
That run also exposed a provenance packaging defect: its source manifest named the raw
protocol hash, while the bundle contained canonically identical re-formatted JSON. The
launcher now archives the exact bytes and validation checks both raw and canonical hashes;
the CoM outcome is retained as a scientifically interpretable negative but is not promoted
past the strengthened raw-byte gate.

A separate payload-mass alternative was then frozen before any payload playback. It adds
50, 100, 150 or 200 g to the 199 g `trunk_base`, scaling mass and pseudo-inertia together
from each seeded startup state while leaving the inertial position, observation contract
and policy bytes unchanged. The ladder, disjoint banks, three optimizer seeds, nine-
generation ceiling, runtime-trace rule, release gates and claim boundaries are fixed in
[`walking_trunk_payload_cross_failure_protocol_v1.json`](walking_trunk_payload_cross_failure_protocol_v1.json).
It is a single bounded ladder, not an adaptive search.

That payload calibration also completed as a valid negative result. Job
`6a9b1e16e686246ca69a26b9` ran from exact source commit
`2533f2e1a21de7fce1902dd1189e685f7cac8375` on A10G Large and independently verified the
exact raw protocol bytes. Nominal and +50 g passed 32/32; +100 g, +150 g and +200 g each
passed 31/32 with the same single `0.36 m/s` miss. Mean forward-velocity RMSE rose
monotonically from `0.19439` nominal to `0.22403` at +200 g, but mean uprightness remained
1.0 for every profile. No payload met the frozen 16--28/32 incident band. The calibration
therefore stopped the branch before source-behavior preflight or training: 160 source-only
episodes, 40,000 requested simulator steps, zero retries and zero optimizer evaluations.
The in-job and independent validation files are byte-identical with SHA-256
`a60ed1d4c8437c805bda6ae41b5aaab38439a9af27c62f1f7493d8de7c6d8eb5`. The complete
summary, immutable repository identities, costs and two zero-simulator infrastructure
attempts are recorded in
[`walking_trunk_payload_calibration_result_v1.json`](walking_trunk_payload_calibration_result_v1.json).
Extending the ladder after observing this result would convert a predeclared test into
condition hunting, so no wider grid or three-seed optimization was launched.

## Implemented workflow

### Exact interaction accounting and phase profiling

`InteractionCost` and its append-only ledger now keep candidate evaluations, world
rollouts, requested simulator steps, executed vector-slot simulator steps, behaviorally
active interaction steps, policy-forward rows, physics substeps, world constructions,
wall time and accelerator time as separate fields. New locomotion summaries include the
actual vector simulator tick count. This matters when one candidate terminates early but
the vector environment continues stepping other candidates: active steps fall while the
terminated slot still incurs simulation and inference work.

Every source baseline, training source-reference bank, candidate bank, campaign-side
selection bank and failed qualification stage is assigned a named ledger phase. Ledger
and profiler state are checkpointed. A resume request refuses a legacy checkpoint that
lacks exact interaction state rather than silently restarting its counters. Optional
synchronised phase profiling covers startup/world construction, reset, production
transport, Torch-to-JAX and JAX-to-Torch transfer, policy inference, MJLab stepping,
objective accumulation and update, campaign selection, and artifact generation. Timers
are inclusive and profiling generations should be treated as separate instrumentation
runs because per-phase synchronisation changes wall time.

### Gate-triggered qualification

The qualification controller uses a content-addressed, predeclared plan. At each declared
interval it first applies the campaign-side target and nominal gates, including consecutive
pass counts. A failed cheap screen does not call the expensive backend. A plausible
candidate must then pass, in order: release-scope retention, ONNX parity, production
runtime replay, independent confirmation, profile routing, and signed activation plus
rollback. Each passing stage requires an evidence SHA-256. Any failure is recorded and
training continues; only all six stages set the stop generation. Task return is rejected
as a selection metric. Attempts, rejection reasons, stage evidence, costs and stop state
all survive resume.

The historical and qualification-only plans retain the original five-generation cadence.
The v2 10x study plan declares a one-generation cadence. This does not relax a threshold or
shorten an episode; it pays for more frozen-bank screens so the required two consecutive
passes can occur before generation 10. The cadence is hashed into the qualification plan
and written into every run config.

The trainer exposes both a typed backend boundary and a content-addressed direct-command
backend. The command path never invokes a shell: every stage uses a predeclared argv and
timeout, verifies the exact candidate checkpoint hash, and must emit a candidate-bound JSON
result with its interaction cost. Command failures and timeouts produce hashed transcripts
and are billed as failed stages. The backend specification hash and resolved working
directory are bound into the run config and checkpoint, so resume refuses orchestration
drift. Two checked-in plans cover the unchanged v1 objective and the new v2 objective. No
dummy or predeclared-success backend is enabled by the CLI.

Commit `9d26a4b3b78a4fcf48b99dda2e7eec61cd27e8ec` binds the concrete production backend.
Its canonical command-spec SHA-256 is
`e896b926de8335b44d1b2801bbaa444be039021f0fa6e6fd29553ec15acb8da9`. For each exact
checkpoint it exports only the final affine derivative, independently checks ONNX parity,
runs complete paired source/adapted replays on the frozen production and confirmation banks,
audits every Rust runtime trace, proves exact-profile and fail-closed unknown-profile routing
through the signed updater engine, then proves fresh-process rollback to the exact source.
The two bank hashes are respectively
`ba760ae8dcbb6c0b5827ab8c38bcbe6c4f4a5b41bc85864c0447af24f55eff01` and
`106a0c05307852fc6c0b05c383ab658ce2c54fef7d161105cdf4ca97c983d307`.
Complete rejected banks retain their exact interaction cost; infrastructure failures remain
fatal rather than being mislabeled as candidate rejections. Source behavior is sealed by
the exact failure case set for each stage/platform/device combination rather than by an
aggregate pass count; unsupported environments and equal-count case swaps are rejected.
The HF wrapper mounts the fixed
production runtime and reference-policy volumes, verifies their identities before building,
and exports only the qualification `stop_generation` checkpoint. Local Python and Rust
integration passes, including the real signed updater engine; the exact remote composition
preflight also passes as described below.

### Release-scope-aware fitness

The new `locomotion-release-scope-lexicographic-v2` objective is a separate campaign, not
a mutation of the historical v1 definition. Within a profile-specific activation scope,
each fresh training-only case is run once through the immutable source policy and through
all matched EGGROLL candidates. Candidates are ordered by:

1. count of source successes retained;
2. count of source failures repaired;
3. worst profile-command success rate;
4. terminal stability; and
5. worst and mean uprightness, progress, velocity error and action-rate quality.

Thus one lost source success dominates any number of repairs or any shaping improvement.
The aggregation code also supports multiple explicitly declared profile groups and refuses
missing or extra retention roles. The current walking runner enables only profile-specific
mode because multi-profile mode still needs predeclared training banks for each activation
profile. Original and unknown profiles retain exact source bytes under the checked-in
release scope; the nominal adapted-policy run remains diagnostic and the existing gate is
still evaluated. Task return remains diagnostic only.

### Failure-frontier result

The adapter-level prototype content-addresses a training-only snapshot and refuses any bank
whose role or identifier is held-out, confirmation or release. Its contract requires
physics, environment, command, previous raw action, action-filter, scheduler, deployment
profile, random-number, episode-counter and objective-accumulator state. It exposes the
same even/odd candidate-pair identity used by HyperscaleES EGGROLL, requires two exact
restores to match the uninterrupted continuation, and bills suffix optimisation separately
from mandatory full, from-reset selection.

The prototype passes deterministic tests against a stateful reference adapter, including
previous-action and filter dependence. It is deliberately **not enabled** in the MJLab
trainer: the installed task/runtime does not yet expose and restore a demonstrated complete
production/MJLab state bundle. Enabling it now would violate the deterministic-replay stop
condition. This is a preserved negative result, not an efficiency claim. The next step is
an actual-runtime snapshot adapter and exact replay proof; only then should suffix training
be tried, followed by full-episode selection.

## Reproducible commands

Recreate the historical accounting summary without running a simulator:

```bash
PYTHONPATH=src uv run python -m mjlab_microduck.autopatch.cli efficiency-report \
  --run-dir /private/tmp/eggroll-walking-wedge-train-20260831-v1/run \
  --campaign docs/experiments/campaigns/walking_wedge_autopatch_v1.json \
  --requested-steps-per-world 250 \
  --output /private/tmp/eggroll-autopatch-historical-summary.json
```

Replay the release-scope-aware gate and exact cumulative optimisation accounting for one
downloaded seed without running a simulator:

```bash
PYTHONPATH=src uv run python -m mjlab_microduck.autopatch.cli efficiency-report \
  --run-dir /path/to/seed/run \
  --campaign docs/experiments/campaigns/walking_wedge_autopatch_release_scope_v2.json \
  --release-scope \
    docs/experiments/release_scopes/walking_wedge_gen85_profile_specific_v1.json \
  --requested-steps-per-world 250 \
  --output /private/tmp/eggroll-autopatch-scoped-summary.json
```

The versioned accelerator training command underlying the remote launcher is:

```bash
PYTHONPATH=src uv run --extra eggroll python -m mjlab_microduck.autopatch.cli \
  train-walking-campaign \
  --campaign docs/experiments/campaigns/walking_wedge_autopatch_release_scope_v2.json \
  --release-scope docs/experiments/release_scopes/walking_wedge_gen85_profile_specific_v1.json \
  --runtime-repo ../microduck --output-dir /path/to/new-empty-run \
  --device cuda:0 --profile-generation 0 --profile-generation 1
```

For a qualified run, add both of the following. The command spec must contain the six plan
stages in exact order and point at real release tools that emit
`eggroll-autopatch-qualification-command-result-v1` manifests:

```bash
  --qualification-plan \
    docs/experiments/qualification_plans/walking_wedge_release_v1.json \
  --qualification-command-spec \
    docs/experiments/qualification_plans/walking_wedge_release_command_spec_v1.json
```

The CLI refuses either argument without the other. The same pair can be bundled by the HF
launcher. A passing result is hashed from the file actually emitted by the stage command;
the CLI cannot synthesize or trust placeholder release evidence.

## Ten-times budget envelope and accelerator evidence

The v2 objective evaluates the immutable source once on each fresh four-case training bank.
That cost is part of optimisation rather than free metadata. Nine generations therefore
have the following predeclared per-seed ceiling before qualification:

| Phase | Candidate evaluations | World rollouts | Requested simulator steps |
| --- | ---: | ---: | ---: |
| EGGROLL candidates | 4,608 | 18,432 | 4,608,000 |
| Training-only source reference | 0 | 36 | 9,000 |
| Source baseline | 0 | 64 | 16,000 |
| Selection every generation | 0 | 576 | 144,000 |
| Total before qualification | 4,608 | 19,108 | 4,777,000 |

The optimisation subtotal is 4,617,000 requested steps, leaving 503,000 steps below the
5.12 million optimisation target. Training is capped at generation 9; it is a failed seed
unless a checkpoint completes the unchanged release gate. This is why the study does not
run generation 10 merely to obtain two screens: candidate plus training-source optimisation
would then total 5.13 million requested steps.

Three immutable campaign records predeclare seeds 20262031, 20272038 and 20282045, each at
population 512, rank 4 and nine generations. Their canonical SHA-256 values are respectively
`8856e98b5539f559f2426fc0a670db3f1bf8701363883091c6f1217ea41df73f`,
`ecf00adc513a794728a438be3e560b1b8ea0864ac526152d153f51003eefcea0`, and
`f52d18e4ca1c95b83e32c7bf80d708c69de55a7b4dbb530acf1ec339ef1fe273`.
The release plan SHA-256 is
`8d7758dbd02c5e8148cb05261943e4b5a35eb4a1db09829cd3f196b6fbd52a41` and the unchanged
release scope is
`4435a7837cedc7f480e961d2f43d5abc3800eeb5d8356aed1b1bc27a1dbcfd00`.

One explicitly authorised one-generation, population-16 A10G Large smoke was launched on
2026-09-03 from exact clean commit
`5add050ee62d746a8524bae00d4d6325a974aece`. Its derived campaign hash is
`8a4950e35507eff807631b0c49f5af8be293aa5cd07a14304865fad319fc5a7b` and its Hugging Face
job ID is `6a98ee4b21c5aa7c8364f5fc`. The actual NVIDIA A10G passed the Torch, JAX and zero-copy
DLPack preflight. The trainer completed one generation with 16 distinct fitness values and
a non-zero `0.1274895711` parameter-delta norm. Its ledger exactly records 16 candidates,
68 optimisation rollouts, 17,000 requested optimisation steps, 196 total rollouts and
49,000 total requested steps.

The provider job nevertheless ended `ERROR`: the original wrapper required
`select-export` to find a release-retained candidate after every run. The tiny smoke passed
nominal retention but, correctly, did not pass release-scope source-success retention, so
the trainer published only its valid generation checkpoint. Requiring a release candidate
made the non-evidence smoke test the wrong boundary; it did not indicate a CUDA, EGGROLL,
accounting or checkpoint failure. The content-addressed post-hoc verifier passes the
uploaded artifacts. The corrective implementation culminates in commit
`21be49798ee5a6016140d6a00c846c3bc5ccd590`: smoke mode now verifies CUDA, diverse finite
fitness, non-zero update, exact accounting and checkpoint integrity, whereas full train
mode still requires a retained candidate and export. Historical v1 scope omission and v2
scope enforcement remain distinct.

The corrected retest then ran from exact clean commit
`21be49798ee5a6016140d6a00c846c3bc5ccd590` as Hugging Face job
`6a99099a21c5aa7c8364f815`. The provider finished `COMPLETED`, the in-job validation passed,
and an independent post-hoc replay produced byte-identical validation SHA-256
`f5cc051a0ed8c2d8b98bad099c72fa37e12399f78ce2e181287d0605e69e0ffd`.
It preserved the same 16 candidates, 68 optimisation rollouts, 17,000 requested
optimisation steps, 196 total rollouts and 49,000 total requested steps. The checkpoint SHA
is `6baf311698bafef9340c14b4cd7e867807fbdb58dc9fbd94b245b2b068991130`.
The smoke remains non-evidence for repair quality: shifted source-success retention was
80%, so it correctly produced no release-retained candidate.

The first smoke trainer process took 930.9 seconds, of which the generation record attributes
66.7 seconds to evaluation/update. The 864.2-second residual includes initial CUDA
compilation, 85 world constructions, baseline and selection. The original smoke did not
enable phase profiling, so it would be false precision to divide that residual further.
Its complete immutable record is
[`eggroll_autopatch_efficiency_remote_smoke_20260903_v1.json`](eggroll_autopatch_efficiency_remote_smoke_20260903_v1.json).

The corrected smoke trainer process took 901.5 seconds and enabled synchronized profiling.
Inclusive phase time was dominated by MJLab environment stepping (653.4 seconds), the
fixed campaign-side selection evaluation (352.0 seconds), and startup identity-world
construction (121.9 seconds). Policy inference took 11.5 seconds, Torch-to-JAX transfer
6.6 seconds, JAX-to-Torch transfer 2.9 seconds, and objective aggregation plus the EGGROLL
update 3.1 seconds. These phase measurements are nested; their 1,236.5-second sum is not
elapsed time. The immutable corrected record is
[`eggroll_autopatch_efficiency_remote_smoke_20260903_v2.json`](eggroll_autopatch_efficiency_remote_smoke_20260903_v2.json).

All three full population-512 seeds completed on 2026-09-03. The initial one-hour jobs
`6a9949570718b0f6d8914326`, `6a99496a21c5aa7c8364fe64`, and
`6a99497d21c5aa7c8364fe66` persisted five generations before their wall-time limit. Exact
audited resumes `6a9960a20718b0f6d8914815`, `6a9960a20718b0f6d8914817`, and
`6a9960a3a88f45a8aec52017` completed generation 9 with 4,608 candidates and 4,617,000
requested optimisation steps apiece. The final exact executed optimisation counts were
4,615,892, 4,616,239 and 4,616,345; all-prequalification executed counts were 4,773,874,
4,773,922 and 4,774,489. The full-run totals prove complete accounting and resumability;
the generation-6 prefix establishes the interaction-efficiency result.

The candidate-specific production evidence backend is bound to the qualification
controller at commit `9d26a4b3b78a4fcf48b99dda2e7eec61cd27e8ec`, platform/device-specific exact source
behavior is sealed at commit `7f28ca702940d18b8e0fc66618440b63ee10f5d5`, and the
profile-specific gate interpretation is fixed at commit
`7e427c16cb8acfd09066a47cee7f76b470770e5b`.

The bounded remote composition preflight on the already-known generation-85 derivative
completed as Hugging Face job `6a993a140718b0f6d8914057` from that exact clean commit. All
six candidate-bound stages passed. The Linux CPU production bank recorded source 21/32 and
adapted 32/32; the independent Linux CUDA confirmation bank recorded source 23/32 and
adapted 32/32. Every source success was retained case-by-case and every Rust trace passed.
ONNX maximum absolute error was `1.6689300537109375e-6`, below the exclusive `1e-5` limit.
The real signed updater selected the exact adapted bytes for the attested profile, retained
the exact source bytes for an unknown profile, and restored the exact source bytes during
fresh-process rollback. This validates the remote environment and orchestration only; the
historical policy is not a result of the new efficiency campaign. The immutable record is
[`eggroll_autopatch_qualification_preflight_20260903_v3.json`](eggroll_autopatch_qualification_preflight_20260903_v3.json).

Two negative preflights are retained rather than hidden. Job `6a991d8421c5aa7c8364f9c9`
exposed that integral episode-step counts arrive as JSON floats. Job
`6a99307621c5aa7c8364fb69` then exposed that the production source aggregate differs between
macOS CPU and Linux CPU, even though the adapted policy still passed 32/32. The fixes accept
only finite integral real counts and bind exact source-failure case sets by platform/device;
the adapted and casewise-retention gates remain unchanged.

The user granted standing authority on 2026-09-03 to start and manage Hugging Face jobs and
explicitly authorized uploading the local policy, source bundle and qualification artifacts
to the named private Hugging Face repositories.
The exact limits, artifact destinations and stop conditions are recorded in
[`eggroll_autopatch_efficiency_remote_plan_v1.json`](eggroll_autopatch_efficiency_remote_plan_v1.json).
The Hugging Face launcher now bundles and hashes the release scope and, when supplied, the
qualification plan and command spec alongside the source policy and campaign. It refuses
dirty source trees, mismatched source identities, incomplete qualification pairs, or stage
order drift. Historical v1 campaigns still omit release scope exactly as before; v2 requires
it explicitly.

## Verification

An earlier full local suite passed with 294 tests, one architecture-specific test skipped
and one upstream deprecation warning. The current payload-focused contract, validator,
launcher, source-behavior and qualification-runtime suite passes 43/43 using an import-only
Torch stub to avoid the local Mac's unrelated PyTorch import hang; the real Torch, CUDA,
MJLab and production Rust paths were exercised by the completed A10G calibration. Focused
Ruff lint and formatting checks pass across
the Autopatch source, launcher and tests, and `git diff --check` passes. The whole-repository
Ruff audit still reports 180 lint findings and 57 files that would be reformatted outside
the clean focused scope; those pre-existing or unrelated files were not mass-rewritten.
The three successful qualification downloads contain 993, 993 and 998 declared evidence
files respectively, and every declared SHA-256 was verified against the downloaded bytes.
The candidate checkpoint hashes also match the frozen training artifacts. No generated
checkpoint, remote evidence tree, cache or credential is included in the Git changes.

## Results table

| Configuration | Seeds | Release-eligible seeds | Requested optimisation steps | Executed optimisation steps | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen reference | 1 | 1 historical patch; routing gate still external | 51,200,000 | unavailable | completed historical run |
| Qualification only | same historical seed, retrospective | 0 newly qualified | 28,160,000 to first cheap trigger; 43,520,000 to only independently replayed checkpoint | unavailable | controller validated locally; no new accelerator run |
| A10G CUDA invariant smoke, original wrapper | 1 non-evidence smoke | not applicable; release retention failed | 17,000 optimisation; 49,000 including baseline and selection | 16,786 optimisation; 45,452 total vector-slot steps | trainer complete and invariants pass; original wrapper exited 1 |
| A10G CUDA invariant smoke, corrected profiled wrapper | 1 non-evidence smoke | not applicable; release retention failed | 17,000 optimisation; 49,000 including baseline and selection | 16,786 optimisation; 44,780 total vector-slot steps | provider completed; in-job and post-hoc validation pass |
| Release-scope fitness + qualification, generation-6 prefix | 3 | 3 | 3,078,000 each; median 3,078,000 | 3,077,101; 3,077,239; 3,077,345 | strong success; all six stages pass for all seeds |
| Same result including baseline, per-generation selection and all sequential qualification attempts | 3 | 3 | 3,254,000; 3,286,000; 3,254,000 | optimisation and qualification separately exact; generation-6 cumulative selection execution not separately checkpointed | median all-in requested cost 3,254,000, or 15.734x below reference |
| Actual completed nine-generation training records | 3 | post-hoc first eligible generation 6 | 4,617,000 optimisation each; 4,777,000 prequalification each | 4,615,892; 4,616,239; 4,616,345 optimisation | jobs ran through 9 because the integrated cheap gate received the corrected profile-specific semantics only afterward |
| Integrated first-eligible-stop control flow, new seed | 1 | 1 | 2,565,000 optimisation; 2,709,000 including all qualification attempts | 2,564,376 optimisation; 2,702,639 including qualification | stopped inside the campaign at generation 5; all four earlier candidates recorded and rejected |
| Orthogonal trunk-CoM source calibration | source-only | not applicable | 40,000; zero optimisation | complete runtime evidence for 160 episodes | negative: every profile 32/32, so no eligible incident and no training |
| Orthogonal trunk-payload source calibration | source-only | not applicable | 40,000; zero optimisation | complete runtime evidence for 160 episodes; zero rejected attempts | negative: 32/32 at 0 and +50 g, 31/32 at +100/+150/+200 g; no eligible incident and no training |
| Frontier branching + best preceding | 0 | 0 | not run | not run | adapter prototype passes; actual MJLab branch rejected pending complete-state replay |

## Current conclusion

The 10x evaluation-efficiency target is demonstrated in simulation and the production
runtime digital twin: 3/3 seeds are independently release eligible at a median 3.078 million
requested optimisation steps, 16.634x below the 51.2 million reference. Candidate quality
did not move backward on the declared gate: all three are 32/32 on both frozen banks, retain
every sealed source success, satisfy final-affine ONNX parity, pass every Rust trace, route
only the attested profile to the derivative, retain exact source bytes for unknown profiles,
and restore exact source bytes on rollback.

The recommended virtual-fleet workflow is now release-scope-aware fitness, a one-generation
cheap screen, ascending candidate-bound qualification through the complete unchanged gate,
and immediate stop at the first full pass. The integrated end-to-end run confirms that this
control flow works operationally without post-hoc candidate stitching.

Cross-failure generality is **not established**. Both predeclared orthogonal follow-ons
failed at source-only calibration for the scientifically benign reason that the sealed
policy was too robust: CoM scored 32/32 throughout, while payload never fell below 31/32.
The payload three-seed protocol remains a reproducible future test if an independently
motivated incident profile satisfies its frozen calibration prerequisite, but these
observed ladders must not now be extended or tuned to manufacture one. Frontier branching
stays off until complete MJLab state capture and deterministic restore are proven.
Physical-robot validation remains blocked and no simulation result here should be used as
a deployment, transfer, fleet or general optimizer-superiority claim.
