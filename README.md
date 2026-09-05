# DuckEgg

**Deployment-time learning for robot policies, powered by EGGROLL.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/c2f7c245-8217-46a1-8d1e-e0ba967cd969" alt="The MicroDuck bipedal robot" width="820">
</p>

> [!IMPORTANT]
> DuckEgg is an independent extension built on Pollen Robotics' original
> [`microduck`](https://github.com/pollen-robotics/microduck) robot runtime and
> [`microduck_rl`](https://github.com/pollen-robotics/microduck_rl) policy-training
> repositories. Their complete source is preserved in this monorepo under
> [`microduck/`](microduck/) and [`microduck_rl/`](microduck_rl/), with the EGGROLL
> integration layered on top. Both upstream projects are Apache-2.0 licensed. See
> [PROVENANCE.md](PROVENANCE.md) for the exact source revisions and modification boundary.

**Can a deployed robot policy be repaired against the outcome that actually matters,
without gradients, without changing its interface, and without rerunning its original
training pipeline?**

This repository is a working answer. **DuckEgg** takes an existing PPO policy as a sealed
ONNX artifact, evaluates narrow policy derivatives through the robot's real Rust control
loop inside its registered MuJoCo environment, and produces a validated, reversible
policy update against explicit behavioral outcomes.

The result goes well beyond a higher simulation reward. In two controlled MicroDuck
experiments, DuckEgg optimized hard, non-differentiable deployment outcomes:

- a standing policy recovered under hidden power and actuator-delay conditions, improving
  stable terminal success from **17/32 to 32/32** while retaining **32/32** nominal success;
- a walking policy adapted to a simulated **15° replacement-foot geometry**, improving
  terminal success from **47/64 to 64/64 across two independent sealed banks**, repairing
  all 17 observed source failures while preserving every source success through
  MicroDuck's production Rust runtime.

Both derivatives preserve the production `obs[1,61] -> actions[1,14]` ONNX contract and
change only the final affine layer: **1,806 of 197,774 parameters (0.91%)**.

The walking workflow now also qualifies and stops candidates against the complete release
gate instead of assuming a fixed 100-generation search. Three independent seeds reached
eligibility at generation 6 with a median **3.078 million requested optimization steps**,
6.012% of the frozen 51.2-million-step reference. A separate integrated campaign stopped
inside training at its first full pass, generation 5, after **2.565 million requested
optimization steps**, while preserving the behavioral, runtime, routing and rollback gates.

## Watch the repairs

Click either image to play the evidence-bound side-by-side simulation. The left panel is
the unmodified production PPO policy; the right panel is the DuckEgg derivative. These are
matched rollouts in the actual registered MicroDuck task, not a hand-authored animation.

### StandUp: hidden voltage, sag and actuator delay

[![Production PPO falls and remains down while DuckEgg recovers and finishes upright](assets/standup-comparison.gif)](microduck_rl/docs/assets/eggroll_posttraining/eggroll_posttraining_hero_v1.mp4)

The policy is not told the hidden condition. It must recover from standing, sitting,
face-down and face-up starts and remain height- and orientation-qualified through the
terminal window. The inline sequence opens on the terminal contrast, then replays the
matched case: both policies start upright and fall; the PPO policy remains down, while
DuckEgg recovers and finishes upright.

| Held-out profile | Production PPO | EGGROLL derivative |
| --- | ---: | ---: |
| 6.5 V, voltage sag 0.2, 16-step delay | 17/32 | **32/32** |
| Nominal | **32/32** | **32/32** |

Every reset category improved under the shifted condition:

| Initial pose | Production PPO | EGGROLL derivative |
| --- | ---: | ---: |
| Standing | 3/8 | **8/8** |
| Sitting | 2/8 | **8/8** |
| Face-down | 5/8 | **8/8** |
| Face-up | 7/8 | **8/8** |

Mean task return fell from 35.94 to 33.67 while terminal success rose from 17/32
to 32/32. That matters: selection was driven by the explicit deployment acceptance test,
not by finding a policy that gamed the environment's shaped reward.

### Walking: a changed foot geometry

[![Original PPO walking policy falls with a 15-degree replacement foot while DuckEgg completes the matched episode](assets/walking-wedge-comparison.gif)](microduck_rl/docs/assets/eggroll_autopatch/walking_wedge_gen85_hero.mp4)

The second proof moved beyond episodic recovery into continuous locomotion. A hidden,
symmetric 15° wedge was applied to the replacement feet. Evaluation used four forward
commands—0.28, 0.32, 0.36 and 0.40 m/s—with matched world identities and identical case
seeds for the source and derivative. The video above is a new-bank 0.40 m/s case: the PPO
rollout terminates when the robot falls, while DuckEgg remains upright and completes the
full five-second episode under the same command, hardware profile and seed.

| 15° replacement-foot evaluation | Production PPO | DuckEgg | Failures repaired | Source successes lost |
| --- | ---: | ---: | ---: | ---: |
| First sealed bank | 24/32 | **32/32** | 8 | **0** |
| Independent confirmation bank | 23/32 | **32/32** | 9 | **0** |
| **Combined** | **47/64** | **64/64** | **17** | **0** |

The confirmation bank used 32 new, seed-disjoint cases. DuckEgg scored 8/8 at every speed
on both banks. Across the combined evidence, source success by speed was 12/16, 12/16,
14/16 and 9/16; DuckEgg achieved **16/16 at all four speeds**. All 128 accepted in-profile
source/adapted episodes passed independent production-runtime trace audits.

This establishes **zero observed in-profile regression across two independent sealed
banks**: DuckEgg retained all 47 cases the source passed and repaired all 17 cases it
failed. This is a finite behavioral result under the declared 15° replacement-foot
profile, not a claim about every possible robot state.

The original-foot cross-profile diagnostic was source 32/32 versus derivative 31/32. That
demonstrates why DuckEgg releases are deployment-scoped: a wedge-foot robot receives the
wedge-foot derivative, while an original-foot or unknown robot keeps the exact source
policy. DuckEgg turns policy specialization into an explicit, testable routing decision
rather than asking one network to compromise across every possible hardware configuration.

## What DuckEgg has proven

The result is a concrete product proof:

> Starting with an existing production-format robot policy, DuckEgg can perform
> RL-like, closed-loop post-training against a non-differentiable behavioral objective,
> materially repair a defined simulated deployment failure, preserve the policy's runtime
> interface, and produce a scoped, testable and reversible policy derivative.

The important product idea is larger than either experiment. DuckEgg is a
**policy-maintenance and release layer** for embodied intelligence:

```text
deployed policy + measured failure + acceptance test + retention constraints
                                      │
                                      ▼
                         forward evaluations only
                                      │
                                      ▼
                      structured low-rank EGGROLL search
                                      │
                                      ▼
                   paired production-runtime validation
                                      │
                                      ▼
                 scoped derivative + evidence + rollback
```

The base policy can come from PPO, imitation learning, a foundation model or another
training pipeline. DuckEgg complements that stack with a way to improve the deployed
artifact against tests that may be binary, delayed, discontinuous, hardware-in-the-loop,
proprietary or otherwise non-differentiable.

## Why this matters

Robot policies are trained against a model of the world and deployed into the world
itself. Actuator latency, battery state, payload, wear, manufacturing variation,
calibration, surface properties and hardware revisions can all move a robot outside its
training distribution. Sim-to-real locomotion work explicitly identifies actuator
dynamics and latency as important transfer gaps, while rapid motor adaptation research
targets changing payloads, wear and terrain
([Tan et al., 2018](https://arxiv.org/abs/1804.10332),
[Kumar et al., 2021](https://www.roboticsproceedings.org/rss17/p011.html)).

The usual response—anticipate every variation during initial training—is necessary but
incomplete. It requires developers to predict the failure, encode it in a training
distribution, preserve the original trainer and simulator, and choose a differentiable or
dense reward before deployment. Real acceptance tests are often much simpler and harder:
did the robot finish, stay upright, satisfy the safety verifier, meet an energy budget or
pass a customer's test?

EGGROLL closes the loop between **what can be evaluated** and **what can improve a
policy**. It needs scores from forward execution, not gradients through the simulator,
robot, runtime or acceptance test. Its structured perturbations are designed to retain
batched-inference efficiency; the EGGROLL authors report up to 91% of pure batch-inference
throughput and substantial scaling advantages over naive evolution strategies
([Evolution Strategies at the Hyperscale](https://eshyperscale.github.io/)).

## How DuckEgg works

### 1. Seal the deployed artifact

The source ONNX file, observation and action dimensions, runtime slot, registered task and
SHA-256 identity become an immutable `PolicyArtifactSpec`. The source is never inferred
from a filename.

### 2. Define capability and deployment contracts

A `CapabilitySpec` defines commands, resets, lifecycle and behavioral success. A
`DeploymentCondition` defines one reversible actuator, sensor, runtime, terrain, object or
model mutation. A `PatchCampaign` binds those contracts to disjoint optimization,
selection and held-out banks.

### 3. Calibrate before searching

The source policy is evaluated first. Calibration selects the useful frontier: conditions
that expose a real, partial capability failure while leaving enough behavior for search to
improve. Irrelevant and catastrophic conditions are filtered out before compute is spent.
Predeclared trunk center-of-mass and payload ladders both failed this prerequisite and
therefore stopped at source-only calibration with zero optimizer evaluations.

### 4. Search a narrow parameter surface

The demonstrations freeze the observation normalizer, inference graph and representation.
EGGROLL perturbs low-rank factors and changes only the output weight matrix and bias.
Candidates are evaluated with forward rollouts; no policy gradient, differentiable reward
or PPO update is used. The current walking objective is release-scope-aware: retaining a
source success outranks repairing an additional failure, and shaping cannot compensate for
behavioral failure.

### 5. Select on behavior, not task return

Selection is lexicographic and terminal-success-first. Stable hold, minimum performance
per command and nominal retention are hard gates. Task return remains a diagnostic;
behavioral success decides which policy advances. Plausible candidates proceed through
candidate-bound qualification, and training stops only after the complete six-stage
release gate passes.

### 6. Replay through the production control loop

Python supplies raw MuJoCo sensors and independent task diagnostics. The real Rust
`robotd` remains authoritative for:

- the exact 61D observation and command scheduler;
- previous raw action state;
- ONNX Runtime inference and policy identity;
- home offsets and mode-specific action scaling;
- head and leg target filters;
- safety clamps; and
- the absolute 14D targets written through the `RobotIo` seam at 50 Hz.

Every accepted playback episode carries a trace audit. A missing write, mismatched policy
hash or hidden-stage divergence is a rejected transport attempt, never a policy result.

### 7. Build a scoped release envelope

A derivative is eligible only for declared deployment profiles. Profile-specific releases
must retain every source success across at least two disjoint paired banks for the profile
they target. Unknown profiles fail closed to the exact source bytes. Multi-profile releases
must pass the same case-by-case retention rule on every declared profile.

### 8. Package, activate and roll back

The Rust updater packages independently versioned model components, verifies evidence and
policy identity, health-gates activation, and persists the exact previous source version
for rollback.

## Inside DuckEgg

### DuckEgg's policy engine

[`microduck_rl/src/mjlab_microduck/autopatch/`](microduck_rl/src/mjlab_microduck/autopatch/)
contains DuckEgg's current policy-agnostic implementation:

- a sealed inventory of all nine production policies;
- artifact, capability, condition, campaign and release-scope contracts;
- deterministic source-fleet and generic paired A/B evaluators;
- registered-task diagnostics and exact seed-bank binding;
- requested and executed candidate, rollout and simulator-step accounting;
- resumable, gate-triggered candidate qualification and first-eligible stopping;
- non-pickle, campaign-bound checkpoint envelopes;
- output-layer-only ONNX export with numerical parity gates;
- capability-node and scheduler-edge coverage;
- case-by-case non-regression verification; and
- release-envelope construction that fails closed without routing evidence.

The implementation namespace and CLI remain `autopatch` and `eggroll-autopatch` for now.
The older StandUp-specific wrappers reproduce the first release; new policies use the
generic DuckEgg engine.

### Production Rust integration

[`microduck/`](microduck/) contains the robot software and the integration used by the
proof:

- `robotd --sim-eval` runs the real scheduler and control loop against simulated
  `RobotIo`;
- an evaluation-only policy trace exposes otherwise hidden control stages for independent
  auditing;
- policy loading checks the `61 -> 14` interface and exact ONNX hash;
- `xtask` packages evidence-bound model components;
- the updater supports signed, health-gated activation and exact rollback; and
- unknown or ineligible deployment profiles retain the source-policy route.

Physical and fake-device startup behavior remain unchanged and separate from the
evaluation-only path.

### One contract across a policy fleet

The source-acceptance suite covers all nine production artifacts and eight scheduler
transitions:

| Policy | Capability | Runtime role |
| --- | --- | --- |
| `alpha_walking.onnx` | legged locomotion | walk |
| `alpha_stand.onnx` | recovery and body pose | stand |
| `alpha_sitstand.onnx` | sit/rise transition | sitstand |
| `alpha_ground_pick.onnx` | ground pick | ground_pick |
| `ball_kick_left.onnx` | left kick | kick_left |
| `ball_kick_right.onnx` | right kick | kick_right |
| `roller.onnx` | roller locomotion | walk in roller mode |
| `roller_crouch.onnx` | roller crouch | ground_pick in roller mode |
| `roulade.onnx` | forward roll | roulade |

The deterministic fleet check passed **9/9 capability nodes and 8/8 transition edges**
through the production-runtime trace. That establishes one common DuckEgg evaluation and
release contract across the full policy fleet. StandUp and walking are the first two
adapted policy classes; the remaining seven are ready-made expansion targets.

## Evidence and reproducibility

| Proof | Evidence role | Candidate evaluations | Requested optimization steps | Timing evidence | Result |
| --- | --- | ---: | ---: | --- | --- |
| StandUp lag/power | historical fixed search | 51,200 | 61,440,000 | ~5.5 h reported, one A10G Large | 17/32 → **32/32**, nominal **32/32** |
| Walking wedge foot | frozen historical reference | 51,200 | 51,200,000 | 13,698.7 s campaign, one A10G Large | **47/64 → 64/64** across two sealed banks |
| Walking release-aware, three seeds | exact cost to first eligibility | 3,072 per seed | 3,078,000 per seed | no early-stop wall-time claim; jobs completed generation 9 | **3/3** eligible at generation 6 |
| Walking release-aware, integrated | actual first-eligible stop | 2,560 | 2,565,000, plus 144,000 qualification | 4,840.9 s qualification only; end-to-end not retained | stopped at generation 5; complete gate passed |

Against DuckEgg's own frozen walking reference, the three-seed median uses 6.012% of the
requested optimization interactions, a 16.634x improvement; the separate integrated run
uses 5.010%, a 19.961x improvement. The original 3.8- and 5.5-hour figures are historical,
not current time-to-patch estimates. The integrated record retains a qualification-only
wall time, so it would be misleading to present that value as an end-to-end replacement.
No retraining or alternative-optimizer baseline was run, and no such comparative claim is
made.

Evidence entry points:

- [machine-readable public evidence index](evidence/results.json);
- [complete walking interaction-accounting study](microduck_rl/docs/experiments/eggroll_autopatch_efficiency_v1.md);
- [three-seed first-eligibility record](microduck_rl/docs/experiments/eggroll_autopatch_efficiency_multiseed_20260903_v1.json);
- [integrated first-eligible-stop record](microduck_rl/docs/experiments/eggroll_autopatch_efficiency_seed4_integrated_20260904_v8.json);
- [walking proof and supersession record](microduck_rl/docs/experiments/eggroll_autopatch_findings.md);
- [walking machine-readable experiment record](microduck_rl/docs/experiments/eggroll_walking_autopatch_2026-09.artifacts.json);
- [walking two-bank non-regression decision](microduck_rl/docs/experiments/walking_wedge_gen85_two_bank_non_regression_v2.json);
- [StandUp experiment record](microduck_rl/docs/experiments/eggroll_posttraining_2026-08.md);
- [StandUp release artifact and exact verification](microduck_rl/policies/eggroll_posttraining/alpha_stand_lag16_v1/README.md); and
- [production deployment boundary](microduck/docs/robot/eggroll-policy.md).

The releasable example derivatives are in [`policies/`](policies/). Production PPO source
policies are in [`microduck/example_policies/`](microduck/example_policies/).

## Repository map

```text
.
├── README.md                         # product claim, evidence and operating model
├── PROVENANCE.md                     # upstream revisions and modification boundary
├── evidence/results.json             # compact public claim/evidence index
├── policies/                         # EGGROLL-derived ONNX examples and scope notes
├── assets/                           # README posters linked to evidence videos
├── microduck_rl/                     # MuJoCo tasks, PPO stack and DuckEgg engine
│   ├── src/mjlab_microduck/autopatch # current DuckEgg implementation namespace
│   ├── src/mjlab_microduck/tasks     # registered robot tasks and behavioral semantics
│   ├── scripts/                      # export, playback and Hugging Face job tools
│   ├── tests/                        # environment and DuckEgg regression tests
│   ├── policies/                     # evidence-bound StandUp reference release
│   └── docs/                         # product contracts and experiment records
└── microduck/                        # production Rust robot runtime
    ├── robotd/                       # 50 Hz scheduler, policy loop and sim-eval seam
    ├── duck-control/                 # observations, actions, filters and safety
    ├── updater/                      # signed activation, health gates and rollback
    ├── xtask/                        # evidence-bound model packaging
    ├── example_policies/             # nine production PPO source policies
    └── docs/robot/eggroll-policy.md  # deployment contract
```

Local runs, downloaded private evidence, caches and agent working state belong in ignored
`.local/`, `.scratch/` or `.codex/` directories. They are intentionally absent from the
public source snapshot.

## Quick start

### Inspect and test DuckEgg

Requirements: Python 3.12, [`uv`](https://docs.astral.sh/uv/) and, for actual-task
evaluation or search, a CUDA GPU.

```bash
cd microduck_rl
uv sync --extra eggroll
uv run eggroll-autopatch registry
uv run --with pytest pytest tests/
```

`HyperscaleES` is not vendored. The exact EGGROLL implementation is pinned by commit in
`microduck_rl/pyproject.toml` and is fetched by `uv`; the unrelated local Hyperscale
checkout used during development is deliberately excluded. That optional dependency is
licensed GPL-3.0-only, so the EGGROLL-enabled runtime environment is also subject to the
upstream HyperscaleES license; see [PROVENANCE.md](PROVENANCE.md).

### Build and test the robot runtime

Requirements: Rust 1.89+ stable. On macOS:

```bash
cd microduck
cargo test --workspace
cargo build --release --locked -p robotd
```

Linux also requires the system libraries documented in
[`microduck/CONTRIBUTING.md`](microduck/CONTRIBUTING.md).

### Reproduce a paired production-runtime evaluation

Build `robotd`, identify the ONNX Runtime dynamic library from the Python environment, and
run the generic evaluator against a sealed case bank:

```bash
cd microduck_rl
uv run eggroll-autopatch evaluate-ab \
  --runtime-repo ../microduck \
  --robotd ../microduck/target/release/robotd \
  --ort-dylib /absolute/path/to/libonnxruntime.so-or-dylib \
  --artifact alpha-walking \
  --adapted-policy ../policies/alpha_walking_wedge15_eggroll_gen85.onnx \
  --profile shifted=replacement-wedge-foot-pitch-15deg-v1 \
  --bank /path/to/sealed-bank.json \
  --output-dir ../.local/walking-ab
```

The committed experiment records contain the exact campaign, profile and policy hashes.
Private sealed-bank artifacts are not silently replaced with generated examples; supply
the declared bank if reproducing the published metric.

## Next steps

DuckEgg already closes the simulated deployment loop end to end: registered tasks, real
61D observation semantics, production Rust scheduling, ONNX loading, filters, safety,
simulated `RobotIo` writes, identity-bound traces, signed profile routing, activation and
rollback. Gate-aware stopping has reduced the walking benchmark's requested optimization
interactions by more than 10x. The next milestones turn that working digital-twin system
into a broadly deployable product:

1. **Run the first physical A/B.** When a MicroDuck is available, replay the same source,
   failure, derivative and rollback sequence with current limits, stop conditions and
   exact policy identity visible to the operator.
2. **Turn telemetry into patches.** Connect deployment failures, customer acceptance tests
   and hardware profiles directly to sealed DuckEgg campaigns and evidence-bound model
   releases.
3. **Expand across the policy fleet.** Apply the same engine to sit/stand, ground pick,
   kicks, roller locomotion, roller crouch and roulade. The shared 61D/14D interface and
   9/9 source acceptance suite already provide the common foundation.
4. **Measure operational latency.** Capture comparable end-to-end campaign wall time and
   accelerator cost, then reduce qualification and orchestration overhead without weakening
   held-out validation, runtime parity or non-regression gates.
5. **Test another externally motivated incident.** The predeclared center-of-mass and
   payload ladders found a robust source policy rather than a repairable failure; use a real
   independent incident to test cross-failure generality instead of tuning those ladders
   after seeing the result.
6. **Generalize beyond MicroDuck.** Demonstrate the same forward-only improvement loop on
   another robot, policy architecture and non-differentiable acceptance test.

The ambition is a system that keeps a fleet's policies improving as hardware, operating
conditions and customer requirements change—without reopening every original training
project.

## License and attribution

DuckEgg's committed source and both upstream projects are licensed under Apache License
2.0, except for third-party components that retain their own notices. The upstream
`microduck` and `microduck_rl` license files are preserved inside their directories, and
source attribution is recorded in [PROVENANCE.md](PROVENANCE.md). In particular,
`microduck/tof/vendor/` retains its upstream license notice, and the optional, non-vendored
HyperscaleES dependency is GPL-3.0-only. MicroDuck model assets may carry additional
license terms documented by the upstream project.
