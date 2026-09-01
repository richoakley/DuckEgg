# EGGROLL post-training plan and operating contract

Status: **experiment complete; go for the scoped post-training claim**.

The predeclared run and independent actual-environment replay are recorded in
[the August 2026 result](experiments/eggroll_posttraining_2026-08.md) and its
machine-readable artifact index. On a fresh paired lag-16 bank, the production
policy achieved 17/32 terminal successes and the EGGROLL-adapted policy achieved
32/32, while both achieved 32/32 under the nominal profile.

## Canonical product proof

The durable release is
[`policies/eggroll_posttraining/alpha_stand_lag16_v1`](../policies/eggroll_posttraining/alpha_stand_lag16_v1/README.md).
It includes the generation-100 ONNX, complete machine manifest, four held-out
summary payloads, the real MicroDuck production-loader verification, and the
unsigned native updater bundle. The adapted-policy SHA-256 is
`bd2bfbb22d7a0942a2f83e3164c035fa887580e29e66eb28c9a809e9ed8be8db`.

The 35-second [hero A/B playback](assets/eggroll_posttraining/eggroll_posttraining_hero_v1.mp4)
is generated deterministically from the complete 32-world paired banks. For each
shifted reset pose it selects the lowest-index source failure that the adapted
policy turns into terminal success, then shows a paired nominal success. The video
itself is therefore illustrative; the overlaid 17/32 to 32/32 and per-pose counts
come from every episode, not from the five clips.

The actual MicroDuck source tree provides `cargo xtask package-model`, the production
`Policy::load` probe, loaded-policy SHA logging, a `model-stand` updater component,
atomic selection and rollback. See its `docs/robot/eggroll-policy.md`. The source
snapshot supplied for this work has no Git metadata, so the release manifest records
content hashes for the loader, observation builder, probe, Cargo lockfile, and ONNX
Runtime rather than inventing a runtime commit. The full changed-file hash inventory and
validation record is retained in
[`microduck_runtime_integration_2026-08.json`](experiments/microduck_runtime_integration_2026-08.json).

## One-command reproduction

On a CUDA host with both repositories and a qualifying calibration artifact, the
complete training-to-release path is:

```bash
uv run --extra eggroll --no-editable python scripts/eggroll_posttrain.py workflow \
  --policy /path/to/alpha_stand.onnx \
  --config configs/eggroll_posttrain/alpha_stand_output_layer_v1.toml \
  --calibration /path/to/calibration.json \
  --release-dir runs/releases/alpha-stand-lag16-v1 \
  --runtime-repo ../microduck \
  --cargo /path/to/cargo \
  --ort-dylib /path/to/libonnxruntime.so \
  --source-commit <committed-training-source-sha> \
  --checkpoint-repository <private-checkpoint-repository> \
  --derivative-id alpha-stand-lag16-v1 \
  --model-version 1.0.0 \
  --final-seed 20260901 \
  --episodes-per-pose 8 \
  --video
```

This is an expensive command and is documentation, not authorization to rerun the
completed experiment. It produces checkpoints, generation metrics, independent
export verification, four paired actual-environment evaluations, bank hashes, a
release-or-rejection manifest, production-loader evidence, the updater bundle, and
the hero video. It refuses a selected checkpoint other than generation 100 and the
release builder rejects overlap among optimization, selection, retention, shifted
final, and nominal final scenarios.

## The ambitious claim

The demo is not “EGGROLL can train a toy policy” and it is not another attempt
to rediscover a 197,774-parameter StandUp actor from random initialization. The
claim is more relevant to deployment:

> Starting with the exact PPO policy already deployed on Microduck, EGGROLL can
> improve a behavior under an arbitrary black-box deployment evaluation without
> gradients, reward-model changes, simulator differentiation, or access to PPO
> training state, while retaining the original nominal capability and producing
> another drop-in ONNX policy.

Version 1 tests one hidden actuator condition: fixed battery voltage, voltage
sag, and command latency. These values are not included in the actor's 61D
observation, so the policy must adapt its feedback behavior rather than read a
new command or context bit.

## Why this is a clean experiment

| Contract | Implementation |
| --- | --- |
| Starting point | Exact production ONNX bytes; SHA-256 recorded |
| Actor graph | Strictly `Sub, Div, (Gemm, ELU) x3, Gemm`, opset 18, IR 8 |
| Frozen state | Observation normalizer and first three linear layers |
| Search state | Final `14 x 128` weight plus 14D bias: 1,806 parameters |
| Simulator | Registered `Mjlab-StandUp-Flat-MicroDuck` environment |
| Commands | Zero 13D twist/head/body command, as a fixed deployment scenario |
| Reset bank | Equal standing, sitting, face-down, and face-up categories |
| Candidate fairness | Complete common reset, model, sensor-noise, motor, and delay realization |
| Fitness | Terminal stable-standing lexicographic ordering; task return is telemetry |
| Model selection | Fixed shifted held-out bank plus a hard nominal-retention gate |
| Deployment | Replace only the final ONNX weight and bias; runtime parity `<1e-5` |

No task reward, environment observation, action scaling, PPO code, network
architecture, or production normalizer is changed.

## Efficiency

The adapted subspace is 1,806 parameters, **0.91%** of the 197,774-parameter
production actor. Each of 100 generations evaluated 512 antithetic candidates on
one scenario from each of four reset categories: 51,200 candidate policies and
204,800 six-second optimization episodes, or about 341 simulated robot-hours. This
work is embarrassingly parallel; it completed in roughly 5.5 wall-clock hours on the
A10 Large job, followed by independent final replay jobs.

The fixed selection bank first crossed the predeclared 52/64 robustness threshold at
generation 35 (54/64), after about 71,680 optimization episodes. It reached 64/64 at
generation 80 and generation 100 won on secondary terminal-hold metrics. These figures
are not claims of PPO-level sample efficiency. They are the cost of adapting 0.91% of
a sealed production network with forward evaluations only and no backpropagation,
reward-model access, or optimizer state from PPO.

## Experiment sequence and stop gates

### Gate 0: policy identity and numerical equivalence

Run locally; it does not require a simulator or CUDA:

```bash
uv run --extra eggroll --no-editable python scripts/eggroll_posttrain.py validate-policy \
  --policy /path/to/alpha_stand.onnx
```

The importer must report the expected graph, 1,806 trainable parameters, the
known source hash, and maximum NumPy-versus-ONNX error below `1e-5`. Any graph
or shape difference is rejected rather than guessed.

### Gate 1: measure a useful deployment gap on A10 Large

```bash
uv run --extra eggroll --no-editable python scripts/hf/eggroll_posttrain_hf.py calibrate \
  --policy /path/to/alpha_stand.onnx \
  --namespace richoakley \
  --flavor a10g-large \
  --episodes-per-pose 8 \
  --output-repo eggroll-posttrain-calibration-v1
```

The fixed ladder is nominal median hardware followed by 6.8 V/lag 12,
6.5 V/lag 14, 6.5 V/lag 16, and 6.2 V/lag 20 with non-decreasing sag. At a
5 ms physics step, the shifted points represent 60, 70, 80, and 100 ms of
motor-command latency, beyond the PPO task's 15--30 ms randomization. A local
CPU bracket using one deterministic episode per pose found 4/4 terminal
successes at 60 ms, 3/4 at 70 ms, 1/4 at 80 ms, and 0/4 at 100 ms. Those
figures are rehearsal evidence, not the result: CUDA calibration with eight
episodes per pose remains authoritative. A profile is selected only when:

- nominal terminal success is at least 75%;
- shifted success is between 10% and 90% and at least ten points below nominal;
- every reset category still has at least one terminal success; and
- among eligible conditions, it is the hardest measured profile.

The resulting `calibration.json` is bound to the production policy hash and a
minimum of eight episodes per pose. Training refuses a missing, catastrophic,
under-sampled, or different-policy artifact. If no profile qualifies, stop and
redesign the deployment condition; do not compensate with EGGROLL tuning.

The launcher also refuses a dirty Git worktree. It uploads tracked committed
source only, then embeds a manifest containing the exact commit, branch,
production-policy hash, and calibration hash. Untracked files, local runs,
caches, archives, and credentials are not silently swept into a remote bundle.

### Gate 2: one-generation end-to-end smoke

```bash
uv run --extra eggroll --no-editable python scripts/hf/eggroll_posttrain_hf.py smoke \
  --policy /path/to/alpha_stand.onnx \
  --calibration /path/to/calibration.json \
  --namespace richoakley \
  --flavor a10g-large \
  --output-repo eggroll-posttrain-smoke-v1
```

The smoke uses 16 candidates and one balanced scenario per pose. It must prove:

- Torch and JAX share CUDA through DLPack;
- reset and actor command slots are correct;
- candidate fitness is not identical;
- non-identical fitness changes the output parameters;
- checkpoint state round-trips;
- held-out and nominal evaluations finish; and
- export changes only the output layer and retains `<1e-5` runtime parity.

Any failure stops the experiment before a long job.

### Local CPU contract rehearsal

Evaluation and video playback can also run in the actual registered task on a
Mac or other CPU-only machine. This uses NumPy for the imported actor and is
deliberately labelled non-authoritative in `summary.json`; it is for reset,
observation, action, metric, and rendering reproduction, not calibration or
training:

```bash
uv run --extra eggroll --no-editable python scripts/eggroll_posttrain.py evaluate \
  --policy /path/to/alpha_stand.onnx \
  --profile calibration-6p5V-lag14 \
  --device cpu --episodes-per-pose 1 --video \
  --output-dir runs/eggroll-posttrain/cpu-source-rehearsal
```

The same command accepts an exported EGGROLL ONNX policy, giving deterministic
source/adapted A/B playback locally. `calibrate-shift` and `train` remain
CUDA-only by contract.

The 2026-08-30 local acceptance run produced one 6.02-second, 301-frame,
320x240 MP4 at 50 fps for each of standing, sitting, face-down, and face-up.
The videos and metrics agreed: standing, sitting, and face-up finished standing;
face-down visibly failed and was the sole terminal failure. The artifact was
written outside the repository under `/tmp/eggroll-cpu-video-rehearsal`.

### Gate 3: the single predeclared experiment

```bash
uv run --extra eggroll --no-editable python scripts/hf/eggroll_posttrain_hf.py train \
  --policy /path/to/alpha_stand.onnx \
  --calibration /path/to/calibration.json \
  --config configs/eggroll_posttrain/alpha_stand_output_layer_v1.toml \
  --namespace richoakley \
  --flavor a10g-large \
  --timeout 12h \
  --output-repo eggroll-posttrain-alpha-v1
```

The predeclared settings are population 512, rank 4, sigma 0.015, Adam learning
rate 0.003, one fresh balanced CRN bank per generation, and 100 generations.
Population 512 is deliberately large relative to the 1,806-parameter search
space: it provides 256 antithetic directions per generation. Rank 4 is a
meaningful low-rank perturbation for a `14 x 128` output matrix without making
each direction full-rank. These are one fixed hypothesis, not a tuning sweep.

Every five generations, the unperturbed policy is evaluated on a fixed
64-episode shifted bank and a fixed 32-episode nominal bank. A checkpoint is
eligible as `best.pkl` only if its shifted lexicographic key improves and its
nominal terminal-success rate remains within five points of the production
baseline. The registered task return never selects a checkpoint.

### Gate 4: export and A/B playback

After downloading the best checkpoint:

```bash
uv run --extra eggroll --no-editable python scripts/eggroll_posttrain.py export \
  --checkpoint best.pkl \
  --output alpha_stand_eggroll_v1.onnx
```

Record the source and adapted policies with the same profile, reset categories,
and seeds:

```bash
uv run --extra eggroll --no-editable python scripts/hf/eggroll_posttrain_hf.py evaluate \
  --policy /path/to/alpha_stand.onnx \
  --profile calibration-6p5V-lag14 --video \
  --episodes-per-pose 3 --output-repo eggroll-posttrain-source-demo

uv run --extra eggroll --no-editable python scripts/hf/eggroll_posttrain_hf.py evaluate \
  --policy /path/to/alpha_stand_eggroll_v1.onnx \
  --profile calibration-6p5V-lag14 --video \
  --episodes-per-pose 3 --output-repo eggroll-posttrain-adapted-demo
```

The actual-environment artifact contains the scenario bank, max/final trunk
height, max/final upright cosine, upright time, transient and terminal success,
stable hold, recovery time, and total registered return for every episode.
The adapted ONNX can then be loaded unchanged in `scripts/infer_policy.py` for
the familiar local viewer and, after the normal robot safety review, by the
production runtime. The actual-environment A/B is the proof; the viewer is a
deployment-interface rehearsal.

## Decision rule

Call the experiment a success only if the selected adapted checkpoint:

1. improves terminal stable-standing success on the fixed calibrated bank;
2. improves at least one initially weak reset category rather than only an easy
   aggregate;
3. retains the nominal terminal-success gate;
4. reproduces in same-seed video playback; and
5. exports as a numerically verified drop-in production graph.

A flat run is an informative no-go for this output-layer search. A nominal
regression is a failure even if shifted performance improves. No result licenses
changing the objective, rank, population, profile, or parameter scope in the
same run; each would be a separately declared experiment.

## Repository cleanup and retained evidence

The live branch intentionally does not contain the previous from-scratch
trainer, capability curricula, objective switches, analysis scripts, remote
launchers, resume checkpoints, production-policy binaries, or generated run and
evaluation directories. Their conclusions and exact provenance are consolidated
in [the August experiment record](experiments/eggroll_standup_from_scratch_2026-08.md)
and its machine-readable artifact index.

The recoverable source snapshot is:

- branch and tag: `archive/eggroll-standup-pre-posttrain-2026-08-30`;
- commit: `5cafd566f88214dbbb9c2e5d6986ab784beed40c`;
- verified bundle: `../archives/eggroll-standup-pre-posttrain-2026-08-30.bundle`;
- bundle SHA-256: `1f1c64eaeed30fe47333fe0e988407d9955a1ef2683a8c811cb83657c44dd096`.

This keeps the negative evidence accessible without presenting dead experiment
modes as supported product functionality.

## Next derivative and hardware boundary

The single best second objective is the
[predeclared asymmetric left-leg derivative](experiments/eggroll_next_objective_asymmetric_leg.md):
reduce the simulated torque authority of the left knee and ankle without exposing
that condition to the actor, then optimize terminal stable recovery with hard
per-pose and nominal-retention gates. It is designed but neither implemented nor
launched. A substantial run requires a fresh approval after calibration and smoke.

The [bounded physical A/B plan](eggroll_physical_ab.md) first proves signed source
installation, adapted activation, exact SHA identity and rollback with motors disabled,
then permits only supported nominal motion. It expressly does not authorize hardware
execution or assume that the simulated lag/sag profile maps directly to a safe physical
intervention.
