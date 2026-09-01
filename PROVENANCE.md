# Provenance and modification boundary

This repository is a source snapshot assembled for the DuckEgg demonstration. It combines
two Apache-2.0 Pollen Robotics projects with the DuckEgg runtime, policy-improvement and
public-evidence changes required to understand and reproduce the work.

## Upstream projects

| Directory | Upstream | Snapshot basis | License |
| --- | --- | --- | --- |
| `microduck/` | <https://github.com/pollen-robotics/microduck> | commit `590b986bd8c0d50ae02cb3ea2f59c463b6828168` plus the DuckEgg runtime working tree described below | Apache-2.0, with retained third-party notices |
| `microduck_rl/` | <https://github.com/pollen-robotics/microduck_rl> | EGGROLL integration commit `8b85e9d` on branch `feat/eggroll-posttraining` | Apache-2.0 |
| EGGROLL dependency | <https://github.com/ESHyperscale/HyperscaleES> | fetched by `uv` at commit `b77f7d6f91238fd575313e946b9cad21e0a74b32`; not vendored | GPL-3.0-only |

The upstream license files remain at `microduck/LICENSE` and `microduck_rl/LICENSE`.
The repository-root `LICENSE` is the same Apache License 2.0 text.

The Apache-2.0 root license applies to DuckEgg's committed source, not to separately
licensed dependencies fetched at install time. In particular, an EGGROLL-enabled
environment installed with `uv sync --extra eggroll` includes the pinned GPL-3.0-only
HyperscaleES package. The ST time-of-flight vendor code under `microduck/tof/vendor/`
retains `microduck/tof/vendor/LICENSE.txt` (BSD-3-Clause fallback when no package license
was supplied). Downstream users should preserve and comply with those notices.

## What changed in `microduck_rl`

The training repository contains DuckEgg's generic `mjlab_microduck.autopatch` engine,
registered-task evaluation, runtime trace auditing, sealed campaign and release contracts,
EGGROLL output-layer post-training, paired non-regression verification, evidence-bound
export, release packaging records, tests and product/experiment documentation.

The clean source was exported from commit `8b85e9d` so local virtual environments, logs,
checkpoints, caches, private job artifacts and the unrelated Hyperscale development
checkout are absent.

## What changed in `microduck`

The runtime snapshot begins at upstream commit
`590b986bd8c0d50ae02cb3ea2f59c463b6828168`. Its DuckEgg integration adds:

- evaluation-only `robotd --sim-eval` transport;
- production-loop policy traces for independent parity auditing;
- policy identity and observation/action contract checks;
- generic evidence-bound model packaging;
- updater activation and rollback examples; and
- deployment documentation and configuration for independently versioned policy models.

In this public snapshot, the runtime working-tree diff has SHA-256
`43630b9465c831a59ee020f00258d2462d8015ad488ca784d78d628d19100d6d`.
The public snapshot omits `.git`, Rust `target/`, OS metadata, local verification output
and signing material. Production PPO artifacts are intentionally retained under
`microduck/example_policies/` because they define the source-policy contract and make the
runtime demo self-contained.

## Public evidence versus private operational artifacts

Committed evidence includes compact metrics, policy hashes, release records, two short
comparison videos and the selected ONNX derivatives. Large raw rollouts, private Hugging
Face job repositories, complete per-tick traces, temporary CUDA build trees and signing
keys are not committed.

Some immutable verification records preserve the absolute path printed by the original
loader probe. Those strings are historical stdout, not runtime dependencies; all public
commands and source paths are repository-relative or explicit user-supplied paths.

Paths under `.local/`, `.scratch/`, `.codex/` and `evidence-private/` are ignored for
future local work. No public code path should depend on those directories.
