# Alpha Stand lag-16 EGGROLL derivative

This directory is the canonical, evidence-bound release of generation 100 from
the August 2026 output-layer post-training experiment. It is deliberately small:
the deployable ONNX, a machine-readable manifest, the production-runtime probe,
the four complete held-out summary payloads, and an unsigned MicroDuck updater
bundle. The 128 source videos remain in the private artifact repositories;
the deterministic hero renderer consumes those without treating a chosen clip as
the result.

Verify the release locally:

```bash
uv run --extra eggroll python scripts/eggroll_release.py verify \
  --manifest policies/eggroll_posttraining/alpha_stand_lag16_v1/manifest.json \
  --source-policy ../microduck/example_policies/alpha_stand.onnx
```

The verifier checks the exact policy and evidence hashes, output-layer-only tensor
change, independent ONNX parity, paired evaluation banks, terminal-success release
gate, nominal retention, and the separation of optimization, checkpoint-selection,
nominal-retention, and final held-out episodes. It also verifies that the exact ONNX
passed the real `duck_control::policy::Policy::load` path with the 61D/14D model API.

`microduck_updater/` is the deterministic `model-stand` package produced by the
actual MicroDuck runtime's `cargo xtask package-model` command. It contains the
policy, this evidence manifest, runtime verification, and version metadata. The
directory is intentionally unsigned: release and developer signing keys must never
live in a source tree. Sign it with MicroDuck's normal `cargo xtask sign` workflow
before sideloading it through `robotctl update apply model-stand --from ...`; the
runtime repository's `docs/robot/eggroll-policy.md` covers setup, identity checks,
selection, and rollback.

`rollback/` contains the same production-loader proof and unsigned version-`0.9.0`
updater bundle for the exact PPO source bytes. Install and verify that version before
the derivative so `robotctl update rollback model-stand` has a real previous release.

This artifact has not been deployed on a physical MicroDuck. Its `released` status
means released for a bounded hardware A/B trial, not production fleet rollout.
