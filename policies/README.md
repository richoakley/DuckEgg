# DuckEgg policy derivatives

These ONNX files are the selected derivatives used by the public simulation evidence.
They preserve MicroDuck's production `obs[1,61] -> actions[1,14]` interface and modify
only the source policy's final affine layer.

| File | Source | Derivative SHA-256 | Scope | Reported result |
| --- | --- | --- | --- | --- |
| `alpha_stand_lag16_eggroll_v1.onnx` | `microduck/example_policies/alpha_stand.onnx` | `bd2bfbb22d7a0942a2f83e3164c035fa887580e29e66eb28c9a809e9ed8be8db` | low-voltage, sag-0.2, lag-16 plus nominal retention | shifted 17/32 → 32/32; nominal 32/32 → 32/32 |
| `alpha_walking_wedge15_eggroll_gen85.onnx` | `microduck/example_policies/alpha_walking.onnx` | `f6c2378b415cbf5449e21d1bd4f1c1df72ac7817e8d4f39caa943c8c22221b5c` | attested 15° replacement-foot profile only | two sealed banks: 47/64 → 64/64; 17 failures repaired, 0 source successes lost |

The walking derivative is a profile-specific DuckEgg release: matching wedge-foot robots
receive the derivative, while original-foot and unknown robots retain the exact source
bytes. Its original-foot playback was 31/32 versus source 32/32, so it is not a universal
multi-profile replacement. The two-bank in-profile behavioral gate has passed; production
routing attestation remains the release gate.

Physical A/B validation is the next milestone for both simulation-proven derivatives.
