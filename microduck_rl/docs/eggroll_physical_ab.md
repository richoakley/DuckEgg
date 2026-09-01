# Bounded physical MicroDuck A/B plan

Status: **ready for review; no hardware execution authorized**

This plan validates deployment mechanics and behavior without treating simulation as a
hardware safety claim. It compares the exact production PPO standing policy with the
generation-100 EGGROLL derivative through MicroDuck's native `model-stand` updater slot.

## Required people and equipment

- one named operator at the robot and one observer owning the stop call;
- a rigid support or harness that prevents an uncontrolled fall in every direction;
- a current-limited bench supply or freshly checked battery within MicroDuck's documented
  voltage range;
- direct physical access to the power cut throughout the test;
- a clear floor area, eye protection, no bystanders, and no remote unattended execution;
- serial/journal capture plus an external camera showing the full robot and support.

Do not begin if the robot has a known mechanical fault, hot actuator, damaged cable,
unexpected backlash, unreliable bus, or missing safety support.

## Frozen identities

| Role | Version | SHA-256 |
| --- | --- | --- |
| production rollback target | `model-stand` 0.9.0 | `1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b` |
| EGGROLL derivative | `model-stand` 1.0.0 | `bd2bfbb22d7a0942a2f83e3164c035fa887580e29e66eb28c9a809e9ed8be8db` |

Both bundles must be signed through the existing MicroDuck release process. Do not accept
filenames as identity: after every transition, record `robotctl update status` and the
`stand_sha256` field from robotd's `policy loaded` log.

## Preflight with no motion

1. Verify the ONNX and both updater bundle hashes against the checked-in manifests.
2. Put the robot in its supported, motor-disabled configuration.
3. Configure `/etc/robot/robotd.toml` to use
   `/opt/robot/model/stand/current/policy.onnx` and configure the `model-stand` component
   exactly as documented in the actual MicroDuck repository.
4. Install signed source version 0.9.0. Confirm the socket health gate passes, the updater
   reports 0.9.0 current, and robotd logs the source SHA above.
5. Install signed adapted version 1.0.0 without enabling motion. Confirm version, SHA, 61D/14D
   load, normal health, bus rate, voltage, and temperatures.
6. Run `robotctl update rollback model-stand`; confirm version 0.9.0 and its SHA. Select 1.0.0,
   confirm again, then finish preflight on 0.9.0. Any ambiguity or failed health gate stops
   the test.

This proves activation and rollback before either policy is allowed to move hardware.

## Motion phases

Each episode starts from power-on/supported rest with zero velocity, head, and body-pose
commands. The operator announces policy version, SHA, initial pose, supply voltage and run
number on camera. Run the source block first, rollback/selection-check between blocks, then
the adapted block. Do not interleave versions without rechecking SHA.

### Phase 1 — supported nominal standing

- source 0.9.0: three standing starts, maximum five seconds each;
- adapted 1.0.0: three identical supported standing starts;
- support remains load-bearing enough that a bad command cannot create a fall.

Pass only if both hold a visually upright pose without violent oscillation, repeated joint
limit contact, bus faults, or current/temperature excursions. This phase tests nominal
retention and runtime semantics, not get-up ability.

### Phase 2 — supported nominal recovery

For each policy, run one sitting, one face-down and one face-up start, with the support taking
the robot's weight and limiting travel. Increase assistance only enough to prevent impact;
record that intervention. A run with operator assistance cannot count as autonomous success.

Proceed only if the derivative is no worse than source on stability and shows no new unsafe
motion. Do not recreate the simulated 80 ms lag by delaying the whole control loop.

### Phase 3 — bounded deployment shift, only after a separate go-ahead

The simulation profile combines 6.5 V, sag gain 0.2 and a 16-step actuator-command delay.
Those are model parameters, not a validated recipe for modifying the real robot. Before this
phase, measure the actual command-to-actuator latency and supply behavior, then choose the
nearest safe, reversible hardware or runtime condition. Present that mapping and obtain a
second explicit approval. Keep the harness and current limit in place.

## Immediate stop criteria

Either person calls stop, and the operator cuts motor power, on any of:

- unexpected policy SHA or updater version;
- health-gate failure, rollback, bus fault, stale sensor, or control-rate degradation;
- support taking a sudden load or the robot moving outside the bounded envelope;
- repeated oscillation, joint-limit contact, cable loading, smoke, odor, or unusual noise;
- current above the predeclared supply limit or any actuator/board temperature above its
  predeclared ceiling;
- behavior that differs materially from simulation or cannot be attributed to one named
  policy and condition;
- loss of camera/log capture, observer attention, power-cut access, or network/console access
  needed for identity confirmation.

After any stop, return to source version 0.9.0 while supported and motors disabled. Do not
retry by changing gains, action scale, filters, commands, or safety thresholds in the same
session.

## Evidence and decision

Retain for every episode: wall-clock timestamp, policy version and SHA, source/adapted bundle
manifest, robotd/updater journal, supply voltage/current trace, temperatures, initial pose,
operator interventions, external video, final pose, and stop reason. Report success by pose
and policy; keep task return out of the physical decision.

The bounded A/B is a go only if activation/rollback identity is perfect, both policies pass
supported nominal standing, and the adapted policy shows no new hazard. A claim that the
simulation improvement transfers requires a separately approved shifted-condition phase and
repeatable unassisted terminal stable standing. A visually impressive single recovery is not
that claim.
