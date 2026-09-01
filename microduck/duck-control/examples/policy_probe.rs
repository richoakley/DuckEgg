//! Deterministic trace probe for the production observation and ONNX contract.
//!
//! Reads raw sensor fixtures as JSON Lines on stdin and emits the exact 61D
//! observation, raw policy action and absolute joint targets used by the robot,
//! one JSON result per input line.  The policy is loaded once so a full trace
//! can be checked without changing inference state between samples.
//! This is intentionally an example around the production types, not a second
//! implementation of their indexing or normalization semantics.

use std::io::{self, Read};
use std::path::PathBuf;

use duck_control::imu::ImuData;
use duck_control::model::{DEFAULT_POSITION, NUM_JOINTS};
use duck_control::obs::{ACTION_LEN, BodyPose, Command, Observation};
use duck_control::policy::{Net, Policy, PolicyPaths};
use duck_control::safety::{ACTUATOR_MAX, ACTUATOR_MIN};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Fixture {
    positions: [f64; NUM_JOINTS],
    velocities: [f64; NUM_JOINTS],
    imu: ImuDataFixture,
    #[serde(default)]
    previous_action: [f32; ACTION_LEN],
    #[serde(default)]
    twist: [f64; 3],
    #[serde(default)]
    head: [f64; 4],
    #[serde(default)]
    body: BodyFixture,
    #[serde(default = "one")]
    action_scale: f64,
}

#[derive(Deserialize)]
struct ImuDataFixture {
    gyro: [f64; 3],
    gravity: [f64; 3],
    #[serde(default = "identity_quat")]
    quat: [f64; 4],
}

#[derive(Default, Deserialize)]
struct BodyFixture {
    z: f64,
    roll: f64,
    pitch: f64,
}

#[derive(Serialize)]
struct ResultFrame {
    observation: Vec<f32>,
    action: [f32; ACTION_LEN],
    /// Controller output before the safety boundary.
    targets: [f64; NUM_JOINTS],
    /// What crosses `RobotIo` after the production actuator-range rule.
    applied_targets: [f64; NUM_JOINTS],
}

fn one() -> f64 {
    1.0
}

fn identity_quat() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let policy_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: cargo run -p duck-control --example policy_probe -- POLICY.onnx")?;
    let sequential = args.any(|arg| arg == "--sequential");
    let paths = PolicyPaths {
        walk: policy_path.clone(),
        stand: Some(policy_path),
        ..PolicyPaths::default()
    };
    let mut policy = Policy::load(&paths, 0.1)?;
    let mut last_action = [0.0f32; ACTION_LEN];
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    for line in input.lines().filter(|line| !line.trim().is_empty()) {
        let fixture: Fixture = serde_json::from_str(line)?;
        let imu = ImuData {
            gyro: fixture.imu.gyro,
            gravity: fixture.imu.gravity,
            quat: fixture.imu.quat,
        };
        let command = Command {
            twist: fixture.twist,
            head: fixture.head,
            body: BodyPose {
                z: fixture.body.z,
                roll: fixture.body.roll,
                pitch: fixture.body.pitch,
            },
        };
        let observation = Observation::build(
            &imu,
            &fixture.positions,
            &fixture.velocities,
            &DEFAULT_POSITION,
            if sequential {
                &last_action
            } else {
                &fixture.previous_action
            },
            &command,
        );
        let action = policy.infer(&observation, Net::Stand)?;
        if sequential {
            last_action = action;
        }
        let offsets = Observation::scatter_action(&action);
        let targets = std::array::from_fn(|joint| {
            DEFAULT_POSITION[joint] + fixture.action_scale * offsets[joint]
        });
        let applied_targets = targets.map(|value| value.clamp(ACTUATOR_MIN, ACTUATOR_MAX));
        let output = ResultFrame {
            observation: observation.as_slice().to_vec(),
            action,
            targets,
            applied_targets,
        };
        println!("{}", serde_json::to_string(&output)?);
    }
    Ok(())
}
